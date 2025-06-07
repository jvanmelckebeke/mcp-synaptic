"""RAG database implementation using ChromaDB with new models."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from ..config.logging import LoggerMixin
from ..config.settings import Settings
from ..core.exceptions import DatabaseError, DocumentNotFoundError, RAGError
from ..models.rag import Document, DocumentSearchResult
from .embeddings import EmbeddingManager


class RAGDatabase(LoggerMixin):
    """RAG database for document storage and similarity search."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.client = None
        self.collection = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the RAG database."""
        if chromadb is None:
            raise RAGError("ChromaDB not available. Install with: pip install chromadb")

        try:
            # Create persistent directory
            self.settings.CHROMADB_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.settings.CHROMADB_PERSIST_DIRECTORY),
                settings=ChromaSettings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="synaptic_documents",
                metadata={"description": "MCP Synaptic document collection"}
            )

            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager(self.settings)
            await self.embedding_manager.initialize()

            self._initialized = True
            
            self.logger.info(
                "RAG database initialized",
                persist_dir=str(self.settings.CHROMADB_PERSIST_DIRECTORY),
                collection_count=self.collection.count()
            )

        except Exception as e:
            self.logger.error("Failed to initialize RAG database", error=str(e))
            raise RAGError(f"RAG database initialization failed: {e}")

    async def close(self) -> None:
        """Close the RAG database."""
        if self.embedding_manager:
            await self.embedding_manager.close()
        
        self._initialized = False
        self.logger.info("RAG database closed")

    def _ensure_initialized(self) -> None:
        """Ensure the database is initialized."""
        if not self._initialized:
            raise RAGError("RAG database not initialized")

    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> Document:
        """Add a document to the RAG database."""
        self._ensure_initialized()

        try:
            # Create new document with proper ID generation
            document = Document(
                id=document_id or str(uuid4()),
                content=content,
                metadata=metadata or {},
                embedding_model=self.embedding_manager.model_name if self.embedding_manager else None,
                embedding_dimension=self.embedding_manager.dimension if self.embedding_manager else None,
            )

            # Generate embedding
            embedding = None
            if self.embedding_manager:
                embedding = await self.embedding_manager.embed_text(content)

            # Add to ChromaDB
            if embedding:
                self.collection.add(
                    ids=[document.id],
                    documents=[document.content],
                    embeddings=[embedding],
                    metadatas=[{
                        **document.metadata,
                        "created_at": document.created_at.isoformat(),
                        "updated_at": document.updated_at.isoformat(),
                        "embedding_model": document.embedding_model,
                        "embedding_dimension": document.embedding_dimension,
                    }]
                )

            self.logger.info("Document added to RAG database", document_id=document.id)
            return document

        except Exception as e:
            self.logger.error("Failed to add document", error=str(e))
            raise RAGError(f"Failed to add document: {e}")

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        self._ensure_initialized()

        try:
            result = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if not result["ids"]:
                return None

            # Reconstruct document
            content = result["documents"][0]
            metadata = result["metadatas"][0] or {}
            
            # Extract system metadata
            created_at_str = metadata.pop("created_at", datetime.utcnow().isoformat())
            updated_at_str = metadata.pop("updated_at", None)
            embedding_model = metadata.pop("embedding_model", None)
            embedding_dimension = metadata.pop("embedding_dimension", None)
            
            # Parse timestamps
            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            updated_at = None
            if updated_at_str:
                updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))

            document = Document(
                id=document_id,
                content=content,
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )

            self.logger.debug("Document retrieved", document_id=document_id)
            return document

        except Exception as e:
            self.logger.error("Failed to get document", document_id=document_id, error=str(e))
            raise RAGError(f"Failed to get document: {e}")

    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Document]:
        """Update an existing document."""
        self._ensure_initialized()

        try:
            # Get existing document
            existing = await self.get_document(document_id)
            if not existing:
                return None

            # Update fields
            new_content = content if content is not None else existing.content
            new_metadata = metadata if metadata is not None else existing.metadata
            
            # Create updated document
            updated_doc = Document(
                id=document_id,
                content=new_content,
                metadata=new_metadata,
                created_at=existing.created_at,
                updated_at=datetime.utcnow(),
                embedding_model=self.embedding_manager.model_name if self.embedding_manager else existing.embedding_model,
                embedding_dimension=self.embedding_manager.dimension if self.embedding_manager else existing.embedding_dimension,
            )

            # Generate new embedding if content changed
            embedding = None
            if content is not None and self.embedding_manager:
                embedding = await self.embedding_manager.embed_text(new_content)

            # Update in ChromaDB
            self.collection.delete(ids=[document_id])
            
            if embedding:
                self.collection.add(
                    ids=[updated_doc.id],
                    documents=[updated_doc.content],
                    embeddings=[embedding],
                    metadatas=[{
                        **updated_doc.metadata,
                        "created_at": updated_doc.created_at.isoformat(),
                        "updated_at": updated_doc.updated_at.isoformat(),
                        "embedding_model": updated_doc.embedding_model,
                        "embedding_dimension": updated_doc.embedding_dimension,
                    }]
                )

            self.logger.info("Document updated", document_id=document_id)
            return updated_doc

        except Exception as e:
            self.logger.error("Failed to update document", document_id=document_id, error=str(e))
            raise RAGError(f"Failed to update document: {e}")

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        self._ensure_initialized()

        try:
            # Check if document exists
            existing = await self.get_document(document_id)
            if not existing:
                return False

            # Delete from ChromaDB
            self.collection.delete(ids=[document_id])
            
            self.logger.info("Document deleted", document_id=document_id)
            return True

        except Exception as e:
            self.logger.error("Failed to delete document", document_id=document_id, error=str(e))
            raise RAGError(f"Failed to delete document: {e}")

    async def search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentSearchResult]:
        """Search for similar documents."""
        self._ensure_initialized()

        try:
            # Generate query embedding
            if not self.embedding_manager:
                raise RAGError("Embedding manager not available")

            query_embedding = await self.embedding_manager.embed_text(query)
            
            # Use similarity threshold from settings if not provided
            threshold = similarity_threshold or self.settings.RAG_SIMILARITY_THRESHOLD
            limit = min(limit, self.settings.MAX_RAG_RESULTS)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
                where=metadata_filter
            )

            # Convert to DocumentSearchResult objects
            search_results = []
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i] or {}
                distance = results["distances"][0][i]
                
                # Calculate similarity score from distance
                similarity_score = max(0.0, 1.0 - distance)
                
                # Skip results below threshold
                if threshold and similarity_score < threshold:
                    continue
                
                # Extract system metadata
                created_at_str = metadata.pop("created_at", datetime.utcnow().isoformat())
                updated_at_str = metadata.pop("updated_at", None)
                embedding_model = metadata.pop("embedding_model", None)
                embedding_dimension = metadata.pop("embedding_dimension", None)
                
                # Parse timestamps
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                updated_at = None
                if updated_at_str:
                    updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))

                # Create document
                document = Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    created_at=created_at,
                    updated_at=updated_at,
                    embedding_model=embedding_model,
                    embedding_dimension=embedding_dimension,
                )

                # Create search result
                search_result = DocumentSearchResult(
                    item=document,
                    score=similarity_score,
                    rank=len(search_results) + 1,
                    distance=distance,
                    embedding_model=embedding_model,
                    match_type="vector"
                )
                
                search_results.append(search_result)

            self.logger.info("Document search completed", query=query, results=len(search_results))
            return search_results

        except Exception as e:
            self.logger.error("Failed to search documents", query=query, error=str(e))
            raise RAGError(f"Failed to search documents: {e}")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        self._ensure_initialized()

        try:
            # Get basic collection stats
            count = self.collection.count()
            
            if count == 0:
                return {
                    "total_documents": 0,
                    "total_embeddings": 0,
                    "embedding_dimension": None,
                    "embedding_models": [],
                    "total_content_length": 0,
                    "average_content_length": 0,
                    "total_word_count": 0,
                    "collection_size_bytes": 0,
                    "metadata_keys": [],
                }

            # Get all documents for detailed stats
            all_docs = self.collection.get(
                include=["documents", "metadatas"],
                limit=count
            )

            # Calculate statistics
            total_content_length = sum(len(doc) for doc in all_docs["documents"])
            total_word_count = sum(len(doc.split()) for doc in all_docs["documents"])
            
            # Collect metadata keys and embedding models
            metadata_keys = set()
            embedding_models = set()
            
            for metadata in all_docs["metadatas"]:
                if metadata:
                    metadata_keys.update(metadata.keys())
                    if "embedding_model" in metadata and metadata["embedding_model"]:
                        embedding_models.add(metadata["embedding_model"])

            # Get embedding dimension from first document with dimension info
            embedding_dimension = None
            for metadata in all_docs["metadatas"]:
                if metadata and "embedding_dimension" in metadata:
                    embedding_dimension = metadata["embedding_dimension"]
                    break

            return {
                "total_documents": count,
                "total_embeddings": count,  # Assuming all docs have embeddings
                "embedding_dimension": embedding_dimension,
                "embedding_models": list(embedding_models),
                "total_content_length": total_content_length,
                "average_content_length": total_content_length / count if count > 0 else 0,
                "total_word_count": total_word_count,
                "collection_size_bytes": None,  # ChromaDB doesn't provide this easily
                "metadata_keys": list(metadata_keys),
            }

        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            raise RAGError(f"Failed to get collection stats: {e}")