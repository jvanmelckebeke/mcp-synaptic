"""RAG database implementation using ChromaDB."""

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
from .embeddings import EmbeddingManager


class Document:
    """Represents a document in the RAG database."""
    
    def __init__(
        self,
        id: Optional[str] = None,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.id = id or str(uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SearchResult:
    """Represents a search result from the RAG database."""
    
    def __init__(
        self,
        document: Document,
        score: float,
        distance: Optional[float] = None,
    ):
        self.document = document
        self.score = score
        self.distance = distance

    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "distance": self.distance,
        }


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
            document = Document(
                id=document_id,
                content=content,
                metadata=metadata or {}
            )

            # Generate embedding
            if self.embedding_manager:
                document.embedding = await self.embedding_manager.embed_text(content)

            # Add to ChromaDB
            if document.embedding:
                self.collection.add(
                    ids=[document.id],
                    documents=[document.content],
                    embeddings=[document.embedding],
                    metadatas=[{
                        **document.metadata,
                        "created_at": document.created_at.isoformat(),
                        "updated_at": document.updated_at.isoformat(),
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
            embedding = result["embeddings"][0] if result["embeddings"] else None

            # Extract timestamps from metadata
            created_at = datetime.fromisoformat(metadata.pop("created_at", datetime.utcnow().isoformat()))
            updated_at = datetime.fromisoformat(metadata.pop("updated_at", datetime.utcnow().isoformat()))

            document = Document(
                id=document_id,
                content=content,
                metadata=metadata,
                embedding=embedding,
                created_at=created_at,
                updated_at=updated_at,
            )

            self.logger.debug("Document retrieved", document_id=document_id)
            return document

        except Exception as e:
            self.logger.error("Failed to get document", document_id=document_id, error=str(e))
            raise RAGError(f"Failed to get document: {e}")

    async def search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
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

            # Convert to SearchResult objects
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else None
                    
                    # Convert distance to similarity score (1 - normalized_distance)
                    score = 1.0 - (distance or 0.0)
                    
                    # Filter by similarity threshold
                    if score < threshold:
                        continue

                    content = results["documents"][0][i]
                    metadata = results["metadatas"][0][i] or {}
                    
                    # Extract timestamps
                    created_at = datetime.fromisoformat(metadata.pop("created_at", datetime.utcnow().isoformat()))
                    updated_at = datetime.fromisoformat(metadata.pop("updated_at", datetime.utcnow().isoformat()))

                    document = Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        created_at=created_at,
                        updated_at=updated_at,
                    )

                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        distance=distance
                    ))

            self.logger.info(
                "Document search completed",
                query_length=len(query),
                results_count=len(search_results),
                threshold=threshold
            )

            return search_results

        except Exception as e:
            self.logger.error("Failed to search documents", error=str(e))
            raise RAGError(f"Failed to search documents: {e}")

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
            existing_doc = await self.get_document(document_id)
            if not existing_doc:
                raise DocumentNotFoundError(document_id)

            # Update fields
            updated_content = content or existing_doc.content
            updated_metadata = {**existing_doc.metadata, **(metadata or {})}
            updated_at = datetime.utcnow()

            # Generate new embedding if content changed
            embedding = existing_doc.embedding
            if content and content != existing_doc.content and self.embedding_manager:
                embedding = await self.embedding_manager.embed_text(updated_content)

            # Create updated document
            updated_doc = Document(
                id=document_id,
                content=updated_content,
                metadata=updated_metadata,
                embedding=embedding,
                created_at=existing_doc.created_at,
                updated_at=updated_at,
            )

            # Update in ChromaDB
            self.collection.update(
                ids=[document_id],
                documents=[updated_content],
                embeddings=[embedding] if embedding else None,
                metadatas=[{
                    **updated_metadata,
                    "created_at": existing_doc.created_at.isoformat(),
                    "updated_at": updated_at.isoformat(),
                }]
            )

            self.logger.info("Document updated", document_id=document_id)
            return updated_doc

        except DocumentNotFoundError:
            raise
        except Exception as e:
            self.logger.error("Failed to update document", document_id=document_id, error=str(e))
            raise RAGError(f"Failed to update document: {e}")

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the database."""
        self._ensure_initialized()

        try:
            # Check if document exists
            existing_doc = await self.get_document(document_id)
            if not existing_doc:
                return False

            # Delete from ChromaDB
            self.collection.delete(ids=[document_id])

            self.logger.info("Document deleted", document_id=document_id)
            return True

        except Exception as e:
            self.logger.error("Failed to delete document", document_id=document_id, error=str(e))
            raise RAGError(f"Failed to delete document: {e}")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        self._ensure_initialized()

        try:
            count = self.collection.count()
            
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "persist_directory": str(self.settings.CHROMADB_PERSIST_DIRECTORY),
            }

        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            raise RAGError(f"Failed to get collection stats: {e}")