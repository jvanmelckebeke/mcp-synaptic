"""Document CRUD operations handler for RAG database."""

from datetime import datetime, UTC
from typing import Any, Dict, Optional
from uuid import uuid4

from ...config.settings import Settings
from ...core.exceptions import RAGError
from ...models.rag import Document
from ..embeddings import EmbeddingManager


class DocumentOperations:
    """Handles CRUD operations for document management."""

    def __init__(self, collection, embedding_manager: EmbeddingManager, settings: Settings, logger):
        self.collection = collection
        self.embedding_manager = embedding_manager
        self.settings = settings
        self.logger = logger

    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> Document:
        """Add a document to the RAG database."""
        try:
            # Create new document with proper ID generation
            document = Document(
                id=document_id or str(uuid4()),
                content=content,
                metadata=metadata or {},
                embedding_model=self.settings.EMBEDDING_MODEL if self.embedding_manager else None,
                embedding_dimension=self.embedding_manager.get_embedding_dimension() if self.embedding_manager else None,
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
            created_at_str = metadata.pop("created_at", datetime.now(UTC).isoformat())
            updated_at_str = metadata.pop("updated_at", None)
            embedding_model = metadata.pop("embedding_model", None)
            embedding_dimension = metadata.pop("embedding_dimension", None)
            
            # Parse timestamps
            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
            
            # Build document kwargs - only include updated_at if we have a value
            doc_kwargs = {
                "id": document_id,
                "content": content,
                "metadata": metadata,
                "created_at": created_at,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
            }
            
            # Only include updated_at if we have a value, otherwise let model use default
            if updated_at_str:
                doc_kwargs["updated_at"] = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))

            document = Document(**doc_kwargs)

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
                updated_at=datetime.now(UTC),
                embedding_model=self.settings.EMBEDDING_MODEL if self.embedding_manager else existing.embedding_model,
                embedding_dimension=self.embedding_manager.get_embedding_dimension() if self.embedding_manager else existing.embedding_dimension,
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