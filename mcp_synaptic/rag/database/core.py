"""Core RAG database with lifecycle management and coordination."""

import asyncio
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None

from ...config.logging import LoggerMixin
from ...config.settings import Settings
from ...core.exceptions import RAGError
from ...models.rag import Document, DocumentSearchResult
from ..embeddings import EmbeddingManager
from .documents import DocumentOperations
from .search import SearchOperations
from .stats import StatsOperations


class RAGDatabase(LoggerMixin):
    """RAG database for document storage and similarity search."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.client = None
        self.collection = None
        self._initialized = False
        
        # Delegate operation handlers
        self._documents: Optional[DocumentOperations] = None
        self._search: Optional[SearchOperations] = None
        self._stats: Optional[StatsOperations] = None

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

            # Initialize operation handlers
            self._documents = DocumentOperations(
                self.collection, self.embedding_manager, self.settings, self.logger
            )
            self._search = SearchOperations(
                self.collection, self.embedding_manager, self.settings, self.logger
            )
            self._stats = StatsOperations(
                self.collection, self.settings, self.logger
            )

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
            self.embedding_manager = None
        
        self.client = None
        self.collection = None
        self._documents = None
        self._search = None
        self._stats = None
        self._initialized = False
        self.logger.info("RAG database closed")

    def _ensure_initialized(self) -> None:
        """Ensure the database is initialized."""
        if not self._initialized or not self._documents or not self._search or not self._stats:
            raise RAGError("RAG database not initialized")

    # Document Operations - delegated to DocumentOperations
    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> Document:
        """Add a document to the RAG database."""
        self._ensure_initialized()
        return await self._documents.add_document(content, metadata, document_id)

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID."""
        self._ensure_initialized()
        return await self._documents.get_document(document_id)

    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Document]:
        """Update an existing document."""
        self._ensure_initialized()
        return await self._documents.update_document(document_id, content, metadata)

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        self._ensure_initialized()
        return await self._documents.delete_document(document_id)

    # Search Operations - delegated to SearchOperations
    async def search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentSearchResult]:
        """Search for similar documents."""
        self._ensure_initialized()
        return await self._search.search(query, limit, similarity_threshold, metadata_filter)

    # Statistics Operations - delegated to StatsOperations
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        self._ensure_initialized()
        return await self._stats.get_collection_stats()