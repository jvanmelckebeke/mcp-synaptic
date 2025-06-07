"""RAG-related MCP tools."""

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ..config.logging import LoggerMixin
from ..models.rag import CollectionStats, Document, DocumentSearchResult
from ..rag.database import RAGDatabase


class RAGTools(LoggerMixin):
    """RAG-related MCP tools."""

    def __init__(self, mcp: FastMCP, rag_database: RAGDatabase):
        self.mcp = mcp
        self.rag_database = rag_database
        self._register_tools()

    def _register_tools(self) -> None:
        """Register RAG tools with FastMCP server."""

        @self.mcp.tool()
        async def rag_add_document(
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
            document_id: Optional[str] = None,
        ) -> Document:
            """Add a document to the RAG database."""
            document = await self.rag_database.add_document(
                content=content,
                metadata=metadata,
                document_id=document_id,
            )

            self.logger.info("Document added", document_id=document.id)
            return document

        @self.mcp.tool()
        async def rag_get_document(document_id: str) -> Optional[Document]:
            """Get a document by ID."""
            document = await self.rag_database.get_document(document_id)
            
            if document:
                self.logger.debug("Document retrieved", document_id=document_id)
            else:
                self.logger.debug("Document not found", document_id=document_id)
                
            return document

        @self.mcp.tool()
        async def rag_update_document(
            document_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Optional[Document]:
            """Update an existing document."""
            document = await self.rag_database.update_document(
                document_id=document_id,
                content=content,
                metadata=metadata,
            )

            if document:
                self.logger.info("Document updated", document_id=document_id)
            else:
                self.logger.debug("Document not found for update", document_id=document_id)

            return document

        @self.mcp.tool()
        async def rag_delete_document(document_id: str) -> bool:
            """Delete a document by ID."""
            deleted = await self.rag_database.delete_document(document_id)

            if deleted:
                self.logger.info("Document deleted", document_id=document_id)
            else:
                self.logger.debug("Document not found for deletion", document_id=document_id)

            return deleted

        @self.mcp.tool()
        async def rag_search(
            query: str,
            limit: int = 10,
            similarity_threshold: Optional[float] = None,
            metadata_filter: Optional[Dict[str, Any]] = None,
        ) -> List[DocumentSearchResult]:
            """Search for similar documents."""
            results = await self.rag_database.search(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter,
            )

            self.logger.info("Document search performed", query=query, results=len(results))
            return results

        @self.mcp.tool()
        async def rag_collection_stats() -> CollectionStats:
            """Get RAG collection statistics."""
            stats_dict = await self.rag_database.get_collection_stats()
            stats = CollectionStats(**stats_dict)
            
            self.logger.debug("RAG collection stats retrieved")
            return stats