"""RAG-related MCP tools."""

from typing import Annotated, Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

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
            content: Annotated[str, Field(description="Text content of the document to store and index")],
            metadata: Annotated[Optional[Dict[str, Any]], Field(
                description="Optional metadata for categorization and filtering"
            )] = None,
            document_id: Annotated[Optional[str], Field(
                description="Optional custom ID (auto-generated if not provided)"
            )] = None,
        ) -> dict:
            """Add a document to the RAG database for vector search.
            
            Returns:
                Document dictionary with ID, embeddings, and timestamps
                
            Note:
                The document content will be embedded using the configured
                embedding provider for semantic search capabilities.
            """
            document = await self.rag_database.add_document(
                content=content,
                metadata=metadata,
                document_id=document_id,
            )

            self.logger.info("Document added", document_id=document.id)
            return document.model_dump()

        @self.mcp.tool()
        async def rag_get_document(
            document_id: Annotated[str, Field(description="Unique identifier of the document")],
        ) -> Optional[dict]:
            """Retrieve a document by its unique ID.
            
            Returns:
                Document dictionary if found, None if not found
            """
            document = await self.rag_database.get_document(document_id)
            
            if document:
                self.logger.debug("Document retrieved", document_id=document_id)
            else:
                self.logger.debug("Document not found", document_id=document_id)
                
            return document.model_dump() if document else None

        @self.mcp.tool()
        async def rag_update_document(
            document_id: Annotated[str, Field(description="Unique identifier of the document to update")],
            content: Annotated[Optional[str], Field(
                description="New content (triggers re-embedding if provided)"
            )] = None,
            metadata: Annotated[Optional[Dict[str, Any]], Field(
                description="New metadata (replaces existing if provided)"
            )] = None,
        ) -> Optional[dict]:
            """Update an existing document's content or metadata.
            
            Returns:
                Updated document dictionary if found, None if not found
                
            Note:
                Updating content will regenerate embeddings, which may take time.
                Only provided parameters are updated.
            """
            document = await self.rag_database.update_document(
                document_id=document_id,
                content=content,
                metadata=metadata,
            )

            if document:
                self.logger.info("Document updated", document_id=document_id)
            else:
                self.logger.debug("Document not found for update", document_id=document_id)

            return document.model_dump() if document else None

        @self.mcp.tool()
        async def rag_delete_document(
            document_id: Annotated[str, Field(description="Unique identifier of the document to delete")],
        ) -> bool:
            """Delete a document by its unique ID.
            
            Returns:
                True if document was found and deleted, False if not found
                
            Note:
                This permanently removes the document and its embeddings.
                This operation is idempotent.
            """
            deleted = await self.rag_database.delete_document(document_id)

            if deleted:
                self.logger.info("Document deleted", document_id=document_id)
            else:
                self.logger.debug("Document not found for deletion", document_id=document_id)

            return deleted

        @self.mcp.tool()
        async def rag_search(
            query: Annotated[str, Field(description="Text query to search for (will be embedded for comparison)")],
            limit: Annotated[int, Field(
                description="Maximum number of results to return",
                ge=1, le=100
            )] = 10,
            similarity_threshold: Annotated[Optional[float], Field(
                description="Minimum similarity score (0.0-1.0, optional)",
                ge=0.0, le=1.0
            )] = None,
            metadata_filter: Annotated[Optional[Dict[str, Any]], Field(
                description="Filter results by metadata key-value pairs"
            )] = None,
        ) -> List[dict]:
            """Search for documents using semantic similarity.
            
            Returns:
                List of search result dictionaries with similarity scores,
                ordered by relevance (highest similarity first)
                
            Example:
                # Basic search
                results = await rag_search("machine learning algorithms")
                
                # Filtered search with threshold
                results = await rag_search(
                    query="python tutorial",
                    limit=5,
                    similarity_threshold=0.7,
                    metadata_filter={"category": "programming"}
                )
            """
            results = await self.rag_database.search(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter,
            )

            self.logger.info("Document search performed", query=query, results=len(results))
            return [result.model_dump() for result in results]

        @self.mcp.tool()
        async def rag_collection_stats() -> dict:
            """Get comprehensive statistics about the RAG document collection.
            
            Returns:
                Dictionary containing collection statistics:
                - total_documents: Total number of documents in collection
                - total_embeddings: Total number of embedding vectors
                - average_document_length: Average content length
                - embedding_dimensions: Dimensionality of embedding vectors
                - collection_size_bytes: Approximate storage size
                - oldest_document: Creation time of oldest document
                - newest_document: Creation time of newest document
                
            Note:
                Statistics are computed in real-time from the vector database.
            """
            stats_dict = await self.rag_database.get_collection_stats()
            stats = CollectionStats(**stats_dict)
            
            self.logger.debug("RAG collection stats retrieved")
            return stats.model_dump()