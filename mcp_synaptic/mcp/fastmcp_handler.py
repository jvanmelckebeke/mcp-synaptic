"""FastMCP-based server implementation for MCP Synaptic."""

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ..config.logging import LoggerMixin
from ..memory.manager import MemoryManager
from ..models.memory import Memory, MemoryQuery, MemoryStats, MemoryType
from ..models.rag import CollectionStats, Document, DocumentSearchResult
from ..models.base import PaginatedResponse
from ..rag.database import RAGDatabase
from ..sse.server import SSEServer


class FastMCPHandler(LoggerMixin):
    """FastMCP server implementation for MCP Synaptic."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        rag_database: RAGDatabase,
        sse_server: Optional[SSEServer] = None,
    ):
        self.memory_manager = memory_manager
        self.rag_database = rag_database
        self.sse_server = sse_server
        
        # Create FastMCP server
        self.mcp = FastMCP()
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all tools with the FastMCP server."""
        self._register_memory_tools()
        self._register_rag_tools()

    def _register_memory_tools(self) -> None:
        """Register memory-related tools."""
        
        @self.mcp.tool()
        async def memory_add(
            key: str,
            data: Dict[str, Any],
            memory_type: str = "short_term",
            ttl_seconds: Optional[int] = None,
            tags: Optional[Dict[str, str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Memory:
            """Add a new memory with optional expiration.
            
            Args:
                key: Unique identifier for the memory
                data: The data to store in memory
                memory_type: Type of memory (ephemeral, short_term, long_term, permanent)
                ttl_seconds: Time to live in seconds (optional)
                tags: Optional tags for the memory
                metadata: Optional metadata for the memory
                
            Returns:
                The created Memory object with full type safety
            """
            try:
                # Validate memory type
                try:
                    mem_type = MemoryType(memory_type)
                except ValueError:
                    raise ValueError(f"Invalid memory type: {memory_type}")

                # Add memory
                memory = await self.memory_manager.add(
                    key=key,
                    data=data,
                    memory_type=mem_type,
                    ttl_seconds=ttl_seconds,
                    tags=tags,
                    metadata=metadata,
                )

                self.logger.info("Memory added via FastMCP", key=key, memory_type=memory_type)
                return memory

            except Exception as e:
                self.logger.error("Failed to add memory via FastMCP", key=key, error=str(e))
                raise

        @self.mcp.tool()
        async def memory_get(key: str, touch: bool = True) -> Optional[Memory]:
            """Get a memory by key.
            
            Args:
                key: The memory key to retrieve
                touch: Whether to update last accessed time
                
            Returns:
                The Memory object if found, or None if not found
            """
            try:
                memory = await self.memory_manager.get(key, touch=touch)
                
                if memory:
                    self.logger.debug("Memory retrieved via FastMCP", key=key)
                    return memory
                else:
                    self.logger.debug("Memory not found via FastMCP", key=key)
                    return None

            except Exception as e:
                self.logger.error("Failed to get memory via FastMCP", key=key, error=str(e))
                raise

        @self.mcp.tool()
        async def memory_update(
            key: str,
            data: Optional[Dict[str, Any]] = None,
            extend_ttl: Optional[int] = None,
            tags: Optional[Dict[str, str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Optional[Memory]:
            """Update an existing memory.
            
            Args:
                key: The memory key to update
                data: New data to store (optional)
                extend_ttl: Seconds to extend TTL (optional)
                tags: New tags (optional)
                metadata: New metadata (optional)
                
            Returns:
                Updated Memory object, or None if not found
            """
            try:
                memory = await self.memory_manager.update(
                    key=key,
                    data=data,
                    extend_ttl=extend_ttl,
                    tags=tags,
                    metadata=metadata,
                )

                if memory:
                    self.logger.info("Memory updated via FastMCP", key=key)
                    return memory
                else:
                    return None

            except Exception as e:
                self.logger.error("Failed to update memory via FastMCP", key=key, error=str(e))
                raise

        @self.mcp.tool()
        async def memory_delete(key: str) -> bool:
            """Delete a memory by key.
            
            Args:
                key: The memory key to delete
                
            Returns:
                True if deleted, False if not found
            """
            try:
                deleted = await self.memory_manager.delete(key)

                if deleted:
                    self.logger.info("Memory deleted via FastMCP", key=key)

                return deleted

            except Exception as e:
                self.logger.error("Failed to delete memory via FastMCP", key=key, error=str(e))
                raise

        @self.mcp.tool()
        async def memory_list(
            keys: Optional[List[str]] = None,
            memory_types: Optional[List[str]] = None,
            include_expired: bool = False,
            limit: Optional[int] = None,
            offset: int = 0,
        ) -> PaginatedResponse[Memory]:
            """List memories with optional filtering.
            
            Args:
                keys: Optional list of specific keys to retrieve
                memory_types: Optional list of memory types to filter by
                include_expired: Whether to include expired memories
                limit: Maximum number of results
                offset: Number of results to skip
                
            Returns:
                Paginated response containing Memory objects
            """
            try:
                # Convert string memory types to enum
                mem_types = None
                if memory_types:
                    try:
                        mem_types = [MemoryType(mt) for mt in memory_types]
                    except ValueError as e:
                        raise ValueError(f"Invalid memory type: {e}")

                # Create query
                query = MemoryQuery(
                    query="",  # Not used in memory search
                    keys=keys,
                    memory_types=mem_types,
                    include_expired=include_expired,
                    limit=limit or 100,
                    offset=offset,
                )

                # List memories
                memories = await self.memory_manager.list(query)
                
                # Get total count for pagination
                total_count = len(memories) + offset  # Approximate, would need separate count query for exact

                self.logger.debug("Memories listed via FastMCP", count=len(memories))
                return PaginatedResponse(
                    items=memories,
                    total_count=total_count,
                    offset=offset,
                    limit=limit
                )

            except Exception as e:
                self.logger.error("Failed to list memories via FastMCP", error=str(e))
                raise

        @self.mcp.tool()
        async def memory_stats() -> MemoryStats:
            """Get memory statistics.
            
            Returns:
                MemoryStats object with comprehensive usage statistics
            """
            try:
                stats = await self.memory_manager.get_stats()
                
                self.logger.debug("Memory stats retrieved via FastMCP")
                return stats

            except Exception as e:
                self.logger.error("Failed to get memory stats via FastMCP", error=str(e))
                raise

    def _register_rag_tools(self) -> None:
        """Register RAG-related tools."""
        
        @self.mcp.tool()
        async def rag_add_document(
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
            document_id: Optional[str] = None,
        ) -> Document:
            """Add a document to the RAG database.
            
            Args:
                content: The document content to add
                metadata: Optional metadata for the document
                document_id: Optional custom document ID
                
            Returns:
                The created Document object with full type safety
            """
            try:
                document = await self.rag_database.add_document(
                    content=content,
                    metadata=metadata,
                    document_id=document_id,
                )

                self.logger.info("Document added via FastMCP", document_id=document.id)
                return document

            except Exception as e:
                self.logger.error("Failed to add document via FastMCP", error=str(e))
                raise

        @self.mcp.tool()
        async def rag_get_document(document_id: str) -> Optional[Document]:
            """Get a document by ID.
            
            Args:
                document_id: The document ID to retrieve
                
            Returns:
                The Document object if found, or None if not found
            """
            try:
                document = await self.rag_database.get_document(document_id)
                
                if document:
                    self.logger.debug("Document retrieved via FastMCP", document_id=document_id)
                    return document
                else:
                    self.logger.debug("Document not found via FastMCP", document_id=document_id)
                    return None

            except Exception as e:
                self.logger.error("Failed to get document via FastMCP", document_id=document_id, error=str(e))
                raise

        @self.mcp.tool()
        async def rag_update_document(
            document_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Optional[Document]:
            """Update an existing document.
            
            Args:
                document_id: The document ID to update
                content: New document content (optional)
                metadata: New metadata (optional)
                
            Returns:
                Updated Document object, or None if not found
            """
            try:
                document = await self.rag_database.update_document(
                    document_id=document_id,
                    content=content,
                    metadata=metadata,
                )

                if document:
                    self.logger.info("Document updated via FastMCP", document_id=document_id)
                    return document
                else:
                    return None

            except Exception as e:
                self.logger.error("Failed to update document via FastMCP", document_id=document_id, error=str(e))
                raise

        @self.mcp.tool()
        async def rag_delete_document(document_id: str) -> bool:
            """Delete a document by ID.
            
            Args:
                document_id: The document ID to delete
                
            Returns:
                True if deleted, False if not found
            """
            try:
                deleted = await self.rag_database.delete_document(document_id)

                if deleted:
                    self.logger.info("Document deleted via FastMCP", document_id=document_id)

                return deleted

            except Exception as e:
                self.logger.error("Failed to delete document via FastMCP", document_id=document_id, error=str(e))
                raise

        @self.mcp.tool()
        async def rag_search(
            query: str,
            limit: int = 10,
            similarity_threshold: Optional[float] = None,
            metadata_filter: Optional[Dict[str, Any]] = None,
        ) -> List[DocumentSearchResult]:
            """Search for similar documents.
            
            Args:
                query: The search query
                limit: Maximum number of results to return
                similarity_threshold: Minimum similarity score for results
                metadata_filter: Optional metadata filter
                
            Returns:
                List of DocumentSearchResult objects with similarity scores
            """
            try:
                results = await self.rag_database.search(
                    query=query,
                    limit=limit,
                    similarity_threshold=similarity_threshold,
                    metadata_filter=metadata_filter,
                )

                self.logger.info("Document search performed via FastMCP", query=query, results=len(results))
                return results

            except Exception as e:
                self.logger.error("Failed to search documents via FastMCP", query=query, error=str(e))
                raise

        @self.mcp.tool()
        async def rag_collection_stats() -> CollectionStats:
            """Get RAG collection statistics.
            
            Returns:
                CollectionStats object with comprehensive collection information
            """
            try:
                stats_dict = await self.rag_database.get_collection_stats()
                
                # Convert dict to CollectionStats model
                stats = CollectionStats(**stats_dict)
                
                self.logger.debug("RAG collection stats retrieved via FastMCP")
                return stats

            except Exception as e:
                self.logger.error("Failed to get RAG collection stats via FastMCP", error=str(e))
                raise

    def get_mcp_server(self) -> FastMCP:
        """Get the FastMCP server instance."""
        return self.mcp

    def get_app(self):
        """Get the FastAPI app with SSE support."""
        return self.mcp.sse_app()