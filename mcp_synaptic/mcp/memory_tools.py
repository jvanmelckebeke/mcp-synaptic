"""Memory-related MCP tools."""

from typing import Annotated, Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..config.logging import LoggerMixin
from ..memory.manager import MemoryManager
from ..models.memory import Memory, MemoryQuery, MemoryStats, MemoryType


class MemoryTools(LoggerMixin):
    """Memory-related MCP tools."""

    def __init__(self, mcp: FastMCP, memory_manager: MemoryManager):
        self.mcp = mcp
        self.memory_manager = memory_manager
        self._register_tools()

    def _register_tools(self) -> None:
        """Register memory tools with FastMCP server."""

        @self.mcp.tool()
        async def memory_add(
            key: Annotated[str, Field(description="Unique identifier for the memory (must be unique)")],
            data: Annotated[Dict[str, Any], Field(description="Memory data to store (any JSON-serializable structure)")],
            memory_type: Annotated[str, Field(
                description="Type of memory determining default behavior",
                enum=["ephemeral", "short_term", "long_term", "permanent"]
            )] = "short_term",
            ttl_seconds: Annotated[Optional[int], Field(
                description="Custom time-to-live in seconds (overrides type defaults, 0 for permanent)",
                ge=0
            )] = None,
            tags: Annotated[Optional[Dict[str, str]], Field(
                description="Optional key-value tags for categorization and filtering"
            )] = None,
            metadata: Annotated[Optional[Dict[str, Any]], Field(
                description="Additional metadata (not used for filtering)"
            )] = None,
        ) -> Memory:
            """Add a new memory with optional expiration.
            
            Returns:
                Memory object with assigned ID and timestamps
            """
            mem_type = MemoryType(memory_type)

            memory = await self.memory_manager.add(
                key=key,
                data=data,
                memory_type=mem_type,
                ttl_seconds=ttl_seconds,
                tags=tags,
                metadata=metadata,
            )

            self.logger.info("Memory added", key=key, memory_type=memory_type)
            return memory

        @self.mcp.tool()
        async def memory_get(
            key: str,
            touch: bool = True,
        ) -> Optional[Memory]:
            """Retrieve a memory by its unique key.
            
            Args:
                key: Unique identifier of the memory to retrieve
                touch: If True, updates last_accessed_at and access_count (default: True)
                      Set to False for read-only access that doesn't affect TTL
                      
            Returns:
                Memory object if found and not expired, None otherwise
                
            Raises:
                MemoryExpiredError: If memory exists but has expired (auto-removed)
                MemoryError: If storage operation fails
            """
            memory = await self.memory_manager.get(key, touch=touch)
            
            if memory:
                self.logger.debug("Memory retrieved", key=key)
            else:
                self.logger.debug("Memory not found", key=key)
                
            return memory

        @self.mcp.tool()
        async def memory_update(
            key: Annotated[str, Field(description="Unique identifier of the memory to update")],
            data: Annotated[Optional[Dict[str, Any]], Field(
                description="New data to replace existing (if provided)"
            )] = None,
            extend_ttl: Annotated[Optional[int], Field(
                description="New TTL in seconds to extend expiration (if provided, 0 for permanent)",
                ge=0
            )] = None,
            tags: Annotated[Optional[Dict[str, str]], Field(
                description="New tags to replace existing tags (if provided)"
            )] = None,
            metadata: Annotated[Optional[Dict[str, Any]], Field(
                description="New metadata to replace existing metadata (if provided)"
            )] = None,
        ) -> Optional[Memory]:
            """Update an existing memory's data, TTL, or metadata.
            
            Returns:
                Updated memory object if found, None if not found
                
            Note:
                Only provided parameters are updated, others remain unchanged.
                Updates the memory's updated_at timestamp.
            """
            memory = await self.memory_manager.update(
                key=key,
                data=data,
                extend_ttl=extend_ttl,
                tags=tags,
                metadata=metadata,
            )

            if memory:
                self.logger.info("Memory updated", key=key)
            else:
                self.logger.debug("Memory not found for update", key=key)

            return memory

        @self.mcp.tool()
        async def memory_delete(
            key: Annotated[str, Field(description="Unique identifier of the memory to delete")],
        ) -> bool:
            """Delete a memory by its unique key.
            
            Returns:
                True if memory was found and deleted, False if not found
                
            Note:
                This operation is idempotent - deleting a non-existent memory
                returns False but doesn't raise an error.
            """
            deleted = await self.memory_manager.delete(key)

            if deleted:
                self.logger.info("Memory deleted", key=key)
            else:
                self.logger.debug("Memory not found for deletion", key=key)

            return deleted

        @self.mcp.tool()
        async def memory_list(
            keys: Annotated[Optional[List[str]], Field(
                description="Filter to specific memory keys (if provided)"
            )] = None,
            memory_types: Annotated[Optional[List[str]], Field(
                description="Filter by memory types",
                enum=["ephemeral", "short_term", "long_term", "permanent"]
            )] = None,
            include_expired: Annotated[bool, Field(
                description="Include expired memories in results"
            )] = False,
            limit: Annotated[Optional[int], Field(
                description="Maximum number of results to return",
                ge=1, le=100
            )] = None,
            offset: Annotated[int, Field(
                description="Number of results to skip for pagination",
                ge=0
            )] = 0,
        ) -> List[Memory]:
            """List memories with optional filtering and pagination.
            
            Returns:
                List of memory objects matching the criteria
                
            Example:
                # Get all short-term memories
                memories = await memory_list(memory_types=["short_term"])
                
                # Get specific memories by key
                memories = await memory_list(keys=["user_profile", "session_data"])
                
                # Paginated results
                memories = await memory_list(limit=5, offset=10)
            """
            # Convert string memory types to enum
            mem_types = None
            if memory_types:
                mem_types = [MemoryType(mt) for mt in memory_types]

            # Use default limit if None provided
            actual_limit = limit if limit is not None else 10

            # Create query
            query = MemoryQuery(
                keys=keys,
                memory_types=mem_types,
                include_expired=include_expired,
                limit=actual_limit,
                offset=offset,
            )

            memories = await self.memory_manager.list(query)
            self.logger.debug("Memories listed", count=len(memories))
            return memories

        @self.mcp.tool()
        async def memory_stats() -> MemoryStats:
            """Get comprehensive memory usage statistics.
            
            Returns:
                MemoryStats object containing memory statistics:
                - total_memories: Total number of memories stored
                - memories_by_type: Count breakdown by memory type
                - expired_memories: Number of expired memories
                - total_size_bytes: Approximate total size
                - average_ttl_seconds: Average TTL across all memories
                - oldest_memory: Creation time of oldest memory
                - newest_memory: Creation time of newest memory
                - most_accessed_count: Highest access count
                
            Note:
                Statistics are computed in real-time and may include
                expired memories that haven't been cleaned up yet.
            """
            stats = await self.memory_manager.get_stats()
            self.logger.debug("Memory stats retrieved")
            return stats