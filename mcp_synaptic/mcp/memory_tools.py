"""Memory-related MCP tools."""

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

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
            key: str,
            data: Dict[str, Any],
            memory_type: str = "short_term",
            ttl_seconds: Optional[int] = None,
            tags: Optional[Dict[str, str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Memory:
            """Add a new memory with optional expiration."""
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                raise ValueError(f"Invalid memory type: {memory_type}")

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
            """Get a memory by key."""
            memory = await self.memory_manager.get(key, touch=touch)
            
            if memory:
                self.logger.debug("Memory retrieved", key=key)
            else:
                self.logger.debug("Memory not found", key=key)
                
            return memory

        @self.mcp.tool()
        async def memory_update(
            key: str,
            data: Optional[Dict[str, Any]] = None,
            extend_ttl: Optional[int] = None,
            tags: Optional[Dict[str, str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Optional[Memory]:
            """Update an existing memory."""
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
        async def memory_delete(key: str) -> bool:
            """Delete a memory by key."""
            deleted = await self.memory_manager.delete(key)

            if deleted:
                self.logger.info("Memory deleted", key=key)
            else:
                self.logger.debug("Memory not found for deletion", key=key)

            return deleted

        @self.mcp.tool()
        async def memory_list(
            keys: Optional[List[str]] = None,
            memory_types: Optional[List[str]] = None,
            include_expired: bool = False,
            limit: Optional[int] = None,
            offset: int = 0,
        ) -> List[Memory]:
            """List memories with optional filtering."""
            # Convert string memory types to enum
            mem_types = None
            if memory_types:
                try:
                    mem_types = [MemoryType(mt) for mt in memory_types]
                except ValueError as e:
                    raise ValueError(f"Invalid memory type: {e}")

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
            """Get memory usage statistics."""
            stats = await self.memory_manager.get_stats()
            self.logger.debug("Memory stats retrieved")
            return stats