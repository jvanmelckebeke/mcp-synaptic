"""Memory query and utility operations handler."""

from typing import List, Optional

from ...config.settings import Settings
from ...core.exceptions import MemoryError, MemoryExpiredError, MemoryNotFoundError
from ..storage import MemoryStorage
from ...models.memory import Memory, MemoryQuery, MemoryStats


class MemoryQueries:
    """Handles query and utility operations for memory management."""

    def __init__(self, storage: MemoryStorage, settings: Settings, logger):
        self.storage = storage
        self.settings = settings
        self.logger = logger

    async def list(self, query: Optional[MemoryQuery] = None) -> List[Memory]:
        """List memories matching the query."""
        try:
            query = query or MemoryQuery()
            memories = await self.storage.list_memories(query)
            
            self.logger.debug("Listed memories", count=len(memories))
            return memories

        except Exception as e:
            self.logger.error("Failed to list memories", error=str(e))
            raise MemoryError(f"Failed to list memories: {e}")

    async def cleanup_expired(self) -> int:
        """Remove all expired memories."""
        try:
            removed_count = await self.storage.cleanup_expired()
            
            if removed_count > 0:
                self.logger.info("Cleaned up expired memories", count=removed_count)
            
            return removed_count

        except Exception as e:
            self.logger.error("Failed to cleanup expired memories", error=str(e))
            raise MemoryError(f"Failed to cleanup expired memories: {e}")

    async def get_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        try:
            stats = await self.storage.get_stats()
            
            self.logger.debug(
                "Memory stats retrieved",
                total=stats.total_memories,
                expired=stats.expired_memories
            )
            
            return stats

        except Exception as e:
            self.logger.error("Failed to get memory stats", error=str(e))
            raise MemoryError(f"Failed to get memory stats: {e}")

    async def exists(self, key: str) -> bool:
        """Check if a memory exists and is not expired."""
        try:
            # Use the operations module's get method through storage directly
            memory = await self.storage.retrieve(key)
            if not memory:
                return False
            
            # Check if expired
            if memory.is_expired:
                # Remove expired memory
                await self.storage.delete(key)
                self.logger.debug("Removed expired memory during exists check", key=key)
                return False
            
            return True
            
        except Exception:
            return False

    async def touch(self, key: str) -> bool:
        """Touch a memory to update its access time."""
        try:
            memory = await self.storage.retrieve(key)
            if not memory:
                return False
                
            # Check if expired
            if memory.is_expired:
                # Remove expired memory
                await self.storage.delete(key)
                self.logger.debug("Removed expired memory during touch", key=key)
                return False
            
            # Update access information
            memory.touch()
            await self.storage.store(memory)
            
            self.logger.debug("Memory touched", key=key, access_count=memory.access_count)
            return True
            
        except Exception:
            return False