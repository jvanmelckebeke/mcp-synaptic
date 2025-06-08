"""Memory manager for handling memory operations with expiration."""

from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional

from ..config.logging import LoggerMixin
from ..config.settings import Settings
from ..core.exceptions import MemoryError, MemoryExpiredError, MemoryNotFoundError
from .storage import MemoryStorage, RedisMemoryStorage, SQLiteMemoryStorage
from ..models.memory import ExpirationPolicy, Memory, MemoryQuery, MemoryStats, MemoryType


class MemoryManager(LoggerMixin):
    """Manages memory storage with expiration and cleanup capabilities."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage: Optional[MemoryStorage] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize memory storage backend."""
        try:
            # Choose storage backend based on settings
            if self.settings.REDIS_ENABLED:
                self.storage = RedisMemoryStorage(self.settings)
            else:
                self.storage = SQLiteMemoryStorage(self.settings)
            
            await self.storage.initialize()
            self._initialized = True
            
            self.logger.info(
                "Memory manager initialized", 
                backend=type(self.storage).__name__,
                max_entries=self.settings.MAX_MEMORY_ENTRIES
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize memory manager", error=str(e))
            raise MemoryError(f"Memory manager initialization failed: {e}")

    async def close(self) -> None:
        """Close memory manager and storage."""
        if self.storage:
            await self.storage.close()
            self._initialized = False
            self.logger.info("Memory manager closed")

    def _ensure_initialized(self) -> None:
        """Ensure the memory manager is initialized."""
        if not self._initialized or not self.storage:
            raise MemoryError("Memory manager not initialized")

    async def add(
        self,
        key: str,
        data: Dict,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        ttl_seconds: Optional[int] = None,
        expiration_policy: ExpirationPolicy = ExpirationPolicy.ABSOLUTE,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Memory:
        """Add a new memory or update existing one."""
        self._ensure_initialized()

        try:
            # Use default TTL if not specified
            if ttl_seconds is None:
                ttl_seconds = self._get_default_ttl(memory_type)

            # Create memory object
            memory = Memory(
                key=key,
                data=data,
                memory_type=memory_type,
                expiration_policy=expiration_policy,
                ttl_seconds=ttl_seconds,
                tags=tags or {},
                metadata=metadata or {},
            )

            # Set expiration based on policy
            memory.update_expiration(ttl_seconds)

            # Store the memory
            await self.storage.store(memory)

            self.logger.info(
                "Memory added",
                key=key,
                memory_type=memory_type.value,
                ttl_seconds=ttl_seconds,
                expires_at=memory.expires_at.isoformat() if memory.expires_at else None
            )

            return memory

        except Exception as e:
            self.logger.error("Failed to add memory", key=key, error=str(e))
            raise MemoryError(f"Failed to add memory '{key}': {e}")

    async def get(self, key: str, touch: bool = True) -> Optional[Memory]:
        """Retrieve a memory by key."""
        self._ensure_initialized()

        try:
            memory = await self.storage.retrieve(key)
            
            if not memory:
                return None

            # Check if expired
            if memory.is_expired:
                # Remove expired memory
                await self.storage.delete(key)
                self.logger.debug("Removed expired memory", key=key)
                raise MemoryExpiredError(key)

            # Update access information if requested
            if touch:
                memory.touch()
                await self.storage.store(memory)

            self.logger.debug("Memory retrieved", key=key, access_count=memory.access_count)
            return memory

        except MemoryExpiredError:
            raise
        except Exception as e:
            self.logger.error("Failed to get memory", key=key, error=str(e))
            raise MemoryError(f"Failed to get memory '{key}': {e}")

    async def delete(self, key: str) -> bool:
        """Delete a memory by key."""
        self._ensure_initialized()

        try:
            deleted = await self.storage.delete(key)
            
            if deleted:
                self.logger.info("Memory deleted", key=key)
            else:
                self.logger.debug("Memory not found for deletion", key=key)
            
            return deleted

        except Exception as e:
            self.logger.error("Failed to delete memory", key=key, error=str(e))
            raise MemoryError(f"Failed to delete memory '{key}': {e}")

    async def update(
        self,
        key: str,
        data: Optional[Dict] = None,
        extend_ttl: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Memory]:
        """Update an existing memory."""
        self._ensure_initialized()

        try:
            memory = await self.get(key, touch=False)
            if not memory:
                raise MemoryNotFoundError(key)

            # Update fields
            if data is not None:
                memory.data = data
            if tags is not None:
                memory.tags.update(tags)
            if metadata is not None:
                memory.metadata.update(metadata)

            # Update timestamps
            memory.updated_at = datetime.now(UTC)

            # Extend TTL if requested
            if extend_ttl is not None:
                memory.ttl_seconds = extend_ttl
                memory.update_expiration(extend_ttl)

            # Store updated memory
            await self.storage.store(memory)

            self.logger.info("Memory updated", key=key)
            return memory

        except (MemoryNotFoundError, MemoryExpiredError):
            raise
        except Exception as e:
            self.logger.error("Failed to update memory", key=key, error=str(e))
            raise MemoryError(f"Failed to update memory '{key}': {e}")

    async def list(self, query: Optional[MemoryQuery] = None) -> List[Memory]:
        """List memories matching the query."""
        self._ensure_initialized()

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
        self._ensure_initialized()

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
        self._ensure_initialized()

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
            memory = await self.get(key, touch=False)
            return memory is not None
        except (MemoryNotFoundError, MemoryExpiredError):
            return False

    async def touch(self, key: str) -> bool:
        """Touch a memory to update its access time."""
        try:
            memory = await self.get(key, touch=True)
            return memory is not None
        except (MemoryNotFoundError, MemoryExpiredError):
            return False

    def _get_default_ttl(self, memory_type: MemoryType) -> int:
        """Get default TTL for a memory type."""
        defaults = {
            MemoryType.EPHEMERAL: 300,    # 5 minutes
            MemoryType.SHORT_TERM: self.settings.DEFAULT_MEMORY_TTL_SECONDS,
            MemoryType.LONG_TERM: 86400 * 7,  # 1 week
            MemoryType.PERMANENT: 0,      # No expiration
        }
        
        return defaults.get(memory_type, self.settings.DEFAULT_MEMORY_TTL_SECONDS)