"""Core memory manager with lifecycle management and coordination."""

from typing import Dict, List, Optional

from ...config.logging import LoggerMixin
from ...config.settings import Settings
from ...core.exceptions import MemoryError
from ..storage import MemoryStorage, RedisMemoryStorage, SQLiteMemoryStorage
from ...models.memory import Memory, MemoryQuery, MemoryStats, MemoryType
from .operations import MemoryOperations
from .queries import MemoryQueries


class MemoryManager(LoggerMixin):
    """Manages memory storage with expiration and cleanup capabilities."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage: Optional[MemoryStorage] = None
        self._initialized = False
        
        # Delegate operation handlers
        self._operations: Optional[MemoryOperations] = None
        self._queries: Optional[MemoryQueries] = None

    async def initialize(self) -> None:
        """Initialize memory storage backend and operation handlers."""
        try:
            # Choose storage backend based on settings
            if self.settings.REDIS_ENABLED:
                self.storage = RedisMemoryStorage(self.settings)
            else:
                self.storage = SQLiteMemoryStorage(self.settings)
            
            await self.storage.initialize()
            
            # Initialize operation handlers
            self._operations = MemoryOperations(self.storage, self.settings, self.logger)
            self._queries = MemoryQueries(self.storage, self.settings, self.logger)
            
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
            self.storage = None
            
        self._operations = None
        self._queries = None
        self._initialized = False
        self.logger.info("Memory manager closed")

    def _ensure_initialized(self) -> None:
        """Ensure the memory manager is initialized."""
        if not self._initialized or not self.storage or not self._operations or not self._queries:
            raise MemoryError("Memory manager not initialized")

    # CRUD Operations - delegated to MemoryOperations
    async def add(
        self,
        key: str,
        data: Dict,
        memory_type: MemoryType = None,
        ttl_seconds: Optional[int] = None,
        expiration_policy = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
    ) -> Memory:
        """Add a new memory or update existing one."""
        self._ensure_initialized()
        return await self._operations.add(key, data, memory_type, ttl_seconds, expiration_policy, tags, metadata)

    async def get(self, key: str, touch: bool = True) -> Optional[Memory]:
        """Retrieve a memory by key."""
        self._ensure_initialized()
        return await self._operations.get(key, touch)

    async def delete(self, key: str) -> bool:
        """Delete a memory by key."""
        self._ensure_initialized()
        return await self._operations.delete(key)

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
        return await self._operations.update(key, data, extend_ttl, tags, metadata)

    # Query Operations - delegated to MemoryQueries
    async def list(self, query: Optional[MemoryQuery] = None) -> List[Memory]:
        """List memories matching the query."""
        self._ensure_initialized()
        return await self._queries.list(query)

    async def cleanup_expired(self) -> int:
        """Remove all expired memories."""
        self._ensure_initialized()
        return await self._queries.cleanup_expired()

    async def get_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        self._ensure_initialized()
        return await self._queries.get_stats()

    async def exists(self, key: str) -> bool:
        """Check if a memory exists and is not expired."""
        self._ensure_initialized()
        return await self._queries.exists(key)

    async def touch(self, key: str) -> bool:
        """Touch a memory to update its access time."""
        self._ensure_initialized()
        return await self._queries.touch(key)