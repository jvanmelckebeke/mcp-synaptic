"""Abstract base class for memory storage implementations."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ...config.logging import LoggerMixin
from ...models.memory import Memory, MemoryQuery, MemoryStats


class MemoryStorage(ABC, LoggerMixin):
    """Abstract base class for memory storage implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend."""
        pass

    @abstractmethod
    async def store(self, memory: Memory) -> None:
        """Store a memory."""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Memory]:
        """Retrieve a memory by key."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a memory by key. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    async def list_memories(self, query: MemoryQuery) -> List[Memory]:
        """List memories matching the query."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Remove expired memories. Returns count of removed memories."""
        pass

    @abstractmethod
    async def get_stats(self) -> MemoryStats:
        """Get storage statistics."""
        pass