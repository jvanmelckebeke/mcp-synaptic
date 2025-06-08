"""Redis-based memory storage implementation."""

from typing import List, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from ...config.settings import Settings
from ...core.exceptions import ConnectionError, DatabaseError
from ...models.memory import Memory, MemoryQuery, MemoryStats
from .base import MemoryStorage


class RedisMemoryStorage(MemoryStorage):
    """Redis-based memory storage implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.redis_url = settings.REDIS_URL
        self._redis: Optional[Redis] = None

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self._redis = redis.from_url(self.redis_url)
            await self._redis.ping()
            
            self.logger.info("Redis memory storage initialized", url=self.redis_url)
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}", "redis")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self.logger.info("Redis memory storage closed")

    async def store(self, memory: Memory) -> None:
        """Store a memory in Redis."""
        if not self._redis:
            raise ConnectionError("Redis not initialized", "redis")

        try:
            key = f"memory:{memory.key}"
            value = memory.model_dump_json()
            
            if memory.ttl_seconds:
                await self._redis.setex(key, memory.ttl_seconds, value)
            else:
                await self._redis.set(key, value)
                
        except Exception as e:
            raise DatabaseError(f"Failed to store memory in Redis: {e}")

    async def retrieve(self, key: str) -> Optional[Memory]:
        """Retrieve a memory by key from Redis."""
        if not self._redis:
            raise ConnectionError("Redis not initialized", "redis")

        try:
            value = await self._redis.get(f"memory:{key}")
            if value:
                return Memory.model_validate_json(value)
            return None
            
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve memory from Redis: {e}")

    async def delete(self, key: str) -> bool:
        """Delete a memory by key from Redis."""
        if not self._redis:
            raise ConnectionError("Redis not initialized", "redis")

        try:
            result = await self._redis.delete(f"memory:{key}")
            return result > 0
            
        except Exception as e:
            raise DatabaseError(f"Failed to delete memory from Redis: {e}")

    async def list_memories(self, query: MemoryQuery) -> List[Memory]:
        """List memories matching the query from Redis."""
        if not self._redis:
            raise ConnectionError("Redis not initialized", "redis")

        try:
            # For Redis, we'll scan all memory keys and filter
            memories = []
            async for key in self._redis.scan_iter(match="memory:*"):
                if query.limit and len(memories) >= query.limit:
                    break
                    
                value = await self._redis.get(key)
                if value:
                    memory = Memory.model_validate_json(value)
                    
                    # Apply filters
                    if query.keys and memory.key not in query.keys:
                        continue
                    if query.memory_types and memory.memory_type not in query.memory_types:
                        continue
                    if not query.include_expired and memory.is_expired:
                        continue
                    
                    memories.append(memory)
            
            return memories[query.offset:]
            
        except Exception as e:
            raise DatabaseError(f"Failed to list memories from Redis: {e}")

    async def cleanup_expired(self) -> int:
        """Remove expired memories from Redis."""
        # Redis automatically handles TTL expiration
        return 0

    async def get_stats(self) -> MemoryStats:
        """Get storage statistics from Redis."""
        if not self._redis:
            raise ConnectionError("Redis not initialized", "redis")

        try:
            # Count memory keys
            count = 0
            async for _ in self._redis.scan_iter(match="memory:*"):
                count += 1
            
            return MemoryStats(
                total_memories=count,
                memories_by_type={},
                expired_memories=0,
                total_size_bytes=0,
                average_ttl_seconds=None,
                oldest_memory=None,
                newest_memory=None,
            )
            
        except Exception as e:
            raise DatabaseError(f"Failed to get stats from Redis: {e}")