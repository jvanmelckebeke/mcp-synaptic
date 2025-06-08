"""Memory storage implementations."""

import asyncio
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional

import aiosqlite
import redis.asyncio as redis
from redis.asyncio import Redis

from ..config.logging import LoggerMixin
from ..config.settings import Settings
from ..core.exceptions import ConnectionError, DatabaseError, MemoryError
from ..models.memory import Memory, MemoryQuery, MemoryStats


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


class SQLiteMemoryStorage(MemoryStorage):
    """SQLite-based memory storage implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_path = settings.SQLITE_DATABASE_PATH
        self._connection: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize SQLite database."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._connection = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            
            self.logger.info("SQLite memory storage initialized", db_path=str(self.db_path))
            
        except Exception as e:
            raise DatabaseError(f"Failed to initialize SQLite storage: {e}")

    async def close(self) -> None:
        """Close SQLite connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self.logger.info("SQLite memory storage closed")

    async def _create_tables(self) -> None:
        """Create necessary database tables."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            key TEXT NOT NULL UNIQUE,
            data TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            expiration_policy TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            accessed_at TEXT NOT NULL,
            expires_at TEXT,
            ttl_seconds INTEGER,
            access_count INTEGER DEFAULT 0,
            tags TEXT,
            metadata TEXT
        )
        """
        
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
        CREATE INDEX IF NOT EXISTS idx_memories_expires_at ON memories(expires_at);
        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
        """
        
        if self._connection:
            await self._connection.execute(create_table_sql)
            await self._connection.executescript(create_index_sql)
            await self._connection.commit()

    async def store(self, memory: Memory) -> None:
        """Store a memory in SQLite."""
        if not self._connection:
            raise DatabaseError("Storage not initialized")

        try:
            sql = """
            INSERT OR REPLACE INTO memories 
            (id, key, data, memory_type, expiration_policy, created_at, updated_at, 
             accessed_at, expires_at, ttl_seconds, access_count, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                str(memory.id),
                memory.key,
                json.dumps(memory.data),
                memory.memory_type.value,
                memory.expiration_policy.value,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                memory.last_accessed_at.isoformat(),
                memory.expires_at.isoformat() if memory.expires_at else None,
                memory.ttl_seconds,
                memory.access_count,
                json.dumps(memory.tags),
                json.dumps(memory.metadata),
            )
            
            await self._connection.execute(sql, values)
            await self._connection.commit()
            
        except Exception as e:
            raise DatabaseError(f"Failed to store memory: {e}")

    async def retrieve(self, key: str) -> Optional[Memory]:
        """Retrieve a memory by key from SQLite."""
        if not self._connection:
            raise DatabaseError("Storage not initialized")

        try:
            sql = "SELECT * FROM memories WHERE key = ?"
            cursor = await self._connection.execute(sql, (key,))
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_memory(row)
            
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve memory: {e}")

    async def delete(self, key: str) -> bool:
        """Delete a memory by key from SQLite."""
        if not self._connection:
            raise DatabaseError("Storage not initialized")

        try:
            sql = "DELETE FROM memories WHERE key = ?"
            cursor = await self._connection.execute(sql, (key,))
            await self._connection.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            raise DatabaseError(f"Failed to delete memory: {e}")

    async def list_memories(self, query: MemoryQuery) -> List[Memory]:
        """List memories matching the query from SQLite."""
        if not self._connection:
            raise DatabaseError("Storage not initialized")

        try:
            conditions = []
            values = []
            
            if query.keys:
                placeholders = ",".join("?" * len(query.keys))
                conditions.append(f"key IN ({placeholders})")
                values.extend(query.keys)
            
            if query.memory_types:
                placeholders = ",".join("?" * len(query.memory_types))
                conditions.append(f"memory_type IN ({placeholders})")
                values.extend([mt.value for mt in query.memory_types])
            
            if not query.include_expired:
                conditions.append("(expires_at IS NULL OR expires_at > ?)")
                values.append(datetime.now(UTC).isoformat())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            sql = f"SELECT * FROM memories WHERE {where_clause}"
            
            if query.limit:
                sql += f" LIMIT {query.limit}"
            if query.offset:
                sql += f" OFFSET {query.offset}"
            
            cursor = await self._connection.execute(sql, values)
            rows = await cursor.fetchall()
            
            return [self._row_to_memory(row) for row in rows]
            
        except Exception as e:
            raise DatabaseError(f"Failed to list memories: {e}")

    async def cleanup_expired(self) -> int:
        """Remove expired memories from SQLite."""
        if not self._connection:
            raise DatabaseError("Storage not initialized")

        try:
            sql = "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?"
            cursor = await self._connection.execute(sql, (datetime.now(UTC).isoformat(),))
            await self._connection.commit()
            
            return cursor.rowcount
            
        except Exception as e:
            raise DatabaseError(f"Failed to cleanup expired memories: {e}")

    async def get_stats(self) -> MemoryStats:
        """Get storage statistics from SQLite."""
        if not self._connection:
            raise DatabaseError("Storage not initialized")

        try:
            # Total memories
            cursor = await self._connection.execute("SELECT COUNT(*) FROM memories")
            total = (await cursor.fetchone())[0]
            
            # Memories by type
            cursor = await self._connection.execute(
                "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
            )
            type_counts = {row[0]: row[1] for row in await cursor.fetchall()}
            
            # Expired memories
            cursor = await self._connection.execute(
                "SELECT COUNT(*) FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (datetime.now(UTC).isoformat(),)
            )
            expired = (await cursor.fetchone())[0]
            
            return MemoryStats(
                total_memories=total,
                memories_by_type=type_counts,
                expired_memories=expired,
                total_size_bytes=0,  # Would need to calculate
                average_ttl_seconds=None,  # Would need to calculate
                oldest_memory=None,  # Would need to query
                newest_memory=None,  # Would need to query
            )
            
        except Exception as e:
            raise DatabaseError(f"Failed to get stats: {e}")

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object."""
        from ..models.memory import MemoryType, ExpirationPolicy
        
        return Memory(
            id=row[0],
            key=row[1],
            data=json.loads(row[2]),
            memory_type=MemoryType(row[3]),
            expiration_policy=ExpirationPolicy(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            updated_at=datetime.fromisoformat(row[6]),
            last_accessed_at=datetime.fromisoformat(row[7]),
            expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
            ttl_seconds=row[9],
            access_count=row[10],
            tags=json.loads(row[11]) if row[11] else {},
            metadata=json.loads(row[12]) if row[12] else {},
        )


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
            raise DatabaseError(f"Failed to get Redis stats: {e}")