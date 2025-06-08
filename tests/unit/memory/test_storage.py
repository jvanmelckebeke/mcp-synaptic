"""Tests for memory storage implementations."""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.core.exceptions import ConnectionError, DatabaseError
from mcp_synaptic.memory.storage import MemoryStorage, SQLiteMemoryStorage, RedisMemoryStorage
from mcp_synaptic.models.memory import ExpirationPolicy, Memory, MemoryQuery, MemoryStats, MemoryType
from tests.utils import MemoryTestHelper, MockFactory, AssertionHelpers


class TestSQLiteMemoryStorage:
    """Test SQLite memory storage implementation."""

    @pytest_asyncio.fixture
    async def storage(self, test_settings: Settings) -> SQLiteMemoryStorage:
        """Create and initialize SQLite storage for testing."""
        storage = SQLiteMemoryStorage(test_settings)
        await storage.initialize()
        yield storage
        await storage.close()

    @pytest_asyncio.fixture
    async def storage_with_data(self, storage: SQLiteMemoryStorage) -> SQLiteMemoryStorage:
        """SQLite storage with test data pre-loaded."""
        # Add some test memories
        memories = MemoryTestHelper.create_memory_batch(5, "test_key")
        for memory in memories:
            await storage.store(memory)
        return storage

    async def test_initialize_creates_database_file(self, test_settings: Settings):
        """Test that initialization creates the database file."""
        storage = SQLiteMemoryStorage(test_settings)
        
        # Database file should not exist initially
        assert not test_settings.SQLITE_DATABASE_PATH.exists()
        
        await storage.initialize()
        
        # Database file should be created
        assert test_settings.SQLITE_DATABASE_PATH.exists()
        
        await storage.close()

    async def test_initialize_creates_tables_and_indexes(self, storage: SQLiteMemoryStorage):
        """Test that initialization creates proper tables and indexes."""
        # Check that tables exist by querying schema
        cursor = await storage._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        table = await cursor.fetchone()
        assert table is not None
        
        # Check that indexes exist
        cursor = await storage._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_memories_%'"
        )
        indexes = await cursor.fetchall()
        assert len(indexes) >= 3  # At least 3 indexes should be created

    async def test_store_memory_success(self, storage: SQLiteMemoryStorage):
        """Test successful memory storage."""
        memory = MemoryTestHelper.create_test_memory(
            key="test_store",
            data={"value": "test_data"},
            memory_type=MemoryType.SHORT_TERM
        )
        
        await storage.store(memory)
        
        # Verify stored in database
        cursor = await storage._connection.execute(
            "SELECT * FROM memories WHERE key = ?", ("test_store",)
        )
        row = await cursor.fetchone()
        
        assert row is not None
        assert row[1] == "test_store"  # key column
        assert json.loads(row[2]) == {"value": "test_data"}  # data column

    async def test_store_memory_overwrites_existing(self, storage: SQLiteMemoryStorage):
        """Test that storing with same key overwrites existing memory."""
        key = "overwrite_test"
        
        # Store first memory
        memory1 = MemoryTestHelper.create_test_memory(
            key=key,
            data={"version": 1}
        )
        await storage.store(memory1)
        
        # Store second memory with same key
        memory2 = MemoryTestHelper.create_test_memory(
            key=key,
            data={"version": 2}
        )
        await storage.store(memory2)
        
        # Verify only one memory exists with updated data
        cursor = await storage._connection.execute(
            "SELECT COUNT(*), data FROM memories WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        
        assert row[0] == 1  # Only one row
        assert json.loads(row[1]) == {"version": 2}  # Updated data

    async def test_retrieve_existing_memory(self, storage_with_data: SQLiteMemoryStorage):
        """Test retrieving an existing memory."""
        retrieved = await storage_with_data.retrieve("test_key_0")
        
        assert retrieved is not None
        assert retrieved.key == "test_key_0"
        assert retrieved.data["index"] == 0
        assert retrieved.data["batch"] is True

    async def test_retrieve_nonexistent_memory(self, storage: SQLiteMemoryStorage):
        """Test retrieving a non-existent memory returns None."""
        retrieved = await storage.retrieve("nonexistent_key")
        assert retrieved is None

    async def test_delete_existing_memory(self, storage_with_data: SQLiteMemoryStorage):
        """Test deleting an existing memory."""
        # Verify memory exists
        retrieved = await storage_with_data.retrieve("test_key_1")
        assert retrieved is not None
        
        # Delete the memory
        deleted = await storage_with_data.delete("test_key_1")
        assert deleted is True
        
        # Verify memory no longer exists
        retrieved = await storage_with_data.retrieve("test_key_1")
        assert retrieved is None

    async def test_delete_nonexistent_memory(self, storage: SQLiteMemoryStorage):
        """Test deleting a non-existent memory returns False."""
        deleted = await storage.delete("nonexistent_key")
        assert deleted is False

    async def test_list_memories_all(self, storage_with_data: SQLiteMemoryStorage):
        """Test listing all memories."""
        query = MemoryQuery()
        memories = await storage_with_data.list_memories(query)
        
        assert len(memories) == 5
        AssertionHelpers.assert_memory_list_properties(
            memories,
            expected_count=5,
            expected_keys=[f"test_key_{i}" for i in range(5)]
        )

    async def test_list_memories_with_key_filter(self, storage_with_data: SQLiteMemoryStorage):
        """Test listing memories filtered by keys."""
        query = MemoryQuery(keys=["test_key_1", "test_key_3"])
        memories = await storage_with_data.list_memories(query)
        
        assert len(memories) == 2
        keys = [m.key for m in memories]
        assert "test_key_1" in keys
        assert "test_key_3" in keys

    async def test_list_memories_with_type_filter(self, storage: SQLiteMemoryStorage):
        """Test listing memories filtered by memory type."""
        # Store memories of different types
        await storage.store(MemoryTestHelper.create_test_memory(
            "ephemeral_1", memory_type=MemoryType.EPHEMERAL
        ))
        await storage.store(MemoryTestHelper.create_test_memory(
            "short_1", memory_type=MemoryType.SHORT_TERM
        ))
        await storage.store(MemoryTestHelper.create_test_memory(
            "long_1", memory_type=MemoryType.LONG_TERM
        ))
        
        # Filter by short-term only
        query = MemoryQuery(memory_types=[MemoryType.SHORT_TERM])
        memories = await storage.list_memories(query)
        
        assert len(memories) == 1
        assert memories[0].memory_type == MemoryType.SHORT_TERM

    async def test_list_memories_exclude_expired(self, storage: SQLiteMemoryStorage):
        """Test listing memories excludes expired ones by default."""
        # Store valid memory
        valid_memory = MemoryTestHelper.create_test_memory("valid_key")
        await storage.store(valid_memory)
        
        # Store expired memory
        expired_memory = MemoryTestHelper.create_test_memory(
            "expired_key",
            expired=True
        )
        await storage.store(expired_memory)
        
        # List without including expired
        query = MemoryQuery(include_expired=False)
        memories = await storage.list_memories(query)
        
        # Should only get the valid memory
        assert len(memories) == 1
        assert memories[0].key == "valid_key"

    async def test_list_memories_include_expired(self, storage: SQLiteMemoryStorage):
        """Test listing memories can include expired ones."""
        # Store valid memory
        valid_memory = MemoryTestHelper.create_test_memory("valid_key")
        await storage.store(valid_memory)
        
        # Store expired memory
        expired_memory = MemoryTestHelper.create_test_memory(
            "expired_key",
            expired=True
        )
        await storage.store(expired_memory)
        
        # List including expired
        query = MemoryQuery(include_expired=True)
        memories = await storage.list_memories(query)
        
        # Should get both memories
        assert len(memories) == 2
        keys = [m.key for m in memories]
        assert "valid_key" in keys
        assert "expired_key" in keys

    async def test_list_memories_with_pagination(self, storage_with_data: SQLiteMemoryStorage):
        """Test memory listing with pagination."""
        # Test limit
        query = MemoryQuery(limit=3)
        memories = await storage_with_data.list_memories(query)
        assert len(memories) == 3
        
        # Test offset
        query = MemoryQuery(limit=2, offset=2)
        memories = await storage_with_data.list_memories(query)
        assert len(memories) == 2

    async def test_cleanup_expired_removes_expired_memories(self, storage: SQLiteMemoryStorage):
        """Test cleanup removes only expired memories."""
        # Store valid memory
        valid_memory = MemoryTestHelper.create_test_memory("valid_key")
        await storage.store(valid_memory)
        
        # Store expired memory
        expired_memory = MemoryTestHelper.create_test_memory(
            "expired_key",
            expired=True
        )
        await storage.store(expired_memory)
        
        # Run cleanup
        removed_count = await storage.cleanup_expired()
        
        # Should remove 1 expired memory
        assert removed_count == 1
        
        # Verify valid memory still exists
        valid_retrieved = await storage.retrieve("valid_key")
        assert valid_retrieved is not None
        
        # Verify expired memory is gone
        expired_retrieved = await storage.retrieve("expired_key")
        assert expired_retrieved is None

    async def test_cleanup_expired_no_expired_memories(self, storage_with_data: SQLiteMemoryStorage):
        """Test cleanup with no expired memories."""
        removed_count = await storage_with_data.cleanup_expired()
        assert removed_count == 0

    async def test_get_stats_empty_storage(self, storage: SQLiteMemoryStorage):
        """Test getting stats from empty storage."""
        stats = await storage.get_stats()
        
        assert stats.total_memories == 0
        assert stats.expired_memories == 0
        assert isinstance(stats.memories_by_type, dict)

    async def test_get_stats_with_data(self, storage: SQLiteMemoryStorage):
        """Test getting stats with data."""
        # Store memories of different types
        await storage.store(MemoryTestHelper.create_test_memory(
            "short_1", memory_type=MemoryType.SHORT_TERM
        ))
        await storage.store(MemoryTestHelper.create_test_memory(
            "short_2", memory_type=MemoryType.SHORT_TERM
        ))
        await storage.store(MemoryTestHelper.create_test_memory(
            "long_1", memory_type=MemoryType.LONG_TERM
        ))
        
        stats = await storage.get_stats()
        
        assert stats.total_memories == 3
        assert stats.memories_by_type.get("short_term", 0) == 2
        assert stats.memories_by_type.get("long_term", 0) == 1

    async def test_row_to_memory_conversion(self, storage: SQLiteMemoryStorage):
        """Test conversion from database row to Memory object."""
        original_memory = MemoryTestHelper.create_test_memory(
            key="conversion_test",
            data={"test": "conversion"},
            memory_type=MemoryType.LONG_TERM,
            ttl_seconds=7200
        )
        
        await storage.store(original_memory)
        retrieved = await storage.retrieve("conversion_test")
        
        # Verify all fields are properly converted
        assert retrieved.key == original_memory.key
        assert retrieved.data == original_memory.data
        assert retrieved.memory_type == original_memory.memory_type
        assert retrieved.ttl_seconds == original_memory.ttl_seconds

    async def test_error_handling_without_initialization(self):
        """Test that operations fail without initialization."""
        settings = Settings()
        storage = SQLiteMemoryStorage(settings)
        # Don't initialize
        
        memory = MemoryTestHelper.create_test_memory()
        
        with pytest.raises(DatabaseError):
            await storage.store(memory)
        
        with pytest.raises(DatabaseError):
            await storage.retrieve("test")
        
        with pytest.raises(DatabaseError):
            await storage.delete("test")

    async def test_close_cleans_up_connection(self, storage: SQLiteMemoryStorage):
        """Test that close() properly cleans up connection."""
        assert storage._connection is not None
        
        await storage.close()
        
        assert storage._connection is None

    async def test_concurrent_operations(self, storage: SQLiteMemoryStorage):
        """Test concurrent storage operations."""
        memories = [
            MemoryTestHelper.create_test_memory(f"concurrent_{i}")
            for i in range(10)
        ]
        
        # Store memories concurrently
        await asyncio.gather(*[
            storage.store(memory) for memory in memories
        ])
        
        # Retrieve memories concurrently
        retrieved = await asyncio.gather(*[
            storage.retrieve(f"concurrent_{i}") for i in range(10)
        ])
        
        # Verify all were stored and retrieved
        assert len(retrieved) == 10
        assert all(memory is not None for memory in retrieved)


class TestRedisMemoryStorage:
    """Test Redis memory storage implementation."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return MockFactory.create_redis_mock()

    @pytest_asyncio.fixture
    async def storage(self, test_settings: Settings, mock_redis):
        """Create Redis storage with mocked Redis client."""
        storage = RedisMemoryStorage(test_settings)
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            await storage.initialize()
            storage._redis = mock_redis
            yield storage
            await storage.close()

    async def test_initialize_connects_to_redis(self, test_settings: Settings, mock_redis):
        """Test Redis initialization and connection."""
        storage = RedisMemoryStorage(test_settings)
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            await storage.initialize()
            
            mock_redis.ping.assert_called_once()
            assert storage._redis is mock_redis

    async def test_initialize_connection_failure(self, test_settings: Settings):
        """Test Redis initialization with connection failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        storage = RedisMemoryStorage(test_settings)
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            with pytest.raises(ConnectionError):
                await storage.initialize()

    async def test_store_memory_without_ttl(self, storage: RedisMemoryStorage):
        """Test storing memory without TTL in Redis."""
        memory = MemoryTestHelper.create_test_memory(
            key="redis_test",
            ttl_seconds=0
        )
        
        await storage.store(memory)
        
        storage._redis.set.assert_called_once()
        # Should use set() not setex() for permanent memories
        assert not storage._redis.setex.called

    async def test_store_memory_with_ttl(self, storage: RedisMemoryStorage):
        """Test storing memory with TTL in Redis."""
        memory = MemoryTestHelper.create_test_memory(
            key="redis_ttl_test",
            ttl_seconds=3600
        )
        
        await storage.store(memory)
        
        storage._redis.setex.assert_called_once()
        call_args = storage._redis.setex.call_args
        assert call_args[0][0] == "memory:redis_ttl_test"  # key
        assert call_args[0][1] == 3600  # ttl

    async def test_retrieve_existing_memory(self, storage: RedisMemoryStorage):
        """Test retrieving existing memory from Redis."""
        # Mock Redis to return memory data
        test_memory = MemoryTestHelper.create_test_memory("redis_retrieve")
        storage._redis.get.return_value = test_memory.model_dump_json()
        
        retrieved = await storage.retrieve("redis_retrieve")
        
        assert retrieved is not None
        assert retrieved.key == "redis_retrieve"
        storage._redis.get.assert_called_once_with("memory:redis_retrieve")

    async def test_retrieve_nonexistent_memory(self, storage: RedisMemoryStorage):
        """Test retrieving non-existent memory from Redis."""
        storage._redis.get.return_value = None
        
        retrieved = await storage.retrieve("nonexistent")
        
        assert retrieved is None

    async def test_delete_existing_memory(self, storage: RedisMemoryStorage):
        """Test deleting existing memory from Redis."""
        storage._redis.delete.return_value = 1  # 1 key deleted
        
        deleted = await storage.delete("redis_delete")
        
        assert deleted is True
        storage._redis.delete.assert_called_once_with("memory:redis_delete")

    async def test_delete_nonexistent_memory(self, storage: RedisMemoryStorage):
        """Test deleting non-existent memory from Redis."""
        storage._redis.delete.return_value = 0  # 0 keys deleted
        
        deleted = await storage.delete("nonexistent")
        
        assert deleted is False

    async def test_list_memories_with_filters(self, storage: RedisMemoryStorage):
        """Test listing memories with filters in Redis."""
        # Mock scan_iter to return memory keys
        test_memories = [
            MemoryTestHelper.create_test_memory("redis_1", memory_type=MemoryType.SHORT_TERM),
            MemoryTestHelper.create_test_memory("redis_2", memory_type=MemoryType.LONG_TERM),
        ]
        
        async def mock_scan_iter(match):
            for i, memory in enumerate(test_memories):
                yield f"memory:redis_{i+1}"
        
        storage._redis.scan_iter.return_value = mock_scan_iter("memory:*")
        
        # Mock get to return memory data
        def mock_get(key):
            if key == "memory:redis_1":
                return test_memories[0].model_dump_json()
            elif key == "memory:redis_2":
                return test_memories[1].model_dump_json()
            return None
        
        storage._redis.get.side_effect = mock_get
        
        # Test filtering by memory type
        query = MemoryQuery(memory_types=[MemoryType.SHORT_TERM])
        memories = await storage.list_memories(query)
        
        assert len(memories) == 1
        assert memories[0].memory_type == MemoryType.SHORT_TERM

    async def test_cleanup_expired_returns_zero(self, storage: RedisMemoryStorage):
        """Test cleanup_expired returns 0 (Redis handles TTL automatically)."""
        removed_count = await storage.cleanup_expired()
        assert removed_count == 0

    async def test_get_stats(self, storage: RedisMemoryStorage):
        """Test getting stats from Redis."""
        # Mock scan_iter to return 3 memory keys
        async def mock_scan_iter(match):
            for i in range(3):
                yield f"memory:key_{i}"
        
        storage._redis.scan_iter.return_value = mock_scan_iter("memory:*")
        
        stats = await storage.get_stats()
        
        assert stats.total_memories == 3
        assert isinstance(stats.memories_by_type, dict)

    async def test_error_handling_without_initialization(self):
        """Test operations fail without initialization."""
        settings = Settings()
        storage = RedisMemoryStorage(settings)
        # Don't initialize
        
        memory = MemoryTestHelper.create_test_memory()
        
        with pytest.raises(ConnectionError):
            await storage.store(memory)
        
        with pytest.raises(ConnectionError):
            await storage.retrieve("test")

    async def test_close_cleans_up_connection(self, storage: RedisMemoryStorage):
        """Test close() cleans up Redis connection."""
        await storage.close()
        
        storage._redis.close.assert_called_once()
        assert storage._redis is None


class TestMemoryStorageProtocol:
    """Test the MemoryStorage abstract protocol."""

    def test_abstract_methods_defined(self):
        """Test that MemoryStorage defines all required abstract methods."""
        abstract_methods = MemoryStorage.__abstractmethods__
        
        expected_methods = {
            'initialize', 'close', 'store', 'retrieve', 'delete',
            'list_memories', 'cleanup_expired', 'get_stats'
        }
        
        assert abstract_methods == expected_methods

    def test_cannot_instantiate_abstract_class(self):
        """Test that MemoryStorage cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryStorage()

    def test_concrete_implementations_satisfy_protocol(self, test_settings: Settings):
        """Test that concrete implementations satisfy the protocol."""
        sqlite_storage = SQLiteMemoryStorage(test_settings)
        redis_storage = RedisMemoryStorage(test_settings)
        
        assert isinstance(sqlite_storage, MemoryStorage)
        assert isinstance(redis_storage, MemoryStorage)