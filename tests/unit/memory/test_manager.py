"""Tests for memory manager."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.core.exceptions import MemoryError, MemoryExpiredError, MemoryNotFoundError
from mcp_synaptic.memory.manager import MemoryManager
from mcp_synaptic.memory.storage import MemoryStorage
from mcp_synaptic.models.memory import ExpirationPolicy, Memory, MemoryQuery, MemoryStats, MemoryType
from tests.utils import MemoryTestHelper, AsyncContextHelper, AssertionHelpers


class TestMemoryManager:
    """Test memory manager functionality."""

    @pytest_asyncio.fixture
    async def memory_manager(self, test_settings: Settings) -> MemoryManager:
        """Create and initialize memory manager for testing."""
        manager = MemoryManager(test_settings)
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = AsyncMock(spec=MemoryStorage)
        return storage

    @pytest_asyncio.fixture
    async def manager_with_mock_storage(self, test_settings: Settings, mock_storage):
        """Create memory manager with mock storage."""
        manager = MemoryManager(test_settings)
        manager.storage = mock_storage
        manager._initialized = True
        yield manager

    async def test_initialize_with_sqlite_storage(self, test_settings: Settings):
        """Test initialization with SQLite storage backend."""
        test_settings.REDIS_ENABLED = False
        manager = MemoryManager(test_settings)
        
        await manager.initialize()
        
        assert manager._initialized is True
        assert manager.storage is not None
        assert "SQLiteMemoryStorage" in str(type(manager.storage))
        
        await manager.close()

    async def test_initialize_with_redis_storage(self, test_settings: Settings):
        """Test initialization with Redis storage backend."""
        test_settings.REDIS_ENABLED = True
        manager = MemoryManager(test_settings)
        
        # Mock Redis storage initialization
        with patch('mcp_synaptic.memory.manager.RedisMemoryStorage') as mock_redis_cls:
            mock_redis_instance = AsyncMock()
            mock_redis_cls.return_value = mock_redis_instance
            
            await manager.initialize()
            
            assert manager._initialized is True
            assert manager.storage is mock_redis_instance
            mock_redis_instance.initialize.assert_called_once()

    async def test_initialize_failure(self, test_settings: Settings):
        """Test initialization failure handling."""
        manager = MemoryManager(test_settings)
        
        with patch('mcp_synaptic.memory.manager.SQLiteMemoryStorage') as mock_cls:
            mock_storage = AsyncMock()
            mock_storage.initialize.side_effect = Exception("Storage init failed")
            mock_cls.return_value = mock_storage
            
            with pytest.raises(MemoryError, match="Memory manager initialization failed"):
                await manager.initialize()

    async def test_close_cleanup(self, memory_manager: MemoryManager):
        """Test that close properly cleans up resources."""
        assert memory_manager._initialized is True
        
        await memory_manager.close()
        
        assert memory_manager._initialized is False

    async def test_ensure_initialized_check(self, test_settings: Settings):
        """Test that operations require initialization."""
        manager = MemoryManager(test_settings)
        # Don't initialize
        
        with pytest.raises(MemoryError, match="Memory manager not initialized"):
            await manager.add("test_key", {"data": "test"})

    async def test_add_memory_success(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test successful memory addition."""
        test_data = {"user": "test", "action": "login"}
        
        await manager_with_mock_storage.add(
            key="test_add",
            data=test_data,
            memory_type=MemoryType.SHORT_TERM,
            ttl_seconds=3600
        )
        
        # Verify storage was called
        mock_storage.store.assert_called_once()
        stored_memory = mock_storage.store.call_args[0][0]
        
        assert stored_memory.key == "test_add"
        assert stored_memory.data == test_data
        assert stored_memory.memory_type == MemoryType.SHORT_TERM
        assert stored_memory.ttl_seconds == 3600

    async def test_add_memory_with_default_ttl(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test adding memory uses default TTL when not specified."""
        await manager_with_mock_storage.add(
            key="default_ttl",
            data={"test": "data"},
            memory_type=MemoryType.EPHEMERAL
        )
        
        stored_memory = mock_storage.store.call_args[0][0]
        assert stored_memory.ttl_seconds == 300  # EPHEMERAL default is 5 minutes

    async def test_add_memory_with_tags_and_metadata(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test adding memory with tags and metadata."""
        tags = {"type": "user_action", "priority": "high"}
        metadata = {"source": "api", "version": "1.0"}
        
        await manager_with_mock_storage.add(
            key="tagged_memory",
            data={"action": "purchase"},
            tags=tags,
            metadata=metadata
        )
        
        stored_memory = mock_storage.store.call_args[0][0]
        assert stored_memory.tags == tags
        assert stored_memory.metadata == metadata

    async def test_add_memory_failure(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test memory addition failure handling."""
        mock_storage.store.side_effect = Exception("Storage error")
        
        with pytest.raises(MemoryError, match="Failed to add memory 'failed_key'"):
            await manager_with_mock_storage.add("failed_key", {"data": "test"})

    async def test_get_memory_success(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test successful memory retrieval."""
        test_memory = MemoryTestHelper.create_test_memory("get_test")
        mock_storage.retrieve.return_value = test_memory
        
        retrieved = await manager_with_mock_storage.get("get_test")
        
        assert retrieved is test_memory
        mock_storage.retrieve.assert_called_once_with("get_test")
        mock_storage.store.assert_called_once()  # Should update access info

    async def test_get_memory_not_found(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test getting non-existent memory returns None."""
        mock_storage.retrieve.return_value = None
        
        retrieved = await manager_with_mock_storage.get("nonexistent")
        
        assert retrieved is None

    async def test_get_memory_expired(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test getting expired memory raises exception and deletes it."""
        expired_memory = MemoryTestHelper.create_test_memory("expired", expired=True)
        mock_storage.retrieve.return_value = expired_memory
        
        with pytest.raises(MemoryExpiredError):
            await manager_with_mock_storage.get("expired")
        
        # Should delete the expired memory
        mock_storage.delete.assert_called_once_with("expired")

    async def test_get_memory_without_touch(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test getting memory without updating access info."""
        test_memory = MemoryTestHelper.create_test_memory("no_touch")
        mock_storage.retrieve.return_value = test_memory
        
        retrieved = await manager_with_mock_storage.get("no_touch", touch=False)
        
        assert retrieved is test_memory
        # Should not call store to update access info
        mock_storage.store.assert_not_called()

    async def test_get_memory_failure(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test memory retrieval failure handling."""
        mock_storage.retrieve.side_effect = Exception("Storage error")
        
        with pytest.raises(MemoryError, match="Failed to get memory 'error_key'"):
            await manager_with_mock_storage.get("error_key")

    async def test_delete_memory_success(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test successful memory deletion."""
        mock_storage.delete.return_value = True
        
        deleted = await manager_with_mock_storage.delete("delete_test")
        
        assert deleted is True
        mock_storage.delete.assert_called_once_with("delete_test")

    async def test_delete_memory_not_found(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test deleting non-existent memory."""
        mock_storage.delete.return_value = False
        
        deleted = await manager_with_mock_storage.delete("nonexistent")
        
        assert deleted is False

    async def test_delete_memory_failure(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test memory deletion failure handling."""
        mock_storage.delete.side_effect = Exception("Storage error")
        
        with pytest.raises(MemoryError, match="Failed to delete memory 'error_key'"):
            await manager_with_mock_storage.delete("error_key")

    async def test_update_memory_success(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test successful memory update."""
        original_memory = MemoryTestHelper.create_test_memory("update_test")
        mock_storage.retrieve.return_value = original_memory
        
        new_data = {"updated": True}
        new_tags = {"status": "updated"}
        
        updated = await manager_with_mock_storage.update(
            "update_test",
            data=new_data,
            tags=new_tags,
            extend_ttl=7200
        )
        
        assert updated is not None
        assert updated.data == new_data
        assert updated.tags["status"] == "updated"
        assert updated.ttl_seconds == 7200

    async def test_update_memory_not_found(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test updating non-existent memory raises exception."""
        mock_storage.retrieve.return_value = None
        
        with pytest.raises(MemoryNotFoundError):
            await manager_with_mock_storage.update("nonexistent", data={"new": "data"})

    async def test_update_memory_expired(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test updating expired memory raises exception."""
        # Mock get method to raise MemoryExpiredError
        with patch.object(manager_with_mock_storage, 'get', side_effect=MemoryExpiredError("expired")):
            with pytest.raises(MemoryExpiredError):
                await manager_with_mock_storage.update("expired", data={"new": "data"})

    async def test_list_memories_success(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test successful memory listing."""
        test_memories = MemoryTestHelper.create_memory_batch(3)
        mock_storage.list_memories.return_value = test_memories
        
        query = MemoryQuery(limit=10)
        memories = await manager_with_mock_storage.list(query)
        
        assert len(memories) == 3
        mock_storage.list_memories.assert_called_once_with(query)

    async def test_list_memories_with_default_query(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test listing memories with default query."""
        mock_storage.list_memories.return_value = []
        
        memories = await manager_with_mock_storage.list()
        
        # Should create default MemoryQuery
        mock_storage.list_memories.assert_called_once()
        query_arg = mock_storage.list_memories.call_args[0][0]
        assert isinstance(query_arg, MemoryQuery)

    async def test_cleanup_expired_success(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test successful expired memory cleanup."""
        mock_storage.cleanup_expired.return_value = 5
        
        removed_count = await manager_with_mock_storage.cleanup_expired()
        
        assert removed_count == 5
        mock_storage.cleanup_expired.assert_called_once()

    async def test_cleanup_expired_failure(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test cleanup failure handling."""
        mock_storage.cleanup_expired.side_effect = Exception("Cleanup error")
        
        with pytest.raises(MemoryError, match="Failed to cleanup expired memories"):
            await manager_with_mock_storage.cleanup_expired()

    async def test_get_stats_success(self, manager_with_mock_storage: MemoryManager, mock_storage):
        """Test successful stats retrieval."""
        test_stats = MemoryStats(
            total_memories=10,
            memories_by_type={MemoryType.SHORT_TERM: 5, MemoryType.LONG_TERM: 5},
            expired_memories=2,
            total_size_bytes=1024
        )
        mock_storage.get_stats.return_value = test_stats
        
        stats = await manager_with_mock_storage.get_stats()
        
        assert stats is test_stats
        mock_storage.get_stats.assert_called_once()

    async def test_exists_memory_found(self, manager_with_mock_storage: MemoryManager):
        """Test exists() returns True for existing memory."""
        with patch.object(manager_with_mock_storage, 'get') as mock_get:
            mock_get.return_value = MemoryTestHelper.create_test_memory("exists")
            
            exists = await manager_with_mock_storage.exists("exists")
            
            assert exists is True
            mock_get.assert_called_once_with("exists", touch=False)

    async def test_exists_memory_not_found(self, manager_with_mock_storage: MemoryManager):
        """Test exists() returns False for non-existent memory."""
        with patch.object(manager_with_mock_storage, 'get') as mock_get:
            mock_get.return_value = None
            
            exists = await manager_with_mock_storage.exists("nonexistent")
            
            assert exists is False

    async def test_exists_memory_expired(self, manager_with_mock_storage: MemoryManager):
        """Test exists() returns False for expired memory."""
        with patch.object(manager_with_mock_storage, 'get') as mock_get:
            mock_get.side_effect = MemoryExpiredError("expired")
            
            exists = await manager_with_mock_storage.exists("expired")
            
            assert exists is False

    async def test_touch_memory_success(self, manager_with_mock_storage: MemoryManager):
        """Test touch() successfully updates memory access."""
        with patch.object(manager_with_mock_storage, 'get') as mock_get:
            mock_get.return_value = MemoryTestHelper.create_test_memory("touch")
            
            touched = await manager_with_mock_storage.touch("touch")
            
            assert touched is True
            mock_get.assert_called_once_with("touch", touch=True)

    async def test_touch_memory_not_found(self, manager_with_mock_storage: MemoryManager):
        """Test touch() returns False for non-existent memory."""
        with patch.object(manager_with_mock_storage, 'get') as mock_get:
            mock_get.return_value = None
            
            touched = await manager_with_mock_storage.touch("nonexistent")
            
            assert touched is False

    async def test_get_default_ttl_values(self, memory_manager: MemoryManager):
        """Test default TTL values for different memory types."""
        assert memory_manager._get_default_ttl(MemoryType.EPHEMERAL) == 300  # 5 minutes
        assert memory_manager._get_default_ttl(MemoryType.SHORT_TERM) == memory_manager.settings.DEFAULT_MEMORY_TTL_SECONDS
        assert memory_manager._get_default_ttl(MemoryType.LONG_TERM) == 86400 * 7  # 1 week
        assert memory_manager._get_default_ttl(MemoryType.PERMANENT) == 0  # No expiration

    async def test_concurrent_operations(self, memory_manager: MemoryManager):
        """Test memory manager handles concurrent operations correctly."""
        # Add multiple memories concurrently
        add_tasks = [
            memory_manager.add(f"concurrent_{i}", {"index": i})
            for i in range(10)
        ]
        
        memories = await asyncio.gather(*add_tasks)
        assert len(memories) == 10
        
        # Retrieve memories concurrently
        get_tasks = [
            memory_manager.get(f"concurrent_{i}")
            for i in range(10)
        ]
        
        retrieved = await asyncio.gather(*get_tasks)
        assert len(retrieved) == 10
        assert all(memory is not None for memory in retrieved)

    async def test_memory_expiration_integration(self, memory_manager: MemoryManager):
        """Test memory expiration works end-to-end."""
        # Add memory with very short TTL
        memory = await memory_manager.add(
            "expiration_test",
            {"test": "data"},
            ttl_seconds=1
        )
        
        # Should be retrievable immediately
        retrieved = await memory_manager.get("expiration_test")
        assert retrieved is not None
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should now be expired
        with pytest.raises(MemoryExpiredError):
            await memory_manager.get("expiration_test")

    async def test_sliding_expiration_policy(self, memory_manager: MemoryManager):
        """Test sliding expiration policy updates on access."""
        # Add memory with sliding expiration
        memory = await memory_manager.add(
            "sliding_test",
            {"test": "data"},
            expiration_policy=ExpirationPolicy.SLIDING,
            ttl_seconds=2
        )
        
        original_expires_at = memory.expires_at
        
        # Wait a bit then access
        await asyncio.sleep(0.5)
        retrieved = await memory_manager.get("sliding_test")
        
        # Expiration should be extended
        assert retrieved.expires_at > original_expires_at

    async def test_permanent_memory_never_expires(self, memory_manager: MemoryManager):
        """Test permanent memories never expire."""
        memory = await memory_manager.add(
            "permanent_test",
            {"test": "data"},
            memory_type=MemoryType.PERMANENT
        )
        
        assert memory.expires_at is None
        assert not memory.is_expired

    async def test_memory_stats_integration(self, memory_manager: MemoryManager):
        """Test memory stats work end-to-end."""
        # Add memories of different types
        await memory_manager.add("short_1", {"type": "short"}, memory_type=MemoryType.SHORT_TERM)
        await memory_manager.add("long_1", {"type": "long"}, memory_type=MemoryType.LONG_TERM)
        await memory_manager.add("perm_1", {"type": "permanent"}, memory_type=MemoryType.PERMANENT)
        
        stats = await memory_manager.get_stats()
        
        assert stats.total_memories >= 3
        assert stats.memories_by_type.get("short_term", 0) >= 1
        assert stats.memories_by_type.get("long_term", 0) >= 1
        assert stats.memories_by_type.get("permanent", 0) >= 1