"""True unit tests for MemoryManager with full mocking."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta, UTC

from mcp_synaptic.memory.manager import MemoryManager
from mcp_synaptic.memory.storage.base import MemoryStorage
from mcp_synaptic.models.memory import Memory, MemoryType, MemoryQuery, MemoryStats
from mcp_synaptic.config.settings import Settings
from mcp_synaptic.core.exceptions import MemoryError, MemoryNotFoundError, ValidationError


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.DEFAULT_MEMORY_TTL_SECONDS = 3600
    settings.MEMORY_CLEANUP_INTERVAL_SECONDS = 300
    settings.REDIS_ENABLED = False
    settings.MAX_MEMORY_ENTRIES = 10000
    return settings


@pytest.fixture
def mock_memory_storage():
    """Create AsyncMock storage following claude-1's pattern."""
    mock = AsyncMock(spec=MemoryStorage)
    
    # Configure default behaviors
    mock.initialize = AsyncMock()
    mock.close = AsyncMock()
    mock.store = AsyncMock()
    mock.retrieve = AsyncMock(return_value=None)
    mock.delete = AsyncMock(return_value=True)
    mock.list_memories = AsyncMock(return_value=[])
    mock.cleanup_expired = AsyncMock(return_value=0)
    mock.get_stats = AsyncMock(return_value=MemoryStats(
        generated_at=datetime.now(UTC),
        total_memories=0,
        memories_by_type={},
        expired_memories=0,
        total_size_bytes=0,
        average_ttl_seconds=None,
        oldest_memory=None,
        newest_memory=None,
        most_accessed_count=0
    ))
    
    return mock


@pytest.fixture
def memory_manager(mock_settings, mock_memory_storage):
    """Create MemoryManager with mocked dependencies."""
    from mcp_synaptic.memory.manager.operations import MemoryOperations
    from mcp_synaptic.memory.manager.queries import MemoryQueries
    
    manager = MemoryManager(mock_settings)
    manager.storage = mock_memory_storage
    manager._initialized = True
    
    # Mock operation handlers
    manager._operations = AsyncMock(spec=MemoryOperations)
    manager._queries = AsyncMock(spec=MemoryQueries)
    
    return manager


class TestMemoryManagerInitialization:
    """Test MemoryManager initialization without real storage."""

    @pytest.mark.asyncio
    async def test_initialize_creates_storage(self, mock_settings):
        """Test initialization creates appropriate storage backend."""
        # Arrange
        with patch('mcp_synaptic.memory.manager.core.SQLiteMemoryStorage') as mock_sqlite:
            mock_storage_instance = AsyncMock()
            mock_sqlite.return_value = mock_storage_instance
            
            manager = MemoryManager(mock_settings)
            
            # Act
            await manager.initialize()
            
            # Assert
            mock_sqlite.assert_called_once_with(mock_settings)
            mock_storage_instance.initialize.assert_called_once()
            assert manager.storage == mock_storage_instance
            assert manager._initialized

    @pytest.mark.asyncio
    async def test_initialize_with_redis_enabled(self, mock_settings):
        """Test initialization with Redis backend."""
        # Arrange
        mock_settings.REDIS_ENABLED = True
        
        with patch('mcp_synaptic.memory.manager.core.RedisMemoryStorage') as mock_redis:
            mock_storage_instance = AsyncMock()
            mock_redis.return_value = mock_storage_instance
            
            manager = MemoryManager(mock_settings)
            
            # Act
            await manager.initialize()
            
            # Assert
            mock_redis.assert_called_once_with(mock_settings)
            mock_storage_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_storage_failure_raises_error(self, mock_settings):
        """Test initialization failure handling."""
        # Arrange
        with patch('mcp_synaptic.memory.manager.core.SQLiteMemoryStorage') as mock_sqlite:
            mock_storage_instance = AsyncMock()
            mock_storage_instance.initialize.side_effect = Exception("Storage init failed")
            mock_sqlite.return_value = mock_storage_instance
            
            manager = MemoryManager(mock_settings)
            
            # Act & Assert
            with pytest.raises(MemoryError) as exc_info:
                await manager.initialize()
            
            assert "Memory manager initialization failed" in str(exc_info.value)
            assert not manager._initialized

    @pytest.mark.asyncio
    async def test_close_cleanup(self, memory_manager, mock_memory_storage):
        """Test proper cleanup on close."""
        # Act
        await memory_manager.close()
        
        # Assert
        mock_memory_storage.close.assert_called_once()
        assert not memory_manager._initialized


class TestMemoryManagerCRUD:
    """Test CRUD operations with mocked storage."""

    @pytest.mark.asyncio
    async def test_add_memory_success(self, memory_manager, mock_memory_storage):
        """Test successful memory addition."""
        # Arrange
        key = "test_key"
        data = {"test": "data"}
        memory_type = MemoryType.SHORT_TERM
        expected_memory = Memory(key=key, data=data, memory_type=memory_type)
        memory_manager._operations.add.return_value = expected_memory
        
        # Act
        result = await memory_manager.add(
            key=key,
            data=data,
            memory_type=memory_type
        )
        
        # Assert
        memory_manager._operations.add.assert_called_once_with(key, data, memory_type, None, None, None, None)
        assert result == expected_memory

    @pytest.mark.asyncio
    async def test_add_memory_with_default_ttl(self, memory_manager, mock_memory_storage, mock_settings):
        """Test memory addition uses default TTL when not specified."""
        # Arrange
        key = "test_key"
        data = {"test": "data"}
        
        # Act
        await memory_manager.add(key=key, data=data)
        
        # Assert
        mock_memory_storage.store.assert_called_once()
        stored_memory = mock_memory_storage.store.call_args[0][0]
        assert stored_memory.ttl_seconds == mock_settings.DEFAULT_MEMORY_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_add_memory_storage_failure(self, memory_manager, mock_memory_storage):
        """Test add_memory handles storage failures."""
        # Arrange
        mock_memory_storage.store.side_effect = Exception("Storage error")
        
        # Act & Assert
        with pytest.raises(MemoryError) as exc_info:
            await memory_manager.add("key", {"data": "test"})
        
        assert "Failed to add memory" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_memory_success(self, memory_manager, mock_memory_storage):
        """Test successful memory retrieval."""
        # Arrange
        key = "test_key"
        expected_memory = Memory(key=key, data={"test": "data"})
        mock_memory_storage.retrieve.return_value = expected_memory
        
        # Act
        result = await memory_manager.get(key)
        
        # Assert
        mock_memory_storage.retrieve.assert_called_once_with(key)
        assert result == expected_memory

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, memory_manager, mock_memory_storage):
        """Test memory retrieval when key doesn't exist."""
        # Arrange
        mock_memory_storage.retrieve.return_value = None
        
        # Act
        result = await memory_manager.get("nonexistent_key")
        
        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_memory_without_touch(self, memory_manager, mock_memory_storage):
        """Test memory retrieval without touching (updating access info)."""
        # Arrange
        key = "test_key"
        expected_memory = Memory(key=key, data={"test": "data"})
        mock_memory_storage.retrieve.return_value = expected_memory
        
        # Act
        result = await memory_manager.get(key, touch=False)
        
        # Assert
        mock_memory_storage.retrieve.assert_called_once_with(key)
        assert result == expected_memory

    @pytest.mark.asyncio
    async def test_update_memory_success(self, memory_manager, mock_memory_storage):
        """Test successful memory update."""
        # Arrange
        key = "test_key"
        new_data = {"updated": "data"}
        existing_memory = Memory(key=key, data={"old": "data"})
        mock_memory_storage.retrieve.return_value = existing_memory
        
        # Act
        result = await memory_manager.update(key, data=new_data)
        
        # Assert
        mock_memory_storage.retrieve.assert_called_once_with(key)
        mock_memory_storage.store.assert_called_once()
        
        stored_memory = mock_memory_storage.store.call_args[0][0]
        assert stored_memory.key == key
        assert stored_memory.data == new_data
        assert result == stored_memory

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, memory_manager, mock_memory_storage):
        """Test update_memory when memory doesn't exist."""
        # Arrange
        mock_memory_storage.retrieve.return_value = None
        
        # Act & Assert
        with pytest.raises(MemoryNotFoundError) as exc_info:
            await memory_manager.update("nonexistent", data={"data": "new"})
        
        assert "nonexistent" in str(exc_info.value)
        mock_memory_storage.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, memory_manager, mock_memory_storage):
        """Test successful memory deletion."""
        # Arrange
        key = "test_key"
        mock_memory_storage.delete.return_value = True
        
        # Act
        result = await memory_manager.delete(key)
        
        # Assert
        mock_memory_storage.delete.assert_called_once_with(key)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, memory_manager, mock_memory_storage):
        """Test delete_memory when key doesn't exist."""
        # Arrange
        mock_memory_storage.delete.return_value = False
        
        # Act
        result = await memory_manager.delete("nonexistent")
        
        # Assert
        assert result is False


class TestMemoryManagerOperations:
    """Test additional memory manager operations."""

    @pytest.mark.asyncio
    async def test_list_memories_success(self, memory_manager, mock_memory_storage):
        """Test successful memory listing."""
        # Arrange
        query = MemoryQuery(limit=5)
        expected_memories = [
            Memory(key="key1", data={"data": "1"}),
            Memory(key="key2", data={"data": "2"})
        ]
        mock_memory_storage.list_memories.return_value = expected_memories
        
        # Act
        result = await memory_manager.list(query)
        
        # Assert
        mock_memory_storage.list_memories.assert_called_once_with(query)
        assert result == expected_memories

    @pytest.mark.asyncio
    async def test_list_memories_with_default_query(self, memory_manager, mock_memory_storage):
        """Test memory listing with default query."""
        # Act
        await memory_manager.list()
        
        # Assert
        mock_memory_storage.list_memories.assert_called_once()
        called_query = mock_memory_storage.list_memories.call_args[0][0]
        assert isinstance(called_query, MemoryQuery)

    @pytest.mark.asyncio
    async def test_exists_memory_found(self, memory_manager, mock_memory_storage):
        """Test memory existence check when memory exists."""
        # Arrange
        mock_memory_storage.retrieve.return_value = Memory(key="test", data={})
        
        # Act
        result = await memory_manager.exists("test")
        
        # Assert
        mock_memory_storage.retrieve.assert_called_once_with("test")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_memory_not_found(self, memory_manager, mock_memory_storage):
        """Test memory existence check when memory doesn't exist."""
        # Arrange
        mock_memory_storage.retrieve.return_value = None
        
        # Act
        result = await memory_manager.exists("nonexistent")
        
        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_touch_memory_success(self, memory_manager, mock_memory_storage):
        """Test successful memory touch operation."""
        # Arrange
        key = "test_key"
        memory = Memory(key=key, data={"test": "data"})
        mock_memory_storage.retrieve.return_value = memory
        
        # Act
        result = await memory_manager.touch(key)
        
        # Assert
        mock_memory_storage.retrieve.assert_called_once_with(key)
        assert result is True

    @pytest.mark.asyncio
    async def test_touch_memory_not_found(self, memory_manager, mock_memory_storage):
        """Test touch_memory when memory doesn't exist."""
        # Arrange
        mock_memory_storage.retrieve.return_value = None
        
        # Act
        result = await memory_manager.touch("nonexistent")
        
        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_success(self, memory_manager, mock_memory_storage):
        """Test successful cleanup of expired memories."""
        # Arrange
        mock_memory_storage.cleanup_expired.return_value = 5
        
        # Act
        result = await memory_manager.cleanup_expired()
        
        # Assert
        mock_memory_storage.cleanup_expired.assert_called_once()
        assert result == 5

    @pytest.mark.asyncio
    async def test_get_stats_success(self, memory_manager, mock_memory_storage):
        """Test successful statistics retrieval."""
        # Arrange
        expected_stats = MemoryStats(
            generated_at=datetime.now(UTC),
            total_memories=10,
            memories_by_type={MemoryType.SHORT_TERM: 8, MemoryType.LONG_TERM: 2},
            expired_memories=1,
            total_size_bytes=2048,
            average_ttl_seconds=1800.0,
            oldest_memory=datetime.now(UTC) - timedelta(days=1),
            newest_memory=datetime.now(UTC) - timedelta(minutes=5),
            most_accessed_count=25
        )
        mock_memory_storage.get_stats.return_value = expected_stats
        
        # Act
        result = await memory_manager.get_stats()
        
        # Assert
        mock_memory_storage.get_stats.assert_called_once()
        assert result == expected_stats


class TestMemoryManagerValidation:
    """Test validation and error handling."""

    @pytest.mark.asyncio
    async def test_ensure_initialized_check_passes(self, memory_manager):
        """Test that initialized manager passes validation."""
        # Act & Assert - Should not raise
        await memory_manager.get("test")

    @pytest.mark.asyncio
    async def test_ensure_initialized_check_fails(self, mock_settings):
        """Test that uninitialized manager raises error."""
        # Arrange
        manager = MemoryManager(mock_settings)
        # Don't initialize
        
        # Act & Assert
        with pytest.raises(MemoryError) as exc_info:
            await manager.get("test")
        
        assert "Memory manager not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_default_ttl_for_memory_type(self, memory_manager, mock_settings):
        """Test default TTL retrieval for different memory types."""
        # Act & Assert
        ephemeral_ttl = memory_manager._get_default_ttl(MemoryType.EPHEMERAL)
        short_term_ttl = memory_manager._get_default_ttl(MemoryType.SHORT_TERM)
        long_term_ttl = memory_manager._get_default_ttl(MemoryType.LONG_TERM)
        permanent_ttl = memory_manager._get_default_ttl(MemoryType.PERMANENT)
        
        assert ephemeral_ttl == 300  # 5 minutes
        assert short_term_ttl == mock_settings.DEFAULT_MEMORY_TTL_SECONDS
        assert long_term_ttl == 604800  # 1 week
        assert permanent_ttl == 0  # Never expires