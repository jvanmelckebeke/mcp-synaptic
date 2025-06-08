"""True unit tests for Memory models with full mocking."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
from pydantic import ValidationError

from mcp_synaptic.models.memory import Memory, MemoryType, ExpirationPolicy, MemoryQuery, MemoryStats


class TestMemory:
    """True unit tests for Memory model - no external dependencies."""

    def test_memory_creation_with_defaults(self):
        """Test memory creation with default values."""
        # Arrange
        key = "test_key"
        data = {"test": "value"}
        
        # Act
        memory = Memory(key=key, data=data)
        
        # Assert
        assert memory.key == key
        assert memory.data == data
        assert memory.memory_type == MemoryType.SHORT_TERM
        assert memory.expiration_policy == ExpirationPolicy.ABSOLUTE
        assert memory.ttl_seconds is None  # Default is None, set by manager
        assert memory.access_count == 0
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)

    def test_memory_expiration_property_never_expires(self):
        """Test is_expired property for permanent memories."""
        # Arrange
        memory = Memory(
            key="permanent_key",
            data={"test": "data"},
            memory_type=MemoryType.PERMANENT,
            expiration_policy=ExpirationPolicy.NEVER,
            ttl_seconds=0
        )
        
        # Act & Assert
        assert not memory.is_expired

    @patch('mcp_synaptic.models.memory.datetime')
    def test_memory_expiration_property_expired(self, mock_datetime):
        """Test is_expired property for expired memories."""
        # Arrange - Mock current time to be after expiration
        past_time = datetime(2024, 1, 1, 12, 0, 0)
        current_time = datetime(2024, 1, 1, 13, 30, 0)  # 1.5 hours later
        
        mock_datetime.utcnow.return_value = current_time
        
        memory = Memory(
            key="expired_key",
            data={"test": "data"},
            ttl_seconds=3600  # 1 hour
        )
        memory.expires_at = past_time + timedelta(seconds=3600)
        
        # Act & Assert
        assert memory.is_expired

    @patch('mcp_synaptic.models.memory.datetime')
    def test_memory_expiration_property_not_expired(self, mock_datetime):
        """Test is_expired property for valid memories."""
        # Arrange - Mock current time to be before expiration
        current_time = datetime(2024, 1, 1, 12, 0, 0)
        future_time = datetime(2024, 1, 1, 14, 0, 0)  # 2 hours later
        
        mock_datetime.utcnow.return_value = current_time
        
        memory = Memory(
            key="valid_key",
            data={"test": "data"},
            ttl_seconds=7200  # 2 hours
        )
        memory.expires_at = future_time
        
        # Act & Assert
        assert not memory.is_expired

    @patch('mcp_synaptic.models.memory.datetime')
    def test_memory_touch_updates_access_info(self, mock_datetime):
        """Test touch method updates access timestamp and count."""
        # Arrange
        original_time = datetime(2024, 1, 1, 12, 0, 0)
        touch_time = datetime(2024, 1, 1, 12, 30, 0)
        
        mock_datetime.utcnow.return_value = original_time
        memory = Memory(key="test_key", data={"test": "data"})
        original_count = memory.access_count
        
        # Act
        mock_datetime.utcnow.return_value = touch_time
        memory.touch()
        
        # Assert
        assert memory.access_count == original_count + 1
        assert memory.last_accessed_at == touch_time

    @patch('mcp_synaptic.models.memory.datetime')
    def test_memory_touch_updates_sliding_expiration(self, mock_datetime):
        """Test touch method updates sliding expiration times."""
        # Arrange
        touch_time = datetime(2024, 1, 1, 12, 30, 0)
        mock_datetime.utcnow.return_value = touch_time
        
        memory = Memory(
            key="sliding_key",
            data={"test": "data"},
            expiration_policy=ExpirationPolicy.SLIDING,
            ttl_seconds=3600
        )
        
        # Act
        memory.touch()
        
        # Assert
        expected_expiry = touch_time + timedelta(seconds=3600)
        assert memory.expires_at == expected_expiry

    def test_memory_validation_negative_ttl_raises_error(self):
        """Test validation error for negative TTL values."""
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            Memory(
                key="invalid_key",
                data={"test": "data"},
                ttl_seconds=-100
            )
        
        assert "ttl_seconds must be non-negative" in str(exc_info.value)

    def test_memory_validation_accepts_zero_ttl(self):
        """Test that TTL of 0 is accepted (permanent memory)."""
        # Act
        memory = Memory(
            key="permanent_key",
            data={"test": "data"},
            ttl_seconds=0
        )
        
        # Assert
        assert memory.ttl_seconds == 0

    def test_memory_update_expiration_method(self):
        """Test update_expiration method with new TTL."""
        # Arrange
        memory = Memory(key="test_key", data={"test": "data"})
        new_ttl = 7200  # 2 hours
        original_expires_at = memory.expires_at
        
        # Act
        memory.update_expiration(new_ttl)
        
        # Assert
        # Method updates expires_at but not ttl_seconds field
        assert memory.expires_at != original_expires_at
        assert isinstance(memory.expires_at, datetime)

    def test_memory_serialization_roundtrip(self):
        """Test memory can be serialized and deserialized correctly."""
        # Arrange
        original_memory = Memory(
            key="serialization_test",
            data={"complex": {"nested": "data"}},
            memory_type=MemoryType.LONG_TERM,
            metadata={"source": "test"}
        )
        
        # Act
        json_data = original_memory.model_dump_json()
        restored_memory = Memory.model_validate_json(json_data)
        
        # Assert
        assert restored_memory.key == original_memory.key
        assert restored_memory.data == original_memory.data
        assert restored_memory.memory_type == original_memory.memory_type
        assert restored_memory.metadata == original_memory.metadata


class TestMemoryQuery:
    """Unit tests for MemoryQuery model."""

    def test_memory_query_defaults(self):
        """Test MemoryQuery creation with default values."""
        # Act
        query = MemoryQuery()
        
        # Assert
        assert query.query == ""
        assert query.limit == 10
        assert query.offset == 0
        assert query.keys is None
        assert query.memory_types is None
        assert query.tags is None
        assert not query.include_expired

    def test_memory_query_with_filters(self):
        """Test MemoryQuery with various filters."""
        # Arrange
        memory_types = [MemoryType.SHORT_TERM, MemoryType.LONG_TERM]
        keys = ["key1", "key2"]
        tags = {"category": "important", "type": "user_data"}
        
        # Act
        query = MemoryQuery(
            query="search term",
            limit=50,
            offset=10,
            keys=keys,
            memory_types=memory_types,
            tags=tags,
            include_expired=True
        )
        
        # Assert
        assert query.query == "search term"
        assert query.limit == 50
        assert query.offset == 10
        assert query.keys == keys
        assert query.memory_types == memory_types
        assert query.tags == tags
        assert query.include_expired


class TestMemoryStats:
    """Unit tests for MemoryStats model."""

    def test_memory_stats_creation(self):
        """Test MemoryStats model creation."""
        # Arrange
        generated_at = datetime.utcnow()
        memories_by_type = {
            MemoryType.SHORT_TERM: 10,
            MemoryType.LONG_TERM: 5,
            MemoryType.PERMANENT: 2
        }
        
        # Act
        stats = MemoryStats(
            generated_at=generated_at,
            total_memories=17,
            memories_by_type=memories_by_type,
            expired_memories=3,
            total_size_bytes=1024,
            average_ttl_seconds=1800.5,
            oldest_memory=generated_at - timedelta(days=30),
            newest_memory=generated_at - timedelta(hours=1),
            most_accessed_count=15
        )
        
        # Assert
        assert stats.generated_at == generated_at
        assert stats.total_memories == 17
        assert stats.memories_by_type == memories_by_type
        assert stats.expired_memories == 3
        assert stats.total_size_bytes == 1024
        assert stats.average_ttl_seconds == 1800.5
        assert stats.most_accessed_count == 15

    def test_memory_stats_optional_fields(self):
        """Test MemoryStats with optional fields as None."""
        # Act
        stats = MemoryStats(
            generated_at=datetime.utcnow(),
            total_memories=0,
            memories_by_type={},
            expired_memories=0,
            total_size_bytes=0,
            average_ttl_seconds=None,
            oldest_memory=None,
            newest_memory=None,
            most_accessed_count=0
        )
        
        # Assert
        assert stats.total_memories == 0
        assert stats.memories_by_type == {}
        assert stats.average_ttl_seconds is None
        assert stats.oldest_memory is None
        assert stats.newest_memory is None


class TestMemoryTypeEnum:
    """Unit tests for MemoryType enum."""

    def test_memory_type_values(self):
        """Test MemoryType enum values."""
        assert MemoryType.EPHEMERAL.value == "ephemeral"
        assert MemoryType.SHORT_TERM.value == "short_term"
        assert MemoryType.LONG_TERM.value == "long_term"
        assert MemoryType.PERMANENT.value == "permanent"

    def test_memory_type_comparison(self):
        """Test MemoryType enum comparison."""
        assert MemoryType.EPHEMERAL == MemoryType.EPHEMERAL
        assert MemoryType.SHORT_TERM != MemoryType.LONG_TERM


class TestExpirationPolicyEnum:
    """Unit tests for ExpirationPolicy enum."""

    def test_expiration_policy_values(self):
        """Test ExpirationPolicy enum values."""
        assert ExpirationPolicy.ABSOLUTE.value == "absolute"
        assert ExpirationPolicy.SLIDING.value == "sliding"
        assert ExpirationPolicy.NEVER.value == "never"

    def test_expiration_policy_comparison(self):
        """Test ExpirationPolicy enum comparison."""
        assert ExpirationPolicy.ABSOLUTE == ExpirationPolicy.ABSOLUTE
        assert ExpirationPolicy.SLIDING != ExpirationPolicy.ABSOLUTE