"""Tests for memory models."""

import json
from datetime import datetime, timedelta
from typing import Dict, Any

import pytest
from pydantic import ValidationError

from mcp_synaptic.models.memory import (
    ExpirationPolicy, Memory, MemoryCreateRequest, MemoryListResponse,
    MemoryQuery, MemoryStats, MemoryType, MemoryUpdateRequest
)
from tests.utils import MemoryTestHelper


class TestMemoryType:
    """Test MemoryType enum."""

    def test_memory_type_values(self):
        """Test MemoryType enum values."""
        assert MemoryType.EPHEMERAL == "ephemeral"
        assert MemoryType.SHORT_TERM == "short_term"
        assert MemoryType.LONG_TERM == "long_term"
        assert MemoryType.PERMANENT == "permanent"

    def test_memory_type_from_string(self):
        """Test creating MemoryType from string values."""
        assert MemoryType("ephemeral") == MemoryType.EPHEMERAL
        assert MemoryType("short_term") == MemoryType.SHORT_TERM
        assert MemoryType("long_term") == MemoryType.LONG_TERM
        assert MemoryType("permanent") == MemoryType.PERMANENT

    def test_invalid_memory_type(self):
        """Test invalid memory type raises ValueError."""
        with pytest.raises(ValueError):
            MemoryType("invalid_type")


class TestExpirationPolicy:
    """Test ExpirationPolicy enum."""

    def test_expiration_policy_values(self):
        """Test ExpirationPolicy enum values."""
        assert ExpirationPolicy.ABSOLUTE == "absolute"
        assert ExpirationPolicy.SLIDING == "sliding"
        assert ExpirationPolicy.NEVER == "never"

    def test_expiration_policy_from_string(self):
        """Test creating ExpirationPolicy from string values."""
        assert ExpirationPolicy("absolute") == ExpirationPolicy.ABSOLUTE
        assert ExpirationPolicy("sliding") == ExpirationPolicy.SLIDING
        assert ExpirationPolicy("never") == ExpirationPolicy.NEVER

    def test_invalid_expiration_policy(self):
        """Test invalid expiration policy raises ValueError."""
        with pytest.raises(ValueError):
            ExpirationPolicy("invalid_policy")


class TestMemory:
    """Test Memory model."""

    @pytest.fixture
    def sample_memory_data(self) -> Dict[str, Any]:
        """Sample memory data for testing."""
        return {
            "user_id": "test_user",
            "session_data": {"theme": "dark", "language": "en"},
            "timestamp": datetime.utcnow().isoformat()
        }

    def test_memory_creation_minimal(self, sample_memory_data):
        """Test creating memory with minimal required fields."""
        memory = Memory(
            key="test_key",
            data=sample_memory_data
        )
        
        assert memory.key == "test_key"
        assert memory.data == sample_memory_data
        assert memory.memory_type == MemoryType.SHORT_TERM  # default
        assert memory.expiration_policy == ExpirationPolicy.ABSOLUTE  # default
        assert memory.access_count == 0
        assert memory.tags == {}
        assert memory.metadata == {}

    def test_memory_creation_full(self, sample_memory_data):
        """Test creating memory with all fields specified."""
        tags = {"type": "user_data", "priority": "high"}
        metadata = {"source": "api", "version": "1.0"}
        
        memory = Memory(
            key="full_test",
            data=sample_memory_data,
            memory_type=MemoryType.LONG_TERM,
            expiration_policy=ExpirationPolicy.SLIDING,
            ttl_seconds=7200,
            tags=tags,
            metadata=metadata
        )
        
        assert memory.key == "full_test"
        assert memory.data == sample_memory_data
        assert memory.memory_type == MemoryType.LONG_TERM
        assert memory.expiration_policy == ExpirationPolicy.SLIDING
        assert memory.ttl_seconds == 7200
        assert memory.tags == tags
        assert memory.metadata == metadata

    def test_memory_timestamps_auto_generated(self, sample_memory_data):
        """Test that timestamps are automatically generated."""
        memory = Memory(key="timestamp_test", data=sample_memory_data)
        
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)
        assert isinstance(memory.last_accessed_at, datetime)
        assert memory.id is not None

    def test_memory_size_bytes_calculation(self, sample_memory_data):
        """Test memory size calculation."""
        memory = Memory(key="size_test", data=sample_memory_data)
        
        size = memory.size_bytes
        assert isinstance(size, int)
        assert size > 0

    def test_memory_is_expired_never_policy(self, sample_memory_data):
        """Test is_expired with NEVER policy."""
        memory = Memory(
            key="never_expires",
            data=sample_memory_data,
            expiration_policy=ExpirationPolicy.NEVER
        )
        
        assert not memory.is_expired

    def test_memory_is_expired_no_expires_at(self, sample_memory_data):
        """Test is_expired when expires_at is None."""
        memory = Memory(
            key="no_expiry",
            data=sample_memory_data,
            expires_at=None
        )
        
        assert not memory.is_expired

    def test_memory_is_expired_future_expiry(self, sample_memory_data):
        """Test is_expired with future expiry time."""
        future_time = datetime.utcnow() + timedelta(hours=1)
        memory = Memory(
            key="future_expiry",
            data=sample_memory_data,
            expires_at=future_time
        )
        
        assert not memory.is_expired

    def test_memory_is_expired_past_expiry(self, sample_memory_data):
        """Test is_expired with past expiry time."""
        past_time = datetime.utcnow() - timedelta(hours=1)
        memory = Memory(
            key="past_expiry",
            data=sample_memory_data,
            expires_at=past_time
        )
        
        assert memory.is_expired

    def test_memory_touch_updates_access_info(self, sample_memory_data):
        """Test touch() updates access timestamp and count."""
        memory = Memory(key="touch_test", data=sample_memory_data)
        
        original_access_time = memory.last_accessed_at
        original_count = memory.access_count
        
        memory.touch()
        
        assert memory.last_accessed_at > original_access_time
        assert memory.access_count == original_count + 1

    def test_memory_touch_updates_sliding_expiration(self, sample_memory_data):
        """Test touch() updates sliding expiration."""
        memory = Memory(
            key="sliding_test",
            data=sample_memory_data,
            expiration_policy=ExpirationPolicy.SLIDING,
            ttl_seconds=3600
        )
        
        original_expires_at = memory.expires_at
        memory.touch()
        
        if memory.expires_at and original_expires_at:
            assert memory.expires_at > original_expires_at

    def test_memory_touch_no_sliding_update_for_absolute(self, sample_memory_data):
        """Test touch() doesn't update absolute expiration."""
        memory = Memory(
            key="absolute_test",
            data=sample_memory_data,
            expiration_policy=ExpirationPolicy.ABSOLUTE,
            ttl_seconds=3600
        )
        
        # Set initial expiration
        memory.update_expiration(3600)
        original_expires_at = memory.expires_at
        
        memory.touch()
        
        # Expiration should not change for absolute policy
        assert memory.expires_at == original_expires_at

    def test_memory_update_expiration_never_policy(self, sample_memory_data):
        """Test update_expiration with NEVER policy."""
        memory = Memory(
            key="never_policy",
            data=sample_memory_data,
            expiration_policy=ExpirationPolicy.NEVER
        )
        
        memory.update_expiration(3600)
        
        assert memory.expires_at is None

    def test_memory_update_expiration_permanent_ttl(self, sample_memory_data):
        """Test update_expiration with TTL of 0 (permanent)."""
        memory = Memory(
            key="permanent_ttl",
            data=sample_memory_data,
            expiration_policy=ExpirationPolicy.ABSOLUTE
        )
        
        memory.update_expiration(0)
        
        assert memory.expires_at is None

    def test_memory_update_expiration_absolute_policy(self, sample_memory_data):
        """Test update_expiration with ABSOLUTE policy."""
        memory = Memory(
            key="absolute_policy",
            data=sample_memory_data,
            expiration_policy=ExpirationPolicy.ABSOLUTE
        )
        
        memory.update_expiration(3600)
        
        assert memory.expires_at is not None
        expected_expires_at = memory.created_at + timedelta(seconds=3600)
        # Allow small time difference due to execution time
        assert abs((memory.expires_at - expected_expires_at).total_seconds()) < 1

    def test_memory_update_expiration_sliding_policy(self, sample_memory_data):
        """Test update_expiration with SLIDING policy."""
        memory = Memory(
            key="sliding_policy",
            data=sample_memory_data,
            expiration_policy=ExpirationPolicy.SLIDING
        )
        
        memory.update_expiration(3600)
        
        assert memory.expires_at is not None
        expected_expires_at = memory.last_accessed_at + timedelta(seconds=3600)
        # Allow small time difference due to execution time
        assert abs((memory.expires_at - expected_expires_at).total_seconds()) < 1

    def test_memory_ttl_seconds_validation_negative(self, sample_memory_data):
        """Test ttl_seconds validation rejects negative values."""
        with pytest.raises(ValidationError):
            Memory(
                key="negative_ttl",
                data=sample_memory_data,
                ttl_seconds=-1
            )

    def test_memory_ttl_seconds_validation_zero_allowed(self, sample_memory_data):
        """Test ttl_seconds validation allows zero (permanent)."""
        memory = Memory(
            key="zero_ttl",
            data=sample_memory_data,
            ttl_seconds=0
        )
        
        assert memory.ttl_seconds == 0

    def test_memory_serialization(self, sample_memory_data):
        """Test memory model serialization."""
        memory = Memory(
            key="serialize_test",
            data=sample_memory_data,
            memory_type=MemoryType.LONG_TERM,
            tags={"type": "test"}
        )
        
        # Test model_dump
        data = memory.model_dump()
        assert data["key"] == "serialize_test"
        assert data["memory_type"] == "long_term"
        assert data["tags"] == {"type": "test"}
        
        # Test model_dump_json
        json_str = memory.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["key"] == "serialize_test"

    def test_memory_deserialization(self, sample_memory_data):
        """Test memory model deserialization."""
        data = {
            "id": "test-id",
            "key": "deserialize_test",
            "data": sample_memory_data,
            "memory_type": "short_term",
            "expiration_policy": "absolute",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "last_accessed_at": datetime.utcnow().isoformat(),
            "ttl_seconds": 3600,
            "access_count": 5,
            "tags": {"type": "test"},
            "metadata": {"source": "test"}
        }
        
        memory = Memory.model_validate(data)
        
        assert memory.key == "deserialize_test"
        assert memory.data == sample_memory_data
        assert memory.memory_type == MemoryType.SHORT_TERM
        assert memory.access_count == 5


class TestMemoryQuery:
    """Test MemoryQuery model."""

    def test_memory_query_defaults(self):
        """Test MemoryQuery default values."""
        query = MemoryQuery()
        
        assert query.query == ""
        assert query.limit == 10
        assert query.offset == 0
        assert query.keys is None
        assert query.memory_types is None
        assert query.tags is None
        assert query.include_expired is False

    def test_memory_query_with_values(self):
        """Test MemoryQuery with specified values."""
        keys = ["key1", "key2"]
        memory_types = [MemoryType.SHORT_TERM, MemoryType.LONG_TERM]
        tags = {"type": "user", "priority": "high"}
        
        query = MemoryQuery(
            limit=50,
            offset=20,
            keys=keys,
            memory_types=memory_types,
            tags=tags,
            include_expired=True
        )
        
        assert query.limit == 50
        assert query.offset == 20
        assert query.keys == keys
        assert query.memory_types == memory_types
        assert query.tags == tags
        assert query.include_expired is True

    def test_memory_query_validation_limit_range(self):
        """Test MemoryQuery limit validation."""
        # Valid limits
        MemoryQuery(limit=1)  # min
        MemoryQuery(limit=100)  # max
        
        # Invalid limits
        with pytest.raises(ValidationError):
            MemoryQuery(limit=0)  # below min
        
        with pytest.raises(ValidationError):
            MemoryQuery(limit=101)  # above max

    def test_memory_query_validation_offset_negative(self):
        """Test MemoryQuery offset validation."""
        # Valid offset
        MemoryQuery(offset=0)
        MemoryQuery(offset=10)
        
        # Invalid offset
        with pytest.raises(ValidationError):
            MemoryQuery(offset=-1)

    def test_memory_query_datetime_filters(self):
        """Test MemoryQuery with datetime filters."""
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)
        
        query = MemoryQuery(
            created_after=yesterday,
            created_before=tomorrow,
            expires_after=now,
            expires_before=tomorrow
        )
        
        assert query.created_after == yesterday
        assert query.created_before == tomorrow
        assert query.expires_after == now
        assert query.expires_before == tomorrow


class TestMemoryStats:
    """Test MemoryStats model."""

    def test_memory_stats_creation(self):
        """Test MemoryStats model creation."""
        stats = MemoryStats(
            total_memories=100,
            memories_by_type={
                MemoryType.SHORT_TERM: 60,
                MemoryType.LONG_TERM: 30,
                MemoryType.PERMANENT: 10
            },
            expired_memories=5,
            total_size_bytes=1024000,
            average_ttl_seconds=7200.0,
            oldest_memory=datetime.utcnow() - timedelta(days=30),
            newest_memory=datetime.utcnow(),
            most_accessed_count=25
        )
        
        assert stats.total_memories == 100
        assert stats.expired_memories == 5
        assert stats.total_size_bytes == 1024000
        assert stats.average_ttl_seconds == 7200.0
        assert stats.most_accessed_count == 25

    def test_memory_stats_validation(self):
        """Test MemoryStats validation."""
        # Negative values should be rejected
        with pytest.raises(ValidationError):
            MemoryStats(
                total_memories=-1,
                memories_by_type={},
                expired_memories=0,
                total_size_bytes=0
            )
        
        with pytest.raises(ValidationError):
            MemoryStats(
                total_memories=0,
                memories_by_type={},
                expired_memories=-1,
                total_size_bytes=0
            )


class TestMemoryCreateRequest:
    """Test MemoryCreateRequest model."""

    def test_memory_create_request_minimal(self):
        """Test MemoryCreateRequest with minimal fields."""
        request = MemoryCreateRequest(
            key="test_key",
            data={"test": "data"}
        )
        
        assert request.key == "test_key"
        assert request.data == {"test": "data"}
        assert request.memory_type == MemoryType.SHORT_TERM  # default
        assert request.ttl_seconds is None
        assert request.tags is None
        assert request.metadata is None

    def test_memory_create_request_full(self):
        """Test MemoryCreateRequest with all fields."""
        request = MemoryCreateRequest(
            key="full_test",
            data={"test": "data"},
            memory_type=MemoryType.LONG_TERM,
            ttl_seconds=7200,
            tags={"type": "test"},
            metadata={"source": "api"}
        )
        
        assert request.key == "full_test"
        assert request.memory_type == MemoryType.LONG_TERM
        assert request.ttl_seconds == 7200
        assert request.tags == {"type": "test"}
        assert request.metadata == {"source": "api"}


class TestMemoryUpdateRequest:
    """Test MemoryUpdateRequest model."""

    def test_memory_update_request_optional_fields(self):
        """Test MemoryUpdateRequest with optional fields."""
        request = MemoryUpdateRequest()
        
        assert request.data is None
        assert request.extend_ttl_seconds is None
        assert request.tags is None
        assert request.metadata is None

    def test_memory_update_request_with_values(self):
        """Test MemoryUpdateRequest with values."""
        request = MemoryUpdateRequest(
            data={"updated": True},
            extend_ttl_seconds=3600,
            tags={"status": "updated"},
            metadata={"version": "2.0"}
        )
        
        assert request.data == {"updated": True}
        assert request.extend_ttl_seconds == 3600
        assert request.tags == {"status": "updated"}
        assert request.metadata == {"version": "2.0"}

    def test_memory_update_request_ttl_validation(self):
        """Test MemoryUpdateRequest TTL validation."""
        # Valid TTL
        MemoryUpdateRequest(extend_ttl_seconds=1)
        MemoryUpdateRequest(extend_ttl_seconds=86400)
        
        # Invalid TTL (must be >= 1)
        with pytest.raises(ValidationError):
            MemoryUpdateRequest(extend_ttl_seconds=0)
        
        with pytest.raises(ValidationError):
            MemoryUpdateRequest(extend_ttl_seconds=-1)


class TestMemoryListResponse:
    """Test MemoryListResponse model."""

    def test_memory_list_response(self):
        """Test MemoryListResponse model."""
        memories = [
            MemoryTestHelper.create_test_memory("test1"),
            MemoryTestHelper.create_test_memory("test2")
        ]
        
        response = MemoryListResponse(
            items=memories,
            total=2,
            page=1,
            per_page=10,
            has_next=False
        )
        
        assert len(response.items) == 2
        assert response.total == 2
        assert response.page == 1
        assert response.per_page == 10
        assert response.has_next is False