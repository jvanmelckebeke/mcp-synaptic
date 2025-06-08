"""Pytest configuration and shared fixtures for MCP Synaptic tests."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from pydantic import BaseModel

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.memory.manager import MemoryManager
from mcp_synaptic.memory.storage import SQLiteMemoryStorage
from mcp_synaptic.models.memory import ExpirationPolicy, Memory, MemoryType


# Event loop fixture for pytest-asyncio
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """Create test settings with temporary directories."""
    return Settings(
        # Server settings
        SERVER_HOST="127.0.0.1",
        SERVER_PORT=8001,  # Different port to avoid conflicts
        DEBUG=True,
        
        # Storage paths (temporary)
        SQLITE_PATH=str(temp_dir / "test_memory.db"),
        CHROMADB_PATH=str(temp_dir / "test_chromadb"),
        DATA_DIR=str(temp_dir / "test_data"),
        
        # Memory settings
        DEFAULT_MEMORY_TTL_SECONDS=3600,
        MAX_MEMORY_ENTRIES=1000,
        MEMORY_CLEANUP_INTERVAL_SECONDS=300,
        
        # Redis (disabled for tests)
        REDIS_ENABLED=False,
        
        # Embedding settings (mock)
        EMBEDDING_PROVIDER="api",
        EMBEDDING_API_BASE="http://mock-embedding-api:4000",
        EMBEDDING_MODEL="text-embedding-ada-002",
        EMBEDDING_DIMENSIONS=1536,
        
        # RAG settings
        RAG_ENABLED=True,
        RAG_COLLECTION_NAME="test_documents",
        RAG_MAX_RESULTS=10,
        RAG_DISTANCE_THRESHOLD=0.7,
    )


@pytest_asyncio.fixture
async def memory_storage(test_settings: Settings) -> AsyncGenerator[SQLiteMemoryStorage, None]:
    """Create and initialize a test memory storage."""
    storage = SQLiteMemoryStorage(test_settings)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def memory_manager(test_settings: Settings) -> AsyncGenerator[MemoryManager, None]:
    """Create and initialize a test memory manager."""
    manager = MemoryManager(test_settings)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def sample_memory_data() -> Dict[str, Any]:
    """Sample memory data for testing."""
    return {
        "user_id": "test_user_123",
        "session_id": "session_abc",
        "preferences": {
            "theme": "dark",
            "language": "en",
            "notifications": True
        },
        "last_activity": "2024-01-15T10:30:00Z"
    }


@pytest.fixture
def sample_memory(sample_memory_data: Dict[str, Any]) -> Memory:
    """Create a sample memory object for testing."""
    return Memory(
        key="test_user_preferences",
        data=sample_memory_data,
        memory_type=MemoryType.SHORT_TERM,
        expiration_policy=ExpirationPolicy.ABSOLUTE,
        ttl_seconds=3600,
        tags={"user": "test_user_123", "type": "preferences"},
        metadata={"source": "user_settings", "version": "1.0"}
    )


@pytest.fixture
def expired_memory(sample_memory_data: Dict[str, Any]) -> Memory:
    """Create an expired memory object for testing."""
    memory = Memory(
        key="expired_test_memory",
        data=sample_memory_data,
        memory_type=MemoryType.EPHEMERAL,
        expiration_policy=ExpirationPolicy.ABSOLUTE,
        ttl_seconds=1,  # 1 second
        tags={"status": "expired"},
        metadata={"test": True}
    )
    # Set expiration to past time
    memory.expires_at = datetime.utcnow() - timedelta(seconds=10)
    return memory


@pytest.fixture
def permanent_memory(sample_memory_data: Dict[str, Any]) -> Memory:
    """Create a permanent memory object for testing."""
    return Memory(
        key="permanent_test_memory",
        data=sample_memory_data,
        memory_type=MemoryType.PERMANENT,
        expiration_policy=ExpirationPolicy.NEVER,
        ttl_seconds=0,
        tags={"type": "permanent"},
        metadata={"important": True}
    )


@pytest.fixture
def mock_embedding_response() -> Dict[str, Any]:
    """Mock embedding API response."""
    return {
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1] * 1536  # Mock 1536-dimensional embedding
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    }


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp session for API testing."""
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "data": [{"embedding": [0.1] * 1536}]
    })
    mock_session.post.return_value.__aenter__.return_value = mock_response
    return mock_session


@pytest.fixture
def mock_chromadb_collection():
    """Create a mock ChromaDB collection for testing."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_collection.add.return_value = None
    mock_collection.query.return_value = {
        "ids": [["doc1", "doc2"]],
        "distances": [[0.1, 0.2]],
        "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
        "documents": [["Test document 1", "Test document 2"]]
    }
    mock_collection.delete.return_value = None
    return mock_collection


@pytest.fixture
def mock_chromadb_client(mock_chromadb_collection):
    """Create a mock ChromaDB client for testing."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_chromadb_collection
    mock_client.list_collections.return_value = []
    return mock_client


# Utility functions for tests
class TestHelpers:
    """Helper functions for testing."""
    
    @staticmethod
    def assert_memory_equal(actual: Memory, expected: Memory, ignore_timestamps: bool = True) -> None:
        """Assert that two Memory objects are equal, optionally ignoring timestamps."""
        assert actual.key == expected.key
        assert actual.data == expected.data
        assert actual.memory_type == expected.memory_type
        assert actual.expiration_policy == expected.expiration_policy
        assert actual.ttl_seconds == expected.ttl_seconds
        assert actual.tags == expected.tags
        assert actual.metadata == expected.metadata
        
        if not ignore_timestamps:
            assert actual.created_at == expected.created_at
            assert actual.updated_at == expected.updated_at
            assert actual.last_accessed_at == expected.last_accessed_at
            assert actual.expires_at == expected.expires_at
    
    @staticmethod
    async def wait_for_expiration(seconds: float = 0.1) -> None:
        """Wait for a short time to allow expiration to occur."""
        await asyncio.sleep(seconds)
    
    @staticmethod
    def create_test_documents(count: int = 3) -> list[Dict[str, Any]]:
        """Create test documents for RAG testing."""
        return [
            {
                "id": f"doc_{i}",
                "content": f"This is test document {i} with some content for testing.",
                "metadata": {"source": f"test_{i}", "category": "test"}
            }
            for i in range(count)
        ]


@pytest.fixture
def test_helpers() -> TestHelpers:
    """Provide test helper functions."""
    return TestHelpers()


# Test data generators
@pytest.fixture
def memory_test_cases() -> list[Dict[str, Any]]:
    """Generate various memory test cases."""
    return [
        {
            "key": "ephemeral_memory",
            "data": {"temp": "data"},
            "memory_type": MemoryType.EPHEMERAL,
            "ttl_seconds": 5
        },
        {
            "key": "short_term_memory", 
            "data": {"session": "active"},
            "memory_type": MemoryType.SHORT_TERM,
            "ttl_seconds": 3600
        },
        {
            "key": "long_term_memory",
            "data": {"user_profile": "data"},
            "memory_type": MemoryType.LONG_TERM,
            "ttl_seconds": 86400
        },
        {
            "key": "permanent_memory",
            "data": {"system": "config"},
            "memory_type": MemoryType.PERMANENT,
            "ttl_seconds": 0
        }
    ]


# Environment cleanup
@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables before/after tests."""
    # Store original env vars
    original_env = dict(os.environ)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)