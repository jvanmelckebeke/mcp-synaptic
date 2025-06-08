"""Test utilities and helper functions for MCP Synaptic tests."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, UTC
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.memory.manager import MemoryManager
from mcp_synaptic.models.memory import Memory, MemoryType, ExpirationPolicy


class AsyncMockManager:
    """Helper for managing async mocks in tests."""
    
    def __init__(self):
        self.mocks: List[AsyncMock] = []
    
    def create_async_mock(self, **kwargs) -> AsyncMock:
        """Create and track an async mock."""
        mock = AsyncMock(**kwargs)
        self.mocks.append(mock)
        return mock
    
    async def cleanup(self):
        """Cleanup all tracked mocks."""
        for mock in self.mocks:
            if hasattr(mock, 'reset_mock'):
                mock.reset_mock()
        self.mocks.clear()


class MemoryTestHelper:
    """Helper class for memory-related testing."""
    
    @staticmethod
    def create_test_memory(
        key: str = "test_key",
        data: Optional[Dict[str, Any]] = None,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        ttl_seconds: Optional[int] = 3600,
        expired: bool = False,
        **kwargs
    ) -> Memory:
        """Create a test memory with configurable parameters."""
        if data is None:
            data = {"test": "data", "timestamp": datetime.now(UTC).isoformat()}
        
        memory = Memory(
            key=key,
            data=data,
            memory_type=memory_type,
            ttl_seconds=ttl_seconds,
            **kwargs
        )
        
        if expired and ttl_seconds and ttl_seconds > 0:
            # Set expiration to past time
            memory.expires_at = datetime.now(UTC) - timedelta(seconds=10)
        
        return memory
    
    @staticmethod
    def create_memory_batch(
        count: int = 5,
        key_prefix: str = "test_key",
        **kwargs
    ) -> List[Memory]:
        """Create a batch of test memories."""
        return [
            MemoryTestHelper.create_test_memory(
                key=f"{key_prefix}_{i}",
                data={"index": i, "batch": True},
                **kwargs
            )
            for i in range(count)
        ]
    
    @staticmethod
    async def add_memories_to_manager(
        manager: MemoryManager,
        memories: List[Memory]
    ) -> List[Memory]:
        """Add multiple memories to a manager and return the stored versions."""
        stored_memories = []
        for memory in memories:
            stored_memory = await manager.add(
                key=memory.key,
                data=memory.data,
                memory_type=memory.memory_type,
                ttl_seconds=memory.ttl_seconds,
                expiration_policy=memory.expiration_policy,
                tags=memory.tags,
                metadata=memory.metadata
            )
            stored_memories.append(stored_memory)
        return stored_memories


class EmbeddingTestHelper:
    """Helper class for embedding and RAG testing."""
    
    @staticmethod
    def create_mock_embedding(dimensions: int = 1536, value: float = 0.1) -> List[float]:
        """Create a mock embedding vector."""
        return [value] * dimensions
    
    @staticmethod
    def create_mock_embedding_response(
        embeddings: Optional[List[List[float]]] = None,
        model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """Create a mock embedding API response."""
        if embeddings is None:
            embeddings = [EmbeddingTestHelper.create_mock_embedding()]
        
        return {
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embedding
                }
                for i, embedding in enumerate(embeddings)
            ],
            "model": model,
            "usage": {
                "prompt_tokens": len(embeddings) * 5,
                "total_tokens": len(embeddings) * 5
            }
        }
    
    @staticmethod
    def create_test_documents(count: int = 3) -> List[Dict[str, Any]]:
        """Create test documents for RAG testing."""
        return [
            {
                "content": f"This is test document {i} with unique content for testing search and retrieval.",
                "metadata": {
                    "id": f"doc_{i}",
                    "source": f"test_source_{i}",
                    "category": "test",
                    "created_at": datetime.now(UTC).isoformat()
                }
            }
            for i in range(count)
        ]


class AsyncContextHelper:
    """Helper for async context management in tests."""
    
    @staticmethod
    @asynccontextmanager
    async def memory_manager_context(settings: Settings) -> AsyncGenerator[MemoryManager, None]:
        """Context manager for memory manager lifecycle."""
        manager = MemoryManager(settings)
        try:
            await manager.initialize()
            yield manager
        finally:
            await manager.close()
    
    @staticmethod
    async def run_with_timeout(coro, timeout: float = 5.0):
        """Run a coroutine with timeout."""
        return await asyncio.wait_for(coro, timeout=timeout)


class MockFactory:
    """Factory for creating various mocks used in tests."""
    
    @staticmethod
    def create_aiohttp_session_mock(response_data: Any = None, status: int = 200) -> AsyncMock:
        """Create a mock aiohttp session."""
        from unittest.mock import MagicMock
        
        session_mock = AsyncMock()
        response_mock = AsyncMock()
        response_mock.status = status
        
        if response_data is not None:
            response_mock.json = AsyncMock(return_value=response_data)
        
        # Create a proper async context manager mock
        class MockAsyncContextManager:
            def __init__(self, response):
                self.response = response
                
            async def __aenter__(self):
                return self.response
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        context_manager = MockAsyncContextManager(response_mock)
        
        session_mock.post = MagicMock(return_value=context_manager)
        session_mock.get = MagicMock(return_value=context_manager)
        session_mock.close = AsyncMock()  # Add close method for cleanup
        return session_mock
    
    @staticmethod
    def create_chromadb_mock():
        """Create a mock ChromaDB client and collection."""
        collection_mock = MagicMock()
        collection_mock.count.return_value = 0
        collection_mock.add.return_value = None
        collection_mock.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "documents": [["Test document 1", "Test document 2"]]
        }
        collection_mock.delete.return_value = None
        
        client_mock = MagicMock()
        client_mock.get_or_create_collection.return_value = collection_mock
        client_mock.list_collections.return_value = []
        
        return client_mock, collection_mock
    
    @staticmethod
    def create_redis_mock():
        """Create a mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.delete.return_value = 1
        redis_mock.keys.return_value = []
        redis_mock.close.return_value = None
        
        # Create proper async iterator for scan_iter
        async def mock_scan_iter(match="*"):
            """Mock async iterator for Redis scan_iter."""
            return
            yield  # This makes it an async generator
        
        redis_mock.scan_iter = mock_scan_iter
        return redis_mock


class AssertionHelpers:
    """Helper functions for common test assertions."""
    
    @staticmethod
    def assert_memory_fields(
        memory: Memory,
        expected_key: str,
        expected_data: Optional[Dict[str, Any]] = None,
        expected_type: Optional[MemoryType] = None,
        check_timestamps: bool = False
    ):
        """Assert memory has expected field values."""
        assert memory.key == expected_key
        
        if expected_data is not None:
            assert memory.data == expected_data
        
        if expected_type is not None:
            assert memory.memory_type == expected_type
        
        if check_timestamps:
            assert memory.created_at is not None
            assert memory.updated_at is not None
            assert memory.last_accessed_at is not None
    
    @staticmethod
    def assert_memory_list_properties(
        memories: List[Memory],
        expected_count: Optional[int] = None,
        expected_keys: Optional[List[str]] = None,
        all_non_expired: bool = True
    ):
        """Assert properties of a memory list."""
        if expected_count is not None:
            assert len(memories) == expected_count
        
        if expected_keys is not None:
            actual_keys = [m.key for m in memories]
            assert set(actual_keys) == set(expected_keys)
        
        if all_non_expired:
            for memory in memories:
                assert not memory.is_expired, f"Memory {memory.key} should not be expired"
    
    @staticmethod
    def assert_embedding_response(
        response: Dict[str, Any],
        expected_dimensions: int = 1536,
        expected_count: int = 1
    ):
        """Assert embedding response has expected structure."""
        assert "data" in response
        assert len(response["data"]) == expected_count
        
        for item in response["data"]:
            assert "embedding" in item
            assert len(item["embedding"]) == expected_dimensions
            assert all(isinstance(x, (int, float)) for x in item["embedding"])


class TestDataBuilder:
    """Builder for creating complex test data structures."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset builder state."""
        self._memories: List[Memory] = []
        self._documents: List[Dict[str, Any]] = []
        return self
    
    def add_memory(
        self,
        key: str,
        data: Dict[str, Any],
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        **kwargs
    ):
        """Add a memory to the builder."""
        memory = MemoryTestHelper.create_test_memory(
            key=key,
            data=data,
            memory_type=memory_type,
            **kwargs
        )
        self._memories.append(memory)
        return self
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a document to the builder."""
        if metadata is None:
            metadata = {}
        
        document = {
            "content": content,
            "metadata": metadata
        }
        self._documents.append(document)
        return self
    
    def build_memories(self) -> List[Memory]:
        """Build and return memories."""
        return self._memories.copy()
    
    def build_documents(self) -> List[Dict[str, Any]]:
        """Build and return documents."""
        return self._documents.copy()


# Context managers for patching
@asynccontextmanager
async def mock_embedding_api(response_data: Any = None):
    """Mock the embedding API for testing."""
    if response_data is None:
        response_data = EmbeddingTestHelper.create_mock_embedding_response()
    
    session_mock = MockFactory.create_aiohttp_session_mock(response_data)
    
    with patch('aiohttp.ClientSession', return_value=session_mock):
        yield session_mock


@asynccontextmanager  
async def mock_chromadb_client():
    """Mock ChromaDB client for testing."""
    client_mock, collection_mock = MockFactory.create_chromadb_mock()
    
    with patch('chromadb.Client', return_value=client_mock):
        yield client_mock, collection_mock