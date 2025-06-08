"""Integration tests for cross-system coordination after refactoring."""

import asyncio
import pytest
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.memory.manager import MemoryManager
from mcp_synaptic.rag.database import RAGDatabase
from mcp_synaptic.mcp.memory_tools import MemoryTools
from mcp_synaptic.mcp.rag_tools import RAGTools
from mcp_synaptic.models.memory import MemoryType


@pytest.fixture
def system_integration_settings():
    """Settings for system integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        settings = Settings(
            SQLITE_DATABASE_PATH=Path(temp_dir) / "test_memory.db",
            CHROMADB_PERSIST_DIRECTORY=Path(temp_dir) / "chromadb",
            REDIS_ENABLED=False,
            EMBEDDING_PROVIDER="api",
            EMBEDDING_API_BASE="http://mock-api:4000",
            EMBEDDING_MODEL="test-model",
            EMBEDDING_API_KEY="test-key",
            DEFAULT_MEMORY_TTL_SECONDS=3600,
            MAX_RAG_RESULTS=20
        )
        yield settings


class TestMemoryRAGIntegration:
    """Test memory and RAG systems working together."""

    @pytest.mark.asyncio
    async def test_memory_and_rag_lifecycle_integration(self, system_integration_settings):
        """Test complete workflow involving both memory and RAG systems."""
        # Initialize both systems
        memory_manager = MemoryManager(system_integration_settings)
        await memory_manager.initialize()
        
        # Mock RAG database dependencies
        with patch('chromadb.PersistentClient') as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.return_value = mock_client
            
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager') as mock_em:
                mock_embedding_manager = AsyncMock()
                mock_embedding_manager.embed_text.return_value = [0.1] * 1536
                mock_embedding_manager.get_embedding_dimension.return_value = 1536
                mock_em.return_value = mock_embedding_manager
                
                rag_database = RAGDatabase(system_integration_settings)
                await rag_database.initialize()
        
        try:
            # Test workflow: Store document metadata in memory, document content in RAG
            document_content = "This is a test document for cross-system integration."
            document_metadata = {"title": "Integration Test Doc", "author": "Test Suite"}
            
            # Add document to RAG system
            rag_database.embedding_manager = mock_embedding_manager
            rag_database.client = mock_client
            rag_database.collection = mock_collection
            
            document = await rag_database.add_document(document_content, document_metadata)
            
            # Store document reference in memory system
            memory_key = f"doc_ref_{document.id}"
            memory_data = {
                "document_id": document.id,
                "title": document_metadata["title"],
                "content_preview": document_content[:50] + "...",
                "added_at": document.created_at.isoformat()
            }
            
            memory = await memory_manager.add(
                key=memory_key,
                data=memory_data,
                memory_type=MemoryType.LONG_TERM
            )
            
            # Verify cross-system coordination
            assert memory.key == memory_key
            assert memory.data["document_id"] == document.id
            assert memory.data["title"] == document_metadata["title"]
            
            # Test retrieval coordination
            retrieved_memory = await memory_manager.get(memory_key)
            assert retrieved_memory is not None
            assert retrieved_memory.data["document_id"] == document.id
            
            # Mock document retrieval from RAG
            mock_collection.get.return_value = {
                "ids": [document.id],
                "documents": [document_content],
                "metadatas": [{
                    **document_metadata,
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat()
                }]
            }
            
            retrieved_document = await rag_database.get_document(document.id)
            assert retrieved_document is not None
            assert retrieved_document.content == document_content
            
            # Test cleanup coordination
            await memory_manager.delete(memory_key)
            await rag_database.delete_document(document.id)
            
            # Verify cleanup
            deleted_memory = await memory_manager.get(memory_key)
            assert deleted_memory is None
            
        finally:
            await memory_manager.close()
            await rag_database.close()

    @pytest.mark.asyncio
    async def test_memory_rag_search_coordination(self, system_integration_settings):
        """Test coordinated search across memory and RAG systems."""
        memory_manager = MemoryManager(system_integration_settings)
        await memory_manager.initialize()
        
        # Mock RAG database
        with patch('chromadb.PersistentClient') as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.return_value = mock_client
            
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager') as mock_em:
                mock_embedding_manager = AsyncMock()
                mock_embedding_manager.embed_text.return_value = [0.1] * 1536
                mock_em.return_value = mock_embedding_manager
                
                rag_database = RAGDatabase(system_integration_settings)
                await rag_database.initialize()
        
        try:
            rag_database.embedding_manager = mock_embedding_manager
            rag_database.client = mock_client
            rag_database.collection = mock_collection
            
            # Store search history in memory
            search_query = "integration testing"
            search_history_key = "search_history"
            
            existing_history = await memory_manager.get(search_history_key)
            if existing_history is None:
                search_history = {"queries": [search_query], "count": 1}
            else:
                search_history = existing_history.data
                search_history["queries"].append(search_query)
                search_history["count"] += 1
            
            await memory_manager.add(
                key=search_history_key,
                data=search_history,
                memory_type=MemoryType.SHORT_TERM
            )
            
            # Mock search results from RAG
            mock_collection.query.return_value = {
                "ids": [["doc1", "doc2"]],
                "documents": [["Integration document 1", "Integration document 2"]],
                "metadatas": [[{"topic": "testing"}, {"topic": "integration"}]],
                "distances": [[0.1, 0.2]]
            }
            
            # Perform search
            search_results = await rag_database.search(search_query, limit=5)
            
            # Verify search coordination
            assert len(search_results) == 2
            assert search_results[0].item.content == "Integration document 1"
            
            # Verify search history was stored
            updated_history = await memory_manager.get(search_history_key)
            assert updated_history is not None
            assert search_query in updated_history.data["queries"]
            assert updated_history.data["count"] >= 1
            
        finally:
            await memory_manager.close()
            await rag_database.close()


class TestMCPToolsIntegration:
    """Test MCP tools integration with refactored components."""

    @pytest.mark.asyncio
    async def test_memory_tools_integration(self, system_integration_settings):
        """Test memory tools with refactored memory manager."""
        memory_manager = MemoryManager(system_integration_settings)
        await memory_manager.initialize()
        
        try:
            memory_tools = MemoryTools(memory_manager)
            
            # Test memory add tool
            add_result = await memory_tools.add_memory(
                key="mcp_test_key",
                data={"tool": "integration", "test": True},
                memory_type="short_term",
                ttl_seconds=1800
            )
            
            assert add_result.key == "mcp_test_key"
            assert add_result.data["tool"] == "integration"
            assert add_result.memory_type == MemoryType.SHORT_TERM
            
            # Test memory get tool
            get_result = await memory_tools.get_memory("mcp_test_key")
            assert get_result is not None
            assert get_result.key == "mcp_test_key"
            assert get_result.data["test"] is True
            
            # Test memory update tool
            update_result = await memory_tools.update_memory(
                key="mcp_test_key",
                data={"tool": "integration", "test": True, "updated": True}
            )
            assert update_result.data["updated"] is True
            
            # Test memory list tool
            list_result = await memory_tools.list_memories(limit=10)
            assert len(list_result.memories) >= 1
            assert any(m.key == "mcp_test_key" for m in list_result.memories)
            
            # Test memory stats tool
            stats_result = await memory_tools.get_memory_stats()
            assert stats_result.total_memories >= 1
            
            # Test memory delete tool
            delete_result = await memory_tools.delete_memory("mcp_test_key")
            assert delete_result is True
            
        finally:
            await memory_manager.close()

    @pytest.mark.asyncio
    async def test_rag_tools_integration(self, system_integration_settings):
        """Test RAG tools with refactored RAG database."""
        # Mock RAG database dependencies
        with patch('chromadb.PersistentClient') as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 1
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.return_value = mock_client
            
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager') as mock_em:
                mock_embedding_manager = AsyncMock()
                mock_embedding_manager.embed_text.return_value = [0.1] * 1536
                mock_embedding_manager.get_embedding_dimension.return_value = 1536
                mock_em.return_value = mock_embedding_manager
                
                rag_database = RAGDatabase(system_integration_settings)
                await rag_database.initialize()
        
        try:
            rag_database.embedding_manager = mock_embedding_manager
            rag_database.client = mock_client
            rag_database.collection = mock_collection
            
            rag_tools = RAGTools(rag_database)
            
            # Test add document tool
            add_result = await rag_tools.add_document(
                content="MCP integration test document",
                metadata={"source": "mcp_tools", "test": True}
            )
            
            assert add_result.content == "MCP integration test document"
            assert add_result.metadata["source"] == "mcp_tools"
            
            # Mock document retrieval for get test
            mock_collection.get.return_value = {
                "ids": [add_result.id],
                "documents": ["MCP integration test document"],
                "metadatas": [{
                    "source": "mcp_tools", 
                    "test": True,
                    "created_at": add_result.created_at.isoformat(),
                    "updated_at": add_result.updated_at.isoformat()
                }]
            }
            
            # Test get document tool
            get_result = await rag_tools.get_document(add_result.id)
            assert get_result is not None
            assert get_result.content == "MCP integration test document"
            
            # Test search tool
            mock_collection.query.return_value = {
                "ids": [[add_result.id]],
                "documents": [["MCP integration test document"]],
                "metadatas": [[{"source": "mcp_tools", "test": True}]],
                "distances": [[0.1]]
            }
            
            search_result = await rag_tools.search_documents(
                query="integration test",
                limit=5
            )
            assert len(search_result.results) == 1
            assert search_result.results[0].item.content == "MCP integration test document"
            
            # Test collection stats tool
            mock_collection.get.return_value = {
                "documents": ["MCP integration test document"],
                "metadatas": [{"source": "mcp_tools", "embedding_model": "test-model", "embedding_dimension": 1536}]
            }
            
            stats_result = await rag_tools.get_collection_stats()
            assert stats_result.total_documents == 1
            assert stats_result.total_embeddings == 1
            
            # Test delete document tool
            delete_result = await rag_tools.delete_document(add_result.id)
            assert delete_result is True
            
        finally:
            await rag_database.close()

    @pytest.mark.asyncio
    async def test_concurrent_mcp_tools_integration(self, system_integration_settings):
        """Test concurrent MCP tools operations."""
        # Initialize both systems
        memory_manager = MemoryManager(system_integration_settings)
        await memory_manager.initialize()
        
        with patch('chromadb.PersistentClient') as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.return_value = mock_client
            
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager') as mock_em:
                mock_embedding_manager = AsyncMock()
                mock_embedding_manager.embed_text.return_value = [0.1] * 1536
                mock_em.return_value = mock_embedding_manager
                
                rag_database = RAGDatabase(system_integration_settings)
                await rag_database.initialize()
        
        try:
            rag_database.embedding_manager = mock_embedding_manager
            rag_database.client = mock_client
            rag_database.collection = mock_collection
            
            memory_tools = MemoryTools(memory_manager)
            rag_tools = RAGTools(rag_database)
            
            # Concurrent memory operations
            memory_tasks = [
                memory_tools.add_memory(f"concurrent_key_{i}", {"index": i})
                for i in range(5)
            ]
            
            # Concurrent RAG operations
            rag_tasks = [
                rag_tools.add_document(f"Concurrent document {i}", {"index": i})
                for i in range(5)
            ]
            
            # Execute concurrently
            memory_results, rag_results = await asyncio.gather(
                asyncio.gather(*memory_tasks),
                asyncio.gather(*rag_tasks)
            )
            
            # Verify results
            assert len(memory_results) == 5
            assert len(rag_results) == 5
            assert all(m.key.startswith("concurrent_key_") for m in memory_results)
            assert all(d.content.startswith("Concurrent document") for d in rag_results)
            
        finally:
            await memory_manager.close()
            await rag_database.close()


class TestErrorHandlingIntegration:
    """Test error handling across integrated systems."""

    @pytest.mark.asyncio
    async def test_cross_system_error_propagation(self, system_integration_settings):
        """Test error propagation between memory and RAG systems."""
        memory_manager = MemoryManager(system_integration_settings)
        await memory_manager.initialize()
        
        try:
            # Test memory error handling
            from mcp_synaptic.core.exceptions import MemoryNotFoundError
            with pytest.raises(MemoryNotFoundError):
                await memory_manager.update("nonexistent_key", data={"test": "data"})
            
            # Test that memory system continues working after error
            memory = await memory_manager.add("recovery_test", {"recovered": True})
            assert memory.key == "recovery_test"
            
        finally:
            await memory_manager.close()

    @pytest.mark.asyncio
    async def test_initialization_error_handling(self, system_integration_settings):
        """Test error handling during system initialization."""
        # Test memory manager initialization error handling
        from mcp_synaptic.core.exceptions import MemoryError
        
        with patch('mcp_synaptic.memory.manager.core.SQLiteMemoryStorage') as mock_storage:
            mock_storage.side_effect = Exception("Storage initialization failed")
            
            manager = MemoryManager(system_integration_settings)
            
            with pytest.raises(MemoryError, match="Memory manager initialization failed"):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_resource_cleanup_integration(self, system_integration_settings):
        """Test proper resource cleanup across systems."""
        memory_manager = MemoryManager(system_integration_settings)
        await memory_manager.initialize()
        
        # Mock RAG database
        with patch('chromadb.PersistentClient') as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.return_value = mock_client
            
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager') as mock_em:
                mock_embedding_manager = AsyncMock()
                mock_em.return_value = mock_embedding_manager
                
                rag_database = RAGDatabase(system_integration_settings)
                await rag_database.initialize()
        
        # Verify both systems are initialized
        assert memory_manager._initialized
        assert rag_database._initialized
        
        # Close both systems
        await memory_manager.close()
        await rag_database.close()
        
        # Verify proper cleanup
        assert not memory_manager._initialized
        assert not rag_database._initialized
        mock_embedding_manager.close.assert_called_once()