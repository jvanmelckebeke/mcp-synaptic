"""Integration tests for RAG system after refactoring."""

import asyncio
import pytest
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.rag.database import RAGDatabase
from mcp_synaptic.rag.embeddings import EmbeddingManager
from mcp_synaptic.models.rag import Document


@pytest.fixture
def rag_integration_settings():
    """Settings for RAG integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        settings = Settings(
            CHROMADB_PERSIST_DIRECTORY=Path(temp_dir) / "chromadb",
            EMBEDDING_PROVIDER="api",
            EMBEDDING_API_BASE="http://mock-api:4000",
            EMBEDDING_MODEL="text-embedding-ada-002",
            EMBEDDING_API_KEY="test-key",
            MAX_RAG_RESULTS=50,
            RAG_SIMILARITY_THRESHOLD=0.7
        )
        yield settings


@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager for integration tests."""
    manager = AsyncMock(spec=EmbeddingManager)
    manager.model_name = "test-model"
    manager.dimension = 1536
    
    # Mock embedding generation - create deterministic embeddings
    def create_embedding(text):
        # Simple hash-based embedding for testing
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return [(hash_val % 1000) / 1000.0] * 1536
    
    manager.embed_text.side_effect = create_embedding
    manager.embed_texts.side_effect = lambda texts: [create_embedding(t) for t in texts]
    manager.get_embedding_dimension.return_value = 1536
    manager.get_model_info.return_value = {"model": "test-model", "dimension": 1536}
    
    return manager


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB for integration tests."""
    # Mock collection
    collection = MagicMock()
    collection.count.return_value = 0
    collection.add.return_value = None
    collection.delete.return_value = None
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    # Mock client
    client = MagicMock()
    client.get_or_create_collection.return_value = collection
    
    return client, collection


class TestRAGDatabaseIntegration:
    """Test RAG database integration with embeddings and storage."""

    @pytest.mark.asyncio
    async def test_rag_document_lifecycle_integration(self, rag_integration_settings, mock_embedding_manager, mock_chromadb):
        """Test complete document lifecycle with embedding integration."""
        client, collection = mock_chromadb
        
        # Initialize RAG database with mocked dependencies
        database = RAGDatabase(rag_integration_settings)
        
        with patch('chromadb.PersistentClient', return_value=client):
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager', return_value=mock_embedding_manager):
                await database.initialize()
        
        try:
            # Ensure mocked components are properly set
            database.embedding_manager = mock_embedding_manager
            database.client = client
            database.collection = collection
            
            # Test add document with embedding generation
            content = "This is a test document for RAG integration testing."
            metadata = {"source": "integration_test", "category": "testing"}
            
            document = await database.add_document(content, metadata)
            
            # Verify document creation
            assert document.content == content
            assert document.metadata == metadata
            assert document.id is not None
            
            # Verify embedding was generated
            mock_embedding_manager.embed_text.assert_called_once_with(content)
            
            # Verify document was added to ChromaDB
            collection.add.assert_called_once()
            add_call = collection.add.call_args
            assert add_call.kwargs["ids"] == [document.id]
            assert add_call.kwargs["documents"] == [content]
            
            # Mock document retrieval
            collection.get.return_value = {
                "ids": [document.id],
                "documents": [content],
                "metadatas": [{"source": "integration_test", "category": "testing", 
                              "created_at": document.created_at.isoformat(),
                              "updated_at": document.updated_at.isoformat(),
                              "embedding_model": "test-model", "embedding_dimension": 1536}],
                "embeddings": [[0.1] * 1536]
            }
            
            # Test get document
            retrieved_doc = await database.get_document(document.id)
            assert retrieved_doc is not None
            assert retrieved_doc.content == content
            assert retrieved_doc.metadata["source"] == "integration_test"
            
            # Test update document
            new_content = "Updated content for integration testing."
            updated_doc = await database.update_document(document.id, content=new_content)
            
            # Verify embedding was regenerated for new content
            assert mock_embedding_manager.embed_text.call_count == 2
            mock_embedding_manager.embed_text.assert_called_with(new_content)
            
            # Test delete document
            deleted = await database.delete_document(document.id)
            assert deleted is True
            collection.delete.assert_called_once_with(ids=[document.id])
            
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_rag_search_integration(self, rag_integration_settings, mock_embedding_manager, mock_chromadb):
        """Test search functionality with embedding integration."""
        client, collection = mock_chromadb
        
        database = RAGDatabase(rag_integration_settings)
        
        with patch('chromadb.PersistentClient', return_value=client):
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager', return_value=mock_embedding_manager):
                await database.initialize()
        
        try:
            database.embedding_manager = mock_embedding_manager
            database.client = client
            database.collection = collection
            
            # Mock search results
            collection.query.return_value = {
                "ids": [["doc1", "doc2", "doc3"]],
                "documents": [["First document", "Second document", "Third document"]],
                "metadatas": [[
                    {"source": "test1", "created_at": "2024-01-01T00:00:00Z"},
                    {"source": "test2", "created_at": "2024-01-01T00:00:00Z"},
                    {"source": "test3", "created_at": "2024-01-01T00:00:00Z"}
                ]],
                "distances": [[0.1, 0.3, 0.5]]
            }
            
            # Test search
            query = "test search query"
            results = await database.search(query, limit=5)
            
            # Verify embedding was generated for query
            mock_embedding_manager.embed_text.assert_called_with(query)
            
            # Verify ChromaDB query was called
            collection.query.assert_called_once()
            query_call = collection.query.call_args.kwargs
            assert query_call["n_results"] == 5
            
            # Verify results
            assert len(results) == 3
            assert results[0].item.id == "doc1"
            assert results[0].score > results[1].score > results[2].score  # Sorted by similarity
            
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_rag_search_with_filters_integration(self, rag_integration_settings, mock_embedding_manager, mock_chromadb):
        """Test search with metadata filtering."""
        client, collection = mock_chromadb
        
        database = RAGDatabase(rag_integration_settings)
        
        with patch('chromadb.PersistentClient', return_value=client):
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager', return_value=mock_embedding_manager):
                await database.initialize()
        
        try:
            database.embedding_manager = mock_embedding_manager
            database.client = client
            database.collection = collection
            
            # Test search with metadata filter
            query = "filtered search"
            metadata_filter = {"category": "important"}
            similarity_threshold = 0.8
            
            await database.search(
                query=query,
                limit=10,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter
            )
            
            # Verify filters were passed to ChromaDB
            query_call = collection.query.call_args.kwargs
            assert query_call["where"] == metadata_filter
            assert query_call["n_results"] == 10
            
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_rag_collection_stats_integration(self, rag_integration_settings, mock_embedding_manager, mock_chromadb):
        """Test collection statistics with real calculations."""
        client, collection = mock_chromadb
        
        database = RAGDatabase(rag_integration_settings)
        
        with patch('chromadb.PersistentClient', return_value=client):
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager', return_value=mock_embedding_manager):
                await database.initialize()
        
        try:
            database.embedding_manager = mock_embedding_manager
            database.client = client
            database.collection = collection
            
            # Mock collection with documents
            collection.count.return_value = 5
            collection.get.return_value = {
                "documents": [
                    "Short document",
                    "This is a longer document with more content",
                    "Medium length document",
                    "Another test document",
                    "Final document for testing"
                ],
                "metadatas": [
                    {"embedding_model": "test-model", "embedding_dimension": 1536, "category": "test"},
                    {"embedding_model": "test-model", "embedding_dimension": 1536, "category": "test"},
                    {"embedding_model": "other-model", "embedding_dimension": 768, "category": "production"},
                    {"embedding_model": "test-model", "embedding_dimension": 1536, "category": "test"},
                    {"embedding_model": "test-model", "embedding_dimension": 1536, "category": "production"}
                ]
            }
            
            # Test collection stats
            stats = await database.get_collection_stats()
            
            # Verify stats calculations
            assert stats["total_documents"] == 5
            assert stats["total_embeddings"] == 5
            assert stats["embedding_dimension"] == 1536  # From first document
            assert "test-model" in stats["embedding_models"]
            assert "other-model" in stats["embedding_models"]
            assert stats["total_word_count"] > 0
            assert stats["average_content_length"] > 0
            assert "category" in stats["metadata_keys"]
            
        finally:
            await database.close()

    @pytest.mark.asyncio
    async def test_concurrent_rag_operations_integration(self, rag_integration_settings, mock_embedding_manager, mock_chromadb):
        """Test concurrent RAG operations."""
        client, collection = mock_chromadb
        
        database = RAGDatabase(rag_integration_settings)
        
        with patch('chromadb.PersistentClient', return_value=client):
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager', return_value=mock_embedding_manager):
                await database.initialize()
        
        try:
            database.embedding_manager = mock_embedding_manager
            database.client = client
            database.collection = collection
            
            # Concurrent document additions
            documents_data = [
                ("Document 1 content", {"category": "test1"}),
                ("Document 2 content", {"category": "test2"}),
                ("Document 3 content", {"category": "test3"})
            ]
            
            add_tasks = [
                database.add_document(content, metadata)
                for content, metadata in documents_data
            ]
            
            added_docs = await asyncio.gather(*add_tasks)
            assert len(added_docs) == 3
            assert all(doc.id is not None for doc in added_docs)
            
            # Verify embedding manager was called for each document
            assert mock_embedding_manager.embed_text.call_count == 3
            
            # Mock search results for concurrent searches
            collection.query.return_value = {
                "ids": [["doc1"]],
                "documents": [["Found document"]],
                "metadatas": [[{"category": "test"}]],
                "distances": [[0.2]]
            }
            
            # Concurrent searches
            search_queries = ["query1", "query2", "query3"]
            search_tasks = [database.search(query) for query in search_queries]
            
            search_results = await asyncio.gather(*search_tasks)
            assert len(search_results) == 3
            assert all(len(results) == 1 for results in search_results)
            
        finally:
            await database.close()


class TestRAGEmbeddingIntegration:
    """Test RAG database integration with different embedding providers."""

    @pytest.mark.asyncio
    async def test_api_embedding_provider_integration(self, rag_integration_settings):
        """Test RAG database with API embedding provider."""
        rag_integration_settings.EMBEDDING_PROVIDER = "api"
        
        # Mock the embedding manager initialization
        with patch('mcp_synaptic.rag.embeddings.manager.ApiEmbeddingProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.provider_name = "api"
            mock_provider.embed_text.return_value = [0.1] * 1536
            mock_provider.get_embedding_dimension.return_value = 1536
            mock_provider_class.return_value = mock_provider
            
            manager = EmbeddingManager(rag_integration_settings)
            await manager.initialize()
            
            try:
                # Test that manager uses API provider
                assert manager.provider is mock_provider
                
                # Test embedding generation
                embedding = await manager.embed_text("test text")
                assert len(embedding) == 1536
                assert embedding[0] == 0.1
                
                # Test dimension retrieval
                dimension = manager.get_embedding_dimension()
                assert dimension == 1536
                
            finally:
                await manager.close()

    @pytest.mark.asyncio
    async def test_local_embedding_provider_integration(self, rag_integration_settings):
        """Test RAG database with local embedding provider."""
        rag_integration_settings.EMBEDDING_PROVIDER = "local"
        
        # Mock the embedding manager initialization
        with patch('mcp_synaptic.rag.embeddings.manager.LocalEmbeddingProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.provider_name = "local"
            mock_provider.embed_text.return_value = [0.2] * 384
            mock_provider.get_embedding_dimension.return_value = 384
            mock_provider_class.return_value = mock_provider
            
            manager = EmbeddingManager(rag_integration_settings)
            await manager.initialize()
            
            try:
                # Test that manager uses local provider
                assert manager.provider is mock_provider
                
                # Test embedding generation
                embedding = await manager.embed_text("test text")
                assert len(embedding) == 384
                assert embedding[0] == 0.2
                
                # Test dimension retrieval
                dimension = manager.get_embedding_dimension()
                assert dimension == 384
                
            finally:
                await manager.close()

    @pytest.mark.asyncio
    async def test_embedding_similarity_computation_integration(self, rag_integration_settings):
        """Test similarity computation with embedding integration."""
        # Mock consistent embeddings for similarity testing
        with patch('mcp_synaptic.rag.embeddings.manager.ApiEmbeddingProvider') as mock_provider_class:
            mock_provider = AsyncMock()
            mock_provider.provider_name = "api"
            
            # Create different embeddings for similarity testing
            def mock_embed_texts(texts):
                if texts == ["identical", "identical"]:
                    return [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # Should give similarity = 1.0
                elif texts == ["different", "opposite"]:
                    return [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]  # Should give similarity = -1.0
                else:
                    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # Should give similarity = 0.0
            
            mock_provider.embed_texts.side_effect = mock_embed_texts
            mock_provider_class.return_value = mock_provider
            
            manager = EmbeddingManager(rag_integration_settings)
            await manager.initialize()
            
            try:
                # Test identical texts
                similarity = await manager.compute_similarity("identical", "identical")
                assert abs(similarity - 1.0) < 0.001
                
                # Test orthogonal texts
                similarity = await manager.compute_similarity("text1", "text2")
                assert abs(similarity - 0.0) < 0.001
                
                # Test opposite texts
                similarity = await manager.compute_similarity("different", "opposite")
                assert abs(similarity - (-1.0)) < 0.001
                
            finally:
                await manager.close()