"""Tests for refactored RAG database package structure."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.core.exceptions import RAGError
from mcp_synaptic.rag.database import RAGDatabase
from mcp_synaptic.rag.database.core import RAGDatabase as CoreRAGDatabase
from mcp_synaptic.rag.database.documents import DocumentOperations
from mcp_synaptic.rag.database.search import SearchOperations
from mcp_synaptic.rag.database.stats import StatsOperations
from mcp_synaptic.models.rag import Document, DocumentSearchResult


class TestDatabasePackageStructure:
    """Test the refactored database package structure."""

    def test_package_imports(self):
        """Test that all package imports work correctly."""
        # Test main export
        from mcp_synaptic.rag.database import RAGDatabase
        assert RAGDatabase is not None
        
        # Test direct imports
        from mcp_synaptic.rag.database.core import RAGDatabase as CoreDatabase
        from mcp_synaptic.rag.database.documents import DocumentOperations
        from mcp_synaptic.rag.database.search import SearchOperations
        from mcp_synaptic.rag.database.stats import StatsOperations
        
        # Verify they are classes
        assert callable(CoreDatabase)
        assert callable(DocumentOperations)
        assert callable(SearchOperations)
        assert callable(StatsOperations)

    def test_database_delegation_structure(self):
        """Test that database properly delegates to operation handlers."""
        from mcp_synaptic.rag.database.core import RAGDatabase
        
        settings = MagicMock(spec=Settings)
        database = RAGDatabase(settings)
        
        # Should have operation handlers as None initially
        assert database._documents is None
        assert database._search is None
        assert database._stats is None
        assert not database._initialized


class TestDocumentOperations:
    """Test document operations handler."""

    @pytest.fixture
    def mock_collection(self):
        """Create mock ChromaDB collection."""
        return MagicMock()

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        manager = AsyncMock()
        manager.embed_text.return_value = [0.1] * 1536
        # get_embedding_dimension should be a regular method, not async
        manager.get_embedding_dimension = MagicMock(return_value=1536)
        return manager

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.EMBEDDING_MODEL = "test-model"
        return settings

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    @pytest.fixture
    def document_ops(self, mock_collection, mock_embedding_manager, mock_settings, mock_logger):
        """Create DocumentOperations instance."""
        return DocumentOperations(mock_collection, mock_embedding_manager, mock_settings, mock_logger)

    @pytest.mark.asyncio
    async def test_add_document_success(self, document_ops, mock_collection, mock_embedding_manager):
        """Test successful document addition."""
        # Arrange
        content = "Test document content"
        metadata = {"category": "test"}
        
        # Act
        result = await document_ops.add_document(content, metadata)
        
        # Assert
        mock_embedding_manager.embed_text.assert_called_once_with(content)
        mock_collection.add.assert_called_once()
        assert result.content == content
        assert result.metadata == metadata
        assert result.id is not None

    @pytest.mark.asyncio
    async def test_get_document_success(self, document_ops, mock_collection):
        """Test successful document retrieval."""
        # Arrange
        document_id = "test-id"
        mock_collection.get.return_value = {
            "ids": [document_id],
            "documents": ["Test content"],
            "metadatas": [{
                "category": "test",
                "created_at": datetime.now(UTC).isoformat(),
                "updated_at": datetime.now(UTC).isoformat(),
                "embedding_model": "test-model",
                "embedding_dimension": 1536
            }]
        }
        
        # Act
        result = await document_ops.get_document(document_id)
        
        # Assert
        mock_collection.get.assert_called_once()
        assert result is not None
        assert result.id == document_id
        assert result.content == "Test content"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, document_ops, mock_collection):
        """Test document retrieval when document doesn't exist."""
        # Arrange
        mock_collection.get.return_value = {"ids": []}
        
        # Act
        result = await document_ops.get_document("nonexistent-id")
        
        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_document_success(self, document_ops, mock_collection):
        """Test successful document deletion."""
        # Arrange
        document_id = "test-id"
        
        # Mock get_document to return existing document
        with patch.object(document_ops, 'get_document', return_value=Document(id=document_id, content="test")):
            # Act
            result = await document_ops.delete_document(document_id)
            
            # Assert
            mock_collection.delete.assert_called_once_with(ids=[document_id])
            assert result is True


class TestSearchOperations:
    """Test search operations handler."""

    @pytest.fixture
    def mock_collection(self):
        """Create mock ChromaDB collection."""
        return MagicMock()

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        manager = AsyncMock()
        manager.embed_text.return_value = [0.1] * 1536
        # get_embedding_dimension should be a regular method, not async
        manager.get_embedding_dimension = MagicMock(return_value=1536)
        return manager

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.RAG_SIMILARITY_THRESHOLD = 0.7
        settings.MAX_RAG_RESULTS = 10
        return settings

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    @pytest.fixture
    def search_ops(self, mock_collection, mock_embedding_manager, mock_settings, mock_logger):
        """Create SearchOperations instance."""
        return SearchOperations(mock_collection, mock_embedding_manager, mock_settings, mock_logger)

    @pytest.mark.asyncio
    async def test_search_success(self, search_ops, mock_collection, mock_embedding_manager):
        """Test successful document search."""
        # Arrange
        query = "test query"
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Content 1", "Content 2"]],
            "metadatas": [[
                {"category": "test", "created_at": datetime.now(UTC).isoformat()},
                {"category": "test", "created_at": datetime.now(UTC).isoformat()}
            ]],
            "distances": [[0.1, 0.2]]
        }
        
        # Act
        results = await search_ops.search(query, limit=5)
        
        # Assert
        mock_embedding_manager.embed_text.assert_called_once_with(query)
        mock_collection.query.assert_called_once()
        assert len(results) == 2
        assert all(isinstance(r, DocumentSearchResult) for r in results)
        assert results[0].score > results[1].score  # Better similarity

    @pytest.mark.asyncio
    async def test_search_no_embedding_manager(self, search_ops):
        """Test search fails without embedding manager."""
        # Arrange
        search_ops.embedding_manager = None
        
        # Act & Assert
        with pytest.raises(RAGError, match="Embedding manager not available"):
            await search_ops.search("test query")


class TestStatsOperations:
    """Test statistics operations handler."""

    @pytest.fixture
    def mock_collection(self):
        """Create mock ChromaDB collection."""
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        return MagicMock(spec=Settings)

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return MagicMock()

    @pytest.fixture
    def stats_ops(self, mock_collection, mock_settings, mock_logger):
        """Create StatsOperations instance."""
        return StatsOperations(mock_collection, mock_settings, mock_logger)

    @pytest.mark.asyncio
    async def test_get_collection_stats_empty(self, stats_ops, mock_collection):
        """Test statistics for empty collection."""
        # Arrange
        mock_collection.count.return_value = 0
        
        # Act
        stats = await stats_ops.get_collection_stats()
        
        # Assert
        assert stats["total_documents"] == 0
        assert stats["total_embeddings"] == 0
        assert stats["embedding_dimension"] is None

    @pytest.mark.asyncio
    async def test_get_collection_stats_with_documents(self, stats_ops, mock_collection):
        """Test statistics for collection with documents."""
        # Arrange
        mock_collection.count.return_value = 2
        mock_collection.get.return_value = {
            "documents": ["Short doc", "This is a longer document with more words"],
            "metadatas": [
                {"category": "test", "embedding_model": "model1", "embedding_dimension": 1536},
                {"category": "test", "embedding_model": "model2", "embedding_dimension": 1536}
            ]
        }
        
        # Act
        stats = await stats_ops.get_collection_stats()
        
        # Assert
        assert stats["total_documents"] == 2
        assert stats["total_embeddings"] == 2
        assert stats["embedding_dimension"] == 1536
        assert "model1" in stats["embedding_models"]
        assert "model2" in stats["embedding_models"]
        assert stats["total_word_count"] == 10  # 2 + 8 words


class TestCoreRAGDatabase:
    """Test core RAG database coordination."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock(spec=Settings)
        settings.CHROMADB_PERSIST_DIRECTORY = MagicMock()
        settings.CHROMADB_PERSIST_DIRECTORY.mkdir = MagicMock()
        return settings

    def test_database_creation(self, mock_settings):
        """Test database can be created."""
        database = CoreRAGDatabase(mock_settings)
        assert database.client is None
        assert database.collection is None
        assert database._documents is None
        assert database._search is None
        assert database._stats is None
        assert not database._initialized

    @pytest.mark.asyncio
    async def test_database_initialization_no_chromadb(self, mock_settings):
        """Test database initialization fails without ChromaDB."""
        database = CoreRAGDatabase(mock_settings)
        
        with patch('mcp_synaptic.rag.database.core.chromadb', None):
            with pytest.raises(RAGError, match="ChromaDB not available"):
                await database.initialize()

    @pytest.mark.asyncio
    async def test_database_delegation_add_document(self, mock_settings):
        """Test database delegates add_document operation."""
        database = CoreRAGDatabase(mock_settings)
        
        # Mock operation handlers
        mock_documents = AsyncMock()
        mock_documents.add_document.return_value = Document(id="test", content="test")
        mock_search = AsyncMock()
        mock_stats = AsyncMock()
        
        database._documents = mock_documents
        database._search = mock_search
        database._stats = mock_stats
        database._initialized = True
        
        result = await database.add_document("test content")
        
        mock_documents.add_document.assert_called_once()
        assert result.content == "test"

    @pytest.mark.asyncio
    async def test_database_not_initialized_errors(self, mock_settings):
        """Test database raises errors when not initialized."""
        database = CoreRAGDatabase(mock_settings)
        
        with pytest.raises(RAGError, match="not initialized"):
            await database.add_document("test")
        
        with pytest.raises(RAGError, match="not initialized"):
            await database.search("test")
        
        with pytest.raises(RAGError, match="not initialized"):
            await database.get_collection_stats()


class TestDatabaseIntegration:
    """Integration tests for the database package."""

    @pytest.fixture
    def integration_settings(self):
        """Create settings for integration tests."""
        settings = MagicMock(spec=Settings)
        settings.CHROMADB_PERSIST_DIRECTORY = MagicMock()
        settings.CHROMADB_PERSIST_DIRECTORY.mkdir = MagicMock()
        return settings

    @pytest.mark.asyncio
    async def test_error_propagation(self, integration_settings):
        """Test that errors are properly propagated through the chain."""
        database = CoreRAGDatabase(integration_settings)
        
        # Mock ChromaDB to raise error
        with patch('mcp_synaptic.rag.database.core.chromadb') as mock_chromadb:
            mock_client = MagicMock()
            mock_client.get_or_create_collection.side_effect = Exception("ChromaDB error")
            mock_chromadb.PersistentClient.return_value = mock_client
            
            with pytest.raises(RAGError, match="RAG database initialization failed"):
                await database.initialize()

    @pytest.mark.asyncio
    async def test_handlers_coordination(self, integration_settings):
        """Test that different operation handlers work together."""
        database = CoreRAGDatabase(integration_settings)
        
        # Mock successful initialization
        with patch('mcp_synaptic.rag.database.core.chromadb') as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            
            with patch('mcp_synaptic.rag.database.core.EmbeddingManager') as mock_em:
                mock_embedding_manager = AsyncMock()
                mock_em.return_value = mock_embedding_manager
                
                await database.initialize()
                
                # Test that all operation types are available
                assert database._documents is not None
                assert database._search is not None
                assert database._stats is not None
                assert hasattr(database, 'add_document')  # Documents
                assert hasattr(database, 'search')  # Search
                assert hasattr(database, 'get_collection_stats')  # Stats