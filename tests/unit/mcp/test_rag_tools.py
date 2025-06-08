"""Tests for MCP RAG tools."""

from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from mcp_synaptic.core.exceptions import RAGError, DocumentNotFoundError
from mcp_synaptic.mcp.rag_tools import RAGTools
from mcp_synaptic.models.rag import CollectionStats, Document, DocumentSearchResult
from mcp_synaptic.rag.database import RAGDatabase
from tests.utils import EmbeddingTestHelper


class TestRAGTools:
    """Test MCP RAG tools functionality."""

    @pytest.fixture
    def mock_mcp(self):
        """Create a mock FastMCP instance."""
        mcp = MagicMock()
        mcp.tool.return_value = lambda func: func  # Return function unchanged
        return mcp

    @pytest.fixture
    def mock_rag_database(self):
        """Create a mock RAG database."""
        database = AsyncMock(spec=RAGDatabase)
        return database

    @pytest.fixture
    def rag_tools(self, mock_mcp, mock_rag_database):
        """Create RAGTools instance with mocked dependencies."""
        return RAGTools(mock_mcp, mock_rag_database)

    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return Document(
            id="doc-123",
            content="This is a sample document for testing RAG functionality.",
            metadata={"source": "test", "category": "sample"},
            embedding_model="text-embedding-ada-002",
            embedding_dimension=1536
        )

    @pytest.fixture
    def sample_search_results(self, sample_document):
        """Create sample search results."""
        return [
            DocumentSearchResult(
                item=sample_document,
                score=0.85,
                rank=1,
                distance=0.15,
                embedding_model="text-embedding-ada-002",
                match_type="vector"
            )
        ]

    def test_rag_tools_initialization(self, mock_mcp, mock_rag_database):
        """Test RAGTools initialization and tool registration."""
        tools = RAGTools(mock_mcp, mock_rag_database)
        
        assert tools.mcp is mock_mcp
        assert tools.rag_database is mock_rag_database
        
        # Should register tools with FastMCP
        assert mock_mcp.tool.called

    class TestRAGAddDocument:
        """Test rag_add_document tool."""

        async def test_add_document_minimal_params(self, rag_tools, mock_rag_database, sample_document):
            """Test adding document with minimal parameters."""
            mock_rag_database.add_document.return_value = sample_document
            
            content = "Test document content"
            
            result = await mock_rag_database.add_document(
                content=content,
                metadata=None,
                document_id=None
            )
            
            mock_rag_database.add_document.assert_called_once_with(
                content=content,
                metadata=None,
                document_id=None
            )
            assert result == sample_document

        async def test_add_document_with_metadata(self, rag_tools, mock_rag_database, sample_document):
            """Test adding document with metadata."""
            mock_rag_database.add_document.return_value = sample_document
            
            content = "Document with metadata"
            metadata = {"source": "api", "category": "user_input", "priority": "high"}
            
            result = await mock_rag_database.add_document(
                content=content,
                metadata=metadata,
                document_id=None
            )
            
            mock_rag_database.add_document.assert_called_once_with(
                content=content,
                metadata=metadata,
                document_id=None
            )

        async def test_add_document_with_custom_id(self, rag_tools, mock_rag_database, sample_document):
            """Test adding document with custom ID."""
            mock_rag_database.add_document.return_value = sample_document
            
            content = "Document with custom ID"
            document_id = "custom-doc-456"
            
            result = await mock_rag_database.add_document(
                content=content,
                metadata=None,
                document_id=document_id
            )
            
            mock_rag_database.add_document.assert_called_once_with(
                content=content,
                metadata=None,
                document_id=document_id
            )

        async def test_add_document_full_params(self, rag_tools, mock_rag_database, sample_document):
            """Test adding document with all parameters."""
            mock_rag_database.add_document.return_value = sample_document
            
            content = "Complete document with all parameters"
            metadata = {"source": "test", "type": "complete", "version": "1.0"}
            document_id = "full-doc-789"
            
            result = await mock_rag_database.add_document(
                content=content,
                metadata=metadata,
                document_id=document_id
            )
            
            mock_rag_database.add_document.assert_called_once_with(
                content=content,
                metadata=metadata,
                document_id=document_id
            )

        async def test_add_document_database_failure(self, rag_tools, mock_rag_database):
            """Test document addition with database failure."""
            mock_rag_database.add_document.side_effect = RAGError("Database error")
            
            with pytest.raises(RAGError, match="Database error"):
                await mock_rag_database.add_document(
                    content="Test content",
                    metadata=None,
                    document_id=None
                )

    class TestRAGGetDocument:
        """Test rag_get_document tool."""

        async def test_get_document_found(self, rag_tools, mock_rag_database, sample_document):
            """Test retrieving existing document."""
            mock_rag_database.get_document.return_value = sample_document
            
            document_id = "doc-123"
            result = await mock_rag_database.get_document(document_id)
            
            mock_rag_database.get_document.assert_called_once_with(document_id)
            assert result == sample_document

        async def test_get_document_not_found(self, rag_tools, mock_rag_database):
            """Test retrieving non-existent document."""
            mock_rag_database.get_document.return_value = None
            
            document_id = "nonexistent-doc"
            result = await mock_rag_database.get_document(document_id)
            
            mock_rag_database.get_document.assert_called_once_with(document_id)
            assert result is None

        async def test_get_document_database_failure(self, rag_tools, mock_rag_database):
            """Test document retrieval with database failure."""
            mock_rag_database.get_document.side_effect = RAGError("Retrieval error")
            
            with pytest.raises(RAGError, match="Retrieval error"):
                await mock_rag_database.get_document("error-doc")

    class TestRAGUpdateDocument:
        """Test rag_update_document tool."""

        async def test_update_document_content_only(self, rag_tools, mock_rag_database, sample_document):
            """Test updating document content only."""
            updated_document = Document(
                id=sample_document.id,
                content="Updated content",
                metadata=sample_document.metadata
            )
            mock_rag_database.update_document.return_value = updated_document
            
            document_id = "doc-123"
            new_content = "Updated content"
            
            result = await mock_rag_database.update_document(
                document_id=document_id,
                content=new_content,
                metadata=None
            )
            
            mock_rag_database.update_document.assert_called_once_with(
                document_id=document_id,
                content=new_content,
                metadata=None
            )
            assert result.content == new_content

        async def test_update_document_metadata_only(self, rag_tools, mock_rag_database, sample_document):
            """Test updating document metadata only."""
            new_metadata = {"source": "updated", "version": "2.0"}
            updated_document = Document(
                id=sample_document.id,
                content=sample_document.content,
                metadata=new_metadata
            )
            mock_rag_database.update_document.return_value = updated_document
            
            document_id = "doc-123"
            
            result = await mock_rag_database.update_document(
                document_id=document_id,
                content=None,
                metadata=new_metadata
            )
            
            mock_rag_database.update_document.assert_called_once_with(
                document_id=document_id,
                content=None,
                metadata=new_metadata
            )

        async def test_update_document_both_content_and_metadata(self, rag_tools, mock_rag_database, sample_document):
            """Test updating both content and metadata."""
            new_content = "Completely updated content"
            new_metadata = {"source": "full_update", "complete": True}
            updated_document = Document(
                id=sample_document.id,
                content=new_content,
                metadata=new_metadata
            )
            mock_rag_database.update_document.return_value = updated_document
            
            document_id = "doc-123"
            
            result = await mock_rag_database.update_document(
                document_id=document_id,
                content=new_content,
                metadata=new_metadata
            )
            
            mock_rag_database.update_document.assert_called_once_with(
                document_id=document_id,
                content=new_content,
                metadata=new_metadata
            )

        async def test_update_document_not_found(self, rag_tools, mock_rag_database):
            """Test updating non-existent document."""
            mock_rag_database.update_document.return_value = None
            
            result = await mock_rag_database.update_document(
                document_id="nonexistent",
                content="New content",
                metadata=None
            )
            
            assert result is None

        async def test_update_document_database_failure(self, rag_tools, mock_rag_database):
            """Test document update with database failure."""
            mock_rag_database.update_document.side_effect = RAGError("Update error")
            
            with pytest.raises(RAGError, match="Update error"):
                await mock_rag_database.update_document(
                    document_id="error-doc",
                    content="New content",
                    metadata=None
                )

    class TestRAGDeleteDocument:
        """Test rag_delete_document tool."""

        async def test_delete_document_success(self, rag_tools, mock_rag_database):
            """Test successful document deletion."""
            mock_rag_database.delete_document.return_value = True
            
            document_id = "delete-doc"
            result = await mock_rag_database.delete_document(document_id)
            
            mock_rag_database.delete_document.assert_called_once_with(document_id)
            assert result is True

        async def test_delete_document_not_found(self, rag_tools, mock_rag_database):
            """Test deleting non-existent document."""
            mock_rag_database.delete_document.return_value = False
            
            document_id = "nonexistent-doc"
            result = await mock_rag_database.delete_document(document_id)
            
            mock_rag_database.delete_document.assert_called_once_with(document_id)
            assert result is False

        async def test_delete_document_database_failure(self, rag_tools, mock_rag_database):
            """Test document deletion with database failure."""
            mock_rag_database.delete_document.side_effect = RAGError("Delete error")
            
            with pytest.raises(RAGError, match="Delete error"):
                await mock_rag_database.delete_document("error-doc")

    class TestRAGSearch:
        """Test rag_search tool."""

        async def test_search_minimal_params(self, rag_tools, mock_rag_database, sample_search_results):
            """Test search with minimal parameters."""
            mock_rag_database.search.return_value = sample_search_results
            
            query = "test search"
            
            result = await mock_rag_database.search(
                query=query,
                limit=10,
                similarity_threshold=None,
                metadata_filter=None
            )
            
            mock_rag_database.search.assert_called_once_with(
                query=query,
                limit=10,
                similarity_threshold=None,
                metadata_filter=None
            )
            assert len(result) == 1
            assert result[0].item.content == sample_search_results[0].item.content

        async def test_search_with_limit(self, rag_tools, mock_rag_database, sample_search_results):
            """Test search with custom limit."""
            mock_rag_database.search.return_value = sample_search_results
            
            query = "test search with limit"
            limit = 25
            
            result = await mock_rag_database.search(
                query=query,
                limit=limit,
                similarity_threshold=None,
                metadata_filter=None
            )
            
            mock_rag_database.search.assert_called_once_with(
                query=query,
                limit=limit,
                similarity_threshold=None,
                metadata_filter=None
            )

        async def test_search_with_similarity_threshold(self, rag_tools, mock_rag_database, sample_search_results):
            """Test search with similarity threshold."""
            mock_rag_database.search.return_value = sample_search_results
            
            query = "test search with threshold"
            threshold = 0.7
            
            result = await mock_rag_database.search(
                query=query,
                limit=10,
                similarity_threshold=threshold,
                metadata_filter=None
            )
            
            mock_rag_database.search.assert_called_once_with(
                query=query,
                limit=10,
                similarity_threshold=threshold,
                metadata_filter=None
            )

        async def test_search_with_metadata_filter(self, rag_tools, mock_rag_database, sample_search_results):
            """Test search with metadata filter."""
            mock_rag_database.search.return_value = sample_search_results
            
            query = "test search with filter"
            metadata_filter = {"category": "important", "source": "api"}
            
            result = await mock_rag_database.search(
                query=query,
                limit=10,
                similarity_threshold=None,
                metadata_filter=metadata_filter
            )
            
            mock_rag_database.search.assert_called_once_with(
                query=query,
                limit=10,
                similarity_threshold=None,
                metadata_filter=metadata_filter
            )

        async def test_search_full_params(self, rag_tools, mock_rag_database, sample_search_results):
            """Test search with all parameters."""
            mock_rag_database.search.return_value = sample_search_results
            
            query = "comprehensive search"
            limit = 50
            threshold = 0.8
            metadata_filter = {"type": "document", "status": "active"}
            
            result = await mock_rag_database.search(
                query=query,
                limit=limit,
                similarity_threshold=threshold,
                metadata_filter=metadata_filter
            )
            
            mock_rag_database.search.assert_called_once_with(
                query=query,
                limit=limit,
                similarity_threshold=threshold,
                metadata_filter=metadata_filter
            )

        async def test_search_no_results(self, rag_tools, mock_rag_database):
            """Test search with no results."""
            mock_rag_database.search.return_value = []
            
            query = "no results query"
            
            result = await mock_rag_database.search(
                query=query,
                limit=10,
                similarity_threshold=None,
                metadata_filter=None
            )
            
            assert len(result) == 0

        async def test_search_database_failure(self, rag_tools, mock_rag_database):
            """Test search with database failure."""
            mock_rag_database.search.side_effect = RAGError("Search error")
            
            with pytest.raises(RAGError, match="Search error"):
                await mock_rag_database.search(
                    query="error query",
                    limit=10,
                    similarity_threshold=None,
                    metadata_filter=None
                )

    class TestRAGGetCollectionStats:
        """Test rag_get_collection_stats tool."""

        async def test_get_collection_stats_success(self, rag_tools, mock_rag_database):
            """Test successful collection stats retrieval."""
            test_stats = {
                "total_documents": 100,
                "total_embeddings": 95,
                "embedding_dimension": 1536,
                "embedding_models": ["text-embedding-ada-002", "all-MiniLM-L6-v2"],
                "total_content_length": 50000,
                "average_content_length": 500.0,
                "total_word_count": 8000,
                "collection_size_bytes": 1024000,
                "metadata_keys": ["source", "category", "author"]
            }
            mock_rag_database.get_collection_stats.return_value = test_stats
            
            result = await mock_rag_database.get_collection_stats()
            
            mock_rag_database.get_collection_stats.assert_called_once()
            assert result == test_stats
            assert result["total_documents"] == 100
            assert len(result["embedding_models"]) == 2
            assert result["average_content_length"] == 500.0

        async def test_get_collection_stats_empty_collection(self, rag_tools, mock_rag_database):
            """Test collection stats for empty collection."""
            empty_stats = {
                "total_documents": 0,
                "total_embeddings": 0,
                "embedding_dimension": None,
                "embedding_models": [],
                "total_content_length": 0,
                "average_content_length": 0,
                "total_word_count": 0,
                "collection_size_bytes": 0,
                "metadata_keys": []
            }
            mock_rag_database.get_collection_stats.return_value = empty_stats
            
            result = await mock_rag_database.get_collection_stats()
            
            assert result["total_documents"] == 0
            assert result["embedding_models"] == []
            assert result["average_content_length"] == 0

        async def test_get_collection_stats_database_failure(self, rag_tools, mock_rag_database):
            """Test collection stats with database failure."""
            mock_rag_database.get_collection_stats.side_effect = RAGError("Stats error")
            
            with pytest.raises(RAGError, match="Stats error"):
                await mock_rag_database.get_collection_stats()

    class TestToolIntegration:
        """Test tool integration and registration."""

        def test_all_tools_registered(self, mock_mcp, mock_rag_database):
            """Test that all expected tools are registered."""
            RAGTools(mock_mcp, mock_rag_database)
            
            # Verify FastMCP tool decorator was called for each tool
            # We expect 6 tools: add_document, get_document, update_document, 
            # delete_document, search, get_collection_stats
            assert mock_mcp.tool.call_count == 6

        def test_logging_functionality(self, rag_tools):
            """Test that RAGTools has logging capability."""
            # RAGTools inherits from LoggerMixin
            assert hasattr(rag_tools, 'logger')
            assert callable(getattr(rag_tools.logger, 'info', None))
            assert callable(getattr(rag_tools.logger, 'debug', None))
            assert callable(getattr(rag_tools.logger, 'error', None))

    class TestToolParameterValidation:
        """Test tool parameter validation and types."""

        async def test_content_parameter_validation(self, rag_tools):
            """Test content parameter accepts string content."""
            valid_contents = [
                "Simple text content",
                "Multi-line\ncontent\nwith\nnewlines",
                "Content with special characters: !@#$%^&*()",
                "Unicode content: 你好世界 🌍",
                "Very long content " + "word " * 1000
            ]
            
            for content in valid_contents:
                assert isinstance(content, str)
                assert len(content) > 0

        async def test_metadata_parameter_types(self, rag_tools):
            """Test metadata parameter accepts various JSON-serializable types."""
            valid_metadata_types = [
                {"string": "value"},
                {"number": 42},
                {"boolean": True},
                {"array": ["item1", "item2"]},
                {"nested": {"object": {"deep": "value"}}},
                {"mixed": {"str": "text", "num": 123, "bool": False, "list": [1, 2, 3]}}
            ]
            
            for metadata in valid_metadata_types:
                assert isinstance(metadata, dict)

        async def test_search_parameter_ranges(self, rag_tools):
            """Test search parameter ranges and validation."""
            # Valid limit ranges (should be handled by Pydantic)
            valid_limits = [1, 10, 50, 100]
            for limit in valid_limits:
                assert 1 <= limit <= 100
            
            # Valid similarity thresholds
            valid_thresholds = [0.0, 0.5, 0.7, 0.9, 1.0]
            for threshold in valid_thresholds:
                assert 0.0 <= threshold <= 1.0

    class TestErrorHandling:
        """Test error handling and propagation."""

        async def test_rag_error_propagation(self, rag_tools, mock_rag_database):
            """Test that RAGError exceptions are properly propagated."""
            mock_rag_database.add_document.side_effect = RAGError("Test RAG error")
            
            with pytest.raises(RAGError, match="Test RAG error"):
                await mock_rag_database.add_document(
                    content="test content",
                    metadata=None,
                    document_id=None
                )

        async def test_document_not_found_error_propagation(self, rag_tools, mock_rag_database):
            """Test that DocumentNotFoundError exceptions are properly propagated."""
            mock_rag_database.get_document.side_effect = DocumentNotFoundError("Document not found")
            
            with pytest.raises(DocumentNotFoundError, match="Document not found"):
                await mock_rag_database.get_document("nonexistent")

        async def test_multiple_error_types(self, rag_tools, mock_rag_database):
            """Test handling of multiple error types."""
            error_scenarios = [
                (RAGError("RAG database error"), RAGError),
                (DocumentNotFoundError("Doc not found"), DocumentNotFoundError),
                (Exception("Generic error"), Exception)
            ]
            
            for error, expected_type in error_scenarios:
                mock_rag_database.search.side_effect = error
                
                with pytest.raises(expected_type):
                    await mock_rag_database.search(
                        query="test",
                        limit=10,
                        similarity_threshold=None,
                        metadata_filter=None
                    )
                
                mock_rag_database.reset_mock()

    class TestDocumentResultFormatting:
        """Test document result formatting and serialization."""

        async def test_document_model_dump_in_tool_response(self, rag_tools, mock_rag_database, sample_document):
            """Test that document results are properly serialized."""
            mock_rag_database.add_document.return_value = sample_document
            
            # The tool should call model_dump() on the returned document
            result = await mock_rag_database.add_document(
                content="test",
                metadata=None,
                document_id=None
            )
            
            # Verify the document can be serialized
            serialized = result.model_dump()
            assert isinstance(serialized, dict)
            assert "id" in serialized
            assert "content" in serialized
            assert "metadata" in serialized

        async def test_search_results_formatting(self, rag_tools, mock_rag_database, sample_search_results):
            """Test that search results are properly formatted."""
            mock_rag_database.search.return_value = sample_search_results
            
            result = await mock_rag_database.search(
                query="test",
                limit=10,
                similarity_threshold=None,
                metadata_filter=None
            )
            
            # Verify search results structure
            assert isinstance(result, list)
            assert len(result) == 1
            
            search_result = result[0]
            assert hasattr(search_result, 'item')
            assert hasattr(search_result, 'score')
            assert hasattr(search_result, 'rank')
            assert isinstance(search_result.item, Document)

        async def test_collection_stats_formatting(self, rag_tools, mock_rag_database):
            """Test that collection stats are properly formatted."""
            stats_dict = {
                "total_documents": 10,
                "total_embeddings": 10,
                "embedding_models": ["model1", "model2"],
                "metadata_keys": ["source", "category"]
            }
            mock_rag_database.get_collection_stats.return_value = stats_dict
            
            result = await mock_rag_database.get_collection_stats()
            
            # Verify stats structure
            assert isinstance(result, dict)
            assert "total_documents" in result
            assert "embedding_models" in result
            assert isinstance(result["embedding_models"], list)
            assert isinstance(result["metadata_keys"], list)