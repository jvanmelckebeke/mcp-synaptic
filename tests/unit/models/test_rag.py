"""Tests for RAG models."""

from datetime import datetime, UTC
from typing import Dict, Any

import pytest
from pydantic import ValidationError

from mcp_synaptic.models.rag import (
    CollectionStats, Document, DocumentCreateRequest, DocumentListResponse,
    DocumentSearchQuery, DocumentSearchResult, DocumentSearchResponse,
    DocumentUpdateRequest, EmbeddingInfo, SimilaritySearchRequest
)


class TestDocument:
    """Test Document model."""

    @pytest.fixture
    def sample_document_data(self) -> Dict[str, Any]:
        """Sample document data for testing."""
        return {
            "content": "This is a sample document for testing the RAG system.",
            "metadata": {
                "source": "test",
                "category": "sample",
                "author": "test_user"
            },
            "embedding_model": "text-embedding-ada-002",
            "embedding_dimension": 1536,
            "content_hash": "abc123def456"
        }

    def test_document_creation_minimal(self):
        """Test creating document with minimal required fields."""
        content = "Minimal document content."
        
        document = Document(content=content)
        
        assert document.content == content
        assert document.metadata == {}
        assert document.embedding_model is None
        assert document.embedding_dimension is None
        assert document.content_hash is None

    def test_document_creation_full(self, sample_document_data):
        """Test creating document with all fields."""
        document = Document(**sample_document_data)
        
        assert document.content == sample_document_data["content"]
        assert document.metadata == sample_document_data["metadata"]
        assert document.embedding_model == sample_document_data["embedding_model"]
        assert document.embedding_dimension == sample_document_data["embedding_dimension"]
        assert document.content_hash == sample_document_data["content_hash"]

    def test_document_inherits_from_identified_model(self, sample_document_data):
        """Test that Document inherits ID and timestamp fields."""
        document = Document(**sample_document_data)
        
        # Should have inherited fields from IdentifiedModel
        assert hasattr(document, 'id')
        assert hasattr(document, 'created_at')
        assert hasattr(document, 'updated_at')
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.updated_at, datetime)

    def test_document_content_length_property(self, sample_document_data):
        """Test content_length property calculation."""
        document = Document(**sample_document_data)
        
        expected_length = len(sample_document_data["content"])
        assert document.content_length == expected_length

    def test_document_word_count_property(self, sample_document_data):
        """Test word_count property calculation."""
        document = Document(**sample_document_data)
        
        expected_words = len(sample_document_data["content"].split())
        assert document.word_count == expected_words

    def test_document_word_count_empty_content(self):
        """Test word count with empty content."""
        document = Document(content="")
        
        assert document.word_count == 0

    def test_document_word_count_whitespace_only(self):
        """Test word count with whitespace-only content."""
        document = Document(content="   \n\t   ")
        
        assert document.word_count == 0

    def test_document_embedding_dimension_validation(self):
        """Test embedding dimension validation."""
        # Valid dimension
        Document(content="test", embedding_dimension=1536)
        
        # Invalid dimension (must be >= 1)
        with pytest.raises(ValidationError):
            Document(content="test", embedding_dimension=0)
        
        with pytest.raises(ValidationError):
            Document(content="test", embedding_dimension=-1)

    def test_document_serialization(self, sample_document_data):
        """Test document serialization."""
        document = Document(**sample_document_data)
        
        # Test model_dump
        data = document.model_dump()
        assert data["content"] == sample_document_data["content"]
        assert data["metadata"] == sample_document_data["metadata"]
        assert data["embedding_model"] == sample_document_data["embedding_model"]
        
        # Test JSON serialization
        json_str = document.model_dump_json()
        assert isinstance(json_str, str)

    def test_document_deserialization(self, sample_document_data):
        """Test document deserialization."""
        # Add required inherited fields
        full_data = {
            **sample_document_data,
            "id": "test-doc-123",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat()
        }
        
        document = Document.model_validate(full_data)
        
        assert document.content == sample_document_data["content"]
        assert document.metadata == sample_document_data["metadata"]
        assert document.embedding_model == sample_document_data["embedding_model"]


class TestDocumentSearchQuery:
    """Test DocumentSearchQuery model."""

    def test_document_search_query_defaults(self):
        """Test DocumentSearchQuery default values."""
        query = DocumentSearchQuery()
        
        # Inherited from SearchQuery
        assert query.query == ""
        assert query.limit == 10
        assert query.offset == 0
        
        # DocumentSearchQuery specific fields
        assert query.metadata_filter is None
        assert query.content_filter is None
        assert query.similarity_threshold is None
        assert query.embedding_model is None

    def test_document_search_query_with_values(self):
        """Test DocumentSearchQuery with specified values."""
        metadata_filter = {"category": "important", "source": "api"}
        
        query = DocumentSearchQuery(
            query="test search",
            limit=20,
            offset=10,
            metadata_filter=metadata_filter,
            content_filter="specific content",
            similarity_threshold=0.8,
            embedding_model="text-embedding-ada-002"
        )
        
        assert query.query == "test search"
        assert query.limit == 20
        assert query.offset == 10
        assert query.metadata_filter == metadata_filter
        assert query.content_filter == "specific content"
        assert query.similarity_threshold == 0.8
        assert query.embedding_model == "text-embedding-ada-002"

    def test_document_search_query_similarity_threshold_validation(self):
        """Test similarity threshold validation."""
        # Valid thresholds
        DocumentSearchQuery(similarity_threshold=0.0)  # minimum
        DocumentSearchQuery(similarity_threshold=0.5)  # middle
        DocumentSearchQuery(similarity_threshold=1.0)  # maximum
        
        # Invalid thresholds
        with pytest.raises(ValidationError):
            DocumentSearchQuery(similarity_threshold=-0.1)  # below minimum
        
        with pytest.raises(ValidationError):
            DocumentSearchQuery(similarity_threshold=1.1)  # above maximum

    def test_document_search_query_inherits_search_query_validation(self):
        """Test that DocumentSearchQuery inherits SearchQuery validation."""
        # Should inherit limit validation from SearchQuery
        with pytest.raises(ValidationError):
            DocumentSearchQuery(limit=0)  # Invalid limit
        
        with pytest.raises(ValidationError):
            DocumentSearchQuery(offset=-1)  # Invalid offset


class TestDocumentSearchResult:
    """Test DocumentSearchResult model."""

    @pytest.fixture
    def sample_document(self):
        """Sample document for search results."""
        return Document(
            content="Search result document",
            metadata={"source": "test"}
        )

    def test_document_search_result_creation(self, sample_document):
        """Test creating DocumentSearchResult."""
        result = DocumentSearchResult(
            item=sample_document,
            score=0.85,
            rank=1,
            distance=0.15,
            embedding_model="text-embedding-ada-002",
            match_type="vector"
        )
        
        assert result.item == sample_document
        assert result.score == 0.85
        assert result.rank == 1
        assert result.distance == 0.15
        assert result.embedding_model == "text-embedding-ada-002"
        assert result.match_type == "vector"

    def test_document_search_result_default_match_type(self, sample_document):
        """Test default match_type value."""
        result = DocumentSearchResult(
            item=sample_document,
            score=0.85,
            rank=1,
            distance=0.15
        )
        
        assert result.match_type == "vector"  # default value

    def test_document_search_result_different_match_types(self, sample_document):
        """Test different match types."""
        # Vector match
        vector_result = DocumentSearchResult(
            item=sample_document,
            score=0.85,
            rank=1,
            distance=0.15,
            match_type="vector"
        )
        assert vector_result.match_type == "vector"
        
        # Metadata match
        metadata_result = DocumentSearchResult(
            item=sample_document,
            score=1.0,
            rank=1,
            distance=0.0,
            match_type="metadata"
        )
        assert metadata_result.match_type == "metadata"
        
        # Content match
        content_result = DocumentSearchResult(
            item=sample_document,
            score=0.9,
            rank=1,
            distance=0.1,
            match_type="content"
        )
        assert content_result.match_type == "content"


class TestDocumentSearchResponse:
    """Test DocumentSearchResponse model."""

    def test_document_search_response_creation(self):
        """Test creating DocumentSearchResponse."""
        document = Document(content="Test document")
        doc_result = DocumentSearchResult(
            item=document,
            score=0.85,
            rank=1,
            distance=0.15
        )
        
        # DocumentSearchResponse expects SearchResult[DocumentSearchResult] based on inheritance
        from mcp_synaptic.models.base import SearchResult
        search_result = SearchResult(
            item=doc_result,
            score=0.85,
            rank=1
        )
        
        response = DocumentSearchResponse(
            query="test query",
            results=[search_result],
            total_count=1,
            search_time_ms=150,
            embedding_model="text-embedding-ada-002",
            similarity_threshold=0.7
        )
        
        assert len(response.items) == 1
        assert response.query == "test query"
        assert response.total_count == 1
        assert response.search_time_ms == 150
        assert response.embedding_model == "text-embedding-ada-002"
        assert response.similarity_threshold == 0.7

    def test_document_search_response_inherits_search_response(self):
        """Test that DocumentSearchResponse inherits SearchResponse properties."""
        response = DocumentSearchResponse(
            query="test",
            results=[],
            total_count=0,
            search_time_ms=100
        )
        
        # Should have inherited fields
        assert hasattr(response, 'items')  # results become items in PaginatedResponse
        assert hasattr(response, 'query')
        assert hasattr(response, 'total_count')
        assert hasattr(response, 'search_time_ms')


class TestCollectionStats:
    """Test CollectionStats model."""

    def test_collection_stats_creation(self):
        """Test creating CollectionStats."""
        stats = CollectionStats(
            total_documents=100,
            total_embeddings=95,
            embedding_dimension=1536,
            embedding_models=["text-embedding-ada-002", "all-MiniLM-L6-v2"],
            total_content_length=50000,
            average_content_length=500.0,
            total_word_count=8000,
            collection_size_bytes=1024000,
            metadata_keys=["source", "category", "author"]
        )
        
        assert stats.total_documents == 100
        assert stats.total_embeddings == 95
        assert stats.embedding_dimension == 1536
        assert len(stats.embedding_models) == 2
        assert stats.total_content_length == 50000
        assert stats.average_content_length == 500.0
        assert stats.total_word_count == 8000
        assert stats.collection_size_bytes == 1024000
        assert len(stats.metadata_keys) == 3

    def test_collection_stats_validation(self):
        """Test CollectionStats validation."""
        # Valid stats
        CollectionStats(
            total_documents=0,
            total_embeddings=0,
            total_content_length=0,
            average_content_length=0.0,
            total_word_count=0
        )
        
        # Invalid negative values
        with pytest.raises(ValidationError):
            CollectionStats(
                total_documents=-1,
                total_embeddings=0,
                total_content_length=0,
                average_content_length=0.0,
                total_word_count=0
            )
        
        with pytest.raises(ValidationError):
            CollectionStats(
                total_documents=0,
                total_embeddings=0,
                total_content_length=-1,
                average_content_length=0.0,
                total_word_count=0
            )

    def test_collection_stats_embedding_dimension_validation(self):
        """Test embedding dimension validation."""
        # Valid dimension
        CollectionStats(
            total_documents=1,
            total_embeddings=1,
            embedding_dimension=1536,
            total_content_length=100,
            average_content_length=100.0,
            total_word_count=10
        )
        
        # Invalid dimension
        with pytest.raises(ValidationError):
            CollectionStats(
                total_documents=1,
                total_embeddings=1,
                embedding_dimension=0,  # Must be >= 1
                total_content_length=100,
                average_content_length=100.0,
                total_word_count=10
            )

    def test_collection_stats_default_values(self):
        """Test CollectionStats default values."""
        stats = CollectionStats(
            total_documents=10,
            total_embeddings=10,
            total_content_length=1000,
            average_content_length=100.0,
            total_word_count=150
        )
        
        assert stats.embedding_models == []  # default
        assert stats.metadata_keys == []  # default
        assert stats.embedding_dimension is None  # default
        assert stats.collection_size_bytes is None  # default


class TestDocumentCreateRequest:
    """Test DocumentCreateRequest model."""

    def test_document_create_request_minimal(self):
        """Test DocumentCreateRequest with minimal fields."""
        request = DocumentCreateRequest(content="Test document content")
        
        assert request.content == "Test document content"
        assert request.metadata is None
        assert request.document_id is None

    def test_document_create_request_full(self):
        """Test DocumentCreateRequest with all fields."""
        metadata = {"source": "api", "category": "test"}
        
        request = DocumentCreateRequest(
            content="Full test document content",
            metadata=metadata,
            document_id="custom-doc-123"
        )
        
        assert request.content == "Full test document content"
        assert request.metadata == metadata
        assert request.document_id == "custom-doc-123"

    def test_document_create_request_content_validation_empty(self):
        """Test content validation rejects empty content."""
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            DocumentCreateRequest(content="")

    def test_document_create_request_content_validation_whitespace(self):
        """Test content validation rejects whitespace-only content."""
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            DocumentCreateRequest(content="   \n\t   ")

    def test_document_create_request_content_validation_valid(self):
        """Test content validation accepts valid content."""
        # Should not raise validation error
        DocumentCreateRequest(content="Valid content")
        DocumentCreateRequest(content="  Valid content with leading spaces  ")


class TestDocumentUpdateRequest:
    """Test DocumentUpdateRequest model."""

    def test_document_update_request_all_none(self):
        """Test DocumentUpdateRequest with all optional fields."""
        request = DocumentUpdateRequest()
        
        assert request.content is None
        assert request.metadata is None

    def test_document_update_request_with_values(self):
        """Test DocumentUpdateRequest with values."""
        metadata = {"updated": True, "version": 2}
        
        request = DocumentUpdateRequest(
            content="Updated content",
            metadata=metadata
        )
        
        assert request.content == "Updated content"
        assert request.metadata == metadata

    def test_document_update_request_content_validation_empty(self):
        """Test content validation rejects empty content."""
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            DocumentUpdateRequest(content="")

    def test_document_update_request_content_validation_whitespace(self):
        """Test content validation rejects whitespace-only content."""
        with pytest.raises(ValidationError, match="Content cannot be empty"):
            DocumentUpdateRequest(content="   \n\t   ")

    def test_document_update_request_content_validation_none_allowed(self):
        """Test content validation allows None."""
        # Should not raise validation error
        request = DocumentUpdateRequest(content=None)
        assert request.content is None

    def test_document_update_request_content_validation_valid(self):
        """Test content validation accepts valid content."""
        # Should not raise validation error
        DocumentUpdateRequest(content="Valid updated content")


class TestEmbeddingInfo:
    """Test EmbeddingInfo model."""

    def test_embedding_info_creation(self):
        """Test creating EmbeddingInfo."""
        info = EmbeddingInfo(
            model_name="text-embedding-ada-002",
            dimension=1536,
            max_tokens=8192,
            provider="api",
            is_available=True
        )
        
        assert info.model_name == "text-embedding-ada-002"
        assert info.dimension == 1536
        assert info.max_tokens == 8192
        assert info.provider == "api"
        assert info.is_available is True

    def test_embedding_info_minimal(self):
        """Test EmbeddingInfo with minimal required fields."""
        info = EmbeddingInfo(
            model_name="all-MiniLM-L6-v2",
            dimension=384,
            provider="local",
            is_available=False
        )
        
        assert info.model_name == "all-MiniLM-L6-v2"
        assert info.dimension == 384
        assert info.max_tokens is None  # optional
        assert info.provider == "local"
        assert info.is_available is False

    def test_embedding_info_dimension_validation(self):
        """Test dimension validation."""
        # Valid dimension
        EmbeddingInfo(
            model_name="test-model",
            dimension=512,
            provider="test",
            is_available=True
        )
        
        # Invalid dimension
        with pytest.raises(ValidationError):
            EmbeddingInfo(
                model_name="test-model",
                dimension=0,  # Must be >= 1
                provider="test",
                is_available=True
            )

    def test_embedding_info_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        EmbeddingInfo(
            model_name="test-model",
            dimension=512,
            max_tokens=4096,
            provider="test",
            is_available=True
        )
        
        # Invalid max_tokens
        with pytest.raises(ValidationError):
            EmbeddingInfo(
                model_name="test-model",
                dimension=512,
                max_tokens=0,  # Must be >= 1
                provider="test",
                is_available=True
            )


class TestSimilaritySearchRequest:
    """Test SimilaritySearchRequest model."""

    def test_similarity_search_request_minimal(self):
        """Test SimilaritySearchRequest with minimal fields."""
        request = SimilaritySearchRequest(query="test search")
        
        assert request.query == "test search"
        assert request.limit == 10  # default
        assert request.similarity_threshold is None
        assert request.metadata_filter is None
        assert request.include_embeddings is False  # default

    def test_similarity_search_request_full(self):
        """Test SimilaritySearchRequest with all fields."""
        metadata_filter = {"category": "important"}
        
        request = SimilaritySearchRequest(
            query="comprehensive search query",
            limit=25,
            similarity_threshold=0.75,
            metadata_filter=metadata_filter,
            include_embeddings=True
        )
        
        assert request.query == "comprehensive search query"
        assert request.limit == 25
        assert request.similarity_threshold == 0.75
        assert request.metadata_filter == metadata_filter
        assert request.include_embeddings is True

    def test_similarity_search_request_limit_validation(self):
        """Test limit validation."""
        # Valid limits
        SimilaritySearchRequest(query="test", limit=1)  # minimum
        SimilaritySearchRequest(query="test", limit=100)  # maximum
        
        # Invalid limits
        with pytest.raises(ValidationError):
            SimilaritySearchRequest(query="test", limit=0)  # below minimum
        
        with pytest.raises(ValidationError):
            SimilaritySearchRequest(query="test", limit=101)  # above maximum

    def test_similarity_search_request_similarity_threshold_validation(self):
        """Test similarity threshold validation."""
        # Valid thresholds
        SimilaritySearchRequest(query="test", similarity_threshold=0.0)  # minimum
        SimilaritySearchRequest(query="test", similarity_threshold=1.0)  # maximum
        
        # Invalid thresholds
        with pytest.raises(ValidationError):
            SimilaritySearchRequest(query="test", similarity_threshold=-0.1)  # below minimum
        
        with pytest.raises(ValidationError):
            SimilaritySearchRequest(query="test", similarity_threshold=1.1)  # above maximum


class TestDocumentListResponse:
    """Test DocumentListResponse model."""

    def test_document_list_response_creation(self):
        """Test creating DocumentListResponse."""
        documents = [
            Document(content="First document"),
            Document(content="Second document")
        ]
        
        response = DocumentListResponse(
            items=documents,
            total_count=2,
            offset=0,
            limit=10
        )
        
        assert len(response.items) == 2
        assert response.total_count == 2
        assert response.offset == 0
        assert response.limit == 10
        assert response.has_more is False

    def test_document_list_response_inherits_paginated_response(self):
        """Test that DocumentListResponse inherits PaginatedResponse properties."""
        response = DocumentListResponse(
            items=[],
            total_count=0,
            offset=0,
            limit=10
        )
        
        # Should have inherited fields
        assert hasattr(response, 'items')
        assert hasattr(response, 'total_count')
        assert hasattr(response, 'offset')
        assert hasattr(response, 'limit')
        assert hasattr(response, 'has_more')