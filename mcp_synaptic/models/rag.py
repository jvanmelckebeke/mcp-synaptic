"""RAG (Retrieval Augmented Generation) domain models for MCP Synaptic."""

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator

from .base import (
    IdentifiedModel, 
    PaginatedResponse, 
    SearchQuery, 
    SearchResponse,
    SearchResult,
    StatsModel, 
    SynapticBaseModel
)


class Document(IdentifiedModel):
    """A document in the RAG database."""
    
    content: str = Field(description="Document content text")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata and attributes"
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Name of the embedding model used"
    )
    embedding_dimension: Optional[int] = Field(
        default=None,
        ge=1,
        description="Dimension of the embedding vector"
    )
    content_hash: Optional[str] = Field(
        default=None,
        description="Hash of content for deduplication"
    )
    
    @property
    def content_length(self) -> int:
        """Get character length of content."""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Estimate word count in content."""
        return len(self.content.split())


class DocumentSearchQuery(SearchQuery):
    """Query parameters for document search."""
    
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter documents by metadata attributes"
    )
    content_filter: Optional[str] = Field(
        default=None,
        description="Filter documents containing this text"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for results"
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Filter by embedding model used"
    )


class DocumentSearchResult(SearchResult[Document]):
    """Search result for a document with additional RAG metadata."""
    
    model_config = ConfigDict(extra="allow")
    
    distance: float = Field(description="Vector distance (lower = more similar)")
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model used for this result"
    )
    match_type: str = Field(
        default="vector",
        description="Type of match (vector, metadata, content)"
    )


class DocumentSearchResponse(SearchResponse[DocumentSearchResult]):
    """Response for document search operations."""
    
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model used for search"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        description="Similarity threshold applied"
    )


class CollectionStats(StatsModel):
    """Statistics about the RAG document collection."""
    
    total_documents: int = Field(ge=0, description="Total number of documents")
    total_embeddings: int = Field(ge=0, description="Total number of embeddings")
    embedding_dimension: Optional[int] = Field(
        default=None,
        ge=1,
        description="Embedding vector dimension"
    )
    embedding_models: List[str] = Field(
        default_factory=list,
        description="List of embedding models used"
    )
    total_content_length: int = Field(
        ge=0,
        description="Total character length of all documents"
    )
    average_content_length: float = Field(
        ge=0,
        description="Average character length per document"
    )
    total_word_count: int = Field(
        ge=0,
        description="Total estimated word count"
    )
    collection_size_bytes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total storage size in bytes"
    )
    metadata_keys: List[str] = Field(
        default_factory=list,
        description="All metadata keys found across documents"
    )


class DocumentListResponse(PaginatedResponse[Document]):
    """Paginated response for document listings."""
    pass


class DocumentCreateRequest(SynapticBaseModel):
    """Request to create a new document."""
    
    content: str = Field(description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Document metadata"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Custom document ID (auto-generated if not provided)"
    )
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v


class DocumentUpdateRequest(SynapticBaseModel):
    """Request to update an existing document."""
    
    content: Optional[str] = Field(
        default=None,
        description="New content (replaces existing)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="New metadata (replaces existing)"
    )
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError('Content cannot be empty')
        return v


class EmbeddingInfo(SynapticBaseModel):
    """Information about an embedding model."""
    
    model_name: str = Field(description="Name of the embedding model")
    dimension: int = Field(ge=1, description="Embedding vector dimension")
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum input tokens supported"
    )
    provider: str = Field(description="Embedding provider (api, local)")
    is_available: bool = Field(description="Whether the model is currently available")


class SimilaritySearchRequest(SynapticBaseModel):
    """Request for similarity search."""
    
    query: str = Field(description="Text query to search for")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter by document metadata"
    )
    include_embeddings: bool = Field(
        default=False,
        description="Whether to include embedding vectors in response"
    )