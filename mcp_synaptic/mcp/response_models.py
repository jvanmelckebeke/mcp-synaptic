"""Pydantic response models for MCP tool outputs."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryResponse(BaseModel):
    """Response model for memory operations."""
    
    key: str = Field(description="Unique memory identifier")
    data: Dict[str, Any] = Field(description="Stored memory data")
    memory_type: str = Field(description="Type of memory (ephemeral, short_term, long_term, permanent)")
    created_at: datetime = Field(description="Creation timestamp")
    last_accessed_at: Optional[datetime] = Field(None, description="Last access timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    tags: Optional[Dict[str, str]] = Field(None, description="Memory tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    size_bytes: int = Field(description="Memory size in bytes")


class MemoryStatsResponse(BaseModel):
    """Response model for memory statistics."""
    
    total_memories: int = Field(description="Total number of memories stored")
    memories_by_type: Dict[str, int] = Field(description="Count of memories by type")
    expired_memories: int = Field(description="Number of expired memories")
    total_size_bytes: int = Field(description="Total size of all memories in bytes")
    average_ttl_seconds: Optional[float] = Field(None, description="Average TTL in seconds")
    oldest_memory: Optional[datetime] = Field(None, description="Timestamp of oldest memory")
    newest_memory: Optional[datetime] = Field(None, description="Timestamp of newest memory")


class DocumentResponse(BaseModel):
    """Response model for RAG document operations."""
    
    id: str = Field(description="Unique document identifier")
    content: str = Field(description="Document content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")


class SearchResultResponse(BaseModel):
    """Response model for RAG search results."""
    
    document: DocumentResponse = Field(description="Matching document")
    similarity_score: float = Field(description="Similarity score (0-1)")
    distance: float = Field(description="Vector distance")


class CollectionStatsResponse(BaseModel):
    """Response model for RAG collection statistics."""
    
    total_documents: int = Field(description="Total number of documents")
    total_embeddings: int = Field(description="Total number of embeddings")
    embedding_dimension: int = Field(description="Embedding vector dimension")
    collection_size_bytes: Optional[int] = Field(None, description="Collection size in bytes")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")


class BooleanResponse(BaseModel):
    """Simple boolean response for delete operations."""
    
    success: bool = Field(description="Whether the operation succeeded")
    message: Optional[str] = Field(None, description="Optional success/error message")


class ListResponse(BaseModel):
    """Generic list response wrapper."""
    
    items: List[Any] = Field(description="List of items")
    total_count: int = Field(description="Total number of items")
    offset: int = Field(0, description="Offset used for pagination")
    limit: Optional[int] = Field(None, description="Limit used for pagination")


class MemoryListResponse(ListResponse):
    """Typed list response for memories."""
    
    items: List[MemoryResponse] = Field(description="List of memory items")


class SearchListResponse(ListResponse):
    """Typed list response for search results."""
    
    items: List[SearchResultResponse] = Field(description="List of search results")
    query: str = Field(description="Search query used")


# Converter functions
def memory_to_response(memory: Any) -> MemoryResponse:
    """Convert Memory object to MemoryResponse."""
    memory_dict = memory.to_dict() if hasattr(memory, 'to_dict') else memory
    return MemoryResponse(
        key=memory_dict['key'],
        data=memory_dict['data'],
        memory_type=memory_dict['memory_type'],
        created_at=datetime.fromisoformat(memory_dict['created_at'].replace('Z', '+00:00')) if isinstance(memory_dict['created_at'], str) else memory_dict['created_at'],
        last_accessed_at=datetime.fromisoformat(memory_dict['accessed_at'].replace('Z', '+00:00')) if memory_dict.get('accessed_at') and isinstance(memory_dict['accessed_at'], str) else memory_dict.get('accessed_at'),
        expires_at=datetime.fromisoformat(memory_dict['expires_at'].replace('Z', '+00:00')) if memory_dict.get('expires_at') and isinstance(memory_dict['expires_at'], str) else memory_dict.get('expires_at'),
        tags=memory_dict.get('tags'),
        metadata=memory_dict.get('metadata'),
        size_bytes=len(str(memory_dict['data']).encode('utf-8'))  # Estimate size
    )


def document_to_response(document: Any) -> DocumentResponse:
    """Convert Document object to DocumentResponse."""
    doc_dict = document.to_dict() if hasattr(document, 'to_dict') else document
    return DocumentResponse(
        id=doc_dict['id'],
        content=doc_dict['content'],
        metadata=doc_dict.get('metadata'),
        created_at=datetime.fromisoformat(doc_dict['created_at'].replace('Z', '+00:00')) if isinstance(doc_dict['created_at'], str) else doc_dict['created_at'],
        updated_at=datetime.fromisoformat(doc_dict['updated_at'].replace('Z', '+00:00')) if doc_dict.get('updated_at') and isinstance(doc_dict['updated_at'], str) else doc_dict.get('updated_at'),
        embedding_model=doc_dict.get('embedding_model')
    )


def search_result_to_response(result: Any) -> SearchResultResponse:
    """Convert SearchResult object to SearchResultResponse."""
    result_dict = result.to_dict() if hasattr(result, 'to_dict') else result
    return SearchResultResponse(
        document=document_to_response(result_dict['document']),
        similarity_score=result_dict['score'],
        distance=result_dict.get('distance', 0.0)
    )