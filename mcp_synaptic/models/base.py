"""Base model classes and generics for MCP Synaptic."""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# Type variables for generics
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound=BaseModel)


class SynapticBaseModel(BaseModel):
    """Base model with common configuration for all MCP Synaptic models."""
    
    model_config = ConfigDict(
        # Keep enum objects in memory, serialize values only when needed
        use_enum_values=False,
        # Allow population by field name or alias
        populate_by_name=True,
        # Validate assignment after model creation
        validate_assignment=True,
        # Include extra validation info in errors
        extra='forbid',
        # Use ISO format for datetime serialization
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    )
    


class TimestampedModel(SynapticBaseModel):
    """Base model for entities with timestamps."""
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the entity was created"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="When the entity was last updated"
    )


class IdentifiedModel(TimestampedModel):
    """Base model for entities with ID and timestamps."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier")


class OperationResult(SynapticBaseModel, Generic[T]):
    """Generic result wrapper for operations."""
    
    success: bool = Field(description="Whether the operation succeeded")
    data: Optional[T] = Field(default=None, description="Operation result data")
    message: Optional[str] = Field(default=None, description="Success or error message")
    error_code: Optional[str] = Field(default=None, description="Error code if operation failed")


class ListResponse(SynapticBaseModel, Generic[T]):
    """Generic response for list operations."""
    
    items: List[T] = Field(description="List of items")
    total_count: int = Field(description="Total number of items available")
    returned_count: int = Field(description="Number of items in this response")
    
    def __init__(self, items: List[T], total_count: Optional[int] = None, **kwargs):
        returned_count = len(items)
        super().__init__(
            items=items,
            total_count=total_count or returned_count,
            returned_count=returned_count,
            **kwargs
        )


class PaginatedResponse(ListResponse[T]):
    """Generic response for paginated list operations."""
    
    offset: int = Field(default=0, description="Number of items skipped")
    limit: Optional[int] = Field(default=None, description="Maximum items per page")
    has_more: bool = Field(description="Whether more items are available")
    
    def __init__(
        self, 
        items: List[T], 
        total_count: int,
        offset: int = 0,
        limit: Optional[int] = None,
        **kwargs
    ):
        has_more = (offset + len(items)) < total_count
        super().__init__(
            items=items,
            total_count=total_count,
            offset=offset,
            limit=limit,
            has_more=has_more,
            **kwargs
        )


class StatsModel(SynapticBaseModel):
    """Base model for statistics responses."""
    
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When these statistics were generated"
    )


class SearchQuery(SynapticBaseModel):
    """Base model for search queries."""
    
    query: str = Field(description="Search query string")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


class SearchResult(SynapticBaseModel, Generic[T]):
    """Generic search result with scoring."""
    
    item: T = Field(description="The matching item")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score (0-1)")
    rank: int = Field(ge=1, description="Result ranking (1-based)")


class SearchResponse(PaginatedResponse[SearchResult[T]]):
    """Generic response for search operations."""
    
    query: str = Field(description="Original search query")
    max_score: float = Field(description="Highest similarity score in results")
    search_time_ms: Optional[float] = Field(default=None, description="Search execution time in milliseconds")
    
    def __init__(
        self,
        query: str,
        results: List[SearchResult[T]],
        total_count: int,
        offset: int = 0,
        limit: Optional[int] = None,
        search_time_ms: Optional[float] = None,
        **kwargs
    ):
        max_score = max((r.score for r in results), default=0.0)
        super().__init__(
            items=results,
            total_count=total_count,
            offset=offset,
            limit=limit,
            query=query,
            max_score=max_score,
            search_time_ms=search_time_ms,
            **kwargs
        )