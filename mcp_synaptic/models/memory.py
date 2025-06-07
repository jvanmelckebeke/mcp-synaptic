"""Memory domain models for MCP Synaptic."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from .base import IdentifiedModel, PaginatedResponse, SearchQuery, StatsModel, SynapticBaseModel


class MemoryType(str, Enum):
    """Types of memory with different characteristics."""
    
    EPHEMERAL = "ephemeral"      # Very short-lived (seconds to minutes)
    SHORT_TERM = "short_term"    # Session-based (minutes to hours)  
    LONG_TERM = "long_term"      # Persistent (days to weeks)
    PERMANENT = "permanent"      # Never expires


class ExpirationPolicy(str, Enum):
    """How memory expiration should be handled."""
    
    ABSOLUTE = "absolute"        # Expires at a specific datetime
    SLIDING = "sliding"          # Expires after TTL from last access
    NEVER = "never"             # Never expires


class Memory(IdentifiedModel):
    """A memory entry with metadata and expiration."""
    
    key: str = Field(description="Unique key for memory retrieval")
    data: Dict[str, Any] = Field(description="Memory data payload")
    memory_type: MemoryType = Field(
        default=MemoryType.SHORT_TERM,
        description="Type of memory determining default behavior"
    )
    expiration_policy: ExpirationPolicy = Field(
        default=ExpirationPolicy.ABSOLUTE,
        description="How expiration time is calculated"
    )
    
    # Timestamps
    last_accessed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When memory was last accessed"
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When memory expires (null = never expires)"
    )
    
    # Metadata
    ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Time to live in seconds"
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times memory has been accessed"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="User-defined tags for categorization"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @property
    def size_bytes(self) -> int:
        """Calculate approximate memory size in bytes."""
        import sys
        return sys.getsizeof(self.data)
    
    @property
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expiration_policy == ExpirationPolicy.NEVER or not self.expires_at:
            return False
        return datetime.utcnow() >= self.expires_at
    
    def touch(self) -> None:
        """Update access timestamp and count."""
        self.last_accessed_at = datetime.utcnow()
        self.access_count += 1
        
        # Update sliding expiration
        if (self.expiration_policy == ExpirationPolicy.SLIDING and 
            self.ttl_seconds is not None):
            from datetime import timedelta
            self.expires_at = self.last_accessed_at + timedelta(seconds=self.ttl_seconds)


class MemoryQuery(SearchQuery):
    """Query parameters for memory search."""
    
    keys: Optional[List[str]] = Field(
        default=None,
        description="Specific memory keys to retrieve"
    )
    memory_types: Optional[List[MemoryType]] = Field(
        default=None,
        description="Filter by memory types"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None,
        description="Filter by tags (exact match)"
    )
    include_expired: bool = Field(
        default=False,
        description="Whether to include expired memories"
    )
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter memories created after this time"
    )
    created_before: Optional[datetime] = Field(
        default=None,
        description="Filter memories created before this time"
    )
    expires_after: Optional[datetime] = Field(
        default=None,
        description="Filter memories expiring after this time"
    )
    expires_before: Optional[datetime] = Field(
        default=None,
        description="Filter memories expiring before this time"
    )


class MemoryStats(StatsModel):
    """Statistics about memory usage."""
    
    total_memories: int = Field(ge=0, description="Total number of memories stored")
    memories_by_type: Dict[MemoryType, int] = Field(
        description="Count of memories by type"
    )
    expired_memories: int = Field(ge=0, description="Number of expired memories")
    total_size_bytes: int = Field(ge=0, description="Total size of all memories")
    average_ttl_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Average TTL across all memories with TTL"
    )
    oldest_memory: Optional[datetime] = Field(
        default=None,
        description="Creation time of oldest memory"
    )
    newest_memory: Optional[datetime] = Field(
        default=None,
        description="Creation time of newest memory"
    )
    most_accessed_count: int = Field(
        default=0,
        ge=0,
        description="Highest access count among all memories"
    )


class MemoryListResponse(PaginatedResponse[Memory]):
    """Paginated response for memory listings."""
    pass


class MemoryCreateRequest(SynapticBaseModel):
    """Request to create a new memory."""
    
    key: str = Field(description="Unique key for the memory")
    data: Dict[str, Any] = Field(description="Memory data to store")
    memory_type: MemoryType = Field(
        default=MemoryType.SHORT_TERM,
        description="Type of memory"
    )
    ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Time to live in seconds"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional tags"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata"
    )


class MemoryUpdateRequest(SynapticBaseModel):
    """Request to update an existing memory."""
    
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="New data (replaces existing)"
    )
    extend_ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Seconds to extend TTL"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None,
        description="New tags (replaces existing)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="New metadata (replaces existing)"
    )