"""Memory types and data models."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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


class Memory(BaseModel):
    """A memory entry with metadata and expiration."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique memory identifier")
    key: str = Field(..., description="Memory key for retrieval")
    data: Dict[str, Any] = Field(..., description="Memory data payload")
    memory_type: MemoryType = Field(default=MemoryType.SHORT_TERM, description="Type of memory")
    expiration_policy: ExpirationPolicy = Field(default=ExpirationPolicy.ABSOLUTE, description="Expiration policy")
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    accessed_at: datetime = Field(default_factory=datetime.utcnow, description="Last access timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
    
    ttl_seconds: Optional[int] = Field(default=None, description="Time to live in seconds")
    access_count: int = Field(default=0, description="Number of times accessed")
    
    tags: Dict[str, str] = Field(default_factory=dict, description="User-defined tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def is_expired(self, current_time: Optional[datetime] = None) -> bool:
        """Check if the memory has expired."""
        if self.expiration_policy == ExpirationPolicy.NEVER:
            return False
        
        if self.expires_at is None:
            return False
        
        now = current_time or datetime.utcnow()
        return now >= self.expires_at

    def update_expiration(self, ttl_seconds: Optional[int] = None) -> None:
        """Update the expiration time based on policy."""
        if self.expiration_policy == ExpirationPolicy.NEVER:
            self.expires_at = None
            return
        
        ttl = ttl_seconds or self.ttl_seconds
        if ttl is None:
            return
        
        now = datetime.utcnow()
        
        if self.expiration_policy == ExpirationPolicy.ABSOLUTE:
            # Set expiration from creation time
            self.expires_at = self.created_at + timedelta(seconds=ttl)
        elif self.expiration_policy == ExpirationPolicy.SLIDING:
            # Set expiration from last access time
            self.expires_at = now + timedelta(seconds=ttl)
        
        self.ttl_seconds = ttl

    def touch(self) -> None:
        """Update access timestamp and handle sliding expiration."""
        now = datetime.utcnow()
        self.accessed_at = now
        self.access_count += 1
        
        # Update sliding expiration
        if (self.expiration_policy == ExpirationPolicy.SLIDING and 
            self.ttl_seconds is not None):
            self.expires_at = now + timedelta(seconds=self.ttl_seconds)

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary representation."""
        return self.model_dump(mode='json')

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create memory from dictionary representation."""
        return cls.model_validate(data)


class MemoryQuery(BaseModel):
    """Query parameters for memory search."""
    
    keys: Optional[list[str]] = None
    memory_types: Optional[list[MemoryType]] = None
    tags: Optional[Dict[str, str]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    expires_after: Optional[datetime] = None
    expires_before: Optional[datetime] = None
    include_expired: bool = False
    limit: Optional[int] = None
    offset: int = 0


class MemoryStats(BaseModel):
    """Statistics about memory usage."""
    
    total_memories: int
    memories_by_type: Dict[MemoryType, int]
    expired_memories: int
    total_size_bytes: int
    average_ttl_seconds: Optional[float]
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]