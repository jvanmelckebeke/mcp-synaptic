"""Event definitions for Server-Sent Events."""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of events that can be sent via SSE."""
    
    # System events
    HEARTBEAT = "heartbeat"
    CONNECTION = "connection"
    ERROR = "error"
    
    # Memory events
    MEMORY_ADDED = "memory_added"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_EXPIRED = "memory_expired"
    MEMORY_ACCESSED = "memory_accessed"
    
    # RAG events
    DOCUMENT_ADDED = "document_added"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENT_DELETED = "document_deleted"
    SEARCH_PERFORMED = "search_performed"
    
    # MCP events
    MCP_REQUEST = "mcp_request"
    MCP_RESPONSE = "mcp_response"
    MCP_NOTIFICATION = "mcp_notification"


class Event(BaseModel):
    """Base event for SSE communication."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType = Field(..., alias="type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = Field(default_factory=dict)
    retry: Optional[int] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
        populate_by_name = True

    def to_sse_format(self) -> str:
        """Convert event to SSE format."""
        lines = []
        
        # Add event ID
        lines.append(f"id: {self.id}")
        
        # Add event type
        lines.append(f"event: {self.event_type.value}")
        
        # Add retry if specified
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        
        # Add data (JSON encoded)
        event_data = {
            "timestamp": self.timestamp.isoformat(),
            **self.data
        }
        lines.append(f"data: {json.dumps(event_data)}")
        
        # End with double newline
        return "\n".join(lines) + "\n\n"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        return cls.model_validate(data)


class HeartbeatEvent(Event):
    """Heartbeat event to keep SSE connections alive."""
    
    def __init__(self, **kwargs):
        super().__init__(
            event_type=EventType.HEARTBEAT,
            data={"message": "heartbeat"},
            **kwargs
        )


class ConnectionEvent(Event):
    """Connection event for client connect/disconnect."""
    
    def __init__(self, client_id: str, action: str, **kwargs):
        super().__init__(
            event_type=EventType.CONNECTION,
            data={
                "client_id": client_id,
                "action": action,  # "connected" or "disconnected"
            },
            **kwargs
        )


class ErrorEvent(Event):
    """Error event for SSE communication errors."""
    
    def __init__(self, error_message: str, error_code: Optional[str] = None, **kwargs):
        super().__init__(
            event_type=EventType.ERROR,
            data={
                "error": error_message,
                "error_code": error_code,
            },
            **kwargs
        )


class MemoryEvent(Event):
    """Base class for memory-related events."""
    
    def __init__(self, action: str, memory_key: str, memory_data: Optional[Dict] = None, **kwargs):
        event_type_map = {
            "added": EventType.MEMORY_ADDED,
            "updated": EventType.MEMORY_UPDATED,
            "deleted": EventType.MEMORY_DELETED,
            "expired": EventType.MEMORY_EXPIRED,
            "accessed": EventType.MEMORY_ACCESSED,
        }
        
        super().__init__(
            event_type=event_type_map.get(action, EventType.MEMORY_UPDATED),
            data={
                "action": action,
                "memory_key": memory_key,
                "memory_data": memory_data or {},
            },
            **kwargs
        )


class RAGEvent(Event):
    """Base class for RAG-related events."""
    
    def __init__(self, action: str, document_id: str, document_data: Optional[Dict] = None, **kwargs):
        event_type_map = {
            "added": EventType.DOCUMENT_ADDED,
            "updated": EventType.DOCUMENT_UPDATED,
            "deleted": EventType.DOCUMENT_DELETED,
            "searched": EventType.SEARCH_PERFORMED,
        }
        
        super().__init__(
            event_type=event_type_map.get(action, EventType.DOCUMENT_UPDATED),
            data={
                "action": action,
                "document_id": document_id,
                "document_data": document_data or {},
            },
            **kwargs
        )


class MCPEvent(Event):
    """Base class for MCP protocol events."""
    
    def __init__(self, action: str, method: str, mcp_data: Optional[Dict] = None, **kwargs):
        event_type_map = {
            "request": EventType.MCP_REQUEST,
            "response": EventType.MCP_RESPONSE,
            "notification": EventType.MCP_NOTIFICATION,
        }
        
        super().__init__(
            event_type=event_type_map.get(action, EventType.MCP_NOTIFICATION),
            data={
                "action": action,
                "method": method,
                "mcp_data": mcp_data or {},
            },
            **kwargs
        )