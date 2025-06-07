"""Custom exceptions for MCP Synaptic."""

from typing import Any, Dict, Optional


class SynapticError(Exception):
    """Base exception for all MCP Synaptic errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """String representation of the error."""
        parts = [self.message]
        if self.error_code:
            parts.append(f"(code: {self.error_code})")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ConfigurationError(SynapticError):
    """Raised when there's a configuration issue."""

    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        details = {"config_key": config_key} if config_key else {}
        super().__init__(message, "CONFIGURATION_ERROR", details)


class MemoryError(SynapticError):
    """Raised when there's a memory management issue."""

    def __init__(self, message: str, memory_key: Optional[str] = None) -> None:
        details = {"memory_key": memory_key} if memory_key else {}
        super().__init__(message, "MEMORY_ERROR", details)


class MemoryNotFoundError(MemoryError):
    """Raised when a requested memory is not found."""

    def __init__(self, memory_key: str) -> None:
        super().__init__(f"Memory not found: {memory_key}", memory_key)
        self.error_code = "MEMORY_NOT_FOUND"


class MemoryExpiredError(MemoryError):
    """Raised when a memory has expired."""

    def __init__(self, memory_key: str) -> None:
        super().__init__(f"Memory expired: {memory_key}", memory_key)
        self.error_code = "MEMORY_EXPIRED"


class RAGError(SynapticError):
    """Raised when there's a RAG database issue."""

    def __init__(self, message: str, document_id: Optional[str] = None) -> None:
        details = {"document_id": document_id} if document_id else {}
        super().__init__(message, "RAG_ERROR", details)


class DocumentNotFoundError(RAGError):
    """Raised when a requested document is not found."""

    def __init__(self, document_id: str) -> None:
        super().__init__(f"Document not found: {document_id}", document_id)
        self.error_code = "DOCUMENT_NOT_FOUND"


class EmbeddingError(RAGError):
    """Raised when there's an embedding generation issue."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.error_code = "EMBEDDING_ERROR"


class SSEError(SynapticError):
    """Raised when there's an SSE issue."""

    def __init__(self, message: str, client_id: Optional[str] = None) -> None:
        details = {"client_id": client_id} if client_id else {}
        super().__init__(message, "SSE_ERROR", details)


class MCPError(SynapticError):
    """Raised when there's an MCP protocol issue."""

    def __init__(self, message: str, method: Optional[str] = None) -> None:
        details = {"method": method} if method else {}
        super().__init__(message, "MCP_ERROR", details)


class ValidationError(SynapticError):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        details = {"field": field} if field else {}
        super().__init__(message, "VALIDATION_ERROR", details)


class ConnectionError(SynapticError):
    """Raised when there's a connection issue."""

    def __init__(self, message: str, service: Optional[str] = None) -> None:
        details = {"service": service} if service else {}
        super().__init__(message, "CONNECTION_ERROR", details)


class DatabaseError(SynapticError):
    """Raised when there's a database issue."""

    def __init__(self, message: str, operation: Optional[str] = None) -> None:
        details = {"operation": operation} if operation else {}
        super().__init__(message, "DATABASE_ERROR", details)