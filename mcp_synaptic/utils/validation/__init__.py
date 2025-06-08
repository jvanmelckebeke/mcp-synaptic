"""Validation utilities package.

This package provides domain-specific validation functions organized by concern:
- memory: Memory-related validation (keys, data, TTL, tags)
- documents: Document and RAG-related validation (content, metadata, search)
- common: General validation utilities (email, URL, sanitization, JSON schema)

For backward compatibility, all validation functions are also available at the package level.
"""

# Import all validation functions to maintain backward compatibility
from .memory import (
    validate_memory_key,
    validate_memory_data,
    validate_ttl_seconds,
    validate_memory_tags,
)

from .documents import (
    validate_document_content,
    validate_document_metadata,
    validate_search_query,
    validate_limit,
    validate_similarity_threshold,
    validate_rag_data,
)

from .common import (
    validate_email,
    validate_url,
    sanitize_string,
    validate_json_schema,
    _basic_schema_validation,
)

# Export all functions for backward compatibility
__all__ = [
    # Memory validation
    "validate_memory_key",
    "validate_memory_data", 
    "validate_ttl_seconds",
    "validate_memory_tags",
    
    # Document validation
    "validate_document_content",
    "validate_document_metadata",
    "validate_search_query",
    "validate_limit",
    "validate_similarity_threshold",
    "validate_rag_data",
    
    # Common validation
    "validate_email",
    "validate_url",
    "sanitize_string",
    "validate_json_schema",
    "_basic_schema_validation",
]