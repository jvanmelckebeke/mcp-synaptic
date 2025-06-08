"""Document and RAG-related validation utilities."""

import json
from typing import Any, Dict, Optional

from ...core.exceptions import ValidationError


def validate_document_content(content: str) -> None:
    """Validate document content."""
    if not content:
        raise ValidationError("Document content cannot be empty", "content")
    
    if not isinstance(content, str):
        raise ValidationError("Document content must be a string", "content")
    
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise ValidationError("Document content too large (max 10MB)", "content")


def validate_document_metadata(metadata: Optional[Dict[str, Any]]) -> None:
    """Validate document metadata."""
    if metadata is None:
        return
    
    if not isinstance(metadata, dict):
        raise ValidationError("Document metadata must be a dictionary", "metadata")
    
    # Check metadata size
    try:
        json_str = json.dumps(metadata)
        if len(json_str) > 64 * 1024:  # 64KB limit
            raise ValidationError("Document metadata too large (max 64KB)", "metadata")
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Document metadata not JSON serializable: {e}", "metadata")


def validate_search_query(query: str) -> None:
    """Validate search query."""
    if not query:
        raise ValidationError("Search query cannot be empty", "query")
    
    if not isinstance(query, str):
        raise ValidationError("Search query must be a string", "query")
    
    if len(query) > 1000:
        raise ValidationError("Search query too long (max 1000 characters)", "query")


def validate_limit(limit: Optional[int]) -> None:
    """Validate limit parameter."""
    if limit is None:
        return
    
    if not isinstance(limit, int):
        raise ValidationError("Limit must be an integer", "limit")
    
    if limit < 1:
        raise ValidationError("Limit must be positive", "limit")
    
    if limit > 1000:
        raise ValidationError("Limit too large (max 1000)", "limit")


def validate_similarity_threshold(threshold: Optional[float]) -> None:
    """Validate similarity threshold."""
    if threshold is None:
        return
    
    if not isinstance(threshold, (int, float)):
        raise ValidationError("Similarity threshold must be a number", "similarity_threshold")
    
    if not (0.0 <= threshold <= 1.0):
        raise ValidationError("Similarity threshold must be between 0.0 and 1.0", "similarity_threshold")


def validate_rag_data(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
) -> None:
    """Validate all RAG-related data."""
    validate_document_content(content)
    validate_document_metadata(metadata)
    
    if document_id is not None:
        if not isinstance(document_id, str):
            raise ValidationError("Document ID must be a string", "document_id")
        
        if len(document_id) > 255:
            raise ValidationError("Document ID too long (max 255 characters)", "document_id")