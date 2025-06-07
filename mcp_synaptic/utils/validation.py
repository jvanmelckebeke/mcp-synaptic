"""Data validation utilities."""

import re
from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import ValidationError


def validate_memory_key(key: str) -> None:
    """Validate memory key format."""
    if not key:
        raise ValidationError("Memory key cannot be empty", "key")
    
    if not isinstance(key, str):
        raise ValidationError("Memory key must be a string", "key")
    
    if len(key) > 255:
        raise ValidationError("Memory key too long (max 255 characters)", "key")
    
    # Check for valid characters (alphanumeric, underscore, hyphen, dot)
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', key):
        raise ValidationError("Memory key contains invalid characters", "key")


def validate_memory_data(data: Any) -> None:
    """Validate memory data."""
    if data is None:
        raise ValidationError("Memory data cannot be None", "data")
    
    if not isinstance(data, dict):
        raise ValidationError("Memory data must be a dictionary", "data")
    
    # Check data size (rough estimate)
    import json
    try:
        json_str = json.dumps(data)
        if len(json_str) > 1024 * 1024:  # 1MB limit
            raise ValidationError("Memory data too large (max 1MB)", "data")
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Memory data not JSON serializable: {e}", "data")


def validate_ttl_seconds(ttl_seconds: Optional[int]) -> None:
    """Validate TTL seconds value."""
    if ttl_seconds is None:
        return
    
    if not isinstance(ttl_seconds, int):
        raise ValidationError("TTL must be an integer", "ttl_seconds")
    
    if ttl_seconds < 0:
        raise ValidationError("TTL cannot be negative", "ttl_seconds")
    
    # Max TTL of 1 year
    if ttl_seconds > 365 * 24 * 3600:
        raise ValidationError("TTL too large (max 1 year)", "ttl_seconds")


def validate_memory_tags(tags: Optional[Dict[str, str]]) -> None:
    """Validate memory tags."""
    if tags is None:
        return
    
    if not isinstance(tags, dict):
        raise ValidationError("Tags must be a dictionary", "tags")
    
    for key, value in tags.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValidationError("Tag keys and values must be strings", "tags")
        
        if len(key) > 50:
            raise ValidationError("Tag key too long (max 50 characters)", "tags")
        
        if len(value) > 200:
            raise ValidationError("Tag value too long (max 200 characters)", "tags")
    
    if len(tags) > 20:
        raise ValidationError("Too many tags (max 20)", "tags")


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
    import json
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


def validate_email(email: str) -> None:
    """Validate email address format."""
    if not email:
        raise ValidationError("Email cannot be empty", "email")
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        raise ValidationError("Invalid email format", "email")


def validate_url(url: str) -> None:
    """Validate URL format."""
    if not url:
        raise ValidationError("URL cannot be empty", "url")
    
    url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(url_pattern, url):
        raise ValidationError("Invalid URL format", "url")


def sanitize_string(
    text: str,
    max_length: Optional[int] = None,
    allow_html: bool = False,
) -> str:
    """Sanitize string input."""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Remove HTML tags if not allowed
    if not allow_html:
        text = re.sub(r'<[^>]+>', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    # Truncate if necessary
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


def validate_json_schema(data: Any, schema: Dict[str, Any]) -> None:
    """Basic JSON schema validation."""
    try:
        import jsonschema
        jsonschema.validate(data, schema)
    except ImportError:
        # Fallback to basic validation if jsonschema not available
        _basic_schema_validation(data, schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(f"Schema validation failed: {e.message}", e.path)


def _basic_schema_validation(data: Any, schema: Dict[str, Any]) -> None:
    """Basic schema validation without jsonschema dependency."""
    schema_type = schema.get('type')
    
    if schema_type == 'object' and not isinstance(data, dict):
        raise ValidationError("Expected object", "type")
    elif schema_type == 'array' and not isinstance(data, list):
        raise ValidationError("Expected array", "type")
    elif schema_type == 'string' and not isinstance(data, str):
        raise ValidationError("Expected string", "type")
    elif schema_type == 'integer' and not isinstance(data, int):
        raise ValidationError("Expected integer", "type")
    elif schema_type == 'number' and not isinstance(data, (int, float)):
        raise ValidationError("Expected number", "type")
    elif schema_type == 'boolean' and not isinstance(data, bool):
        raise ValidationError("Expected boolean", "type")
    
    # Check required fields for objects
    if schema_type == 'object' and isinstance(data, dict):
        required = schema.get('required', [])
        for field in required:
            if field not in data:
                raise ValidationError(f"Required field missing: {field}", field)