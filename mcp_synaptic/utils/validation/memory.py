"""Memory-related validation utilities."""

import json
import re
from typing import Any, Dict, Optional

from ...core.exceptions import ValidationError


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