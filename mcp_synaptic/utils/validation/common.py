"""Common validation utilities."""

import re
from typing import Any, Dict, Optional

from ...core.exceptions import ValidationError


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