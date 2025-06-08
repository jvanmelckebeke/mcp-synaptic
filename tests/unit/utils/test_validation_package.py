"""Tests for the refactored validation package structure."""

import pytest

from mcp_synaptic.core.exceptions import ValidationError


class TestValidationPackageStructure:
    """Test the new validation package organization."""
    
    def test_memory_module_imports(self):
        """Test that memory validation functions can be imported from memory module."""
        from mcp_synaptic.utils.validation.memory import (
            validate_memory_key,
            validate_memory_data,
            validate_ttl_seconds,
            validate_memory_tags,
        )
        
        # Test that functions work correctly
        validate_memory_key("test_key")
        validate_memory_data({"test": "data"})
        validate_ttl_seconds(3600)
        validate_memory_tags({"tag1": "value1"})
    
    def test_documents_module_imports(self):
        """Test that document validation functions can be imported from documents module."""
        from mcp_synaptic.utils.validation.documents import (
            validate_document_content,
            validate_document_metadata,
            validate_search_query,
            validate_limit,
            validate_similarity_threshold,
            validate_rag_data,
        )
        
        # Test that functions work correctly
        validate_document_content("test content")
        validate_document_metadata({"key": "value"})
        validate_search_query("test query")
        validate_limit(10)
        validate_similarity_threshold(0.8)
        validate_rag_data("test content", {"meta": "data"})
    
    def test_common_module_imports(self):
        """Test that common validation functions can be imported from common module."""
        from mcp_synaptic.utils.validation.common import (
            validate_email,
            validate_url,
            sanitize_string,
            validate_json_schema,
        )
        
        # Test that functions work correctly
        validate_email("test@example.com")
        validate_url("https://example.com")
        sanitized = sanitize_string("<script>test</script>", max_length=10)
        assert sanitized == "test"
        
        # Test JSON schema validation
        schema = {"type": "object", "required": ["name"]}
        validate_json_schema({"name": "test"}, schema)
    
    def test_package_level_imports_backward_compatibility(self):
        """Test that all functions are available at package level for backward compatibility."""
        from mcp_synaptic.utils.validation import (
            # Memory functions
            validate_memory_key,
            validate_memory_data,
            validate_ttl_seconds,
            validate_memory_tags,
            # Document functions
            validate_document_content,
            validate_document_metadata,
            validate_search_query,
            validate_limit,
            validate_similarity_threshold,
            validate_rag_data,
            # Common functions
            validate_email,
            validate_url,
            sanitize_string,
            validate_json_schema,
        )
        
        # Test that all functions are callable
        validate_memory_key("test")
        validate_document_content("test")
        validate_email("test@example.com")
    
    def test_old_module_backward_compatibility(self):
        """Test that the old utils.validation module still works."""
        # This should still work for backward compatibility
        from mcp_synaptic.utils.validation import validate_memory_key
        
        validate_memory_key("backward_compat_test")
    
    def test_memory_validation_functions_work(self):
        """Test memory validation functions in isolation."""
        from mcp_synaptic.utils.validation.memory import (
            validate_memory_key,
            validate_memory_data,
            validate_ttl_seconds,
            validate_memory_tags,
        )
        
        # Test valid cases
        validate_memory_key("valid_key_123")
        validate_memory_data({"key": "value", "number": 42})
        validate_ttl_seconds(0)  # 0 should be valid (permanent)
        validate_ttl_seconds(3600)  # 1 hour
        validate_memory_tags({"env": "test", "version": "1.0"})
        
        # Test invalid cases
        with pytest.raises(ValidationError):
            validate_memory_key("")  # Empty key
        
        with pytest.raises(ValidationError):
            validate_memory_key("invalid@key")  # Invalid characters
        
        with pytest.raises(ValidationError):
            validate_memory_data(None)  # None data
        
        with pytest.raises(ValidationError):
            validate_ttl_seconds(-1)  # Negative TTL
    
    def test_document_validation_functions_work(self):
        """Test document validation functions in isolation."""
        from mcp_synaptic.utils.validation.documents import (
            validate_document_content,
            validate_search_query,
            validate_similarity_threshold,
        )
        
        # Test valid cases
        validate_document_content("This is valid document content")
        validate_search_query("machine learning")
        validate_similarity_threshold(0.75)
        
        # Test invalid cases
        with pytest.raises(ValidationError):
            validate_document_content("")  # Empty content
        
        with pytest.raises(ValidationError):
            validate_search_query("")  # Empty query
        
        with pytest.raises(ValidationError):
            validate_similarity_threshold(1.5)  # Out of range
    
    def test_common_validation_functions_work(self):
        """Test common validation functions in isolation."""
        from mcp_synaptic.utils.validation.common import (
            validate_email,
            validate_url,
            sanitize_string,
        )
        
        # Test valid cases
        validate_email("user@example.com")
        validate_url("https://www.example.com")
        
        # Test sanitization
        result = sanitize_string("<script>alert('xss')</script>Hello World!", max_length=15)
        assert result == "alert('xss')Hel"
        assert "<script>" not in result
        
        # Test invalid cases
        with pytest.raises(ValidationError):
            validate_email("invalid-email")
        
        with pytest.raises(ValidationError):
            validate_url("not-a-url")
    
    def test_package_exports_all_functions(self):
        """Test that the package __all__ exports include all functions."""
        from mcp_synaptic.utils.validation import __all__
        
        expected_functions = {
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
        }
        
        assert set(__all__) == expected_functions
    
    def test_domain_separation_is_logical(self):
        """Test that functions are logically separated by domain."""
        # Memory-related functions should be in memory module
        from mcp_synaptic.utils.validation.memory import validate_memory_key
        from mcp_synaptic.utils.validation.memory import validate_memory_data
        
        # Document/RAG-related functions should be in documents module
        from mcp_synaptic.utils.validation.documents import validate_document_content
        from mcp_synaptic.utils.validation.documents import validate_rag_data
        
        # General utilities should be in common module
        from mcp_synaptic.utils.validation.common import validate_email
        from mcp_synaptic.utils.validation.common import sanitize_string
        
        # All imports should work without issues
        assert callable(validate_memory_key)
        assert callable(validate_document_content)
        assert callable(validate_email)