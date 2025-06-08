"""Data validation utilities.

DEPRECATED: This module has been refactored into domain-specific modules.
Use the validation package instead for better organization:

- mcp_synaptic.utils.validation.memory for memory-related validation
- mcp_synaptic.utils.validation.documents for document/RAG validation  
- mcp_synaptic.utils.validation.common for general validation utilities

This module is maintained for backward compatibility and imports all functions
from the new modular structure.
"""

# Import all validation functions from the new package structure
from .validation import *

# Re-export everything for backward compatibility
from .validation import __all__