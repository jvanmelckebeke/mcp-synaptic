"""
Memory management with expiration capabilities.

This package provides a modular memory management system with:

- **Core Management**: Main MemoryManager class with lifecycle and coordination
- **CRUD Operations**: Create, read, update, delete operations for memories  
- **Query Operations**: List, search, statistics, and utility operations

Architecture:
- MemoryManager: Main coordinator that delegates to operation handlers
- Storage abstraction: Pluggable backends (SQLite/Redis) handled transparently
- Expiration policies: Flexible TTL management with automatic cleanup
- Comprehensive logging: Structured logging for all memory operations

The memory manager automatically handles storage backend selection, lifecycle
management, error propagation, and maintains backward compatibility through
the main MemoryManager class export.
"""

from .core import MemoryManager

__all__ = ["MemoryManager"]