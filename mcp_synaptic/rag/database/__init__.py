"""
RAG database implementation using ChromaDB for document storage and similarity search.

This package provides a modular RAG (Retrieval-Augmented Generation) database system with:

- **Core Management**: Main RAGDatabase class with lifecycle and coordination
- **Document Operations**: CRUD operations for document management
- **Search Operations**: Vector similarity search with filtering and ranking
- **Statistics**: Collection analysis and metadata extraction

Architecture:
- RAGDatabase: Main coordinator that delegates to operation handlers
- ChromaDB integration: Persistent vector storage with configurable embeddings
- Embedding abstraction: Pluggable embedding providers (API/local)
- Comprehensive logging: Structured logging for all database operations

The RAG database automatically handles ChromaDB client management, embedding generation,
document lifecycle, and maintains backward compatibility through the main RAGDatabase
class export.
"""

from .core import RAGDatabase

__all__ = ["RAGDatabase"]