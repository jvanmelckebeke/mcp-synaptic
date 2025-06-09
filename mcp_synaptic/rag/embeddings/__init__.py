"""
Embedding generation and management for RAG functionality.

This package provides a flexible embedding system with multiple provider backends:

- **API Provider**: Uses external OpenAI-compatible endpoints for embeddings
- **Local Provider**: Uses sentence-transformers for local embedding generation

Architecture:
- EmbeddingProvider: Abstract base class for all embedding providers
- ApiEmbeddingProvider: API-based provider for external services  
- LocalEmbeddingProvider: Local sentence-transformers provider
- EmbeddingManager: Main coordinator that selects and manages providers

The embedding system automatically handles provider selection based on configuration,
supports both synchronous and asynchronous operations, and provides comprehensive
error handling and logging.
"""

from .manager import EmbeddingManager

__all__ = ["EmbeddingManager"]