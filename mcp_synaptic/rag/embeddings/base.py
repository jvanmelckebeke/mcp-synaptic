"""Abstract base classes for embedding providers."""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from ...config.logging import LoggerMixin
from ...config.settings import Settings
from ...core.exceptions import EmbeddingError


class EmbeddingProvider(ABC, LoggerMixin):
    """Abstract base class for embedding providers."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding provider."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the embedding provider and clean up resources."""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of this provider."""
        pass
    
    def _ensure_initialized(self) -> None:
        """Ensure the provider is initialized."""
        if not self._initialized:
            raise EmbeddingError(f"{self.provider_name} provider not initialized")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        self._ensure_initialized()
        
        return {
            "model_name": self.settings.EMBEDDING_MODEL,
            "provider": self.provider_name,
            "dimension": self.get_embedding_dimension(),
        }