"""Main embedding manager that coordinates between providers."""

import numpy as np
from typing import List, Dict, Any

from ...config.logging import LoggerMixin
from ...config.settings import Settings
from ...core.exceptions import EmbeddingError
from .base import EmbeddingProvider
from .api import ApiEmbeddingProvider
from .local import LocalEmbeddingProvider


class EmbeddingManager(LoggerMixin):
    """Manages text embeddings for RAG functionality."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider: EmbeddingProvider = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the embedding manager with appropriate provider."""
        try:
            # Select provider based on settings
            if self.settings.EMBEDDING_PROVIDER == "api":
                self.provider = ApiEmbeddingProvider(self.settings)
            else:
                self.provider = LocalEmbeddingProvider(self.settings)
            
            # Initialize the selected provider
            await self.provider.initialize()
            self._initialized = True
            
            self.logger.info(
                "Embedding manager initialized",
                provider=self.provider.provider_name,
                model=self.settings.EMBEDDING_MODEL
            )

        except Exception as e:
            self.logger.error("Failed to initialize embedding manager", error=str(e))
            raise EmbeddingError(f"Embedding manager initialization failed: {e}")

    async def close(self) -> None:
        """Close the embedding manager."""
        if self.provider:
            await self.provider.close()
            self.provider = None
        
        self._initialized = False
        self.logger.info("Embedding manager closed")

    def _ensure_initialized(self) -> None:
        """Ensure the embedding manager is initialized."""
        if not self._initialized or not self.provider:
            raise EmbeddingError("Embedding manager not initialized")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._ensure_initialized()
        return await self.provider.embed_text(text)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._ensure_initialized()
        return await self.provider.embed_texts(texts)

    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        self._ensure_initialized()

        try:
            embeddings = await self.embed_texts([text1, text2])
            
            if len(embeddings) != 2:
                raise EmbeddingError("Failed to generate embeddings for similarity computation")

            # Compute cosine similarity
            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])
            
            # Cosine similarity formula
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            self.logger.debug("Similarity computed", similarity=float(similarity))
            return float(similarity)

        except Exception as e:
            self.logger.error("Failed to compute similarity", error=str(e))
            raise EmbeddingError(f"Failed to compute similarity: {e}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        self._ensure_initialized()
        return self.provider.get_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        self._ensure_initialized()
        return self.provider.get_model_info()