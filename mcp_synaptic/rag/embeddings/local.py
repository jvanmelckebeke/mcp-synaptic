"""Local sentence-transformers embedding provider implementation."""

import asyncio
from typing import List, Optional, Dict

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ...core.exceptions import EmbeddingError
from .base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embedding provider."""
    
    def __init__(self, settings):
        super().__init__(settings)
        self.model: Optional[SentenceTransformer] = None
    
    @property
    def provider_name(self) -> str:
        return "local"
    
    async def initialize(self) -> None:
        """Initialize local sentence-transformers model."""
        if SentenceTransformer is None:
            raise EmbeddingError(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )

        try:
            # Load model in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.settings.EMBEDDING_MODEL)
            )

            self._initialized = True
            
            self.logger.info(
                "Local embedding provider initialized",
                model=self.settings.EMBEDDING_MODEL,
                dimensions=self.model.get_sentence_embedding_dimension()
            )

        except Exception as e:
            self.logger.error("Failed to initialize local embedding provider", error=str(e))
            raise EmbeddingError(f"Local embedding provider initialization failed: {e}")
    
    async def close(self) -> None:
        """Close the local embedding provider."""
        self.model = None
        self._initialized = False
        self.logger.info("Local embedding provider closed")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        self._ensure_initialized()
        
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid texts to embed")

        try:
            embeddings = await self._local_embed_texts(valid_texts)
            
            self.logger.debug(
                "Texts embedded locally",
                count=len(valid_texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings

        except Exception as e:
            self.logger.error("Failed to embed texts locally", count=len(texts), error=str(e))
            raise EmbeddingError(f"Failed to embed texts locally: {e}")
    
    async def _local_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model."""
        if not self.model:
            raise EmbeddingError("Local model not initialized")

        # Generate embeddings in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: [emb.tolist() for emb in self.model.encode(texts, convert_to_tensor=False)]
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the local model."""
        self._ensure_initialized()
        
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            self.logger.error("Failed to get local embedding dimension", error=str(e))
            raise EmbeddingError(f"Failed to get embedding dimension: {e}")
    
    def get_model_info(self) -> Dict:
        """Get local provider model information."""
        info = super().get_model_info()
        if self.model:
            info["max_sequence_length"] = getattr(self.model, 'max_seq_length', 'unknown')
        return info