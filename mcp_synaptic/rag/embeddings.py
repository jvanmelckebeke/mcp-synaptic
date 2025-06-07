"""Embedding generation and management for RAG."""

import asyncio
import aiohttp
import numpy as np
from typing import List, Optional, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..config.logging import LoggerMixin
from ..config.settings import Settings
from ..core.exceptions import EmbeddingError


class EmbeddingManager(LoggerMixin):
    """Manages text embeddings for RAG functionality."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self._initialized = False
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize the embedding manager."""
        if self.settings.EMBEDDING_PROVIDER == "api":
            await self._initialize_api()
        else:
            await self._initialize_local()

    async def _initialize_api(self) -> None:
        """Initialize API-based embedding provider."""
        if not self.settings.EMBEDDING_API_BASE:
            raise EmbeddingError("EMBEDDING_API_BASE required for API provider")

        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(timeout=timeout)

        # Test API connection
        try:
            await self._test_api_connection()
            self._initialized = True
            
            self.logger.info(
                "API embedding manager initialized",
                api_base=self.settings.EMBEDDING_API_BASE,
                model=self.settings.EMBEDDING_MODEL
            )
        except Exception as e:
            if self._session:
                await self._session.close()
                self._session = None
            raise EmbeddingError(f"API embedding manager initialization failed: {e}")

    async def _initialize_local(self) -> None:
        """Initialize local sentence-transformers model."""
        if SentenceTransformer is None:
            raise EmbeddingError("sentence-transformers not available. Install with: pip install sentence-transformers")

        try:
            # Load model in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.settings.EMBEDDING_MODEL)
            )

            self._initialized = True
            
            self.logger.info(
                "Local embedding manager initialized",
                model=self.settings.EMBEDDING_MODEL,
                dimensions=self.model.get_sentence_embedding_dimension()
            )

        except Exception as e:
            self.logger.error("Failed to initialize local embedding manager", error=str(e))
            raise EmbeddingError(f"Local embedding manager initialization failed: {e}")

    async def _test_api_connection(self) -> None:
        """Test API connection with a simple embedding request."""
        try:
            await self._api_embed_texts(["test"])
        except Exception as e:
            raise EmbeddingError(f"API connection test failed: {e}")

    async def close(self) -> None:
        """Close the embedding manager."""
        if self._session:
            await self._session.close()
            self._session = None
        
        self.model = None
        self._initialized = False
        self.logger.info("Embedding manager closed")

    def _ensure_initialized(self) -> None:
        """Ensure the embedding manager is initialized."""
        if not self._initialized:
            raise EmbeddingError("Embedding manager not initialized")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._ensure_initialized()

        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._ensure_initialized()

        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid texts to embed")

        try:
            if self.settings.EMBEDDING_PROVIDER == "api":
                embeddings = await self._api_embed_texts(valid_texts)
            else:
                embeddings = await self._local_embed_texts(valid_texts)

            self.logger.debug(
                "Texts embedded",
                count=len(valid_texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings

        except Exception as e:
            self.logger.error("Failed to embed texts", count=len(texts), error=str(e))
            raise EmbeddingError(f"Failed to embed texts: {e}")

    async def _api_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using API."""
        if not self._session:
            raise EmbeddingError("HTTP session not initialized")

        headers = {
            "Content-Type": "application/json"
        }
        
        if self.settings.EMBEDDING_API_KEY:
            headers["Authorization"] = f"Bearer {self.settings.EMBEDDING_API_KEY}"

        payload = {
            "model": self.settings.EMBEDDING_MODEL,
            "input": texts
        }

        # Try OpenAI-compatible endpoint first
        url = f"{self.settings.EMBEDDING_API_BASE.rstrip('/')}/v1/embeddings"
        
        try:
            async with self._session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return [item["embedding"] for item in data["data"]]
                else:
                    error_text = await response.text()
                    raise EmbeddingError(f"API request failed: {response.status} - {error_text}")
        except aiohttp.ClientError as e:
            raise EmbeddingError(f"API request error: {e}")

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
        
        if self.settings.EMBEDDING_PROVIDER == "api":
            # Common dimensions for popular models
            model_dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
            }
            return model_dims.get(self.settings.EMBEDDING_MODEL, 1536)  # Default to 1536
        else:
            try:
                return self.model.get_sentence_embedding_dimension()
            except Exception as e:
                self.logger.error("Failed to get embedding dimension", error=str(e))
                raise EmbeddingError(f"Failed to get embedding dimension: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        self._ensure_initialized()
        
        info = {
            "model_name": self.settings.EMBEDDING_MODEL,
            "provider": self.settings.EMBEDDING_PROVIDER,
            "dimension": self.get_embedding_dimension(),
        }
        
        if self.settings.EMBEDDING_PROVIDER == "api":
            info["api_base"] = self.settings.EMBEDDING_API_BASE
        else:
            info["max_sequence_length"] = getattr(self.model, 'max_seq_length', 'unknown')
            
        return info