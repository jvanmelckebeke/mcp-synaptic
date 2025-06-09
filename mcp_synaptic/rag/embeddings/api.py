"""API-based embedding provider implementation."""

import aiohttp
from typing import List, Optional, Dict

from ...core.exceptions import EmbeddingError
from .base import EmbeddingProvider


class ApiEmbeddingProvider(EmbeddingProvider):
    """API-based embedding provider for external services."""
    
    def __init__(self, settings):
        super().__init__(settings)
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def provider_name(self) -> str:
        return "api"
    
    async def initialize(self) -> None:
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
                "API embedding provider initialized",
                api_base=self.settings.EMBEDDING_API_BASE,
                model=self.settings.EMBEDDING_MODEL
            )
        except Exception as e:
            if self._session:
                await self._session.close()
                self._session = None
            raise EmbeddingError(f"API embedding provider initialization failed: {e}")
    
    async def close(self) -> None:
        """Close the API embedding provider."""
        if self._session:
            await self._session.close()
            self._session = None
        
        self._initialized = False
        self.logger.info("API embedding provider closed")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using API."""
        self._ensure_initialized()
        
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid texts to embed")

        try:
            embeddings = await self._api_embed_texts(valid_texts)
            
            self.logger.debug(
                "Texts embedded via API",
                count=len(valid_texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0
            )
            
            return embeddings

        except Exception as e:
            self.logger.error("Failed to embed texts via API", count=len(texts), error=str(e))
            raise EmbeddingError(f"Failed to embed texts via API: {e}")
    
    async def _test_api_connection(self) -> None:
        """Test API connection with a simple embedding request."""
        try:
            await self._api_embed_texts(["test"])
        except Exception as e:
            raise EmbeddingError(f"API connection test failed: {e}")
    
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

        # Try OpenAI-compatible endpoint
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
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the API model."""
        self._ensure_initialized()
        
        # Common dimensions for popular API models
        model_dims = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
        }
        return model_dims.get(self.settings.EMBEDDING_MODEL, 1536)  # Default to 1536
    
    def get_model_info(self) -> Dict:
        """Get API provider model information."""
        info = super().get_model_info()
        info["api_base"] = self.settings.EMBEDDING_API_BASE
        return info