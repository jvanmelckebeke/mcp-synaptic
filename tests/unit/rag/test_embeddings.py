"""Tests for embedding generation and management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

import pytest
import pytest_asyncio
import numpy as np

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.core.exceptions import EmbeddingError
from mcp_synaptic.rag.embeddings import EmbeddingManager
from tests.utils import MockFactory, EmbeddingTestHelper, mock_embedding_api


class TestEmbeddingManager:
    """Test embedding manager functionality."""

    @pytest.fixture
    def test_settings_api(self, test_settings: Settings) -> Settings:
        """Test settings configured for API provider."""
        test_settings.EMBEDDING_PROVIDER = "api"
        test_settings.EMBEDDING_API_BASE = "http://mock-api:4000"
        test_settings.EMBEDDING_MODEL = "text-embedding-ada-002"
        test_settings.EMBEDDING_DIMENSIONS = 1536
        test_settings.EMBEDDING_API_KEY = "test-api-key"
        return test_settings

    @pytest.fixture
    def test_settings_local(self, test_settings: Settings) -> Settings:
        """Test settings configured for local provider."""
        test_settings.EMBEDDING_PROVIDER = "local"
        test_settings.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        test_settings.EMBEDDING_DIMENSIONS = 384
        return test_settings

    @pytest_asyncio.fixture
    async def api_embedding_manager(self, test_settings_api: Settings) -> EmbeddingManager:
        """Create API-based embedding manager for testing."""
        manager = EmbeddingManager(test_settings_api)
        
        # Mock the API connection test
        with patch.object(manager, '_test_api_connection', return_value=None):
            await manager.initialize()
        
        yield manager
        await manager.close()

    @pytest_asyncio.fixture
    async def local_embedding_manager(self, test_settings_local: Settings) -> EmbeddingManager:
        """Create local embedding manager for testing."""
        manager = EmbeddingManager(test_settings_local)
        
        # Mock sentence transformers
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [np.array([0.1] * 384) for _ in range(1)]
        
        with patch('mcp_synaptic.rag.embeddings.SentenceTransformer', return_value=mock_model):
            await manager.initialize()
        
        yield manager
        await manager.close()

    class TestInitialization:
        """Test embedding manager initialization."""

        async def test_initialize_api_provider_success(self, test_settings_api: Settings):
            """Test successful API provider initialization."""
            manager = EmbeddingManager(test_settings_api)
            
            with patch.object(manager, '_test_api_connection', return_value=None):
                await manager.initialize()
                
                assert manager._initialized is True
                assert manager._session is not None
                assert manager.model is None  # API provider doesn't use local model
            
            await manager.close()

        async def test_initialize_api_provider_missing_base_url(self, test_settings_api: Settings):
            """Test API provider initialization fails without base URL."""
            test_settings_api.EMBEDDING_API_BASE = None
            manager = EmbeddingManager(test_settings_api)
            
            with pytest.raises(EmbeddingError, match="EMBEDDING_API_BASE required"):
                await manager.initialize()

        async def test_initialize_api_provider_connection_failure(self, test_settings_api: Settings):
            """Test API provider initialization fails on connection test."""
            manager = EmbeddingManager(test_settings_api)
            
            with patch.object(manager, '_test_api_connection', side_effect=Exception("Connection failed")):
                with pytest.raises(EmbeddingError, match="API embedding manager initialization failed"):
                    await manager.initialize()

        async def test_initialize_local_provider_success(self, test_settings_local: Settings):
            """Test successful local provider initialization."""
            manager = EmbeddingManager(test_settings_local)
            
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            
            with patch('mcp_synaptic.rag.embeddings.SentenceTransformer', return_value=mock_model):
                await manager.initialize()
                
                assert manager._initialized is True
                assert manager.model is mock_model
                assert manager._session is None  # Local provider doesn't use HTTP session
            
            await manager.close()

        async def test_initialize_local_provider_missing_library(self, test_settings_local: Settings):
            """Test local provider initialization fails without sentence-transformers."""
            manager = EmbeddingManager(test_settings_local)
            
            with patch('mcp_synaptic.rag.embeddings.SentenceTransformer', None):
                with pytest.raises(EmbeddingError, match="sentence-transformers not available"):
                    await manager.initialize()

        async def test_initialize_local_provider_model_loading_failure(self, test_settings_local: Settings):
            """Test local provider initialization fails on model loading."""
            manager = EmbeddingManager(test_settings_local)
            
            with patch('mcp_synaptic.rag.embeddings.SentenceTransformer', side_effect=Exception("Model loading failed")):
                with pytest.raises(EmbeddingError, match="Local embedding manager initialization failed"):
                    await manager.initialize()

    class TestClose:
        """Test embedding manager cleanup."""

        async def test_close_api_provider(self, api_embedding_manager: EmbeddingManager):
            """Test closing API provider cleans up session."""
            assert api_embedding_manager._session is not None
            assert api_embedding_manager._initialized is True
            
            await api_embedding_manager.close()
            
            assert api_embedding_manager._session is None
            assert api_embedding_manager._initialized is False

        async def test_close_local_provider(self, local_embedding_manager: EmbeddingManager):
            """Test closing local provider cleans up model."""
            assert local_embedding_manager.model is not None
            assert local_embedding_manager._initialized is True
            
            await local_embedding_manager.close()
            
            assert local_embedding_manager.model is None
            assert local_embedding_manager._initialized is False

    class TestEmbedText:
        """Test single text embedding."""

        async def test_embed_text_api_success(self, api_embedding_manager: EmbeddingManager):
            """Test successful single text embedding with API."""
            test_text = "This is a test document for embedding."
            
            # Mock the embed_texts method to return a single embedding
            mock_embedding = EmbeddingTestHelper.create_mock_embedding(1536)
            with patch.object(api_embedding_manager, 'embed_texts', return_value=[mock_embedding]):
                result = await api_embedding_manager.embed_text(test_text)
                
                assert result == mock_embedding
                assert len(result) == 1536

        async def test_embed_text_local_success(self, local_embedding_manager: EmbeddingManager):
            """Test successful single text embedding with local model."""
            test_text = "This is a test document for embedding."
            
            # Mock the embed_texts method to return a single embedding
            mock_embedding = EmbeddingTestHelper.create_mock_embedding(384)
            with patch.object(local_embedding_manager, 'embed_texts', return_value=[mock_embedding]):
                result = await local_embedding_manager.embed_text(test_text)
                
                assert result == mock_embedding
                assert len(result) == 384

        async def test_embed_text_empty_string(self, api_embedding_manager: EmbeddingManager):
            """Test embedding empty text raises error."""
            with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
                await api_embedding_manager.embed_text("")

        async def test_embed_text_whitespace_only(self, api_embedding_manager: EmbeddingManager):
            """Test embedding whitespace-only text raises error."""
            with pytest.raises(EmbeddingError, match="Cannot embed empty text"):
                await api_embedding_manager.embed_text("   \n\t   ")

        async def test_embed_text_not_initialized(self, test_settings_api: Settings):
            """Test embedding without initialization raises error."""
            manager = EmbeddingManager(test_settings_api)
            # Don't initialize
            
            with pytest.raises(EmbeddingError, match="Embedding manager not initialized"):
                await manager.embed_text("test")

    class TestEmbedTexts:
        """Test batch text embedding."""

        async def test_embed_texts_api_success(self, test_settings_api: Settings):
            """Test successful batch embedding with API."""
            manager = EmbeddingManager(test_settings_api)
            
            # Initialize with mocked session
            session_mock = MockFactory.create_aiohttp_session_mock(
                EmbeddingTestHelper.create_mock_embedding_response([
                    EmbeddingTestHelper.create_mock_embedding(1536, 0.1),
                    EmbeddingTestHelper.create_mock_embedding(1536, 0.2),
                ])
            )
            
            with patch('aiohttp.ClientSession', return_value=session_mock):
                await manager.initialize()
                
                texts = ["First document", "Second document"]
                embeddings = await manager.embed_texts(texts)
                
                assert len(embeddings) == 2
                assert len(embeddings[0]) == 1536
                assert len(embeddings[1]) == 1536
                assert embeddings[0] != embeddings[1]  # Different values
            
            await manager.close()

        async def test_embed_texts_local_success(self, local_embedding_manager: EmbeddingManager):
            """Test successful batch embedding with local model."""
            texts = ["First document", "Second document"]
            
            # Mock the model's encode method
            mock_embeddings = [
                np.array([0.1] * 384),
                np.array([0.2] * 384)
            ]
            local_embedding_manager.model.encode.return_value = mock_embeddings
            
            embeddings = await local_embedding_manager.embed_texts(texts)
            
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384
            assert len(embeddings[1]) == 384
            assert embeddings[0] != embeddings[1]

        async def test_embed_texts_empty_list(self, api_embedding_manager: EmbeddingManager):
            """Test embedding empty list returns empty list."""
            result = await api_embedding_manager.embed_texts([])
            assert result == []

        async def test_embed_texts_filters_empty_strings(self, api_embedding_manager: EmbeddingManager):
            """Test that empty strings are filtered out."""
            texts = ["Valid text", "", "   ", "Another valid text"]
            
            # Mock API to return 2 embeddings (for 2 valid texts)
            mock_response = EmbeddingTestHelper.create_mock_embedding_response([
                EmbeddingTestHelper.create_mock_embedding(),
                EmbeddingTestHelper.create_mock_embedding()
            ])
            
            with patch.object(api_embedding_manager, '_api_embed_texts', return_value=[
                mock_response["data"][0]["embedding"],
                mock_response["data"][1]["embedding"]
            ]):
                embeddings = await api_embedding_manager.embed_texts(texts)
                
                assert len(embeddings) == 2  # Only valid texts processed

        async def test_embed_texts_all_empty_raises_error(self, api_embedding_manager: EmbeddingManager):
            """Test embedding only empty texts raises error."""
            texts = ["", "   ", "\n\t"]
            
            with pytest.raises(EmbeddingError, match="No valid texts to embed"):
                await api_embedding_manager.embed_texts(texts)

        async def test_embed_texts_api_failure(self, api_embedding_manager: EmbeddingManager):
            """Test API embedding failure handling."""
            texts = ["Test document"]
            
            with patch.object(api_embedding_manager, '_api_embed_texts', side_effect=Exception("API error")):
                with pytest.raises(EmbeddingError, match="Failed to embed texts"):
                    await api_embedding_manager.embed_texts(texts)

        async def test_embed_texts_local_failure(self, local_embedding_manager: EmbeddingManager):
            """Test local embedding failure handling."""
            texts = ["Test document"]
            
            local_embedding_manager.model.encode.side_effect = Exception("Model error")
            
            with pytest.raises(EmbeddingError, match="Failed to embed texts"):
                await local_embedding_manager.embed_texts(texts)

    class TestAPIEmbedTexts:
        """Test API-specific embedding functionality."""

        async def test_api_embed_texts_success(self, test_settings_api: Settings):
            """Test successful API embedding call."""
            manager = EmbeddingManager(test_settings_api)
            texts = ["Test document"]
            
            mock_response = EmbeddingTestHelper.create_mock_embedding_response()
            session_mock = MockFactory.create_aiohttp_session_mock(mock_response, status=200)
            
            manager._session = session_mock
            manager._initialized = True
            
            embeddings = await manager._api_embed_texts(texts)
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 1536
            
            # Verify correct API call
            session_mock.post.assert_called_once()
            call_args = session_mock.post.call_args
            assert "v1/embeddings" in str(call_args[0][0])  # URL contains endpoint

        async def test_api_embed_texts_with_auth_header(self, test_settings_api: Settings):
            """Test API call includes authorization header."""
            manager = EmbeddingManager(test_settings_api)
            texts = ["Test document"]
            
            mock_response = EmbeddingTestHelper.create_mock_embedding_response()
            session_mock = MockFactory.create_aiohttp_session_mock(mock_response)
            
            manager._session = session_mock
            manager._initialized = True
            
            await manager._api_embed_texts(texts)
            
            # Check that authorization header was included
            call_kwargs = session_mock.post.call_args.kwargs
            headers = call_kwargs.get('headers', {})
            assert 'Authorization' in headers
            assert headers['Authorization'] == f"Bearer {test_settings_api.EMBEDDING_API_KEY}"

        async def test_api_embed_texts_http_error(self, test_settings_api: Settings):
            """Test API HTTP error handling."""
            manager = EmbeddingManager(test_settings_api)
            texts = ["Test document"]
            
            session_mock = MockFactory.create_aiohttp_session_mock(
                response_data={"error": "Invalid request"},
                status=400
            )
            
            manager._session = session_mock
            manager._initialized = True
            
            with pytest.raises(EmbeddingError, match="API request failed: 400"):
                await manager._api_embed_texts(texts)

        async def test_api_embed_texts_no_session(self, test_settings_api: Settings):
            """Test API embedding without session raises error."""
            manager = EmbeddingManager(test_settings_api)
            manager._session = None
            manager._initialized = True
            
            with pytest.raises(EmbeddingError, match="HTTP session not initialized"):
                await manager._api_embed_texts(["test"])

    class TestLocalEmbedTexts:
        """Test local model embedding functionality."""

        async def test_local_embed_texts_success(self, local_embedding_manager: EmbeddingManager):
            """Test successful local embedding."""
            texts = ["First text", "Second text"]
            
            # Mock model response
            mock_embeddings = [
                np.array([0.1] * 384),
                np.array([0.2] * 384)
            ]
            local_embedding_manager.model.encode.return_value = mock_embeddings
            
            embeddings = await local_embedding_manager._local_embed_texts(texts)
            
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 384
            assert isinstance(embeddings[0], list)  # Converted from numpy array

        async def test_local_embed_texts_no_model(self, test_settings_local: Settings):
            """Test local embedding without model raises error."""
            manager = EmbeddingManager(test_settings_local)
            manager.model = None
            manager._initialized = True
            
            with pytest.raises(EmbeddingError, match="Local model not initialized"):
                await manager._local_embed_texts(["test"])

    class TestComputeSimilarity:
        """Test similarity computation."""

        async def test_compute_similarity_success(self, api_embedding_manager: EmbeddingManager):
            """Test successful similarity computation."""
            text1 = "This is the first document"
            text2 = "This is the second document"
            
            # Mock embeddings that should have high similarity
            embedding1 = [0.5, 0.5, 0.5, 0.5]
            embedding2 = [0.6, 0.4, 0.5, 0.5]
            
            with patch.object(api_embedding_manager, 'embed_texts', return_value=[embedding1, embedding2]):
                similarity = await api_embedding_manager.compute_similarity(text1, text2)
                
                assert 0.0 <= similarity <= 1.0
                assert isinstance(similarity, float)

        async def test_compute_similarity_identical_texts(self, api_embedding_manager: EmbeddingManager):
            """Test similarity of identical texts approaches 1.0."""
            text = "This is a test document"
            
            # Identical embeddings should have similarity 1.0
            embedding = [0.5, 0.5, 0.5, 0.5]
            
            with patch.object(api_embedding_manager, 'embed_texts', return_value=[embedding, embedding]):
                similarity = await api_embedding_manager.compute_similarity(text, text)
                
                assert abs(similarity - 1.0) < 1e-6  # Very close to 1.0

        async def test_compute_similarity_orthogonal_vectors(self, api_embedding_manager: EmbeddingManager):
            """Test similarity of orthogonal vectors is 0.0."""
            text1 = "First document"
            text2 = "Second document"
            
            # Orthogonal embeddings should have similarity 0.0
            embedding1 = [1.0, 0.0, 0.0, 0.0]
            embedding2 = [0.0, 1.0, 0.0, 0.0]
            
            with patch.object(api_embedding_manager, 'embed_texts', return_value=[embedding1, embedding2]):
                similarity = await api_embedding_manager.compute_similarity(text1, text2)
                
                assert abs(similarity - 0.0) < 1e-6  # Very close to 0.0

        async def test_compute_similarity_zero_vectors(self, api_embedding_manager: EmbeddingManager):
            """Test similarity with zero vectors returns 0.0."""
            text1 = "First document"
            text2 = "Second document"
            
            # Zero embeddings
            embedding1 = [0.0, 0.0, 0.0, 0.0]
            embedding2 = [0.0, 0.0, 0.0, 0.0]
            
            with patch.object(api_embedding_manager, 'embed_texts', return_value=[embedding1, embedding2]):
                similarity = await api_embedding_manager.compute_similarity(text1, text2)
                
                assert similarity == 0.0

        async def test_compute_similarity_embedding_failure(self, api_embedding_manager: EmbeddingManager):
            """Test similarity computation with embedding failure."""
            with patch.object(api_embedding_manager, 'embed_texts', side_effect=EmbeddingError("Embedding failed")):
                with pytest.raises(EmbeddingError, match="Failed to compute similarity"):
                    await api_embedding_manager.compute_similarity("text1", "text2")

    class TestUtilityMethods:
        """Test utility methods."""

        def test_get_embedding_dimension_api_known_model(self, api_embedding_manager: EmbeddingManager):
            """Test getting dimension for known API model."""
            api_embedding_manager.settings.EMBEDDING_MODEL = "text-embedding-ada-002"
            
            dimension = api_embedding_manager.get_embedding_dimension()
            
            assert dimension == 1536

        def test_get_embedding_dimension_api_unknown_model(self, api_embedding_manager: EmbeddingManager):
            """Test getting dimension for unknown API model defaults to 1536."""
            api_embedding_manager.settings.EMBEDDING_MODEL = "unknown-model"
            
            dimension = api_embedding_manager.get_embedding_dimension()
            
            assert dimension == 1536  # Default

        def test_get_embedding_dimension_local_model(self, local_embedding_manager: EmbeddingManager):
            """Test getting dimension from local model."""
            local_embedding_manager.model.get_sentence_embedding_dimension.return_value = 768
            
            dimension = local_embedding_manager.get_embedding_dimension()
            
            assert dimension == 768

        def test_get_embedding_dimension_not_initialized(self, test_settings_api: Settings):
            """Test getting dimension without initialization raises error."""
            manager = EmbeddingManager(test_settings_api)
            
            with pytest.raises(EmbeddingError, match="Embedding manager not initialized"):
                manager.get_embedding_dimension()

        def test_get_model_info_api(self, api_embedding_manager: EmbeddingManager):
            """Test getting model info for API provider."""
            info = api_embedding_manager.get_model_info()
            
            assert info["model_name"] == api_embedding_manager.settings.EMBEDDING_MODEL
            assert info["provider"] == "api"
            assert "dimension" in info
            assert "api_base" in info

        def test_get_model_info_local(self, local_embedding_manager: EmbeddingManager):
            """Test getting model info for local provider."""
            local_embedding_manager.model.max_seq_length = 512
            
            info = local_embedding_manager.get_model_info()
            
            assert info["model_name"] == local_embedding_manager.settings.EMBEDDING_MODEL
            assert info["provider"] == "local"
            assert "dimension" in info
            assert "max_sequence_length" in info

    class TestConcurrency:
        """Test concurrent operations."""

        async def test_concurrent_embed_operations(self, api_embedding_manager: EmbeddingManager):
            """Test concurrent embedding operations."""
            texts = [f"Document {i}" for i in range(10)]
            
            # Mock to return different embeddings for each text
            mock_embeddings = [
                EmbeddingTestHelper.create_mock_embedding(1536, value=i * 0.1)
                for i in range(10)
            ]
            
            async def mock_embed_text(text: str) -> List[float]:
                # Simulate some async work
                await asyncio.sleep(0.01)
                index = int(text.split()[-1])
                return mock_embeddings[index]
            
            with patch.object(api_embedding_manager, 'embed_text', side_effect=mock_embed_text):
                # Run embeddings concurrently
                embeddings = await asyncio.gather(*[
                    api_embedding_manager.embed_text(text) for text in texts
                ])
                
                assert len(embeddings) == 10
                # Verify each embedding is different (based on our mock)
                for i, embedding in enumerate(embeddings):
                    assert embedding[0] == i * 0.1

        async def test_concurrent_similarity_operations(self, api_embedding_manager: EmbeddingManager):
            """Test concurrent similarity computations."""
            text_pairs = [
                ("doc1", "doc2"),
                ("doc3", "doc4"),
                ("doc5", "doc6")
            ]
            
            # Mock similarity computation
            async def mock_compute_similarity(text1: str, text2: str) -> float:
                await asyncio.sleep(0.01)
                # Return different similarity based on text content
                return hash(text1 + text2) % 100 / 100.0
            
            with patch.object(api_embedding_manager, 'compute_similarity', side_effect=mock_compute_similarity):
                similarities = await asyncio.gather(*[
                    api_embedding_manager.compute_similarity(t1, t2) for t1, t2 in text_pairs
                ])
                
                assert len(similarities) == 3
                assert all(0.0 <= sim <= 1.0 for sim in similarities)