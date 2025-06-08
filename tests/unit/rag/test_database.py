"""Tests for RAG database implementation."""

import asyncio
from datetime import datetime, UTC
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from mcp_synaptic.config.settings import Settings
from mcp_synaptic.core.exceptions import RAGError, DocumentNotFoundError
from mcp_synaptic.models.rag import Document, DocumentSearchResult
from mcp_synaptic.rag.database import RAGDatabase
from mcp_synaptic.rag.embeddings import EmbeddingManager
from tests.utils import EmbeddingTestHelper, MockFactory, mock_chromadb_client


class TestRAGDatabase:
    """Test RAG database functionality."""

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager."""
        manager = AsyncMock(spec=EmbeddingManager)
        manager.model_name = "test-model"
        manager.dimension = 1536
        manager.embed_text.return_value = EmbeddingTestHelper.create_mock_embedding(1536)
        manager.embed_texts.return_value = [EmbeddingTestHelper.create_mock_embedding(1536)]
        return manager

    @pytest.fixture
    def mock_collection(self):
        """Create a mock ChromaDB collection."""
        collection = MagicMock()
        collection.count.return_value = 0
        collection.add.return_value = None
        collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "embeddings": []
        }
        collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        collection.delete.return_value = None
        return collection

    @pytest.fixture
    def mock_chromadb_client_with_collection(self, mock_collection):
        """Create a mock ChromaDB client with collection."""
        client = MagicMock()
        client.get_or_create_collection.return_value = mock_collection
        return client, mock_collection

    @pytest_asyncio.fixture
    async def rag_database(self, test_settings: Settings, mock_chromadb_client_with_collection, mock_embedding_manager):
        """Create RAG database with mocked dependencies."""
        client, collection = mock_chromadb_client_with_collection
        
        database = RAGDatabase(test_settings)
        
        # Mock ChromaDB initialization
        with patch('chromadb.PersistentClient', return_value=client):
            with patch('mcp_synaptic.rag.database.EmbeddingManager', return_value=mock_embedding_manager):
                await database.initialize()
        
        # Set mocked components
        database.embedding_manager = mock_embedding_manager
        database.client = client
        database.collection = collection
        
        yield database
        await database.close()

    class TestInitialization:
        """Test RAG database initialization."""

        async def test_initialize_success(self, test_settings: Settings, mock_chromadb_client_with_collection, mock_embedding_manager):
            """Test successful RAG database initialization."""
            client, collection = mock_chromadb_client_with_collection
            collection.count.return_value = 5
            
            database = RAGDatabase(test_settings)
            
            with patch('chromadb.PersistentClient', return_value=client):
                with patch('mcp_synaptic.rag.database.EmbeddingManager', return_value=mock_embedding_manager):
                    await database.initialize()
            
            assert database._initialized is True
            assert database.client is client
            assert database.collection is collection
            assert database.embedding_manager is mock_embedding_manager
            
            # Verify collection was created/retrieved
            client.get_or_create_collection.assert_called_once_with(
                name="synaptic_documents",
                metadata={"description": "MCP Synaptic document collection"}
            )
            
            # Verify embedding manager was initialized
            mock_embedding_manager.initialize.assert_called_once()

        async def test_initialize_chromadb_not_available(self, test_settings: Settings):
            """Test initialization fails when ChromaDB is not available."""
            database = RAGDatabase(test_settings)
            
            with patch('mcp_synaptic.rag.database.chromadb', None):
                with pytest.raises(RAGError, match="ChromaDB not available"):
                    await database.initialize()

        async def test_initialize_embedding_manager_failure(self, test_settings: Settings, mock_chromadb_client_with_collection):
            """Test initialization fails when embedding manager fails."""
            client, collection = mock_chromadb_client_with_collection
            
            database = RAGDatabase(test_settings)
            
            # Mock embedding manager that fails to initialize
            mock_manager = AsyncMock()
            mock_manager.initialize.side_effect = Exception("Embedding init failed")
            
            with patch('chromadb.PersistentClient', return_value=client):
                with patch('mcp_synaptic.rag.database.EmbeddingManager', return_value=mock_manager):
                    with pytest.raises(RAGError, match="RAG database initialization failed"):
                        await database.initialize()

        async def test_initialize_chromadb_failure(self, test_settings: Settings):
            """Test initialization fails when ChromaDB client creation fails."""
            database = RAGDatabase(test_settings)
            
            with patch('chromadb.PersistentClient', side_effect=Exception("ChromaDB init failed")):
                with pytest.raises(RAGError, match="RAG database initialization failed"):
                    await database.initialize()

    class TestClose:
        """Test RAG database cleanup."""

        async def test_close_success(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test successful database close."""
            assert rag_database._initialized is True
            
            await rag_database.close()
            
            assert rag_database._initialized is False
            mock_embedding_manager.close.assert_called_once()

    class TestAddDocument:
        """Test document addition."""

        async def test_add_document_success(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test successful document addition."""
            content = "This is a test document for the RAG database."
            metadata = {"source": "test", "category": "example"}
            
            # Mock embedding generation
            mock_embedding = EmbeddingTestHelper.create_mock_embedding(1536)
            mock_embedding_manager.embed_text.return_value = mock_embedding
            
            document = await rag_database.add_document(content, metadata)
            
            # Verify document creation
            assert document.content == content
            assert document.metadata == metadata
            assert document.id is not None
            assert isinstance(document.created_at, datetime)
            
            # Verify embedding was generated
            mock_embedding_manager.embed_text.assert_called_once_with(content)
            
            # Verify document was added to ChromaDB
            rag_database.collection.add.assert_called_once()
            call_args = rag_database.collection.add.call_args
            
            assert call_args.kwargs["ids"] == [document.id]
            assert call_args.kwargs["documents"] == [content]
            assert call_args.kwargs["embeddings"] == [mock_embedding]
            
            # Check metadata includes system fields
            stored_metadata = call_args.kwargs["metadatas"][0]
            assert stored_metadata["source"] == "test"
            assert stored_metadata["category"] == "example"
            assert "created_at" in stored_metadata
            assert "updated_at" in stored_metadata

        async def test_add_document_with_custom_id(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test adding document with custom ID."""
            content = "Test document"
            custom_id = "custom-doc-123"
            
            document = await rag_database.add_document(content, document_id=custom_id)
            
            assert document.id == custom_id

        async def test_add_document_no_embedding_manager(self, rag_database: RAGDatabase):
            """Test adding document without embedding manager."""
            rag_database.embedding_manager = None
            content = "Test document"
            
            document = await rag_database.add_document(content)
            
            # Should still create document, but without embedding
            assert document.content == content
            assert document.embedding_model is None
            assert document.embedding_dimension is None

        async def test_add_document_embedding_failure(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test document addition with embedding failure."""
            mock_embedding_manager.embed_text.side_effect = Exception("Embedding failed")
            
            with pytest.raises(RAGError, match="Failed to add document"):
                await rag_database.add_document("Test content")

        async def test_add_document_not_initialized(self, test_settings: Settings):
            """Test adding document without initialization."""
            database = RAGDatabase(test_settings)
            # Don't initialize
            
            with pytest.raises(RAGError, match="RAG database not initialized"):
                await database.add_document("Test content")

    class TestGetDocument:
        """Test document retrieval."""

        async def test_get_document_success(self, rag_database: RAGDatabase):
            """Test successful document retrieval."""
            doc_id = "test-doc-123"
            content = "This is a test document."
            metadata = {"source": "test"}
            created_at = datetime.now(UTC)
            
            # Mock ChromaDB response
            rag_database.collection.get.return_value = {
                "ids": [doc_id],
                "documents": [content],
                "metadatas": [{
                    **metadata,
                    "created_at": created_at.isoformat(),
                    "updated_at": created_at.isoformat(),
                    "embedding_model": "test-model",
                    "embedding_dimension": 1536
                }],
                "embeddings": [[0.1] * 1536]
            }
            
            document = await rag_database.get_document(doc_id)
            
            assert document is not None
            assert document.id == doc_id
            assert document.content == content
            assert document.metadata == metadata
            assert document.embedding_model == "test-model"
            assert document.embedding_dimension == 1536
            
            # Verify correct ChromaDB call
            rag_database.collection.get.assert_called_once_with(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )

        async def test_get_document_not_found(self, rag_database: RAGDatabase):
            """Test retrieving non-existent document."""
            doc_id = "nonexistent-doc"
            
            # Mock empty ChromaDB response
            rag_database.collection.get.return_value = {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "embeddings": []
            }
            
            document = await rag_database.get_document(doc_id)
            
            assert document is None

        async def test_get_document_minimal_metadata(self, rag_database: RAGDatabase):
            """Test retrieving document with minimal metadata."""
            doc_id = "minimal-doc"
            content = "Minimal document"
            
            # Mock ChromaDB response with minimal metadata
            rag_database.collection.get.return_value = {
                "ids": [doc_id],
                "documents": [content],
                "metadatas": [{}],  # Empty metadata
                "embeddings": [[0.1] * 1536]
            }
            
            document = await rag_database.get_document(doc_id)
            
            assert document is not None
            assert document.id == doc_id
            assert document.content == content
            assert document.metadata == {}

        async def test_get_document_chromadb_failure(self, rag_database: RAGDatabase):
            """Test document retrieval with ChromaDB failure."""
            rag_database.collection.get.side_effect = Exception("ChromaDB error")
            
            with pytest.raises(RAGError, match="Failed to get document"):
                await rag_database.get_document("test-doc")

    class TestUpdateDocument:
        """Test document updates."""

        async def test_update_document_content_only(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test updating document content only."""
            doc_id = "test-doc"
            original_content = "Original content"
            new_content = "Updated content"
            metadata = {"source": "test"}
            created_at = datetime.now(UTC)
            
            # Mock existing document
            existing_doc = Document(
                id=doc_id,
                content=original_content,
                metadata=metadata,
                created_at=created_at,
                embedding_model="test-model",
                embedding_dimension=1536
            )
            
            with patch.object(rag_database, 'get_document', return_value=existing_doc):
                mock_embedding = EmbeddingTestHelper.create_mock_embedding(1536)
                mock_embedding_manager.embed_text.return_value = mock_embedding
                
                updated_doc = await rag_database.update_document(doc_id, content=new_content)
                
                assert updated_doc is not None
                assert updated_doc.content == new_content
                assert updated_doc.metadata == metadata  # Unchanged
                assert updated_doc.created_at == created_at  # Unchanged
                assert updated_doc.updated_at > created_at  # Should be updated
                
                # Verify new embedding was generated
                mock_embedding_manager.embed_text.assert_called_once_with(new_content)
                
                # Verify ChromaDB operations
                rag_database.collection.delete.assert_called_once_with(ids=[doc_id])
                rag_database.collection.add.assert_called_once()

        async def test_update_document_metadata_only(self, rag_database: RAGDatabase):
            """Test updating document metadata only."""
            doc_id = "test-doc"
            content = "Document content"
            original_metadata = {"source": "test"}
            new_metadata = {"source": "updated", "category": "new"}
            
            existing_doc = Document(
                id=doc_id,
                content=content,
                metadata=original_metadata,
                embedding_model="test-model"
            )
            
            with patch.object(rag_database, 'get_document', return_value=existing_doc):
                updated_doc = await rag_database.update_document(doc_id, metadata=new_metadata)
                
                assert updated_doc is not None
                assert updated_doc.content == content  # Unchanged
                assert updated_doc.metadata == new_metadata
                
                # Should not generate new embedding for metadata-only update
                rag_database.embedding_manager.embed_text.assert_not_called()

        async def test_update_document_not_found(self, rag_database: RAGDatabase):
            """Test updating non-existent document."""
            with patch.object(rag_database, 'get_document', return_value=None):
                result = await rag_database.update_document("nonexistent", content="new content")
                
                assert result is None

        async def test_update_document_chromadb_failure(self, rag_database: RAGDatabase):
            """Test document update with ChromaDB failure."""
            existing_doc = Document(id="test", content="content", metadata={})
            
            with patch.object(rag_database, 'get_document', return_value=existing_doc):
                rag_database.collection.delete.side_effect = Exception("ChromaDB error")
                
                with pytest.raises(RAGError, match="Failed to update document"):
                    await rag_database.update_document("test", content="new content")

    class TestDeleteDocument:
        """Test document deletion."""

        async def test_delete_document_success(self, rag_database: RAGDatabase):
            """Test successful document deletion."""
            doc_id = "test-doc"
            existing_doc = Document(id=doc_id, content="content", metadata={})
            
            with patch.object(rag_database, 'get_document', return_value=existing_doc):
                deleted = await rag_database.delete_document(doc_id)
                
                assert deleted is True
                rag_database.collection.delete.assert_called_once_with(ids=[doc_id])

        async def test_delete_document_not_found(self, rag_database: RAGDatabase):
            """Test deleting non-existent document."""
            with patch.object(rag_database, 'get_document', return_value=None):
                deleted = await rag_database.delete_document("nonexistent")
                
                assert deleted is False
                # Should not call ChromaDB delete
                rag_database.collection.delete.assert_not_called()

        async def test_delete_document_chromadb_failure(self, rag_database: RAGDatabase):
            """Test document deletion with ChromaDB failure."""
            existing_doc = Document(id="test", content="content", metadata={})
            
            with patch.object(rag_database, 'get_document', return_value=existing_doc):
                rag_database.collection.delete.side_effect = Exception("ChromaDB error")
                
                with pytest.raises(RAGError, match="Failed to delete document"):
                    await rag_database.delete_document("test")

    class TestSearch:
        """Test document search."""

        async def test_search_success(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test successful document search."""
            query = "test query"
            mock_query_embedding = EmbeddingTestHelper.create_mock_embedding(1536)
            mock_embedding_manager.embed_text.return_value = mock_query_embedding
            
            # Mock ChromaDB search results
            rag_database.collection.query.return_value = {
                "ids": [["doc1", "doc2"]],
                "documents": [["First document", "Second document"]],
                "metadatas": [[
                    {"source": "test1", "created_at": datetime.now(UTC).isoformat()},
                    {"source": "test2", "created_at": datetime.now(UTC).isoformat()}
                ]],
                "distances": [[0.2, 0.4]]
            }
            
            results = await rag_database.search(query, limit=5)
            
            assert len(results) == 2
            assert all(isinstance(r, DocumentSearchResult) for r in results)
            
            # Check first result
            assert results[0].item.id == "doc1"
            assert results[0].item.content == "First document"
            assert results[0].score == pytest.approx(0.8, abs=0.01)  # 1.0 - 0.2
            assert results[0].rank == 1
            assert results[0].distance == 0.2
            
            # Check second result
            assert results[1].item.id == "doc2"
            assert results[1].score == pytest.approx(0.6, abs=0.01)  # 1.0 - 0.4
            assert results[1].rank == 2
            
            # Verify query embedding was generated
            mock_embedding_manager.embed_text.assert_called_once_with(query)
            
            # Verify ChromaDB query
            rag_database.collection.query.assert_called_once()
            call_kwargs = rag_database.collection.query.call_args.kwargs
            assert call_kwargs["query_embeddings"] == [mock_query_embedding]
            assert call_kwargs["n_results"] == 5

        async def test_search_with_similarity_threshold(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test search with similarity threshold filtering."""
            query = "test query"
            threshold = 0.7
            
            # Mock results with varying distances
            rag_database.collection.query.return_value = {
                "ids": [["doc1", "doc2", "doc3"]],
                "documents": [["Doc 1", "Doc 2", "Doc 3"]],
                "metadatas": [[{}, {}, {}]],
                "distances": [[0.1, 0.4, 0.8]]  # Similarities: 0.9, 0.6, 0.2
            }
            
            results = await rag_database.search(query, similarity_threshold=threshold)
            
            # Only first result should pass threshold (0.9 > 0.7)
            assert len(results) == 1
            assert results[0].item.id == "doc1"
            assert results[0].score > threshold

        async def test_search_with_metadata_filter(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test search with metadata filtering."""
            query = "test query"
            metadata_filter = {"category": "important"}
            
            await rag_database.search(query, metadata_filter=metadata_filter)
            
            # Verify metadata filter was passed to ChromaDB
            call_kwargs = rag_database.collection.query.call_args.kwargs
            assert call_kwargs["where"] == metadata_filter

        async def test_search_respects_max_results_setting(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test search respects MAX_RAG_RESULTS setting."""
            rag_database.settings.MAX_RAG_RESULTS = 3
            
            await rag_database.search("query", limit=10)  # Request more than max
            
            # Should be limited to MAX_RAG_RESULTS
            call_kwargs = rag_database.collection.query.call_args.kwargs
            assert call_kwargs["n_results"] == 3

        async def test_search_no_embedding_manager(self, rag_database: RAGDatabase):
            """Test search without embedding manager."""
            rag_database.embedding_manager = None
            
            with pytest.raises(RAGError, match="Embedding manager not available"):
                await rag_database.search("test query")

        async def test_search_embedding_failure(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test search with embedding generation failure."""
            mock_embedding_manager.embed_text.side_effect = Exception("Embedding failed")
            
            with pytest.raises(RAGError, match="Failed to search documents"):
                await rag_database.search("test query")

        async def test_search_chromadb_failure(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test search with ChromaDB failure."""
            rag_database.collection.query.side_effect = Exception("ChromaDB error")
            
            with pytest.raises(RAGError, match="Failed to search documents"):
                await rag_database.search("test query")

    class TestCollectionStats:
        """Test collection statistics."""

        async def test_get_collection_stats_empty(self, rag_database: RAGDatabase):
            """Test getting stats for empty collection."""
            rag_database.collection.count.return_value = 0
            
            stats = await rag_database.get_collection_stats()
            
            assert stats["total_documents"] == 0
            assert stats["total_embeddings"] == 0
            assert stats["embedding_dimension"] is None
            assert stats["embedding_models"] == []
            assert stats["total_content_length"] == 0
            assert stats["average_content_length"] == 0

        async def test_get_collection_stats_with_documents(self, rag_database: RAGDatabase):
            """Test getting stats for collection with documents."""
            # Mock collection with documents
            rag_database.collection.count.return_value = 3
            rag_database.collection.get.return_value = {
                "documents": [
                    "Short doc",
                    "This is a longer document with more words",
                    "Medium length document here"
                ],
                "metadatas": [
                    {"embedding_model": "model1", "embedding_dimension": 1536, "source": "test"},
                    {"embedding_model": "model1", "embedding_dimension": 1536, "category": "example"},
                    {"embedding_model": "model2", "embedding_dimension": 768, "type": "data"}
                ]
            }
            
            stats = await rag_database.get_collection_stats()
            
            assert stats["total_documents"] == 3
            assert stats["total_embeddings"] == 3
            assert stats["embedding_dimension"] == 1536  # From first document
            assert set(stats["embedding_models"]) == {"model1", "model2"}
            assert stats["total_content_length"] == sum(len(doc) for doc in [
                "Short doc", "This is a longer document with more words", "Medium length document here"
            ])
            assert stats["average_content_length"] > 0
            assert stats["total_word_count"] > 0
            assert set(stats["metadata_keys"]) >= {"embedding_model", "embedding_dimension", "source", "category", "type"}

        async def test_get_collection_stats_chromadb_failure(self, rag_database: RAGDatabase):
            """Test collection stats with ChromaDB failure."""
            rag_database.collection.count.side_effect = Exception("ChromaDB error")
            
            with pytest.raises(RAGError, match="Failed to get collection stats"):
                await rag_database.get_collection_stats()

    class TestConcurrency:
        """Test concurrent operations."""

        async def test_concurrent_document_operations(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test concurrent document add/get operations."""
            documents_data = [
                ("doc1", "First document content"),
                ("doc2", "Second document content"),
                ("doc3", "Third document content")
            ]
            
            # Mock embedding generation
            mock_embedding_manager.embed_text.side_effect = lambda text: [0.1] * 1536
            
            # Add documents concurrently
            add_tasks = [
                rag_database.add_document(content, document_id=doc_id)
                for doc_id, content in documents_data
            ]
            
            added_docs = await asyncio.gather(*add_tasks)
            
            assert len(added_docs) == 3
            assert all(doc.id in ["doc1", "doc2", "doc3"] for doc in added_docs)

        async def test_concurrent_search_operations(self, rag_database: RAGDatabase, mock_embedding_manager):
            """Test concurrent search operations."""
            queries = ["query1", "query2", "query3"]
            
            # Mock search results
            rag_database.collection.query.return_value = {
                "ids": [["doc1"]],
                "documents": [["Test document"]],
                "metadatas": [[{}]],
                "distances": [[0.3]]
            }
            
            # Perform searches concurrently
            search_tasks = [
                rag_database.search(query)
                for query in queries
            ]
            
            results = await asyncio.gather(*search_tasks)
            
            assert len(results) == 3
            assert all(len(result) == 1 for result in results)

    class TestEnsureInitialized:
        """Test initialization checking."""

        async def test_operations_require_initialization(self, test_settings: Settings):
            """Test that operations require initialization."""
            database = RAGDatabase(test_settings)
            # Don't initialize
            
            with pytest.raises(RAGError, match="RAG database not initialized"):
                await database.add_document("test")
            
            with pytest.raises(RAGError, match="RAG database not initialized"):
                await database.get_document("test")
            
            with pytest.raises(RAGError, match="RAG database not initialized"):
                await database.search("test")
            
            with pytest.raises(RAGError, match="RAG database not initialized"):
                await database.get_collection_stats()