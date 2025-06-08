"""Statistics operations handler for RAG database."""

from typing import Any, Dict

from ...config.settings import Settings
from ...core.exceptions import RAGError


class StatsOperations:
    """Handles statistics and collection analysis operations."""

    def __init__(self, collection, settings: Settings, logger):
        self.collection = collection
        self.settings = settings
        self.logger = logger

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            # Get basic collection stats
            count = self.collection.count()
            
            if count == 0:
                return {
                    "total_documents": 0,
                    "total_embeddings": 0,
                    "embedding_dimension": None,
                    "embedding_models": [],
                    "total_content_length": 0,
                    "average_content_length": 0,
                    "total_word_count": 0,
                    "collection_size_bytes": 0,
                    "metadata_keys": [],
                }

            # Get all documents for detailed stats
            all_docs = self.collection.get(
                include=["documents", "metadatas"],
                limit=count
            )

            # Calculate statistics
            total_content_length = sum(len(doc) for doc in all_docs["documents"])
            total_word_count = sum(len(doc.split()) for doc in all_docs["documents"])
            
            # Collect metadata keys and embedding models
            metadata_keys = set()
            embedding_models = set()
            
            for metadata in all_docs["metadatas"]:
                if metadata:
                    metadata_keys.update(metadata.keys())
                    if "embedding_model" in metadata and metadata["embedding_model"]:
                        embedding_models.add(metadata["embedding_model"])

            # Get embedding dimension from first document with dimension info
            embedding_dimension = None
            for metadata in all_docs["metadatas"]:
                if metadata and "embedding_dimension" in metadata:
                    embedding_dimension = metadata["embedding_dimension"]
                    break

            return {
                "total_documents": count,
                "total_embeddings": count,  # Assuming all docs have embeddings
                "embedding_dimension": embedding_dimension,
                "embedding_models": list(embedding_models),
                "total_content_length": total_content_length,
                "average_content_length": total_content_length / count if count > 0 else 0,
                "total_word_count": total_word_count,
                "collection_size_bytes": None,  # ChromaDB doesn't provide this easily
                "metadata_keys": list(metadata_keys),
            }

        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            raise RAGError(f"Failed to get collection stats: {e}")