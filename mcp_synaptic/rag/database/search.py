"""Search operations handler for RAG database."""

from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

from ...config.settings import Settings
from ...core.exceptions import RAGError
from ...models.rag import Document, DocumentSearchResult
from ..embeddings import EmbeddingManager


class SearchOperations:
    """Handles search operations and similarity matching."""

    def __init__(self, collection, embedding_manager: EmbeddingManager, settings: Settings, logger):
        self.collection = collection
        self.embedding_manager = embedding_manager
        self.settings = settings
        self.logger = logger

    async def search(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentSearchResult]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            if not self.embedding_manager:
                raise RAGError("Embedding manager not available")

            query_embedding = await self.embedding_manager.embed_text(query)
            
            # Use similarity threshold from settings if not provided
            threshold = similarity_threshold or self.settings.RAG_SIMILARITY_THRESHOLD
            limit = min(limit, self.settings.MAX_RAG_RESULTS)

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
                where=metadata_filter
            )

            # Convert to DocumentSearchResult objects
            search_results = []
            for i, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i] or {}
                distance = results["distances"][0][i]
                
                # Calculate similarity score from distance
                similarity_score = max(0.0, 1.0 - distance)
                
                # Skip results below threshold
                if threshold and similarity_score < threshold:
                    continue
                
                # Extract system metadata
                created_at_str = metadata.pop("created_at", datetime.now(UTC).isoformat())
                updated_at_str = metadata.pop("updated_at", None)
                embedding_model = metadata.pop("embedding_model", None)
                embedding_dimension = metadata.pop("embedding_dimension", None)
                
                # Parse timestamps
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                
                # Build document kwargs - only include updated_at if we have a value
                doc_kwargs = {
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "created_at": created_at,
                    "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension,
                }
                
                # Only include updated_at if we have a value, otherwise let model use default
                if updated_at_str:
                    doc_kwargs["updated_at"] = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))

                # Create document
                document = Document(**doc_kwargs)

                # Create search result
                search_result = DocumentSearchResult(
                    item=document,
                    score=similarity_score,
                    rank=len(search_results) + 1,
                    distance=distance,
                    embedding_model=embedding_model,
                    match_type="vector"
                )
                
                search_results.append(search_result)

            self.logger.info("Document search completed", query=query, results=len(search_results))
            return search_results

        except Exception as e:
            self.logger.error("Failed to search documents", query=query, error=str(e))
            raise RAGError(f"Failed to search documents: {e}")