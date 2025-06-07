"""Document retrieval utilities for RAG."""

from typing import Any, Dict, List, Optional

from ..config.logging import LoggerMixin
from ..core.exceptions import RAGError
from .database import RAGDatabase
from ..models.rag import Document, DocumentSearchResult


class DocumentRetriever(LoggerMixin):
    """High-level document retrieval interface."""

    def __init__(self, rag_database: RAGDatabase):
        self.rag_database = rag_database

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentSearchResult]:
        """Search for documents similar to the query."""
        try:
            results = await self.rag_database.search(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold,
                metadata_filter=filters,
            )

            self.logger.info(
                "Document search completed",
                query=query[:100] + "..." if len(query) > 100 else query,
                results_count=len(results),
                limit=limit
            )

            return results

        except Exception as e:
            self.logger.error("Failed to search documents", query=query, error=str(e))
            raise RAGError(f"Document search failed: {e}")

    async def get_relevant_context(
        self,
        query: str,
        max_context_length: int = 2000,
        max_documents: int = 5,
        similarity_threshold: float = 0.7,
    ) -> str:
        """Get relevant context for a query by combining top search results."""
        try:
            # Search for relevant documents
            results = await self.search_documents(
                query=query,
                limit=max_documents,
                similarity_threshold=similarity_threshold,
            )

            if not results:
                self.logger.debug("No relevant documents found", query=query)
                return ""

            # Combine document contents
            context_parts = []
            total_length = 0

            for result in results:
                content = result.document.content
                
                # Check if adding this content would exceed max length
                if total_length + len(content) > max_context_length:
                    # Truncate to fit
                    remaining_space = max_context_length - total_length
                    if remaining_space > 100:  # Only add if there's meaningful space
                        content = content[:remaining_space - 3] + "..."
                        context_parts.append(content)
                    break
                
                context_parts.append(content)
                total_length += len(content)

            context = "\n\n".join(context_parts)

            self.logger.info(
                "Context generated",
                query=query[:50] + "..." if len(query) > 50 else query,
                context_length=len(context),
                documents_used=len(context_parts)
            )

            return context

        except Exception as e:
            self.logger.error("Failed to get relevant context", query=query, error=str(e))
            raise RAGError(f"Failed to get relevant context: {e}")

    async def find_similar_documents(
        self,
        document_id: str,
        limit: int = 5,
        similarity_threshold: float = 0.8,
    ) -> List[DocumentSearchResult]:
        """Find documents similar to a given document."""
        try:
            # Get the reference document
            reference_doc = await self.rag_database.get_document(document_id)
            if not reference_doc:
                raise RAGError(f"Reference document not found: {document_id}")

            # Search using the reference document's content
            results = await self.search_documents(
                query=reference_doc.content,
                limit=limit + 1,  # +1 to account for the reference document itself
                similarity_threshold=similarity_threshold,
            )

            # Filter out the reference document from results
            similar_docs = [
                result for result in results 
                if result.document.id != document_id
            ][:limit]

            self.logger.info(
                "Similar documents found",
                reference_id=document_id,
                similar_count=len(similar_docs)
            )

            return similar_docs

        except Exception as e:
            self.logger.error("Failed to find similar documents", document_id=document_id, error=str(e))
            raise RAGError(f"Failed to find similar documents: {e}")

    async def get_document_summary(
        self,
        documents: List[Document],
        max_summary_length: int = 500,
    ) -> str:
        """Generate a summary from multiple documents."""
        try:
            if not documents:
                return ""

            # Simple extractive summary - take first sentences up to max length
            summary_parts = []
            total_length = 0

            for doc in documents:
                # Split into sentences (simple approach)
                sentences = doc.content.split('. ')
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # Add period if not present
                    if not sentence.endswith('.'):
                        sentence += '.'
                    
                    # Check if adding this sentence would exceed max length
                    if total_length + len(sentence) > max_summary_length:
                        break
                    
                    summary_parts.append(sentence)
                    total_length += len(sentence) + 1  # +1 for space
                
                if total_length >= max_summary_length:
                    break

            summary = ' '.join(summary_parts)

            self.logger.debug(
                "Document summary generated",
                input_documents=len(documents),
                summary_length=len(summary)
            )

            return summary

        except Exception as e:
            self.logger.error("Failed to generate document summary", error=str(e))
            raise RAGError(f"Failed to generate document summary: {e}")

    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        limit: int = 20,
    ) -> List[Document]:
        """Search documents by metadata filters."""
        try:
            # Use empty query to get all documents matching metadata
            results = await self.rag_database.search(
                query="",  # Empty query
                limit=limit,
                similarity_threshold=0.0,  # Very low threshold
                metadata_filter=filters,
            )

            documents = [result.document for result in results]

            self.logger.info(
                "Metadata search completed",
                filters=filters,
                results_count=len(documents)
            )

            return documents

        except Exception as e:
            self.logger.error("Failed to search by metadata", filters=filters, error=str(e))
            raise RAGError(f"Metadata search failed: {e}")