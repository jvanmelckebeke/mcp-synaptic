"""MCP message handlers for memory and RAG operations."""

from typing import Any, Dict, List, Optional

from ..config.logging import LoggerMixin
from ..core.exceptions import MCPError, MemoryError, RAGError
from ..memory.manager import MemoryManager
from ..models.memory import MemoryQuery, MemoryType
from ..rag.database import RAGDatabase
from ..sse.events import MemoryEvent, RAGEvent
from ..sse.server import SSEServer


class BaseHandler(LoggerMixin):
    """Base class for MCP handlers."""

    def __init__(self, sse_server: Optional[SSEServer] = None):
        self.sse_server = sse_server

    async def emit_event(self, event) -> None:
        """Emit an SSE event if server is available."""
        if self.sse_server:
            await self.sse_server.broadcast_event(event)


class MemoryHandler(BaseHandler):
    """Handles memory-related MCP operations."""

    def __init__(self, memory_manager: MemoryManager, sse_server: Optional[SSEServer] = None):
        super().__init__(sse_server)
        self.memory_manager = memory_manager

    async def add_memory(
        self,
        key: str,
        data: Dict[str, Any],
        memory_type: str = "short_term",
        ttl_seconds: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a new memory."""
        try:
            # Validate memory type
            try:
                mem_type = MemoryType(memory_type)
            except ValueError:
                raise MCPError(f"Invalid memory type: {memory_type}")

            # Add memory
            memory = await self.memory_manager.add(
                key=key,
                data=data,
                memory_type=mem_type,
                ttl_seconds=ttl_seconds,
                tags=tags,
                metadata=metadata,
            )

            # Emit event
            event = MemoryEvent("added", key, memory.to_dict())
            await self.emit_event(event)

            self.logger.info("Memory added via MCP", key=key, memory_type=memory_type)
            return memory.to_dict()

        except MemoryError as e:
            self.logger.error("Failed to add memory via MCP", key=key, error=str(e))
            raise MCPError(f"Failed to add memory: {e}")

    async def get_memory(self, key: str, touch: bool = True) -> Optional[Dict[str, Any]]:
        """Get a memory by key."""
        try:
            memory = await self.memory_manager.get(key, touch=touch)
            
            if memory:
                # Emit access event if touched
                if touch:
                    event = MemoryEvent("accessed", key, {"access_count": memory.access_count})
                    await self.emit_event(event)
                
                self.logger.debug("Memory retrieved via MCP", key=key)
                return memory.to_dict()
            else:
                self.logger.debug("Memory not found via MCP", key=key)
                return None

        except MemoryError as e:
            self.logger.error("Failed to get memory via MCP", key=key, error=str(e))
            raise MCPError(f"Failed to get memory: {e}")

    async def update_memory(
        self,
        key: str,
        data: Optional[Dict[str, Any]] = None,
        extend_ttl: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing memory."""
        try:
            memory = await self.memory_manager.update(
                key=key,
                data=data,
                extend_ttl=extend_ttl,
                tags=tags,
                metadata=metadata,
            )

            if memory:
                # Emit event
                event = MemoryEvent("updated", key, memory.to_dict())
                await self.emit_event(event)

                self.logger.info("Memory updated via MCP", key=key)
                return memory.to_dict()
            else:
                return None

        except MemoryError as e:
            self.logger.error("Failed to update memory via MCP", key=key, error=str(e))
            raise MCPError(f"Failed to update memory: {e}")

    async def delete_memory(self, key: str) -> bool:
        """Delete a memory by key."""
        try:
            deleted = await self.memory_manager.delete(key)

            if deleted:
                # Emit event
                event = MemoryEvent("deleted", key)
                await self.emit_event(event)
                
                self.logger.info("Memory deleted via MCP", key=key)

            return deleted

        except MemoryError as e:
            self.logger.error("Failed to delete memory via MCP", key=key, error=str(e))
            raise MCPError(f"Failed to delete memory: {e}")

    async def list_memories(
        self,
        keys: Optional[List[str]] = None,
        memory_types: Optional[List[str]] = None,
        include_expired: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List memories with optional filtering."""
        try:
            # Convert string memory types to enum
            mem_types = None
            if memory_types:
                try:
                    mem_types = [MemoryType(mt) for mt in memory_types]
                except ValueError as e:
                    raise MCPError(f"Invalid memory type: {e}")

            # Create query
            query = MemoryQuery(
                keys=keys,
                memory_types=mem_types,
                include_expired=include_expired,
                limit=limit,
                offset=offset,
            )

            # List memories
            memories = await self.memory_manager.list(query)

            self.logger.debug("Memories listed via MCP", count=len(memories))
            return [memory.to_dict() for memory in memories]

        except MemoryError as e:
            self.logger.error("Failed to list memories via MCP", error=str(e))
            raise MCPError(f"Failed to list memories: {e}")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            stats = await self.memory_manager.get_stats()
            
            self.logger.debug("Memory stats retrieved via MCP")
            return {
                "total_memories": stats.total_memories,
                "memories_by_type": stats.memories_by_type,
                "expired_memories": stats.expired_memories,
                "total_size_bytes": stats.total_size_bytes,
                "average_ttl_seconds": stats.average_ttl_seconds,
                "oldest_memory": stats.oldest_memory.isoformat() if stats.oldest_memory else None,
                "newest_memory": stats.newest_memory.isoformat() if stats.newest_memory else None,
            }

        except MemoryError as e:
            self.logger.error("Failed to get memory stats via MCP", error=str(e))
            raise MCPError(f"Failed to get memory stats: {e}")


class RAGHandler(BaseHandler):
    """Handles RAG-related MCP operations."""

    def __init__(self, rag_database: RAGDatabase, sse_server: Optional[SSEServer] = None):
        super().__init__(sse_server)
        self.rag_database = rag_database

    async def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a document to the RAG database."""
        try:
            document = await self.rag_database.add_document(
                content=content,
                metadata=metadata,
                document_id=document_id,
            )

            # Emit event
            event = RAGEvent("added", document.id, document.to_dict())
            await self.emit_event(event)

            self.logger.info("Document added via MCP", document_id=document.id)
            return document.to_dict()

        except RAGError as e:
            self.logger.error("Failed to add document via MCP", error=str(e))
            raise MCPError(f"Failed to add document: {e}")

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        try:
            document = await self.rag_database.get_document(document_id)
            
            if document:
                self.logger.debug("Document retrieved via MCP", document_id=document_id)
                return document.to_dict()
            else:
                self.logger.debug("Document not found via MCP", document_id=document_id)
                return None

        except RAGError as e:
            self.logger.error("Failed to get document via MCP", document_id=document_id, error=str(e))
            raise MCPError(f"Failed to get document: {e}")

    async def update_document(
        self,
        document_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update an existing document."""
        try:
            document = await self.rag_database.update_document(
                document_id=document_id,
                content=content,
                metadata=metadata,
            )

            if document:
                # Emit event
                event = RAGEvent("updated", document_id, document.to_dict())
                await self.emit_event(event)

                self.logger.info("Document updated via MCP", document_id=document_id)
                return document.to_dict()
            else:
                return None

        except RAGError as e:
            self.logger.error("Failed to update document via MCP", document_id=document_id, error=str(e))
            raise MCPError(f"Failed to update document: {e}")

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        try:
            deleted = await self.rag_database.delete_document(document_id)

            if deleted:
                # Emit event
                event = RAGEvent("deleted", document_id)
                await self.emit_event(event)
                
                self.logger.info("Document deleted via MCP", document_id=document_id)

            return deleted

        except RAGError as e:
            self.logger.error("Failed to delete document via MCP", document_id=document_id, error=str(e))
            raise MCPError(f"Failed to delete document: {e}")

    async def search_documents(
        self,
        query: str,
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            results = await self.rag_database.search(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter,
            )

            # Emit event
            event = RAGEvent("searched", "query", {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "results_count": len(results),
                "limit": limit,
            })
            await self.emit_event(event)

            self.logger.info("Document search performed via MCP", query=query, results=len(results))
            return [result.to_dict() for result in results]

        except RAGError as e:
            self.logger.error("Failed to search documents via MCP", query=query, error=str(e))
            raise MCPError(f"Failed to search documents: {e}")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get RAG collection statistics."""
        try:
            stats = await self.rag_database.get_collection_stats()
            
            self.logger.debug("RAG stats retrieved via MCP")
            return stats

        except RAGError as e:
            self.logger.error("Failed to get RAG stats via MCP", error=str(e))
            raise MCPError(f"Failed to get RAG stats: {e}")