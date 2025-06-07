"""MCP protocol handler implementation."""

import asyncio
from typing import Any, Dict, List, Optional, Union

import mcp
from mcp import types
from mcp.server import Server
from mcp.server.models import InitializationOptions

from ..config.logging import LoggerMixin
from ..config.settings import Settings
from ..core.exceptions import MCPError
from ..memory.manager import MemoryManager
from ..rag.database import RAGDatabase
from ..sse.server import SSEServer


class MCPProtocolHandler(LoggerMixin):
    """Handles MCP protocol operations."""

    def __init__(
        self,
        settings: Settings,
        memory_manager: MemoryManager,
        rag_database: RAGDatabase,
        sse_server: SSEServer,
    ):
        self.settings = settings
        self.memory_manager = memory_manager
        self.rag_database = rag_database
        self.sse_server = sse_server
        self.server: Optional[Server] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the MCP protocol handler."""
        if mcp is None:
            raise MCPError("MCP library not available. Install with: pip install mcp")

        try:
            # Create MCP server
            self.server = Server(self.settings.MCP_SERVER_NAME)
            
            # Register handlers
            await self._register_handlers()
            
            self._initialized = True
            
            self.logger.info(
                "MCP protocol handler initialized",
                server_name=self.settings.MCP_SERVER_NAME,
                version=self.settings.MCP_SERVER_VERSION
            )

        except Exception as e:
            self.logger.error("Failed to initialize MCP protocol handler", error=str(e))
            raise MCPError(f"MCP protocol handler initialization failed: {e}")

    async def _register_handlers(self) -> None:
        """Register MCP method handlers."""
        if not self.server:
            return

        # Memory-related handlers
        @self.server.list_resources()
        async def handle_list_resources() -> List[types.Resource]:
            """List available resources."""
            resources = []
            
            # Add memory resources
            resources.append(
                types.Resource(
                    uri="memory://",
                    name="Memory Management",
                    description="Access to memory storage with expiration",
                    mimeType="application/json"
                )
            )
            
            # Add RAG resources
            resources.append(
                types.Resource(
                    uri="rag://",
                    name="RAG Database",
                    description="Access to document storage and retrieval",
                    mimeType="application/json"
                )
            )
            
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read a resource by URI."""
            if uri.startswith("memory://"):
                # Handle memory resource requests
                key = uri.replace("memory://", "")
                if key:
                    memory = await self.memory_manager.get(key)
                    if memory:
                        return memory.model_dump_json()
                    else:
                        raise MCPError(f"Memory not found: {key}")
                else:
                    # List all memories
                    memories = await self.memory_manager.list()
                    return json.dumps([m.to_dict() for m in memories])
            
            elif uri.startswith("rag://"):
                # Handle RAG resource requests
                doc_id = uri.replace("rag://", "")
                if doc_id:
                    document = await self.rag_database.get_document(doc_id)
                    if document:
                        return json.dumps(document.to_dict())
                    else:
                        raise MCPError(f"Document not found: {doc_id}")
                else:
                    # Return collection stats
                    stats = await self.rag_database.get_collection_stats()
                    return json.dumps(stats)
            
            else:
                raise MCPError(f"Unknown resource URI: {uri}")

        @self.server.list_tools()
        async def handle_list_tools() -> List[types.Tool]:
            """List available tools."""
            tools = []
            
            # Memory tools
            tools.extend([
                types.Tool(
                    name="memory_add",
                    description="Add a new memory with optional expiration",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Memory key"},
                            "data": {"type": "object", "description": "Memory data"},
                            "ttl_seconds": {"type": "integer", "description": "Time to live in seconds"},
                            "memory_type": {"type": "string", "enum": ["ephemeral", "short_term", "long_term", "permanent"]},
                        },
                        "required": ["key", "data"]
                    }
                ),
                types.Tool(
                    name="memory_get",
                    description="Get a memory by key",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Memory key"},
                        },
                        "required": ["key"]
                    }
                ),
                types.Tool(
                    name="memory_delete",
                    description="Delete a memory by key",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "description": "Memory key"},
                        },
                        "required": ["key"]
                    }
                ),
                types.Tool(
                    name="memory_search",
                    description="Search memories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "keys": {"type": "array", "items": {"type": "string"}},
                            "memory_types": {"type": "array", "items": {"type": "string"}},
                            "include_expired": {"type": "boolean", "default": False},
                        }
                    }
                ),
            ])
            
            # RAG tools
            tools.extend([
                types.Tool(
                    name="rag_add_document",
                    description="Add a document to the RAG database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "Document content"},
                            "metadata": {"type": "object", "description": "Document metadata"},
                            "document_id": {"type": "string", "description": "Optional document ID"},
                        },
                        "required": ["content"]
                    }
                ),
                types.Tool(
                    name="rag_search",
                    description="Search for similar documents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "default": 10, "description": "Maximum results"},
                            "similarity_threshold": {"type": "number", "description": "Similarity threshold"},
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="rag_get_document",
                    description="Get a document by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string", "description": "Document ID"},
                        },
                        "required": ["document_id"]
                    }
                ),
                types.Tool(
                    name="rag_delete_document",
                    description="Delete a document by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string", "description": "Document ID"},
                        },
                        "required": ["document_id"]
                    }
                ),
            ])
            
            return tools

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls."""
            try:
                result = await self._execute_tool(name, arguments)
                return [types.TextContent(type="text", text=str(result))]
            except Exception as e:
                self.logger.error("Tool execution failed", tool=name, error=str(e))
                raise MCPError(f"Tool execution failed: {e}")

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        # Memory tools
        if name == "memory_add":
            memory = await self.memory_manager.add(
                key=arguments["key"],
                data=arguments["data"],
                ttl_seconds=arguments.get("ttl_seconds"),
                memory_type=arguments.get("memory_type", "short_term"),
            )
            return memory.to_dict()
        
        elif name == "memory_get":
            memory = await self.memory_manager.get(arguments["key"])
            return memory.to_dict() if memory else None
        
        elif name == "memory_delete":
            deleted = await self.memory_manager.delete(arguments["key"])
            return {"deleted": deleted}
        
        elif name == "memory_search":
            from ..models.memory import MemoryQuery
            query = MemoryQuery(**arguments)
            memories = await self.memory_manager.list(query)
            return [m.to_dict() for m in memories]
        
        # RAG tools
        elif name == "rag_add_document":
            document = await self.rag_database.add_document(
                content=arguments["content"],
                metadata=arguments.get("metadata"),
                document_id=arguments.get("document_id"),
            )
            return document.to_dict()
        
        elif name == "rag_search":
            results = await self.rag_database.search(
                query=arguments["query"],
                limit=arguments.get("limit", 10),
                similarity_threshold=arguments.get("similarity_threshold"),
            )
            return [r.to_dict() for r in results]
        
        elif name == "rag_get_document":
            document = await self.rag_database.get_document(arguments["document_id"])
            return document.to_dict() if document else None
        
        elif name == "rag_delete_document":
            deleted = await self.rag_database.delete_document(arguments["document_id"])
            return {"deleted": deleted}
        
        else:
            raise MCPError(f"Unknown tool: {name}")

    async def close(self) -> None:
        """Close the MCP protocol handler."""
        self._initialized = False
        self.logger.info("MCP protocol handler closed")

    @property
    def is_initialized(self) -> bool:
        """Check if the handler is initialized."""
        return self._initialized