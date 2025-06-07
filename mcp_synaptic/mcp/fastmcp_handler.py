"""FastMCP-based server implementation for MCP Synaptic."""

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config.logging import LoggerMixin
from ..memory.manager import MemoryManager
from ..rag.database import RAGDatabase
from .memory_tools import MemoryTools
from .rag_tools import RAGTools


class FastMCPHandler(LoggerMixin):
    """FastMCP server implementation for MCP Synaptic."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        rag_database: RAGDatabase,
    ):
        self.memory_manager = memory_manager
        self.rag_database = rag_database
        
        # Create FastMCP server
        self.mcp = FastMCP("MCP Synaptic Server")
        self._register_custom_routes()
        self._register_tools()
        
    def _register_custom_routes(self) -> None:
        """Register custom HTTP routes."""
        
        @self.mcp.custom_route("/health", methods=["GET"])
        async def health_check(request: Request) -> JSONResponse:
            """Health check endpoint."""
            return JSONResponse({
                "status": "healthy",
                "service": "MCP Synaptic Server",
                "components": {
                    "memory": self.memory_manager is not None,
                    "rag": self.rag_database is not None,
                }
            })

    def _register_tools(self) -> None:
        """Register all tools with the FastMCP server."""
        # Register memory tools
        MemoryTools(self.mcp, self.memory_manager)
        
        # Register RAG tools  
        RAGTools(self.mcp, self.rag_database)

    def get_mcp_server(self) -> FastMCP:
        """Get the FastMCP server instance."""
        return self.mcp