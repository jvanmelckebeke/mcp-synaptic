"""Main MCP Synaptic server implementation."""

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config.logging import LoggerMixin, setup_logging
from ..config.settings import Settings
from ..memory.manager import MemoryManager
from ..mcp.protocol import MCPProtocolHandler
from ..rag.database import RAGDatabase
from ..sse.server import SSEServer
from .exceptions import SynapticError


class SynapticServer(LoggerMixin):
    """Main MCP Synaptic server with memory and RAG capabilities."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the server with configuration."""
        self.settings = settings or Settings()
        self.settings.create_directories()
        
        # Set up logging
        setup_logging(self.settings)
        self.logger.info("Initializing MCP Synaptic server", version=self.settings.MCP_SERVER_VERSION)
        
        # Initialize components
        self.memory_manager: Optional[MemoryManager] = None
        self.rag_database: Optional[RAGDatabase] = None
        self.sse_server: Optional[SSEServer] = None
        self.mcp_handler: Optional[MCPProtocolHandler] = None
        
        # FastAPI app will be created in the lifespan context
        self.app: Optional[FastAPI] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """FastAPI lifespan context manager."""
        try:
            await self._startup()
            yield
        finally:
            await self._shutdown()

    async def _startup(self) -> None:
        """Start up all server components."""
        self.logger.info("Starting MCP Synaptic server components")
        
        try:
            # Initialize memory manager
            self.memory_manager = MemoryManager(self.settings)
            await self.memory_manager.initialize()
            
            # Initialize RAG database
            self.rag_database = RAGDatabase(self.settings)
            await self.rag_database.initialize()
            
            # Initialize SSE server
            self.sse_server = SSEServer(self.settings)
            
            # Initialize MCP protocol handler
            self.mcp_handler = MCPProtocolHandler(
                self.settings,
                self.memory_manager,
                self.rag_database,
                self.sse_server
            )
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._running = True
            self.logger.info("All server components started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start server components", error=str(e))
            raise

    async def _shutdown(self) -> None:
        """Shut down all server components."""
        self.logger.info("Shutting down MCP Synaptic server")
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components in reverse order
        if self.sse_server:
            await self.sse_server.shutdown()
        
        if self.rag_database:
            await self.rag_database.close()
        
        if self.memory_manager:
            await self.memory_manager.close()
        
        self.logger.info("Server shutdown complete")

    async def _cleanup_loop(self) -> None:
        """Background cleanup task for expired memories."""
        while self._running:
            try:
                if self.memory_manager:
                    await self.memory_manager.cleanup_expired()
                
                await asyncio.sleep(self.settings.MEMORY_CLEANUP_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    def create_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="MCP Synaptic",
            description="Memory-enhanced MCP Server with RAG capabilities",
            version=self.settings.MCP_SERVER_VERSION,
            lifespan=self.lifespan,
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add exception handler
        @app.exception_handler(SynapticError)
        async def synaptic_exception_handler(request, exc: SynapticError):
            return JSONResponse(
                status_code=400,
                content=exc.to_dict()
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request, exc: Exception):
            self.logger.error("Unhandled exception", error=str(exc), exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "version": self.settings.MCP_SERVER_VERSION,
                "components": {
                    "memory": self.memory_manager is not None,
                    "rag": self.rag_database is not None,
                    "sse": self.sse_server is not None,
                    "mcp": self.mcp_handler is not None,
                }
            }
        
        # Include routers when they're implemented
        # app.include_router(memory_router, prefix="/memory", tags=["memory"])
        # app.include_router(rag_router, prefix="/rag", tags=["rag"])
        # app.include_router(sse_router, prefix="/events", tags=["sse"])
        # app.include_router(mcp_router, prefix="/mcp", tags=["mcp"])
        
        self.app = app
        return app

    async def start(self) -> None:
        """Start the server using uvicorn."""
        app = self.create_app()
        
        # Set up signal handlers
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully")
            asyncio.create_task(self._shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=self.settings.SERVER_HOST,
            port=self.settings.SERVER_PORT,
            log_level=self.settings.LOG_LEVEL.lower(),
            reload=self.settings.DEBUG,
            access_log=self.settings.DEBUG,
        )
        
        server = uvicorn.Server(config)
        await server.serve()

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running