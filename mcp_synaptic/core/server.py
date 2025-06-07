"""Main MCP Synaptic server implementation."""

import asyncio
from typing import Optional

from ..config.logging import LoggerMixin, setup_logging
from ..config.settings import Settings
from ..memory.manager import MemoryManager
from ..mcp.fastmcp_handler import FastMCPHandler
from ..rag.database import RAGDatabase


class SynapticServer(LoggerMixin):
    """Main MCP Synaptic server with memory and RAG capabilities."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the server with configuration."""
        self.settings = settings or Settings()
        self.settings.create_directories()
        
        # Set up logging
        setup_logging(self.settings)
        self.logger.info("Initializing MCP Synaptic server")
        
        # Initialize components
        self.memory_manager: Optional[MemoryManager] = None
        self.rag_database: Optional[RAGDatabase] = None
        self.mcp_handler: Optional[FastMCPHandler] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def _startup(self) -> None:
        """Initialize all server components."""
        if self._running:
            return
            
        self.logger.info("Starting MCP Synaptic server")
        
        try:
            # Initialize memory manager
            self.memory_manager = MemoryManager(self.settings)
            await self.memory_manager.initialize()
            
            # Initialize RAG database
            self.rag_database = RAGDatabase(self.settings)
            await self.rag_database.initialize()
            
            # Initialize FastMCP handler
            self.mcp_handler = FastMCPHandler(
                memory_manager=self.memory_manager,
                rag_database=self.rag_database
            )
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self._running = True
            self.logger.info("MCP Synaptic server started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start server", error=str(e))
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
        
        # Shutdown components
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

    async def run(self) -> None:
        """Run the server using FastMCP SSE transport."""
        # Initialize components first
        await self._startup()
        
        try:
            # Get FastMCP server and run with SSE transport
            if not self.mcp_handler:
                raise RuntimeError("MCP handler not initialized")
            
            mcp_server = self.mcp_handler.get_mcp_server()
            
            # Get the SSE FastAPI app and run with uvicorn for host/port control
            import uvicorn
            sse_app = mcp_server.sse_app()
            
            # Configure uvicorn with proper host binding for Docker/Traefik
            config = uvicorn.Config(
                sse_app,
                host=self.settings.SERVER_HOST,
                port=self.settings.SERVER_PORT,
                log_level=self.settings.LOG_LEVEL.lower()
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error("Server error", error=str(e))
            raise
        finally:
            await self._shutdown()

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running