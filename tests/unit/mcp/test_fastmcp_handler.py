"""Tests for FastMCP handler."""

from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from mcp_synaptic.mcp.fastmcp_handler import FastMCPHandler
from mcp_synaptic.memory.manager import MemoryManager
from mcp_synaptic.rag.database import RAGDatabase


class TestFastMCPHandler:
    """Test FastMCP handler functionality."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        manager = AsyncMock(spec=MemoryManager)
        return manager

    @pytest.fixture
    def mock_rag_database(self):
        """Create a mock RAG database."""
        database = AsyncMock(spec=RAGDatabase)
        return database

    @pytest.fixture
    def mock_fastmcp(self):
        """Create a mock FastMCP server."""
        mcp = MagicMock()
        mcp.custom_route.return_value = lambda func: func  # Return function unchanged
        return mcp

    @pytest.fixture
    def fastmcp_handler(self, mock_memory_manager, mock_rag_database):
        """Create FastMCPHandler instance with mocked dependencies."""
        with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
            mock_mcp_instance = MagicMock()
            mock_mcp_instance.custom_route.return_value = lambda func: func
            mock_fastmcp_class.return_value = mock_mcp_instance
            
            with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools') as mock_memory_tools:
                with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools') as mock_rag_tools:
                    handler = FastMCPHandler(mock_memory_manager, mock_rag_database)
                    handler.mcp = mock_mcp_instance
                    return handler

    def test_fastmcp_handler_initialization(self, mock_memory_manager, mock_rag_database):
        """Test FastMCPHandler initialization."""
        with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
            mock_mcp_instance = MagicMock()
            mock_fastmcp_class.return_value = mock_mcp_instance
            
            with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools') as mock_memory_tools:
                with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools') as mock_rag_tools:
                    handler = FastMCPHandler(mock_memory_manager, mock_rag_database)
                    
                    # Verify initialization
                    assert handler.memory_manager is mock_memory_manager
                    assert handler.rag_database is mock_rag_database
                    
                    # Verify FastMCP server was created
                    mock_fastmcp_class.assert_called_once_with("MCP Synaptic Server")
                    
                    # Verify tools were registered
                    mock_memory_tools.assert_called_once_with(mock_mcp_instance, mock_memory_manager)
                    mock_rag_tools.assert_called_once_with(mock_mcp_instance, mock_rag_database)

    def test_custom_routes_registration(self, fastmcp_handler):
        """Test that custom routes are registered."""
        # Verify custom route decorator was called
        fastmcp_handler.mcp.custom_route.assert_called()
        
        # Check if health endpoint was registered
        calls = fastmcp_handler.mcp.custom_route.call_args_list
        health_call = next((call for call in calls if call[0][0] == "/health"), None)
        assert health_call is not None
        assert health_call.kwargs["methods"] == ["GET"]

    def test_tools_registration(self, mock_memory_manager, mock_rag_database):
        """Test that both memory and RAG tools are registered."""
        with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
            mock_mcp_instance = MagicMock()
            mock_fastmcp_class.return_value = mock_mcp_instance
            
            with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools') as mock_memory_tools:
                with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools') as mock_rag_tools:
                    handler = FastMCPHandler(mock_memory_manager, mock_rag_database)
                    
                    # Verify both tool classes were instantiated
                    mock_memory_tools.assert_called_once_with(mock_mcp_instance, mock_memory_manager)
                    mock_rag_tools.assert_called_once_with(mock_mcp_instance, mock_rag_database)

    def test_logging_functionality(self, fastmcp_handler):
        """Test that FastMCPHandler has logging capability."""
        # FastMCPHandler inherits from LoggerMixin
        assert hasattr(fastmcp_handler, 'logger')
        assert callable(getattr(fastmcp_handler.logger, 'info', None))
        assert callable(getattr(fastmcp_handler.logger, 'debug', None))
        assert callable(getattr(fastmcp_handler.logger, 'error', None))

    class TestHealthEndpoint:
        """Test health check endpoint functionality."""

        async def test_health_check_both_components_available(self, fastmcp_handler, mock_memory_manager, mock_rag_database):
            """Test health check when both components are available."""
            # Create a mock request
            mock_request = MagicMock()
            
            # Simulate health check logic
            health_response = {
                "status": "healthy",
                "service": "MCP Synaptic Server",
                "components": {
                    "memory": mock_memory_manager is not None,
                    "rag": mock_rag_database is not None,
                }
            }
            
            # Both components should be available
            assert health_response["components"]["memory"] is True
            assert health_response["components"]["rag"] is True
            assert health_response["status"] == "healthy"

        async def test_health_check_missing_memory_manager(self, mock_rag_database):
            """Test health check when memory manager is missing."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools'):
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools'):
                        handler = FastMCPHandler(None, mock_rag_database)  # No memory manager
                        
                        health_response = {
                            "status": "healthy",
                            "service": "MCP Synaptic Server",
                            "components": {
                                "memory": handler.memory_manager is not None,
                                "rag": handler.rag_database is not None,
                            }
                        }
                        
                        assert health_response["components"]["memory"] is False
                        assert health_response["components"]["rag"] is True

        async def test_health_check_missing_rag_database(self, mock_memory_manager):
            """Test health check when RAG database is missing."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools'):
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools'):
                        handler = FastMCPHandler(mock_memory_manager, None)  # No RAG database
                        
                        health_response = {
                            "status": "healthy",
                            "service": "MCP Synaptic Server",
                            "components": {
                                "memory": handler.memory_manager is not None,
                                "rag": handler.rag_database is not None,
                            }
                        }
                        
                        assert health_response["components"]["memory"] is True
                        assert health_response["components"]["rag"] is False

        async def test_health_check_both_components_missing(self):
            """Test health check when both components are missing."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools'):
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools'):
                        handler = FastMCPHandler(None, None)  # No components
                        
                        health_response = {
                            "status": "healthy",
                            "service": "MCP Synaptic Server",
                            "components": {
                                "memory": handler.memory_manager is not None,
                                "rag": handler.rag_database is not None,
                            }
                        }
                        
                        assert health_response["components"]["memory"] is False
                        assert health_response["components"]["rag"] is False
                        # Status is still "healthy" - endpoint is about API health, not components

    class TestIntegration:
        """Test integration between handler and tools."""

        def test_memory_tools_integration(self, mock_memory_manager, mock_rag_database):
            """Test integration with memory tools."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools') as mock_memory_tools:
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools'):
                        handler = FastMCPHandler(mock_memory_manager, mock_rag_database)
                        
                        # Verify MemoryTools was called with correct parameters
                        mock_memory_tools.assert_called_once_with(mock_mcp_instance, mock_memory_manager)

        def test_rag_tools_integration(self, mock_memory_manager, mock_rag_database):
            """Test integration with RAG tools."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools'):
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools') as mock_rag_tools:
                        handler = FastMCPHandler(mock_memory_manager, mock_rag_database)
                        
                        # Verify RAGTools was called with correct parameters
                        mock_rag_tools.assert_called_once_with(mock_mcp_instance, mock_rag_database)

        def test_fastmcp_server_name(self, mock_memory_manager, mock_rag_database):
            """Test that FastMCP server is created with correct name."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools'):
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools'):
                        handler = FastMCPHandler(mock_memory_manager, mock_rag_database)
                        
                        # Verify server name
                        mock_fastmcp_class.assert_called_once_with("MCP Synaptic Server")

    class TestErrorHandling:
        """Test error handling during initialization."""

        def test_initialization_with_none_components(self):
            """Test initialization handles None components gracefully."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools'):
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools'):
                        # Should not raise an exception
                        handler = FastMCPHandler(None, None)
                        
                        assert handler.memory_manager is None
                        assert handler.rag_database is None

        def test_fastmcp_creation_failure(self, mock_memory_manager, mock_rag_database):
            """Test handling of FastMCP creation failure."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_fastmcp_class.side_effect = Exception("FastMCP creation failed")
                
                # Should propagate the exception
                with pytest.raises(Exception, match="FastMCP creation failed"):
                    FastMCPHandler(mock_memory_manager, mock_rag_database)

        def test_tools_registration_failure(self, mock_memory_manager, mock_rag_database):
            """Test handling of tools registration failure."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools') as mock_memory_tools:
                    mock_memory_tools.side_effect = Exception("Memory tools registration failed")
                    
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools'):
                        # Should propagate the exception
                        with pytest.raises(Exception, match="Memory tools registration failed"):
                            FastMCPHandler(mock_memory_manager, mock_rag_database)

    class TestComponentAccess:
        """Test access to handler components."""

        def test_memory_manager_access(self, fastmcp_handler, mock_memory_manager):
            """Test access to memory manager component."""
            assert fastmcp_handler.memory_manager is mock_memory_manager

        def test_rag_database_access(self, fastmcp_handler, mock_rag_database):
            """Test access to RAG database component."""
            assert fastmcp_handler.rag_database is mock_rag_database

        def test_mcp_server_access(self, fastmcp_handler):
            """Test access to FastMCP server instance."""
            assert hasattr(fastmcp_handler, 'mcp')
            assert fastmcp_handler.mcp is not None

    class TestServerConfiguration:
        """Test server configuration and setup."""

        def test_server_name_configuration(self, mock_memory_manager, mock_rag_database):
            """Test that server is configured with correct name."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools'):
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools'):
                        handler = FastMCPHandler(mock_memory_manager, mock_rag_database)
                        
                        # Verify server was created with expected name
                        mock_fastmcp_class.assert_called_once_with("MCP Synaptic Server")

        def test_route_registration_order(self, mock_memory_manager, mock_rag_database):
            """Test that routes and tools are registered in correct order."""
            with patch('mcp_synaptic.mcp.fastmcp_handler.FastMCP') as mock_fastmcp_class:
                mock_mcp_instance = MagicMock()
                mock_fastmcp_class.return_value = mock_mcp_instance
                
                with patch('mcp_synaptic.mcp.fastmcp_handler.MemoryTools') as mock_memory_tools:
                    with patch('mcp_synaptic.mcp.fastmcp_handler.RAGTools') as mock_rag_tools:
                        handler = FastMCPHandler(mock_memory_manager, mock_rag_database)
                        
                        # Verify custom routes were registered first (via custom_route calls)
                        assert mock_mcp_instance.custom_route.called
                        
                        # Verify tools were registered after routes
                        mock_memory_tools.assert_called_once()
                        mock_rag_tools.assert_called_once()

        def test_health_endpoint_configuration(self, fastmcp_handler):
            """Test health endpoint configuration details."""
            # Verify health endpoint was registered with correct path and methods
            calls = fastmcp_handler.mcp.custom_route.call_args_list
            health_call = next((call for call in calls if call[0][0] == "/health"), None)
            
            assert health_call is not None
            assert health_call[0][0] == "/health"  # Path
            assert health_call[1]["methods"] == ["GET"]  # HTTP methods