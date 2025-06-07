"""
MCP Synaptic - A Memory-enhanced MCP Server with RAG capabilities.

This package provides an MCP (Model Context Protocol) server with:
- Local RAG (Retrieval-Augmented Generation) database
- Expiring memory management
- Server-Sent Events (SSE) communication
- Vector-based document storage and retrieval
"""

__version__ = "0.1.0"
__author__ = "MCP Synaptic Team"
__email__ = "dev@mcp-synaptic.com"

from .core.server import SynapticServer
from .config.settings import Settings

__all__ = ["SynapticServer", "Settings"]