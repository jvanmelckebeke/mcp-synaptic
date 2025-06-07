# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Package Management:**
```bash
# Install dependencies
uv sync                          # Core dependencies (lightweight, API embeddings)
uv sync --extra local-embeddings # Include PyTorch for local embeddings (heavy)
uv sync --group dev             # Add development tools

# Quality Assurance
uv run pytest                   # Run all tests
uv run pytest tests/unit/       # Run only unit tests
uv run pytest -k "test_name"    # Run specific test
uv run mypy mcp_synaptic        # Type checking
uv run ruff check mcp_synaptic  # Linting
uv run black mcp_synaptic       # Code formatting

# Combined checks (run before commits)
uv run pytest && uv run mypy mcp_synaptic && uv run ruff check mcp_synaptic
```

**Server Operations:**
```bash
# Development server
uv run mcp-synaptic server --debug

# Production server  
uv run python main.py

# Docker development
docker-compose up --build
```

## Architecture Overview

**Core System Design:**
MCP Synaptic is a modular async Python application with these key layers:

1. **SynapticServer** (`core/server.py`) - Central orchestrator managing component lifecycle
2. **Memory Management** - Pluggable backend (SQLite/Redis) with TTL-based expiration
3. **RAG Database** - ChromaDB for vector storage with flexible embedding providers
4. **MCP Protocol** - Model Context Protocol implementation with tool registration
5. **SSE Communication** - Real-time event streaming for memory/RAG operations

**Component Initialization Flow:**
Settings → Memory Manager → RAG Database → SSE Server → MCP Handler → FastAPI App

**Critical Patterns:**
- **Async Context Managers**: All components use `async with` for resource management
- **Dependency Injection**: Components receive dependencies via constructor
- **Background Tasks**: Memory cleanup and SSE heartbeats run as asyncio tasks
- **Error Propagation**: Custom exception hierarchy with structured logging

## Embedding System Architecture

**Dual Provider Support:**
- **API Provider** (default): Calls external OpenAI-compatible endpoints (LiteLLM, OpenAI)
- **Local Provider**: Uses sentence-transformers with PyTorch (optional heavy dependency)

Configuration determines provider:
```python
# Lightweight API-based (recommended)
EMBEDDING_PROVIDER=api
EMBEDDING_API_BASE=http://localhost:4000  # Your LiteLLM container

# Heavy local processing  
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**Key Implementation Details:**
- API calls use aiohttp sessions with proper timeout/error handling
- Local embeddings run in thread pools to avoid blocking the event loop
- Embedding dimensions are model-specific and cached for performance

## Memory System Design

**Storage Abstraction:**
- `MemoryStorage` protocol defines the interface
- `SQLiteMemoryStorage` and `RedisMemoryStorage` implement backends
- Backend selection based on `REDIS_ENABLED` configuration

**Memory Types & TTL:**
- Ephemeral (5 min), Short-term (1 hr), Long-term (1 week), Permanent (no expiry)
- Both absolute and sliding TTL expiration policies
- Background cleanup task removes expired memories every 5 minutes

**Access Patterns:**
- `touch=True` updates last_accessed_at for sliding expiration
- Memory statistics track usage patterns for monitoring

## Configuration System

**Settings Architecture:**
- Pydantic-based validation in `config/settings.py`
- Environment variables with sensible defaults
- Automatic directory creation for data storage paths
- Type-safe configuration with validation errors

**Key Configuration Categories:**
- Server settings (host, port, debug mode)
- Database paths (SQLite, ChromaDB persistence)
- Memory management (TTL defaults, cleanup intervals)
- Embedding configuration (provider, API endpoints)
- Redis settings (optional distributed storage)

## Testing Strategy

**Test Structure:**
- `tests/unit/` - Component isolation tests
- `tests/integration/` - Multi-component interaction tests
- Async test support with `pytest-asyncio`
- Mock external dependencies (Redis, API calls)

**Coverage Targets:**
- Memory management operations (CRUD, expiration)
- Embedding provider switching (API vs local)
- Configuration validation and error handling
- Server lifecycle management

## Docker & Deployment

**Container Strategy:**
- Base image: Python 3.11-slim with UV package manager
- Multi-stage build for production optimization
- Non-root user for security
- Health check endpoint at `/health`

**Deployment Variants:**
- `docker-compose.yml` - Basic setup
- `docker-compose.dev.yml` - Development with Redis commander
- `docker-compose.prod.yml` - Production optimizations

**Important Notes:**
- Default embedding provider is API-based to keep containers lightweight
- Local embeddings require `--extra local-embeddings` and significantly increase image size
- Redis is optional - SQLite works fine for single-instance deployments

## Development Workflow

**Code Quality:**
- Pre-commit hooks run mypy, ruff, and black automatically
- Type hints required on all public functions
- Structured logging using `structlog` with correlation IDs
- Custom exception hierarchy for proper error handling

**Adding New Features:**
1. Define configuration in `settings.py` with validation
2. Implement core logic with proper async patterns
3. Add MCP tool registration in `protocol.py` if exposing to MCP clients
4. Add SSE events if real-time updates needed
5. Write unit tests covering happy path and error cases
6. Update integration tests for component interactions

**Component Dependencies:**
- All async operations must handle `asyncio.CancelledError`
- Database operations should use the storage abstraction layer
- External API calls need proper timeout and retry logic
- Memory operations should validate TTL and data size limits