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

**Expert-Level Dockerfile Requirements:**
All Dockerfiles must demonstrate expert-level containerization practices:
- Multi-stage builds with proper dependency separation
- Mount caches for build performance optimization (`--mount=type=cache`)
- Official uv image usage (`ghcr.io/astral-sh/uv`) with proper version pinning
- Deterministic builds with UV environment variables (UV_COMPILE_BYTECODE, UV_NO_INSTALLER_METADATA)
- Optimized layer ordering and caching strategies
- Minimal runtime images with only necessary dependencies
- Proper security practices (non-root users, minimal attack surface)
- BuildKit features utilization for advanced patterns
- Enterprise-grade configuration (health checks, proper signal handling)

**Docker Compose Structure:**
- `docker/docker-compose.yaml` - Base configuration (standard ports)
- `docker/overrides/laptop.yaml` - Traefik HTTP entrypoint override
- `docker/overrides/desktop.yaml` - Traefik WEB entrypoint override
- `docker/variants/dev.yaml` - Development variant (Redis commander, debug)
- `docker/variants/prod.yaml` - Production variant (optimized resources)

**Usage Examples:**
```bash
# Standard (ports exposed)
cd docker && docker-compose up -d

# Laptop (Traefik HTTP)
cd docker && docker-compose -f docker-compose.yaml -f overrides/laptop.yaml up -d

# Desktop (Traefik WEB)  
cd docker && docker-compose -f docker-compose.yaml -f overrides/desktop.yaml up -d

# Development mode
cd docker && docker-compose -f docker-compose.yaml -f overrides/laptop.yaml -f variants/dev.yaml up -d
```

**Important Notes:**
- Default embedding provider is API-based to keep containers lightweight
- Local embeddings require `--extra local-embeddings` and significantly increase image size
- Redis is optional - SQLite works fine for single-instance deployments

## Development Workflow

**Semantic Versioning Responsibility:**
- **CRITICAL**: Claude Code MUST maintain semantic versioning tags for all releases
- User will NOT diligently manage version tags - this is Claude's responsibility
- **REQUIRED WORKFLOW**: For every significant change commit to main branch:
  1. Assess impact: patch (bug fix), minor (new feature), major (breaking change)
  2. Create appropriate git tag: `git tag v1.2.3` (follow semver.org strictly)
  3. Push tag: `git push --tags`
  4. Verify GHCR publishes both `:latest` and `:v1.2.3` images
- **Examples**:
  - Bug fixes, small improvements: v1.0.1, v1.0.2, etc.
  - New features, tool additions: v1.1.0, v1.2.0, etc.  
  - Breaking API changes, major refactors: v2.0.0, v3.0.0, etc.
- **Version Tag Format**: Always `v` prefix followed by semantic version (v1.2.3)
- **When in doubt**: Err on the side of incrementing minor version for features, patch for fixes

**Backward Compatibility Policy:**
- **DO NOT** consider backward compatibility when making architectural changes
- This is a new project with rapidly evolving architecture
- Prioritize clean, maintainable code over preserving legacy interfaces
- Breaking changes are acceptable and expected during early development
- Focus on getting the architecture right rather than maintaining compatibility with previous versions

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

## Task Management & Development Workflow

**GitHub Issues & Projects:**
- Use GitHub issues as task board for all development work
- Create GitHub Projects kanban board with columns: todo, selected for dev, doing, review, done
- Follow workflow: create issue → add to project → select for dev → implement → commit regularly → close issue

**Development Process:**
1. Create GitHub issue for feature/fix with detailed description and acceptance criteria
2. Add issue to GitHub project: `gh project item-add <project-id> --owner <username> --url <issue-url>`
3. Move to "selected for dev" and start implementation
4. Commit regularly with descriptive messages ending with Claude Code signature
5. Close issue when complete: `gh issue close <issue-number> --comment "Description of completion"`

**GitHub Issue Feedback Loop Workflow (Complex Tasks):**
For major architectural changes, refactors, or complex implementations:

1. **Issue Creation**: Create detailed GitHub issue with problem analysis, solution requirements, and implementation plan
2. **Progress Logging**: Use `gh issue comment <issue-number> --body "progress update"` throughout implementation to:
   - Log each major step attempted (Step 1: Analysis, Step 2: Implementation, etc.)
   - Document what works and what doesn't work
   - Record error messages, solutions tried, and outcomes
   - Keep running commentary of approach and decision-making process
3. **Real-time Documentation**: Update issue comments with:
   - Architectural decisions and reasoning
   - Code changes and their impact
   - Testing results and verification steps
   - Breakthrough moments and roadblocks encountered
4. **User Verification**: Do NOT close issue yourself - ask user to verify functionality first
5. **Final Closure**: Only close issue after user confirms success: `gh issue close <issue-number> --comment "Success confirmation"`

**Benefits of GitHub Issue Feedback Loop:**
- Creates permanent record of implementation process
- Allows user to follow progress in real-time
- Documents decision-making process for future reference
- Builds confidence through transparency
- Enables early course correction if approach is wrong
- Provides detailed post-mortem for successful implementations

**When to Use Issue Feedback Loop:**
- Architectural refactors (like the FastMCP migration)
- Major feature implementations
- Complex debugging sessions
- Performance optimization projects  
- Any task expected to take >30 minutes with multiple steps

## CRITICAL: When to STOP and Ask for User Feedback

**MANDATORY STOP Points - DO NOT PROCEED WITHOUT USER INPUT:**

1. **External Service Dependencies**
   - When encountering connection failures to external services (APIs, databases, etc.)
   - Example: "Cannot connect to host 172.17.0.1:4000" for embedding API
   - **STOP** and ask user about their specific service setup (localhost vs litellm.lan vs other)
   - Do NOT assume service locations or create new services without permission

2. **Deprecated/Changed Framework Features**  
   - When discovering deprecated features that affect architecture decisions
   - Example: FastMCP SSE transport deprecation discovery
   - **STOP** and ask user about their compatibility requirements before proceeding
   - Do NOT make assumptions about acceptable alternatives

3. **Infrastructure Configuration**
   - When encountering environment-specific configuration issues
   - Example: Docker host networking, service discovery, DNS resolution
   - **STOP** and ask user about their specific infrastructure setup
   - Do NOT assume standard configurations apply

4. **Large Dependency Changes**
   - When solutions require significant dependency additions (>500MB, new runtimes, etc.)
   - Example: Adding PyTorch for local embeddings (~2GB)
   - **STOP** and ask user about resource constraints and preferences
   - Do NOT make trade-offs without user input

**Pattern Recognition:**
- Connection timeouts → ASK about user's service setup
- Missing services → ASK about user's infrastructure  
- Deprecated features → ASK about user's compatibility needs
- Large downloads → ASK about user's resource constraints

**Example Proper Response:**
"I discovered the embedding API is trying to connect to 172.17.0.1:4000 but getting connection refused. This suggests you have a LiteLLM or similar service running somewhere else. Where is your embedding service located? (localhost, litellm.lan, or other host?)"

**Why This Matters:**
- Prevents wasted time on wrong assumptions
- Avoids creating unnecessary services/complexity  
- Respects user's existing infrastructure
- Enables targeted solutions based on actual setup

**Traefik Integration Workflow:**
- Use existing Traefik instance instead of adding new containers
- Check running containers: `docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}" | grep -i traefik`
- Remove port mappings from services, add Traefik labels instead:
  ```yaml
  labels:
    - "traefik.enable=true"
    - "traefik.http.routers.service-name.rule=Host(`service.domain.localhost`)"
    - "traefik.http.routers.service-name.entrypoints=web"
    - "traefik.http.services.service-name.loadbalancer.server.port=8000"
    - "traefik.docker.network=project-network"  # Tell Traefik which network to use
  ```
- Use `traefik.docker.network` label instead of adding external networks to avoid coupling
- Keep services in their own project network for isolation

## Second Instance Protocol

**Claude Code Second Instance Permissions:**
As a second instance of Claude Code assistant, the following protocol applies:

**ALLOWED Operations:**
- Add GitHub issues for tracking problems and solutions
- Research solutions and analyze code architecture
- Audit code for issues, improvements, or security concerns
- Read files and analyze codebase structure
- Provide recommendations and implementation suggestions
- Create documentation and analysis reports

**NOT ALLOWED Operations (unless EXPLICITLY granted permission):**
- Change existing files or write new code
- Perform destructive operations (delete files, containers, etc.)
- Modify configuration files or environment settings
- Execute commands that alter system state
- Make commits or deploy changes
- Install or uninstall dependencies

**Permission Request Process:**
If destructive or file-modifying operations are needed, explicitly request permission from the user before proceeding.