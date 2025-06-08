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
2. **Memory Management** - Modular storage and operation handlers with TTL-based expiration
3. **RAG Database** - Modular document operations with ChromaDB and flexible embedding providers
4. **MCP Protocol** - Model Context Protocol implementation with tool registration
5. **SSE Communication** - Real-time event streaming for memory/RAG operations

**Component Initialization Flow:**
Settings → Memory Manager → RAG Database → SSE Server → MCP Handler → FastAPI App

**Critical Patterns:**
- **Async Context Managers**: All components use `async with` for resource management
- **Dependency Injection**: Components receive dependencies via constructor
- **Modular Architecture**: Domain-driven design with focused, single-responsibility modules
- **Delegation Pattern**: Core managers coordinate between specialized operation handlers
- **Background Tasks**: Memory cleanup and SSE heartbeats run as asyncio tasks
- **Error Propagation**: Custom exception hierarchy with structured logging

**Refactored Module Structure (as of Issues #17-#21):**
The codebase has been refactored from monolithic files into focused, domain-specific modules:
- `memory/storage/` - Storage backend implementations (base, sqlite, redis)
- `memory/manager/` - Memory operation handlers (core, operations, queries)
- `rag/embeddings/` - Embedding provider implementations (base, api, local, manager)
- `rag/database/` - RAG operation handlers (core, documents, search, stats)
- `utils/validation/` - Domain-specific validation modules (memory, documents, common)

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
- **Modular Provider System**: EmbeddingManager delegates to provider-specific implementations
- **Provider Selection**: Configuration-driven provider selection (api vs local)
- **Provider Interface**: Abstract base class ensures consistent provider APIs

## RAG Database Architecture

**Document Operations Structure:**
- **RAGDatabase** (`rag/database/core.py`) - Central coordinator with lifecycle management
- **DocumentOperations** (`rag/database/documents.py`) - Document CRUD operations handler
- **SearchOperations** (`rag/database/search.py`) - Search and similarity operations handler  
- **StatsOperations** (`rag/database/stats.py`) - Collection statistics operations handler
- **Delegation Pattern**: Core database coordinates between operation handlers

**ChromaDB Integration:**
- Persistent collection management with metadata and embedding storage
- Document lifecycle: add, get, update, delete with embedding generation
- Search functionality with similarity thresholds and metadata filtering
- Collection statistics with content analysis and embedding model tracking

## Memory System Design

**Storage Abstraction:**
- `MemoryStorage` protocol defines the interface (`memory/storage/base.py`)
- `SQLiteMemoryStorage` and `RedisMemoryStorage` implement backends (`memory/storage/sqlite.py`, `memory/storage/redis.py`)
- Backend selection based on `REDIS_ENABLED` configuration
- **Modular Storage Package**: Clean separation of storage implementations

**Memory Management Architecture:**
- **MemoryManager** (`memory/manager/core.py`) - Central coordinator with lifecycle management
- **MemoryOperations** (`memory/manager/operations.py`) - CRUD operations handler
- **MemoryQueries** (`memory/manager/queries.py`) - Query and utility operations handler
- **Delegation Pattern**: Core manager coordinates between operation handlers

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
- `tests/unit/` - Component isolation tests with modular package testing
- `tests/integration/` - Multi-component interaction tests (23 comprehensive tests)
- Async test support with `pytest-asyncio`
- Mock external dependencies (Redis, API calls, ChromaDB)

**Modular Testing Approach:**
- **Package Tests**: Test each refactored module's structure and interfaces
  - `test_storage_package.py` - Memory storage backend testing
  - `test_manager_package.py` - Memory manager operation handler testing
  - `test_embeddings_package.py` - Embedding provider testing
  - `test_database_package.py` - RAG database operation handler testing
- **Integration Tests**: Validate cross-module coordination
  - Memory system integration with real SQLite storage
  - RAG system integration with mocked ChromaDB and embeddings
  - Cross-system coordination (Memory ↔ RAG ↔ MCP Tools)

**Coverage Targets:**
- Memory management operations (CRUD, expiration, filtering, concurrency)
- RAG database operations (document lifecycle, search, statistics)
- Embedding provider switching (API vs local with delegation testing)
- Configuration validation and error handling
- Server lifecycle management and resource cleanup
- Cross-system error propagation and recovery

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

**Git Branching Workflow:**
- **Main Development Branch**: `dev` - primary development branch where all feature work merges
- **Feature Branches**: Use descriptive branch names with prefixes:
  - `feat/feature-name` - New features and major functionality
  - `fix/bug-description` - Bug fixes and patches  
  - `refactor/component-name` - Code refactoring and improvements
  - `docs/section-name` - Documentation updates
  - `test/feature-name` - Test additions and improvements

**Branch Workflow Process:**
1. **Start from dev**: Always branch off from the latest `dev` branch
2. **Create feature branch**: `git checkout -b feat/my-feature` 
3. **Implement changes**: Make commits with descriptive messages
4. **Create Pull Request**: Submit PR to merge feature branch into `dev`
5. **Auto-merge allowed**: Most PRs can be auto-merged after basic validation
6. **Delete feature branch**: Clean up after successful merge

**Pull Request Guidelines:**
- **Target**: All feature branches merge into `dev` branch
- **Title**: Use conventional commit format (feat: add new feature)
- **Description**: Include brief summary of changes and testing approach
- **Auto-merge**: Allowed for straightforward changes after validation
- **Review required**: Complex architectural changes should get user review

**Branch Protection:**
- `dev` branch serves as integration branch for ongoing development
- Feature branches are short-lived and deleted after merge
- Keep feature branches focused on single functionality
- Regular commits to feature branches for progress tracking

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
2. Implement core logic with proper async patterns in appropriate modular packages
3. Follow delegation pattern: core managers coordinate between operation handlers
4. Add MCP tool registration in `mcp/` modules if exposing to MCP clients
5. Add SSE events if real-time updates needed
6. Write comprehensive package tests for new modules
7. Write unit tests covering happy path and error cases
8. Update integration tests for cross-component interactions

**Component Dependencies:**
- All async operations must handle `asyncio.CancelledError`
- Database operations should use the storage abstraction layer
- External API calls need proper timeout and retry logic
- Memory operations should validate TTL and data size limits
- **Modular Design**: Follow single responsibility principle within packages
- **Interface Contracts**: Use abstract base classes for pluggable components
- **Error Handling**: Propagate errors through delegation chain properly

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
- Architectural refactors (like the Issues #17-#21 modular refactoring)
- Major feature implementations
- Complex debugging sessions
- Performance optimization projects  
- Any task expected to take >30 minutes with multiple steps

## Architectural Refactoring History

**Issues #17-#21 Modular Architecture Refactoring (Completed):**
A comprehensive refactoring that transformed monolithic files into focused, domain-specific modules:

**Refactoring Scope:**
- **1,608 lines** of monolithic code split into **25+ focused modules**
- **5 GitHub issues** systematically addressed with dependency analysis
- **272 unit tests** + **23 integration tests** ensuring system cohesion
- **No backward compatibility breaking** - all functionality preserved

**Refactored Modules:**
1. **utils/validation.py** (247 lines) → `utils/validation/` package (memory, documents, common)
2. **memory/storage.py** (430 lines) → `memory/storage/` package (base, sqlite, redis)
3. **rag/embeddings.py** (256 lines) → `rag/embeddings/` package (base, api, local, manager)
4. **memory/manager.py** (271 lines) → `memory/manager/` package (core, operations, queries)
5. **rag/database.py** (404 lines) → `rag/database/` package (core, documents, search, stats)

**Architectural Benefits:**
- **Single Responsibility**: Each module has a focused, well-defined purpose
- **Testability**: Individual components can be tested in isolation
- **Maintainability**: Changes are localized to specific functional areas
- **Extensibility**: New providers/handlers can be added without affecting existing code
- **Interface Contracts**: Abstract base classes ensure consistent APIs

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

## Multi-Instance Claude Code Workflow

**Self-Discovery System:**
Each Claude Code instance can automatically identify itself using the working directory:
```bash
pwd  # Shows: /path/to/mcp-synaptic-worktree/claude-1 or claude-2, etc.
```

**Instance-Specific Workflow:**
Each Claude instance operates with their own persistent branch (`claude-1`, `claude-2`, etc.) that serves as their personal development branch while coordinating with the main `dev` branch.

**Step-by-Step Workflow:**
1. **Sync with dev**: `git merge origin/dev --no-ff -m "Sync with latest dev changes"`
2. **Create feature branch FROM claude-#**: `git checkout -b feat/my-feature` 
3. **Implement changes**: Make commits with descriptive messages
4. **Push feature branch**: `git push -u origin feat/my-feature`
5. **Create Pull Request**: Submit PR to merge feature branch into `dev`
6. **After merge**: Sync back with `git merge origin/dev --no-ff`
7. **Cleanup**: `git branch -d feat/my-feature`

**Why This Works:**
- **Worktree Compatible**: Multiple Claude instances can't checkout `dev` simultaneously
- **Personal Base Branch**: Each `claude-#` branch acts as a stable working branch
- **Stay Current**: Regular merges from `dev` keep instances synchronized
- **Clean Features**: Feature branches created from updated `claude-#` branch
- **Proper Integration**: PRs merge features into `dev`, then sync back to `claude-#`

**Multi-Instance Coordination:**
- **Conflict-Free**: Each Claude works from their own base branch
- **Synchronized**: All instances stay current with dev through merges
- **Independent**: Multiple instances can work simultaneously without conflicts
- **Scalable**: Add new instances by creating new worktree directories

**Setup for New Instance:**
1. User creates new worktree: `git worktree add ../claude-X claude-X`
2. Claude runs: `pwd` to discover identity (`claude-X`)
3. Claude runs: `git merge origin/dev --no-ff` to sync
4. Claude proceeds with feature branch workflow from their `claude-X` branch

This system allows unlimited parallel Claude Code sessions while maintaining clean git workflow and proper dev branch integration.

## CRITICAL: Claude Code Development Lessons Learned

**NEVER MAKE THESE MISTAKES AGAIN:**

### 1. Git Workflow Violations
- **NEVER push claude-# branches to remote** - These are personal local branches only
- **ALWAYS create feature branches** for any code changes (feat/, fix/, refactor/, etc.)
- **ALWAYS follow the workflow**: claude-# → feat/branch → PR to dev → merge back to claude-#

### 2. Test-Before-Change Protocol
- **NEVER make code changes without testing first**
- **ALWAYS verify the change works** before committing
- **Use proper testing approach**: Docker for server changes, unit tests for logic
- **For CLI changes**: Test with Docker since that's the production deployment method

### 3. Critical Workflow Steps
1. **Before ANY code change**: Test current functionality to understand baseline
2. **Create feature branch**: `git checkout -b feat/descriptive-name`
3. **Make and test change**: Verify it works in appropriate environment
4. **Commit to feature branch**: With descriptive messages
5. **Push feature branch**: `git push -u origin feat/descriptive-name`
6. **Create PR**: Feature branch → dev
7. **After merge**: Sync back to claude-# branch

### 4. Emergency Recovery Protocol
If you accidentally push claude-# branch:
```bash
git push --delete origin claude-X  # Delete from remote immediately
```

### 5. Testing Methodology by Component
- **CLI changes**: Test with Docker deployment (`docker-compose up --build`)
- **Core logic**: Run unit tests (`python -m pytest`)
- **Server functionality**: Test with appropriate MCP client
- **Configuration**: Test with different environment setups

**Remember**: Excitement about fixes is good, but proper process prevents disasters!