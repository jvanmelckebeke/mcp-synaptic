"""Configuration settings for MCP Synaptic."""

from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    SERVER_HOST: str = Field(default="localhost", description="Server host")
    SERVER_PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Database Configuration
    SQLITE_DATABASE_PATH: Path = Field(
        default=Path("./data/synaptic.db"), description="SQLite database path"
    )
    CHROMADB_PERSIST_DIRECTORY: Path = Field(
        default=Path("./data/chroma"), description="ChromaDB persistence directory"
    )

    # Redis Configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    REDIS_ENABLED: bool = Field(
        default=True, description="Enable Redis for distributed memory"
    )

    # Memory Configuration
    DEFAULT_MEMORY_TTL_SECONDS: int = Field(
        default=3600, description="Default memory TTL in seconds"
    )
    MAX_MEMORY_ENTRIES: int = Field(
        default=10000, description="Maximum number of memory entries"
    )
    MEMORY_CLEANUP_INTERVAL_SECONDS: int = Field(
        default=300, description="Memory cleanup interval in seconds"
    )

    # RAG Configuration
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-3-small", description="Embedding model name"
    )
    EMBEDDING_API_BASE: Optional[str] = Field(
        default=None, description="Embedding API base URL (e.g., http://localhost:4000)"
    )
    EMBEDDING_API_KEY: Optional[str] = Field(
        default=None, description="Embedding API key"
    )
    EMBEDDING_PROVIDER: str = Field(
        default="local", description="Embedding provider: 'local' or 'api'"
    )
    MAX_RAG_RESULTS: int = Field(
        default=10, description="Maximum RAG search results"
    )
    RAG_SIMILARITY_THRESHOLD: float = Field(
        default=0.7, description="RAG similarity threshold"
    )

    # SSE Configuration
    SSE_HEARTBEAT_INTERVAL: int = Field(
        default=30, description="SSE heartbeat interval in seconds"
    )
    SSE_MAX_CONNECTIONS: int = Field(
        default=100, description="Maximum SSE connections"
    )

    # MCP Protocol Configuration
    MCP_SERVER_NAME: str = Field(
        default="synaptic", description="MCP server name"
    )
    MCP_SERVER_VERSION: str = Field(
        default="0.1.0", description="MCP server version"
    )

    def load_from_file(self, config_path: Path) -> None:
        """Load settings from a configuration file."""
        if config_path.exists():
            # For now, we rely on pydantic-settings to load from .env
            # This method can be expanded to support other formats like TOML/YAML
            pass

    def create_directories(self) -> None:
        """Create necessary directories."""
        self.SQLITE_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CHROMADB_PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

    @property
    def database_url(self) -> str:
        """Get the SQLite database URL."""
        return f"sqlite:///{self.SQLITE_DATABASE_PATH}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()

    def __repr__(self) -> str:
        """String representation of settings."""
        return f"Settings(host={self.SERVER_HOST}, port={self.SERVER_PORT}, debug={self.DEBUG})"