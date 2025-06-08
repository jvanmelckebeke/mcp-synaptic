"""Command-line interface for MCP Synaptic."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from .config.settings import Settings
from .core.server import SynapticServer

app = typer.Typer(
    name="mcp-synaptic",
    help="MCP Synaptic - Memory-enhanced MCP Server with RAG capabilities",
    add_completion=False,
)
console = Console()


@app.command("server")
def run_server(
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
) -> None:
    """Start the MCP Synaptic server."""
    try:
        settings = Settings()
        if config:
            settings.load_from_file(config)
        
        if debug:
            settings.DEBUG = True
            settings.LOG_LEVEL = "DEBUG"
        
        settings.SERVER_HOST = host
        settings.SERVER_PORT = port
        
        console.print(f"[green]Starting MCP Synaptic server on {host}:{port}[/green]")
        
        server = SynapticServer(settings)
        asyncio.run(server.run())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")
        sys.exit(1)


@app.command("init")
def init_project(
    directory: Path = typer.Argument(Path("."), help="Directory to initialize"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
) -> None:
    """Initialize a new MCP Synaptic project."""
    directory = directory.resolve()
    
    if not directory.exists():
        directory.mkdir(parents=True)
    
    config_file = directory / ".env"
    data_dir = directory / "data"
    
    if config_file.exists() and not force:
        console.print(f"[yellow]Configuration file already exists: {config_file}[/yellow]")
        console.print("Use --force to overwrite")
        return
    
    data_dir.mkdir(exist_ok=True)
    
    # Create basic configuration
    config_content = """# MCP Synaptic Configuration
SERVER_HOST=localhost
SERVER_PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Database paths
SQLITE_DATABASE_PATH=./data/synaptic.db
CHROMADB_PERSIST_DIRECTORY=./data/chroma

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
REDIS_ENABLED=false

# Memory settings
DEFAULT_MEMORY_TTL_SECONDS=3600
MAX_MEMORY_ENTRIES=10000
"""
    
    config_file.write_text(config_content)
    console.print(f"[green]Initialized MCP Synaptic project in {directory}[/green]")
    console.print(f"Configuration file: {config_file}")
    console.print(f"Data directory: {data_dir}")


@app.command("version")
def show_version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"MCP Synaptic version {__version__}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()