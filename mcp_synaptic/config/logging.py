"""Logging configuration for MCP Synaptic."""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .settings import Settings


def setup_logging(settings: Optional[Settings] = None) -> None:
    """Set up structured logging with Rich formatting."""
    if settings is None:
        settings = Settings()

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True,
            ),
            logging.FileHandler(
                logs_dir / "synaptic.log",
                encoding="utf-8",
            ),
        ],
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.DEBUG else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Convenience function for module-level loggers
def get_module_logger(module_name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger for a specific module."""
    return get_logger(f"mcp_synaptic.{module_name}")


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get a logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


# Pre-configured loggers for common components
server_logger = get_module_logger("server")
memory_logger = get_module_logger("memory")
rag_logger = get_module_logger("rag")
sse_logger = get_module_logger("sse")
mcp_logger = get_module_logger("mcp")