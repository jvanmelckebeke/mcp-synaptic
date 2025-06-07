# MCP Synaptic Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Create application user
RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Set work directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY README.md ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY mcp_synaptic ./mcp_synaptic

# Create data directories
RUN mkdir -p data logs && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uv", "run", "mcp-synaptic", "server", "--host", "0.0.0.0", "--port", "8000"]