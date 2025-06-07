# Docker Deployment Guide

This directory contains organized Docker Compose configurations and GHCR integration for MCP Synaptic deployment.

## Structure

```
docker/
├── docker-compose.yaml      # Base configuration (standard ports)
├── overrides/
│   ├── laptop.yaml         # Traefik HTTP entrypoint 
│   └── desktop.yaml        # Traefik WEB entrypoint
└── variants/
    ├── dev.yaml            # Development with Redis commander
    └── prod.yaml           # Production optimizations
```

## GHCR Package Usage

### Pull Pre-built Images

```bash
# Lightweight variant (API embeddings only)
docker pull ghcr.io/jvanmelckebeke/mcp-synaptic:latest

# Full variant (includes local embeddings with PyTorch)
docker pull ghcr.io/jvanmelckebeke/mcp-synaptic:latest-full

# Specific version
docker pull ghcr.io/jvanmelckebeke/mcp-synaptic:v1.0.0
```

### Quick Start with GHCR

```bash
# Run lightweight version
docker run -d \
  --name mcp-synaptic \
  -p 8000:8000 \
  -e EMBEDDING_API_BASE=http://your-litellm-server \
  -v $(pwd)/data:/app/data \
  ghcr.io/jvanmelckebeke/mcp-synaptic:latest

# Run full version with local embeddings
docker run -d \
  --name mcp-synaptic-full \
  -p 8000:8000 \
  -e EMBEDDING_PROVIDER=local \
  -v $(pwd)/data:/app/data \
  ghcr.io/jvanmelckebeke/mcp-synaptic:latest-full
```

## Docker Compose Usage

### Standard Deployment (with ports)
```bash
cd docker
docker-compose up -d
```
Access: http://localhost:8000

### Laptop (Traefik HTTP)
```bash
cd docker  
docker-compose -f docker-compose.yaml -f overrides/laptop.yaml up -d
```
Access: http://synaptic.localhost

### Desktop (Traefik WEB)
```bash
cd docker
docker-compose -f docker-compose.yaml -f overrides/desktop.yaml up -d  
```
Access: http://synaptic.localhost (via web entrypoint)

### Development Mode
```bash
cd docker
docker-compose -f docker-compose.yaml -f overrides/laptop.yaml -f variants/dev.yaml up -d
```
Features: Redis commander, source code mounting, debug mode

### Production Mode  
```bash
cd docker
docker-compose -f docker-compose.yaml -f overrides/desktop.yaml -f variants/prod.yaml up -d
```
Features: Optimized resources, always restart, production config

## Key Differences

### Base (docker-compose.yaml)
- Exposes port 8000:8000
- No Traefik labels
- Standard for most users

### Laptop Override
- Removes port exposure
- Traefik HTTP entrypoint (for laptop setup)

### Desktop Override  
- Removes port exposure
- Traefik WEB entrypoint (for desktop setup)

### Dev Variant
- Redis commander UI
- Source code volume mounting
- Debug command
- Reduced resource limits

### Prod Variant
- Pre-built image usage
- Always restart policy
- Optimized resource allocation
- Production environment config

## Image Variants

### Lightweight (`latest`)
- **Size**: ~200MB
- **Use case**: API-based embeddings (LiteLLM, OpenAI)
- **Dependencies**: Core Python packages only
- **Performance**: Fast startup, low memory usage

### Full (`latest-full`)
- **Size**: ~2GB  
- **Use case**: Local embeddings with sentence-transformers
- **Dependencies**: PyTorch, transformers, CUDA support
- **Performance**: Slower startup, higher memory usage, no external API needed

## Available Tags

- `latest` - Latest main branch (lightweight)
- `latest-full` - Latest main branch (full)
- `v1.2.3` - Semantic version releases
- `main` - Main branch builds
- `develop` - Development branch builds
- `pr-123` - Pull request builds

## GHCR Integration

### Update Compose to Use GHCR

Edit `docker-compose.yaml` to use pre-built images:

```yaml
services:
  mcp-synaptic:
    # Use GHCR image instead of building locally
    image: ghcr.io/jvanmelckebeke/mcp-synaptic:latest
    # Remove build context
    # build:
    #   context: ..
    #   dockerfile: Dockerfile
```

### Version Pinning

For production, pin to specific versions:

```yaml
services:
  mcp-synaptic:
    image: ghcr.io/jvanmelckebeke/mcp-synaptic:v1.2.3
```

## Environment Files

All configurations use environment files from `../envs/`:
- `synaptic.env` - Base configuration
- `synaptic.dev.env` - Development overrides
- `synaptic.prod.env` - Production overrides
- `redis.env` - Redis commander settings