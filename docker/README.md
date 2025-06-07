# Docker Compose Structure

This directory contains organized Docker Compose configurations for different deployment scenarios.

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

## Usage

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

## Environment Files

All configurations use environment files from `../envs/`:
- `synaptic.env` - Base configuration
- `synaptic.dev.env` - Development overrides
- `synaptic.prod.env` - Production overrides
- `redis.env` - Redis commander settings