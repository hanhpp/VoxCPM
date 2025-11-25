# Docker Setup for VoxCPM

This guide explains how to run VoxCPM using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. Build and run the container:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:1234`

### Option 2: Using Docker directly

1. Build the Docker image:
```bash
docker build -t voxcpm-app .
```

2. Run the container:
```bash
docker run -d \
  -p 1234:1234 \
  -e VOXCPM_FORCE_CPU=1 \
  -e SERVER_PORT=1234 \
  -e SERVER_NAME=0.0.0.0 \
  --name voxcpm-app \
  voxcpm-app
```

3. Access the application at `http://localhost:1234`

## Persisting Models and Cache

To avoid re-downloading models every time you restart the container, you can mount volumes:

### Using Docker Compose

The `docker-compose.yml` already includes volume mounts for:
- `./models` - Local models directory
- HuggingFace cache
- ModelScope cache

### Using Docker directly

```bash
docker run -d \
  -p 1234:1234 \
  -v $(pwd)/models:/app/models \
  -v voxcpm-hf-cache:/root/.cache/huggingface \
  -v voxcpm-ms-cache:/root/.cache/modelscope \
  -e VOXCPM_FORCE_CPU=1 \
  --name voxcpm-app \
  voxcpm-app
```

## Environment Variables

You can customize the container behavior using environment variables:

- `VOXCPM_FORCE_CPU=1` - Force CPU mode (required for Docker, as CUDA support requires special setup)
- `SERVER_PORT=1234` - Port to run the web interface on (default: 7860)
- `SERVER_NAME=0.0.0.0` - Server hostname (use 0.0.0.0 to accept connections from outside the container)
- `HF_REPO_ID` - HuggingFace repository ID for the model (default: "openbmb/VoxCPM-0.5B")

## Viewing Logs

### Docker Compose
```bash
docker-compose logs -f
```

### Docker directly
```bash
docker logs -f voxcpm-app
```

## Stopping the Container

### Docker Compose
```bash
docker-compose down
```

### Docker directly
```bash
docker stop voxcpm-app
docker rm voxcpm-app
```

## Building for GPU Support (Advanced)

If you have NVIDIA GPU support configured with nvidia-docker:

1. Modify the Dockerfile to remove `VOXCPM_FORCE_CPU=1`
2. Use `nvidia-docker` or `docker run --gpus all`:

```bash
docker run -d \
  --gpus all \
  -p 1234:1234 \
  -e SERVER_PORT=1234 \
  --name voxcpm-app \
  voxcpm-app
```

**Note:** The application will automatically fallback to CPU mode if GPU compatibility issues are detected (e.g., older GPUs with CUDA capability < 7.0).

## Troubleshooting

### Container exits immediately
- Check logs: `docker logs voxcpm-app`
- Ensure port 1234 is not already in use
- Verify models can be downloaded (check network connectivity)

### Models not persisting
- Ensure volume mounts are configured correctly
- Check volume permissions

### Out of memory
- The model requires significant memory (several GB)
- Increase Docker memory allocation in Docker Desktop settings
- Consider using CPU mode which requires less VRAM but more RAM

## Model Download

Models will be automatically downloaded on first run. This can take several minutes. You can also pre-download models by mounting a local models directory:

1. Download models locally first (see main README.md)
2. Mount the models directory as shown in the volume examples above

