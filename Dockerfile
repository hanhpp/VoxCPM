# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for setuptools-scm (needed before pip install)
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Copy dependency files first for better caching
COPY pyproject.toml ./
# Copy source code (needed for package installation)
# Note: We copy source after pyproject.toml so dependency installation is cached separately
COPY src/ ./src/

# Install Python dependencies (this layer will be cached if dependencies don't change)
# Using regular install instead of editable for multi-stage builds
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Install only runtime system dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    VOXCPM_FORCE_CPU=1 \
    SERVER_PORT=1234 \
    SERVER_NAME=0.0.0.0 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 \
    PYTHONPATH=/app:$PYTHONPATH

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (this happens after dependency installation for better caching)
# Note: setuptools-scm needs source files, so we copy everything
COPY . .

# Create directories for models and cache (these will be mounted as volumes)
RUN mkdir -p /app/models && \
    mkdir -p /root/.cache/huggingface && \
    mkdir -p /root/.cache/modelscope

# Expose port
EXPOSE 1234

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:1234/ || exit 1

# Run the application
CMD ["python", "app.py"]

