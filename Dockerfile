# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including ffmpeg for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    VOXCPM_FORCE_CPU=1 \
    SERVER_PORT=1234 \
    SERVER_NAME=0.0.0.0 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

# Copy all files (needed for installation)
# Note: setuptools-scm needs source files, so we copy everything
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -e .

# Create directories for models and cache
RUN mkdir -p /app/models && \
    mkdir -p /root/.cache/huggingface && \
    mkdir -p /root/.cache/modelscope

# Expose port
EXPOSE 1234

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:1234/ || exit 1

# Run the application
CMD ["python3", "app.py"]

