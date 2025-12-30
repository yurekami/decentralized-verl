# Decentralized veRL Docker Image
# Supports both CPU and CUDA environments

ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10

# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS cuda-base

# Or use Python base for CPU-only
FROM python:${PYTHON_VERSION}-slim AS cpu-base

# Build stage
FROM cuda-base AS builder

ARG PYTHON_VERSION=3.10

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project files
WORKDIR /app
COPY pyproject.toml README.md ./
COPY decentralized_verl/ ./decentralized_verl/

# Install dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[dev]"

# Runtime stage
FROM cuda-base AS runtime

ARG PYTHON_VERSION=3.10

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY --from=builder /app /app
COPY examples/ ./examples/
COPY tests/ ./tests/

# Create non-root user
RUN useradd -m -u 1000 dverl \
    && chown -R dverl:dverl /app
USER dverl

# Default port for DHT
EXPOSE 31337

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import decentralized_verl; print('ok')" || exit 1

# Default command shows help
CMD ["python", "-m", "decentralized_verl.cli.run_dht", "--help"]
