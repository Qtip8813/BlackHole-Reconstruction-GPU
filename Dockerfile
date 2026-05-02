# Multi-stage build for QRFT Black Hole Toolkit
# Stage 1: Builder - Install dependencies and build artifacts
FROM python:3.13-slim AS builder

WORKDIR /build

# Install system dependencies for scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and setup files
COPY qrft-blackhole-toolkit/requirements.txt .
COPY qrft-blackhole-toolkit/setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Minimal image with only runtime dependencies
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install only runtime libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    liblapack3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

# Copy application code
COPY qrft-blackhole-toolkit/ .

# Create output and data directories
RUN mkdir -p /app/output /app/eht_data

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/core:/app/gpu:/app/entropy:/app/pipeline:/app/viz

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from core import Q4StokesEncoder; Q4StokesEncoder()" || exit 1

# Default entrypoint
ENTRYPOINT ["python"]
CMD ["-m", "tests.test_synthetic"]
