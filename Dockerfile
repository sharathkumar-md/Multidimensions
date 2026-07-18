# ─────────────────────────────────────────────────────────────────────────────
# MultiDimensions RAG — Production Dockerfile
#
# Multi-stage build:
#   builder  → installs all Python dependencies into /opt/venv
#   runtime  → lean image that copies only /opt/venv (no build tools)
#
# Build:  docker build -t multidimensions-rag .
# Run:    docker run --gpus 1 -p 8080:8080 multidimensions-rag
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install Python dependencies
COPY 03_rag/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install Python runtime (no build tools — keeps image lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ── Application Setup ─────────────────────────────────────────────────────────
WORKDIR /app

# Copy application source
COPY . .

# ── Environment Variables ─────────────────────────────────────────────────────
# Runtime behaviour
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HuggingFace optimizations
    HF_HUB_DISABLE_XET=1 \
    TQDM_DISABLE=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TOKENIZERS_PARALLELISM=false \
    # PyTorch memory management
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # Streamlit server config
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# ── Health Check ──────────────────────────────────────────────────────────────
# Cloud Run / Kubernetes readiness probe target
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
EXPOSE 8080
CMD ["python3.11", "-m", "streamlit", "run", "04_demo/app.py", \
     "--server.port", "8080", \
     "--server.headless", "true", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "false"]
