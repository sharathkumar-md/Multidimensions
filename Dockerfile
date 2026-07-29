# ─────────────────────────────────────────────────────────────────────────────
# MultiDimensions RAG — DEMO / STREAMLIT Dockerfile
#
# ⚠️  THIS IMAGE RUNS THE STREAMLIT DEMO APP (04_demo/app.py).
# ⚠️  For the production FastAPI backend, use 05_api/Dockerfile
#     (referenced by docker-compose.yml under the `api` service).
#
# DO NOT deploy this image to production — it has no JWT auth, no rate
# limiting, and no database. It is for internal demos only.
#
# Build:  docker build -t multidimensions-demo .
# Run:    docker run --gpus 1 -p 8080:8080 multidimensions-demo
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

# ── Model Pre-warm ────────────────────────────────────────────────────────────
# Pre-download the embed model into the image so first-request latency is low.
# The LLM (Qwen3-8B, ~17 GB) is too large for the image — it is loaded at
# runtime from HuggingFace into the HF_HOME volume mount.
RUN python3.11 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-large-en-v1.5', cache_folder='/app/.cache/huggingface'); \
print('Embed model cached OK')"

# ── Environment Variables ─────────────────────────────────────────────────────
# Runtime behaviour
ENV PYTHONPATH=/app/03_rag \
    HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
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
