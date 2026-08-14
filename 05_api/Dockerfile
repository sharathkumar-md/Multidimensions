FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependencies first for better caching
COPY 05_api/requirements.txt /app/05_api/
# Ensure RAG requirements are also available if needed
COPY 03_rag/requirements.txt /app/03_rag/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/05_api/requirements.txt

# Copy source code
COPY 03_rag /app/03_rag
COPY 05_api /app/05_api

WORKDIR /app/05_api

# Environment variables for HuggingFace / PyTorch optimization
ENV HF_HOME=/app/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_XET=1 \
    TQDM_DISABLE=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
