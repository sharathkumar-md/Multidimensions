FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for better caching
COPY 05_api/requirements.txt /app/05_api/
# Ensure RAG requirements are also available if needed
COPY 03_rag/requirements.txt /app/03_rag/

RUN pip install --no-cache-dir -r /app/05_api/requirements.txt

# Copy source code
COPY 03_rag /app/03_rag
COPY 05_api /app/05_api

WORKDIR /app/05_api

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
