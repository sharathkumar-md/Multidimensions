"""Health check router — no auth required."""
from __future__ import annotations

from fastapi import APIRouter, Response, status
from loguru import logger

from models import HealthResponse
from rag_service import get_index_stats, is_pipeline_loaded

router = APIRouter(tags=["Health"])


@router.get("/api/live", summary="Liveness probe")
async def liveness() -> dict:
    """Always returns 200 if the process is alive. Used for liveness checks."""
    return {"status": "alive"}


@router.get("/api/health", response_model=HealthResponse, summary="Service health check")
async def health(response: Response) -> HealthResponse:
    """
    Readiness probe.

    Returns 200 if the server is up AND RAG pipeline is loaded.
    Returns 503 if the RAG pipeline is not yet ready.
    """
    stats = get_index_stats()
    rag_loaded = is_pipeline_loaded()
    
    if not rag_loaded:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("Health check: RAG pipeline not loaded")
    
    return HealthResponse(
        status="ok" if rag_loaded else "degraded",
        version="0.1.0",
        rag_loaded=rag_loaded,
        gpu_available=stats["gpu_available"],
    )
