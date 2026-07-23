"""Health check router — no auth required."""
from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

from models import HealthResponse
from rag_service import get_index_stats, is_pipeline_loaded

router = APIRouter(tags=["Health"])


@router.get("/api/health", response_model=HealthResponse, summary="Service health check")
async def health() -> HealthResponse:
    """
    Liveness + readiness probe.

    Returns 200 once the server is up, even if the RAG pipeline is still
    loading. Check `rag_loaded` to know if queries will work.
    """
    stats = get_index_stats()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        rag_loaded=is_pipeline_loaded(),
        gpu_available=stats["gpu_available"],
    )
