"""Admin router — PDF upload, ingestion trigger, index stats."""

from __future__ import annotations

import os
import uuid

from api_config import api_settings
from auth import require_admin
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from loguru import logger
from models import IndexStats, IngestionStatus, UploadResponse, UserInfo
from rag_service import get_index_stats, get_ingestion_status, refresh_index, trigger_ingest

router = APIRouter(prefix="/api", tags=["Admin"])


@router.get(
    "/index/stats",
    response_model=IndexStats,
    summary="Get vector index statistics",
)
async def index_stats(_: UserInfo = Depends(require_admin)) -> IndexStats:
    """Returns chunk count, doc count, GPU status, and last rebuild time."""
    stats = get_index_stats()
    return IndexStats(
        n_chunks=stats["n_chunks"],
        n_docs=stats["n_docs"],
        last_updated=stats["last_updated"],
        gpu_available=stats["gpu_available"],
    )


@router.get(
    "/admin/ingest/status",
    response_model=IngestionStatus,
    summary="Get current ingestion job status",
)
async def ingest_status(_: UserInfo = Depends(require_admin)) -> IngestionStatus:
    """Poll this endpoint to track progress of an active ingestion job."""
    s = get_ingestion_status()
    return IngestionStatus(
        running=s["running"],
        progress=s["progress"],
        current_file=s["current_file"],
        error=s["error"],
    )


@router.post(
    "/admin/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF catalog and trigger ingestion",
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF catalog file"),
    user: UserInfo = Depends(require_admin),
) -> UploadResponse:
    """
    Accepts a PDF file, saves it to the upload directory, then triggers
    a background ingestion job that rebuilds the vector index.

    - Only PDF files accepted.
    - Enforces API_MAX_UPLOAD_SIZE_MB limit.
    - Returns immediately; poll /api/admin/ingest/status for progress.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    # Read and size-check
    content = await file.read()
    if len(content) > api_settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {api_settings.max_upload_size_mb} MB.",
        )

    # Fix 009: strip directory components to prevent path traversal.
    # e.g. filename="../../etc/cron.d/evil.pdf" → "evil.pdf"
    clean_filename = os.path.basename(file.filename or "upload.pdf")
    if not clean_filename.lower().endswith(".pdf"):
        clean_filename += ".pdf"

    # Save with a unique prefix to avoid collisions
    api_settings.upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{clean_filename}"
    dest = api_settings.upload_dir / safe_name

    # Paranoia check: ensure dest is still inside upload_dir after Path resolution
    if not dest.resolve().is_relative_to(api_settings.upload_dir.resolve()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    try:
        dest.write_bytes(content)
        logger.info(f"PDF saved: {dest} ({len(content):,} bytes) by {user.email}")
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        )

    # Kick off background ingestion (atomic check-and-set)
    started = trigger_ingest(dest)
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An ingestion job is already running. Wait for it to finish.",
        )

    return UploadResponse(
        filename=file.filename,
        size_bytes=len(content),
        message="Upload successful. Ingestion started in the background.",
    )


@router.post(
    "/admin/refresh",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Force-reload the vector index from disk",
)
async def refresh(_: UserInfo = Depends(require_admin)) -> None:
    """
    Reloads the Qdrant index from disk without restarting the server.
    Useful after a manual re-ingestion or if the index appears stale.
    """
    logger.info("Admin: force refresh index")
    refresh_index()
