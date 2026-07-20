"""
Structured audit logging for the MultiDimensions RAG pipeline.

Every user interaction is recorded as a single-line JSON event to the audit log.
This provides:
    - Full query/response traceability per user and session
    - Performance monitoring (response latency)
    - Routing analytics (LOCAL vs WEB vs NONE usage)
    - Error tracking for post-incident investigation

The audit log is append-only JSONL. In production, ship it to BigQuery or
Cloud Logging for queryable analytics dashboards.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from config.settings import settings


def _ensure_log_dir() -> None:
    """Create audit log directory if it doesn't exist."""
    settings.audit_log_file.parent.mkdir(parents=True, exist_ok=True)


def log_audit_event(
    user_id: str,
    session_id: str,
    question: str,
    route: str,
    response_time_ms: int,
    success: bool,
    error_message: str = "",
) -> None:
    """
    Append a structured JSON audit event to the audit log file.

    Args:
        user_id: The authenticated user's email or identifier.
        session_id: The current chat session ID.
        question: The raw user question (truncated to 500 chars for log safety).
        route: The routing decision — LOCAL, WEB, or NONE.
        response_time_ms: Total wall-clock time in milliseconds.
        success: Whether the request completed without an unhandled error.
        error_message: Exception message if success=False.
    """
    _ensure_log_dir()

    event = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        # Truncate question to avoid bloating log files; full text is in conversation state
        "question": question[:500] if question else "",
        "route": route,
        "response_time_ms": response_time_ms,
        "success": success,
        "error_message": error_message[:500] if error_message else "",
    }

    try:
        with open(settings.audit_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Fallback to loguru if file write fails — never silently lose audit events
        logger.error(
            f"Failed to write audit log to '{settings.audit_log_file}': {exc}. "
            f"Audit event: {json.dumps(event)}"
        )
