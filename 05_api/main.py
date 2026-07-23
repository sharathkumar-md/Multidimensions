"""
MultiDimensions RAG — FastAPI Backend

Startup sequence:
  1. Validate config (raises on bad env vars)
  2. Create required directories
  3. Kick off RAG pipeline loading in a background thread
  4. Serve requests (health check returns immediately; chat waits for pipeline)

Run locally:
  cd 05_api
  uvicorn main:app --reload --port 8000

Or via Makefile:
  make dev
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# ── Bootstrap: add 03_rag to sys.path before any RAG imports ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAG_DIR = _PROJECT_ROOT / "03_rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.append(str(_RAG_DIR))

from api_config import api_settings                                    # noqa: E402
from routers import admin_router, chat_router, health_router, sessions_router  # noqa: E402
from rag_service import load_pipeline_async                         # noqa: E402


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown logic."""
    logger.info("MultiDimensions API starting up…")
    api_settings.ensure_directories()
    logger.info(
        f"Auth enabled: {api_settings.auth_enabled} | "
        f"CORS origins: {api_settings.cors_origins_list}"
    )
    # Initialize the database (creates tables if missing)
    from session_store import store
    await store.initialize()
    
    # Load RAG pipeline in background so the server responds immediately
    load_pipeline_async()
    yield
    logger.info("MultiDimensions API shut down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MultiDimensions RAG API",
    description=(
        "Streaming AI Q&A over industrial product catalogs. "
        "Backed by a Qwen3-8B LLM + Qdrant vector index."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(sessions_router)
app.include_router(chat_router)
app.include_router(admin_router)

# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."},
    )


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=api_settings.host,
        port=api_settings.port,
        reload=api_settings.reload,
        log_level=api_settings.log_level,
    )
