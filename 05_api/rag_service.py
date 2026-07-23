"""
RAG service — wraps the 03_rag pipeline for use from the FastAPI layer.

Key design decisions:
- Model + index are loaded ONCE at startup (expensive: ~30s on GPU, ~2min on CPU)
  and reused across all requests via module-level singletons.
- `stream_answer()` is an async generator that yields JSON-encoded SSE payloads.
- Web-search routing, HyDE, and reranking all happen inside the existing
  03_rag code — this layer just bridges async FastAPI ↔ sync pipeline.
- Uses asyncio.get_event_loop().run_in_executor() to run the blocking
  token-generation loop in a thread pool so the event loop stays responsive.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import threading
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from loguru import logger

# ── Add 03_rag to sys.path so we can import its modules ───────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAG_DIR = _PROJECT_ROOT / "03_rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.append(str(_RAG_DIR))

# ── Lazy imports — only resolved after sys.path is set ────────────────────────
_pipeline_loaded = False
_load_lock = threading.Lock()

_model = None
_tokenizer = None
_model_id: str = ""
_index_client = None
_index_chunks: list = []
_reranker = None
_embed_model = None
_gpu_available = False
_n_chunks = 0
_n_docs = 0
_last_updated: Optional[float] = None
_ingest_running = False
_ingest_progress = 0.0
_ingest_current_file: Optional[str] = None
_ingest_error: Optional[str] = None


def _do_load() -> None:
    """Blocking load — runs once in a background thread at startup."""
    global _model, _tokenizer, _model_id, _index_client, _index_chunks
    global _reranker, _embed_model, _gpu_available, _pipeline_loaded
    global _n_chunks, _n_docs, _last_updated

    try:
        import torch
        from config.settings import settings
        from src.embed import get_embed_model
        from src.indexer import load_index
        from src.retriever import load_reranker

        _gpu_available = torch.cuda.is_available()
        logger.info(f"GPU available: {_gpu_available}")

        logger.info("Loading embed model…")
        _embed_model = get_embed_model()

        logger.info("Loading reranker…")
        _reranker = load_reranker()

        logger.info("Loading vector index…")
        _index_client, _index_chunks = load_index(settings.index_dir)
        _n_chunks = len(_index_chunks)
        _n_docs = len({c.source_doc for c in _index_chunks})
        _last_updated = time.time()
        logger.info(f"Index loaded: {_n_chunks} chunks, {_n_docs} docs")

        logger.info("Loading LLM…")
        from src.generator import load_model
        _model, _tokenizer, _model_id = load_model()
        logger.info(f"LLM loaded: {_model_id}")

        _pipeline_loaded = True
        logger.info("RAG pipeline fully loaded ✓")

    except Exception as exc:
        logger.error(f"RAG pipeline load failed: {exc}")
        # Don't crash the API — health endpoint will report rag_loaded=False


def load_pipeline_async() -> None:
    """Kick off pipeline loading in a background thread (called at startup)."""
    t = threading.Thread(target=_do_load, daemon=True, name="rag-loader")
    t.start()


# ── Streaming answer ──────────────────────────────────────────────────────────

async def stream_answer(
    question: str,
    web_search: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-ready JSON strings.

    Each yielded string is valid JSON — parse with JSON.parse() on the client.
    Final event has done=True and includes sources, product_images, route.
    """
    from config.settings import settings
    from src.retriever import retrieve
    from src.web_retriever import web_retrieve
    from src.generator import stream_generate
    from src.indexer import find_product_images

    loop = asyncio.get_event_loop()

    # ── 1. Retrieve context (in thread pool — blocking) ───────────────────────
    def _retrieve():
        if not _pipeline_loaded:
            return [], [], "NONE"

        def hyde_fn(prompt: str, max_new_tokens: int = 120) -> str:
            from src.generator import generate_raw
            return generate_raw(prompt, _model, _tokenizer, max_new_tokens=max_new_tokens)

        # Try local index first
        local_chunks = retrieve(
            query=question,
            client=_index_client,
            chunks=_index_chunks,
            reranker=_reranker,
            embed_model=_embed_model,
            generator_fn=hyde_fn if settings.hyde_enabled else None,
        )

        # Decide routing
        route = "LOCAL"
        web_chunks = []
        if web_search and settings.web_search_enabled:
            try:
                web_chunks = web_retrieve(question)
                route = "WEB"
            except Exception as e:
                logger.warning(f"Web retrieval failed, using local: {e}")

        # Merge context: web first (more recent), then local
        all_chunks = web_chunks + local_chunks
        return all_chunks, local_chunks, route

    all_chunks, local_chunks, route = await loop.run_in_executor(None, _retrieve)

    # ── 2. Build context texts ────────────────────────────────────────────────
    context_texts = []
    for chunk in all_chunks:
        if hasattr(chunk, "chunk"):   # RetrievedChunk wrapper
            context_texts.append(chunk.chunk.text)
        elif hasattr(chunk, "text"):  # raw Chunk
            context_texts.append(chunk.text)
        else:
            context_texts.append(str(chunk))

    # ── 3. Stream tokens ──────────────────────────────────────────────────────
    if not _pipeline_loaded or not context_texts:
        # Graceful degradation: stream a single error token
        yield json.dumps({"token": "⚠️ The RAG pipeline is still loading or no context found. Please try again in a moment."})
        yield json.dumps({"done": True, "sources": [], "route": route})
        return

    token_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    def _generate_in_thread():
        try:
            from src.generator import stream_generate
            for token in stream_generate(question, context_texts, _model, _tokenizer, _model_id):
                asyncio.run_coroutine_threadsafe(token_queue.put(token), loop)
        except Exception as exc:
            logger.error(f"Generator error: {exc}")
            asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)
        finally:
            asyncio.run_coroutine_threadsafe(token_queue.put(None), loop)  # sentinel

    gen_thread = threading.Thread(target=_generate_in_thread, daemon=True, name="rag-gen")
    gen_thread.start()

    # Drain the token queue
    while True:
        token = await token_queue.get()
        if token is None:
            break
        yield json.dumps({"token": token})

    # ── 4. Build sources + images ─────────────────────────────────────────────
    sources = [
        {
            "source_doc": chunk.chunk.source_doc if hasattr(chunk, "chunk") else getattr(chunk, "source_doc", ""),
            "page_num": chunk.chunk.page_num if hasattr(chunk, "chunk") else getattr(chunk, "page_num", 0),
            "snippet": (chunk.chunk.text[:200] if hasattr(chunk, "chunk") else getattr(chunk, "text", ""))[:200],
        }
        for chunk in local_chunks[:5]
    ]

    try:
        images = find_product_images(local_chunks[:5])
        product_images = [
            {"image_path": img.image_path, "title": img.title, "source_doc": img.source_doc}
            for img in images
        ]
    except Exception as e:
        logger.warning(f"Failed to find product images: {e}")
        product_images = []

    yield json.dumps({
        "done": True,
        "sources": sources,
        "product_images": product_images,
        "route": route,
    })


# ── Index stats & ingestion ───────────────────────────────────────────────────

def get_index_stats() -> dict:
    return {
        "n_chunks": _n_chunks,
        "n_docs": _n_docs,
        "last_updated": _last_updated,
        "gpu_available": _gpu_available,
    }


def get_ingestion_status() -> dict:
    return {
        "running": _ingest_running,
        "progress": _ingest_progress,
        "current_file": _ingest_current_file,
        "error": _ingest_error,
    }


def is_pipeline_loaded() -> bool:
    return _pipeline_loaded


def refresh_index() -> None:
    """Reload the vector index from disk without restarting."""
    global _index_client, _index_chunks, _n_chunks, _n_docs, _last_updated
    if not _pipeline_loaded:
        return
    from config.settings import settings
    from src.indexer import load_index
    _index_client, _index_chunks = load_index(settings.index_dir)
    _n_chunks = len(_index_chunks)
    _n_docs = len({c.source_doc for c in _index_chunks})
    _last_updated = time.time()
    logger.info(f"Index refreshed: {_n_chunks} chunks, {_n_docs} docs")


def trigger_ingest(pdf_path: Path) -> None:
    """Run ingestion in a background thread."""
    global _ingest_running, _ingest_progress, _ingest_current_file, _ingest_error

    def _ingest():
        global _ingest_running, _ingest_progress, _ingest_current_file, _ingest_error
        _ingest_running = True
        _ingest_progress = 0.0
        _ingest_current_file = pdf_path.name
        _ingest_error = None
        try:
            from config.settings import settings
            from src.pipeline import build_pipeline_index
            logger.info(f"Starting ingestion of {pdf_path}")
            build_pipeline_index(
                ocr_output_dir=settings.ocr_output_dir,
                index_dir=settings.index_dir,
            )
            _ingest_progress = 1.0
            refresh_index()
            logger.info("Ingestion complete")
        except Exception as exc:
            _ingest_error = str(exc)
            logger.error(f"Ingestion error: {exc}")
        finally:
            _ingest_running = False
            _ingest_current_file = None

    t = threading.Thread(target=_ingest, daemon=True, name="rag-ingest")
    t.start()
