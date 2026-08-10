"""
Web retrieval module for the MultiDimensions RAG pipeline.

Uses DuckDuckGo Search (free, no API key required) to fetch web snippets
for queries that fall outside the local product catalog. Results are:
    1. Fetched with 24-hour disk caching to avoid redundant API calls.
    2. Sanitized to strip HTML and limit injection surface area.
    3. Reranked by a cross-encoder model for relevance.
    4. Returned as standard RetrievedChunk objects compatible with the
       local retrieval pipeline.

Security hardening:
    - Web snippets are HTML-stripped and truncated before LLM injection
      to limit prompt injection attack surface.
    - Cache keys are normalized to prevent duplicate calls from typos.
    - Source URLs are stored as-is but never executed or followed further.

When RAG_WEB_SEARCH_ENABLED=False, this module returns an empty list
and the generator falls back to the local catalog gracefully.
"""
from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

from diskcache import Cache
from duckduckgo_search import DDGS
from loguru import logger
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.chunker import Chunk
from src.retriever import RetrievedChunk

# Module-level cache instance — initialized lazily per settings
_cache: Optional[Cache] = None

# Thread pool for DuckDuckGo searches with timeout
_ddg_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ddg-search")
_DDG_TIMEOUT = 30.0  # seconds


def _get_cache() -> Cache:
    """Lazily initialize the disk cache using the path from settings."""
    global _cache
    if _cache is None:
        settings.web_cache_dir.mkdir(parents=True, exist_ok=True)
        _cache = Cache(str(settings.web_cache_dir))
    return _cache


def _sanitize_snippet(text: str, max_chars: int) -> str:
    """
    Strip HTML tags and truncate a web snippet before LLM injection.

    This is a lightweight defense against prompt injection: by stripping
    markup and capping length, we limit the LLM's exposure to malicious
    instructions that could be embedded in scraped web content.
    """
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate to configured max_chars
    return text[:max_chars]


def _build_cache_key(query: str, max_results: int) -> str:
    """
    Normalized, collision-resistant cache key.

    Normalization (lowercase + strip) prevents cache misses from minor
    capitalization or whitespace differences in the same logical query.
    """
    normalized = query.strip().lower()
    return f"ddg_v2_{normalized}_{max_results}"


def _fetch_duckduckgo_results(query: str, max_results: int) -> list[dict]:
    """
    Fetch search results from DuckDuckGo with disk-based caching and timeout.

    Returns a list of result dicts with keys: title, href, body.
    Returns an empty list on any error (fail-safe).
    """
    cache = _get_cache()
    cache_key = _build_cache_key(query, max_results)

    if cache_key in cache:
        logger.info(f"Web search cache hit for query: '{query[:80]}'")
        return cache[cache_key]

    logger.info(f"Web search cache miss — fetching: '{query[:80]}'")
    
    def _ddg_search() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    try:
        future = _ddg_executor.submit(_ddg_search)
        results = future.result(timeout=_DDG_TIMEOUT)

        cache.set(cache_key, results, expire=86_400)  # 24 hours TTL
        logger.info(f"Fetched {len(results)} web results for query: '{query[:80]}'")
        return results

    except FuturesTimeoutError:
        logger.error(f"DuckDuckGo search timed out after {_DDG_TIMEOUT}s for query: '{query[:80]}'")
        return []
    except Exception as exc:
        logger.error(
            f"DuckDuckGo search failed for query '{query[:80]}': {exc}",
            exc_info=True,
        )
        return []


def web_retrieve(
    query: str,
    reranker: CrossEncoder,
    top_k_rerank: Optional[int] = None,
    max_results: Optional[int] = None,
) -> list[RetrievedChunk]:
    """
    Retrieve and rerank web snippets relevant to the given query.

    Args:
        query: The standalone search query (already rewritten by conversational module).
        reranker: A loaded CrossEncoder for relevance scoring.
        top_k_rerank: Number of results to return after reranking (default: settings value).
        max_results: Max raw results to fetch from DuckDuckGo (default: settings value).

    Returns:
        A list of RetrievedChunk objects sorted by cross-encoder relevance score,
        or an empty list if web search is disabled or no results were found.
    """
    if not settings.web_search_enabled:
        logger.info("Web search is disabled (RAG_WEB_SEARCH_ENABLED=False).")
        return []

    top_k = top_k_rerank if top_k_rerank is not None else settings.top_k_rerank
    n_results = max_results if max_results is not None else settings.web_search_max_results
    snippet_max = settings.web_snippet_max_chars

    raw_results = _fetch_duckduckgo_results(query, n_results)

    if not raw_results:
        logger.warning("No web results returned — falling back to empty context.")
        return []

    # Build sanitized Chunk objects from raw DDG results
    chunks: list[Chunk] = []
    for res in raw_results:
        raw_body = res.get("body", "")
        title = _sanitize_snippet(res.get("title", ""), max_chars=200)
        body = _sanitize_snippet(raw_body, max_chars=snippet_max)
        href = res.get("href", "")

        if not body:
            continue

        text = f"Source URL: {href}\nTitle: {title}\nContent: {body}"
        chunk = Chunk(
            chunk_id=uuid.uuid4().hex,
            doc_hash="web_search",
            source_doc=href,
            page_num=0,
            text=text,
        )
        chunks.append(chunk)

    if not chunks:
        logger.warning("All web results had empty body — returning empty context.")
        return []

    # Cross-encoder reranking
    pairs = [[query, c.text] for c in chunks]
    try:
        scores = reranker.predict(pairs)
    except Exception as exc:
        logger.error(
            f"Cross-encoder reranking of web snippets failed: {exc}. "
            "Returning snippets in original fetch order.",
            exc_info=True,
        )
        # Graceful degradation: return unranked results rather than nothing
        return [
            RetrievedChunk(chunk=c, score=0.0, rank=i + 1)
            for i, c in enumerate(chunks[:top_k])
        ]

    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    logger.info(
        f"Web retrieval complete: {len(chunks)} candidates → "
        f"returning top {min(len(scored), top_k)} after reranking."
    )

    return [
        RetrievedChunk(chunk=c, score=float(s), rank=i + 1)
        for i, (c, s) in enumerate(scored[:top_k])
    ]
