from __future__ import annotations

import logging
import uuid
from typing import List

from diskcache import Cache
from duckduckgo_search import DDGS
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.chunker import Chunk
from src.retriever import RetrievedChunk

from pathlib import Path

logger = logging.getLogger(__name__)

_RAG_DIR = Path(__file__).resolve().parent.parent
# Initialize local disk cache for web searches (expires in 24 hours)
cache = Cache(str(_RAG_DIR / ".cache" / "web_search"))


def _fetch_duckduckgo_results(query: str, max_results: int = 10) -> list[dict]:
    """Fetch search results from DuckDuckGo, utilizing disk cache."""
    cache_key = f"ddg_{query}_{max_results}"
    if cache_key in cache:
        logger.info(f"Cache hit for web search query: '{query}'")
        return cache[cache_key]

    logger.info(f"Cache miss. Fetching web search results for query: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        # Cache for 24 hours (86400 seconds)
        cache.set(cache_key, results, expire=86400)
        return results
    except Exception as e:
        logger.error(f"Failed to fetch from DuckDuckGo: {e}", exc_info=True)
        return []


def web_retrieve(
    query: str,
    reranker: CrossEncoder,
    top_k_rerank: int | None = None,
) -> list[RetrievedChunk]:
    """Retrieve and rerank snippets from the web."""
    top_k_rerank = top_k_rerank if top_k_rerank is not None else settings.top_k_rerank
    
    # Fetch top 15 results to give reranker a good pool
    raw_results = _fetch_duckduckgo_results(query, max_results=15)
    
    if not raw_results:
        logger.warning("No results found from web search.")
        return []

    # Convert DDG results to Chunk objects
    chunks = []
    for res in raw_results:
        # DDG returns: 'title', 'href', 'body'
        body = res.get("body", "")
        title = res.get("title", "")
        href = res.get("href", "")
        if not body:
            continue
            
        text = f"Source URL: {href}\nTitle: {title}\nContent: {body}"
        chunk = Chunk(
            chunk_id=uuid.uuid4().hex,
            doc_hash="web_search",  # Dummy hash for web snippets
            source_doc=href,  # Use URL as source
            page_num=0,
            text=text,
        )
        chunks.append(chunk)

    # Cross-encoder reranking
    pairs = [[query, c.text] for c in chunks]
    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.error(f"Failed to rerank web snippets: {e}", exc_info=True)
        return []
        
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Web retrieval complete. Returning top {min(len(scored), top_k_rerank)} reranked chunks.")
    return [
        RetrievedChunk(chunk=c, score=float(s), rank=i + 1)
        for i, (c, s) in enumerate(scored[:top_k_rerank])
    ]
