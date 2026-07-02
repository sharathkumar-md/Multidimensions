from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient

from config.settings import settings
from src.chunker import Chunk

_RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


def _hyde_query(query: str, generator_fn) -> str:
    prompt = (
        "Write a short technical answer (2-3 sentences) to this question based on "
        "engineering documentation.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    try:
        hypo = generator_fn(prompt, max_new_tokens=120)
        logger.debug(f"hyde: {hypo[:80]}...")
        return hypo
    except Exception as e:
        logger.warning(f"hyde failed, using raw query: {e}")
        return query


def retrieve(
    query: str,
    client: QdrantClient,
    chunks: list[Chunk],
    reranker: CrossEncoder,
    top_k_dense: int | None = None,
    top_k_sparse: int | None = None,
    top_k_rerank: int | None = None,
    hyde_enabled: bool | None = None,
    generator_fn=None,
) -> list[RetrievedChunk]:
    top_k_dense = top_k_dense if top_k_dense is not None else settings.top_k_dense
    top_k_sparse = top_k_sparse if top_k_sparse is not None else settings.top_k_sparse
    top_k_rerank = top_k_rerank if top_k_rerank is not None else settings.top_k_rerank
    hyde_enabled = hyde_enabled if hyde_enabled is not None else settings.hyde_enabled

    retrieval_query = query
    if hyde_enabled and generator_fn is not None:
        retrieval_query = _hyde_query(query, generator_fn)

    # Qdrant with fastembed handles dense + sparse + fusion natively
    results = client.query(
        collection_name="rag_chunks",
        query_text=retrieval_query,
        limit=max(top_k_dense, top_k_sparse),
    )
    
    chunk_map = {c.chunk_id: c for c in chunks}
    retrieved = []
    
    for r in results:
        if isinstance(r.id, str) and r.id in chunk_map:
            retrieved.append(chunk_map[r.id])
        elif isinstance(r.id, int) and str(r.id) in chunk_map:
            retrieved.append(chunk_map[str(r.id)])
            
    if not retrieved:
        return []

    # cross-encoder reranking
    pairs = [[query, c.text] for c in retrieved]
    scores = reranker.predict(pairs)
    scored = list(zip(retrieved, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [
        RetrievedChunk(chunk=c, score=float(s), rank=i + 1)
        for i, (c, s) in enumerate(scored[:top_k_rerank])
    ]


def load_reranker(model_name: str | None = None) -> CrossEncoder:
    import torch
    model_name = model_name or settings.reranker_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"loading reranker: {model_name} on {device}")
    return CrossEncoder(model_name, device=device)
