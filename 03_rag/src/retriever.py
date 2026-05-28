from __future__ import annotations

from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi
from loguru import logger
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.chunker import Chunk
from src.embed import embed_texts

_RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


def _rrf(ranked_lists: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def _hyde_query(query: str, generator_fn) -> str:
    """Generate a hypothetical answer to use as the retrieval query."""
    prompt = (
        "Generate a concise technical answer (2-3 sentences) to the following question. "
        "Focus on factual details that would appear in engineering documentation.\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    try:
        hypo = generator_fn(prompt, max_new_tokens=120)
        logger.debug(f"HyDE hypothesis: {hypo[:100]}...")
        return hypo
    except Exception as e:
        logger.warning(f"HyDE generation failed, falling back to raw query: {e}")
        return query


def retrieve(
    query: str,
    collection: chromadb.Collection,
    bm25: BM25Okapi,
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

    # --- Dense retrieval ---
    query_vec = embed_texts([retrieval_query])[0]
    dense_results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k_dense,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids: list[str] = dense_results["ids"][0]
    logger.debug(f"Dense retrieved {len(dense_ids)} chunks")

    # --- Sparse BM25 retrieval ---
    tokenized_query = retrieval_query.lower().split()
    scores_arr = bm25.get_scores(tokenized_query)
    chunk_id_list = [c.chunk_id for c in chunks]
    sparse_ranked = sorted(
        range(len(scores_arr)), key=lambda i: scores_arr[i], reverse=True
    )
    sparse_ids: list[str] = [chunk_id_list[i] for i in sparse_ranked[:top_k_sparse]]
    logger.debug(f"Sparse retrieved {len(sparse_ids)} chunks")

    # --- RRF fusion ---
    fused_scores = _rrf([dense_ids, sparse_ids])
    fused_ranked = sorted(fused_scores, key=fused_scores.get, reverse=True)

    chunk_map = {c.chunk_id: c for c in chunks}
    candidate_chunks = [chunk_map[cid] for cid in fused_ranked if cid in chunk_map]
    logger.debug(f"After RRF fusion: {len(candidate_chunks)} candidates")

    if not candidate_chunks:
        return []

    # --- Cross-encoder reranking ---
    pairs = [[query, c.text] for c in candidate_chunks]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

    results = [
        RetrievedChunk(chunk=c, score=float(s), rank=i + 1)
        for i, (c, s) in enumerate(ranked[:top_k_rerank])
    ]
    logger.debug(f"Reranked to top {len(results)} chunks")
    return results


def load_reranker(model_name: str | None = None) -> CrossEncoder:
    model_name = model_name or settings.reranker_model
    logger.info(f"Loading reranker: {model_name}")
    return CrossEncoder(model_name)
