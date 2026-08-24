from __future__ import annotations

import uuid
from dataclasses import dataclass

from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder
from fastembed import SparseTextEmbedding

from config.settings import settings
from src.chunker import Chunk


_RRF_K = 60

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"

_SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

_SPARSE_EMBEDDER = SparseTextEmbedding(
    model_name=_SPARSE_MODEL
)

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
        hypo = generator_fn(
            prompt,
            max_new_tokens=120,
        )

        logger.debug(
            f"hyde: {hypo[:80]}..."
        )

        return hypo

    except Exception as e:
        logger.warning(
            f"hyde failed, using raw query: {e}"
        )
        return query


def _rrf_merge(
    dense_results,
    sparse_results,
):
    """
    Reciprocal Rank Fusion.

    score(d) = sum(1 / (k + rank))
    """

    scores: dict[str, float] = {}

    for rank, result in enumerate(
        dense_results,
        start=1,
    ):
        point_id = str(result.id)

        scores[point_id] = scores.get(
            point_id,
            0.0,
        ) + 1.0 / (_RRF_K + rank)

    for rank, result in enumerate(
        sparse_results,
        start=1,
    ):
        point_id = str(result.id)

        scores[point_id] = scores.get(
            point_id,
            0.0,
        ) + 1.0 / (_RRF_K + rank)

    return scores


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

    top_k_dense = (
        top_k_dense
        if top_k_dense is not None
        else settings.top_k_dense
    )

    top_k_sparse = (
        top_k_sparse
        if top_k_sparse is not None
        else settings.top_k_sparse
    )

    top_k_rerank = (
        top_k_rerank
        if top_k_rerank is not None
        else settings.top_k_rerank
    )

    hyde_enabled = (
        hyde_enabled
        if hyde_enabled is not None
        else settings.hyde_enabled
    )

    # -------------------------------------------------------------
    # Optional HyDE
    # -------------------------------------------------------------

    retrieval_query = query

    if (
        hyde_enabled
        and generator_fn is not None
    ):
        retrieval_query = _hyde_query(
            query,
            generator_fn,
        )

    # -------------------------------------------------------------
    # Dense retrieval
    # -------------------------------------------------------------

    dense_result = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=models.Document(
            text=retrieval_query,
            model=settings.embed_model,
        ),
        using=_DENSE_VECTOR_NAME,
        limit=top_k_dense,
        with_payload=False,
    )

    dense_results = dense_result.points

    # -------------------------------------------------------------
    # Sparse retrieval
    # -------------------------------------------------------------

    sparse_embedding = next(
    _SPARSE_EMBEDDER.embed(
        [retrieval_query]
    )
)

    sparse_vector = models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist(),
    )

    sparse_result = client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=sparse_vector,
        using=_SPARSE_VECTOR_NAME,
        limit=top_k_sparse,
        with_payload=False,
    )

    sparse_results = sparse_result.points

    # -------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -------------------------------------------------------------

    rrf_scores = _rrf_merge(
        dense_results,
        sparse_results,
    )

    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True,
    )

    # Give reranker a larger candidate pool.
    candidate_ids = ranked_ids[
        : max(
            top_k_dense,
            top_k_sparse,
        ) * 2
    ]

    # -------------------------------------------------------------
    # Match Qdrant IDs back to Chunk objects
    # -------------------------------------------------------------

    chunk_map = {
        uuid.uuid5(
            uuid.NAMESPACE_DNS,
            chunk.chunk_id,
        ).hex: chunk
        for chunk in chunks
    }

    retrieved = []

    for point_id in candidate_ids:

        uuid_str = str(
            point_id
        ).replace(
            "-",
            "",
        )

        chunk = chunk_map.get(
            uuid_str
        )

        if chunk is not None:
            retrieved.append(chunk)

    if not retrieved:
        return []

    # -------------------------------------------------------------
    # Cross-encoder reranking
    # -------------------------------------------------------------

    pairs = [
        [query, chunk.text]
        for chunk in retrieved
    ]

    scores = reranker.predict(
        pairs
    )

    scored = list(
        zip(
            retrieved,
            scores,
        )
    )

    scored.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        RetrievedChunk(
            chunk=chunk,
            score=float(score),
            rank=rank,
        )
        for rank, (
            chunk,
            score,
        ) in enumerate(
            scored[:top_k_rerank],
            start=1,
        )
    ]


def load_reranker(
    model_name: str | None = None,
) -> CrossEncoder:

    import torch

    model_name = (
        model_name
        or settings.reranker_model
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info(
        f"Loading reranker '{model_name}' "
        f"on device '{device}'"
    )

    return CrossEncoder(
        model_name,
        device=device,
    )