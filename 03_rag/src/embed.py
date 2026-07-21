from __future__ import annotations

from functools import lru_cache

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings

# bge-* retrieval models are trained to embed the query with this instruction prefix.
# Passages are embedded without it. Skipping it on queries measurably lowers recall.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=None)
def _load_embed_model(model_name: str, device: str) -> SentenceTransformer:
    logger.info(f"loading embed model: {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)


def get_embed_model(model_name: str | None = None, device: str | None = None) -> SentenceTransformer:
    """Return (and cache) the SentenceTransformer for the given model name.

    BUG-006: device now defaults to GPU when available instead of always
    forcing CPU.  Passing device="cpu" explicitly overrides this (useful
    when the LLM is already holding GPU memory during evaluation).
    """
    name = model_name or settings.embed_model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return _load_embed_model(name, device)


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    vecs = get_embed_model(model_name).encode(
        texts, show_progress_bar=False, normalize_embeddings=True
    )
    return vecs.tolist()


def embed_queries(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    name = model_name or settings.embed_model
    prepared = [_BGE_QUERY_INSTRUCTION + t for t in texts] if "bge" in name.lower() else texts
    vecs = get_embed_model(name).encode(
        prepared, show_progress_bar=False, normalize_embeddings=True
    )
    return vecs.tolist()
