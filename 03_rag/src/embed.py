from __future__ import annotations

from functools import lru_cache

from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings

# bge-* retrieval models are trained to embed the query with this instruction prefix.
# Passages are embedded without it. Skipping it on queries measurably lowers recall.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=None)
def _load_embed_model(model_name: str) -> SentenceTransformer:
    logger.info(f"loading embed model: {model_name}")
    return SentenceTransformer(model_name, device="cpu")


def get_embed_model(model_name: str | None = None) -> SentenceTransformer:
    return _load_embed_model(model_name or settings.embed_model)


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
