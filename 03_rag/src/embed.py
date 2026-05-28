from __future__ import annotations

from functools import lru_cache

from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings


@lru_cache(maxsize=1)
def get_embed_model(model_name: str | None = None) -> SentenceTransformer:
    model_name = model_name or settings.embed_model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Embedding model ready")
    return model


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    model = get_embed_model(model_name)
    logger.debug(f"Embedding {len(texts)} texts")
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vecs.tolist()
