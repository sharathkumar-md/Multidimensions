from __future__ import annotations

import re

from loguru import logger

from src.models import AnswerScores

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence transformer loaded: all-MiniLM-L6-v2")
    return _embed_model


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[\w.]+\b", text.lower()))


def token_f1(expected: str, actual: str) -> float:
    exp_tokens = _tokenize(expected)
    act_tokens = _tokenize(actual)
    if not exp_tokens or not act_tokens:
        return 0.0
    common = exp_tokens & act_tokens
    precision = len(common) / len(act_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def keyword_recall(key_facts: list[str], actual: str) -> float:
    if not key_facts:
        return 1.0
    actual_lower = actual.lower()
    hits = sum(1 for kf in key_facts if kf.lower() in actual_lower)
    return hits / len(key_facts)


def semantic_similarity(expected: str, actual: str) -> float:
    import numpy as np

    model = _get_embed_model()
    embs = model.encode([expected, actual], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def score_answer(
    expected: str,
    key_facts: list[str],
    actual: str,
) -> AnswerScores:
    kr = keyword_recall(key_facts, actual)
    tf1 = token_f1(expected, actual)
    sem = semantic_similarity(expected, actual)
    composite = 0.40 * kr + 0.30 * tf1 + 0.30 * sem
    return AnswerScores(
        keyword_recall=round(kr, 4),
        token_f1=round(tf1, 4),
        semantic_similarity=round(sem, 4),
        composite=round(composite, 4),
    )
