from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import torch
from loguru import logger
from sentence_transformers import CrossEncoder, SentenceTransformer

from config.settings import settings
from src.embed import get_embed_model


@lru_cache(maxsize=1)
def _get_nli_model(model_name: str | None = None) -> CrossEncoder:
    model_name = model_name or settings.nli_model
    logger.info(f"Loading NLI model: {model_name}")
    return CrossEncoder(model_name)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def keyword_recall(answer: str, reference: str) -> float:
    ref_tokens = set(_tokenize(reference))
    if not ref_tokens:
        return 0.0
    ans_tokens = set(_tokenize(answer))
    return len(ref_tokens & ans_tokens) / len(ref_tokens)


def token_f1(answer: str, reference: str) -> float:
    ans_tokens = Counter(_tokenize(answer))
    ref_tokens = Counter(_tokenize(reference))
    common = sum((ans_tokens & ref_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(ans_tokens.values())
    recall = common / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def semantic_sim(answer: str, reference: str, embed_model: SentenceTransformer | None = None) -> float:
    model = embed_model or get_embed_model()
    vecs = model.encode([answer, reference], normalize_embeddings=True)
    sim = float((vecs[0] * vecs[1]).sum())
    return max(0.0, sim)


def faithfulness(answer: str, context_chunks: list[str], nli_model: CrossEncoder | None = None) -> float:
    """
    Fraction of answer sentences entailed by the retrieved context.
    Label index 2 = entailment for nli-deberta-v3-small after softmax.
    """
    nli = nli_model or _get_nli_model()
    context = " ".join(context_chunks)
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    if not sentences:
        return 0.0

    pairs = [[context, s] for s in sentences]
    logits = nli.predict(pairs, apply_softmax=True)

    entailed = sum(1 for row in logits if row[2] > 0.5)
    score = entailed / len(sentences)
    logger.debug(f"Faithfulness: {entailed}/{len(sentences)} sentences entailed")
    return score


def score_answer(
    answer: str,
    reference: str,
    context_chunks: list[str],
    embed_model: SentenceTransformer | None = None,
    nli_model: CrossEncoder | None = None,
) -> dict[str, float]:
    kw = keyword_recall(answer, reference)
    tf = token_f1(answer, reference)
    ss = semantic_sim(answer, reference, embed_model)
    fa = faithfulness(answer, context_chunks, nli_model)
    composite = 0.35 * kw + 0.25 * tf + 0.25 * ss + 0.15 * fa
    scores = {
        "keyword_recall": round(kw, 4),
        "token_f1": round(tf, 4),
        "semantic_sim": round(ss, 4),
        "faithfulness": round(fa, 4),
        "composite": round(composite, 4),
    }
    logger.debug(f"Scores: {scores}")
    return scores
