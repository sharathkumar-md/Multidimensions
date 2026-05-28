from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from loguru import logger
from sentence_transformers import CrossEncoder, SentenceTransformer

from config.settings import settings
from src.embed import get_embed_model


@lru_cache(maxsize=1)
def _get_nli_model(model_name: str | None = None) -> CrossEncoder:
    model_name = model_name or settings.nli_model
    logger.info(f"loading nli model: {model_name}")
    return CrossEncoder(model_name)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def keyword_recall(answer: str, reference: str) -> float:
    ref = set(_tokenize(reference))
    if not ref:
        return 0.0
    return len(ref & set(_tokenize(answer))) / len(ref)


def token_f1(answer: str, reference: str) -> float:
    a = Counter(_tokenize(answer))
    r = Counter(_tokenize(reference))
    common = sum((a & r).values())
    if common == 0:
        return 0.0
    p = common / sum(a.values())
    rec = common / sum(r.values())
    return 2 * p * rec / (p + rec)


def semantic_sim(answer: str, reference: str, embed_model: SentenceTransformer | None = None) -> float:
    model = embed_model or get_embed_model()
    vecs = model.encode([answer, reference], normalize_embeddings=True)
    return max(0.0, float((vecs[0] * vecs[1]).sum()))


def faithfulness(answer: str, context_chunks: list[str], nli_model: CrossEncoder | None = None) -> float:
    # label index 2 = entailment for nli-deberta-v3-small
    nli = nli_model or _get_nli_model()
    context = " ".join(context_chunks)
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    if not sentences:
        return 0.0
    logits = nli.predict([[context, s] for s in sentences], apply_softmax=True)
    entailed = sum(1 for row in logits if row[2] > 0.5)
    return entailed / len(sentences)


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
    return {
        "keyword_recall": round(kw, 4),
        "token_f1": round(tf, 4),
        "semantic_sim": round(ss, 4),
        "faithfulness": round(fa, 4),
        "composite": round(composite, 4),
    }
