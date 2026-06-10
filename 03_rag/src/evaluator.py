from __future__ import annotations

import re
from collections import Counter

from sentence_transformers import SentenceTransformer

from src.embed import get_embed_model

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# common words we don't want to credit/penalise when checking grounding
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of",
    "for", "and", "or", "in", "on", "at", "by", "with", "from", "as", "it", "its",
    "this", "that", "these", "those", "has", "have", "had", "can", "will", "which",
    "up", "out", "about", "into", "over", "between", "range", "value", "values",
}


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


def groundedness(answer: str, context_chunks: list[str]) -> float:
    """Fraction of the answer's facts that actually appear in the retrieved context.
    Hallucination signal = 1 - groundedness. Built for tabular spec data: numbers and
    units, not abstract NLI entailment which fails on markdown tables.
    """
    if not context_chunks:
        return 0.0
    context = " ".join(context_chunks).lower()
    ctx_tokens = set(_tokenize(context))
    ctx_nums = set(_NUM_RE.findall(context))

    ans_lower = answer.lower()
    ans_nums = set(_NUM_RE.findall(ans_lower))
    # the headline signal: did the bot invent a number not in the catalog?
    num_score = len(ans_nums & ctx_nums) / len(ans_nums) if ans_nums else None

    ans_tokens = [t for t in _tokenize(ans_lower) if t not in _STOPWORDS and not t.isdigit()]
    tok_score = (
        sum(1 for t in ans_tokens if t in ctx_tokens) / len(ans_tokens)
        if ans_tokens else 1.0
    )

    if num_score is None:
        return tok_score
    return 0.5 * num_score + 0.5 * tok_score


def score_answer(
    answer: str,
    reference: str,
    context_chunks: list[str],
    embed_model: SentenceTransformer | None = None,
) -> dict[str, float]:
    kw = keyword_recall(answer, reference)
    tf = token_f1(answer, reference)
    ss = semantic_sim(answer, reference, embed_model)
    gr = groundedness(answer, context_chunks)
    composite = 0.35 * kw + 0.25 * tf + 0.25 * ss + 0.15 * gr
    return {
        "keyword_recall": round(kw, 4),
        "token_f1": round(tf, 4),
        "semantic_sim": round(ss, 4),
        "groundedness": round(gr, 4),
        "composite": round(composite, 4),
    }
