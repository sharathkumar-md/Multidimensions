from __future__ import annotations

import pytest

from src.evaluator import keyword_recall, token_f1, semantic_sim, score_answer


def test_keyword_recall_perfect():
    assert keyword_recall("the quick brown fox", "the quick brown fox") == 1.0


def test_keyword_recall_partial():
    score = keyword_recall("the quick fox", "the quick brown fox")
    assert 0.5 < score < 1.0


def test_keyword_recall_empty_reference():
    assert keyword_recall("anything", "") == 0.0


def test_token_f1_perfect():
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_no_overlap():
    assert token_f1("foo bar", "baz qux") == 0.0


def test_token_f1_partial():
    score = token_f1("the cat sat", "the cat lay")
    assert 0.0 < score < 1.0


def test_semantic_sim_identical(monkeypatch):
    import numpy as np

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            vecs = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            return vecs

    score = semantic_sim("anything", "anything", embed_model=_FakeModel())
    assert abs(score - 1.0) < 1e-5


def test_semantic_sim_orthogonal(monkeypatch):
    import numpy as np

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    score = semantic_sim("a", "b", embed_model=_FakeModel())
    assert abs(score) < 1e-5


def test_score_answer_composite_range(monkeypatch):
    import numpy as np
    from sentence_transformers import CrossEncoder

    class _FakeEmbed:
        def encode(self, texts, normalize_embeddings=True):
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    class _FakeNLI:
        def predict(self, pairs, apply_softmax=True):
            return [[0.1, 0.1, 0.8]] * len(pairs)

    scores = score_answer(
        answer="The bolt torque is 25 Nm.",
        reference="Torque specification is 25 Nm for the bolt.",
        context_chunks=["The bolt torque is 25 Nm as per spec sheet."],
        embed_model=_FakeEmbed(),
        nli_model=_FakeNLI(),
    )

    assert 0.0 <= scores["composite"] <= 1.0
    assert set(scores.keys()) == {"keyword_recall", "token_f1", "semantic_sim", "faithfulness", "composite"}
    expected = round(
        0.35 * scores["keyword_recall"]
        + 0.25 * scores["token_f1"]
        + 0.25 * scores["semantic_sim"]
        + 0.15 * scores["faithfulness"],
        4,
    )
    assert abs(scores["composite"] - expected) < 1e-4
