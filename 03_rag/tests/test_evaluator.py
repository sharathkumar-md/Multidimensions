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


def test_groundedness_number_in_context():
    from src.evaluator import groundedness
    # answer number present in context -> well grounded
    g = groundedness("The torque is 25 Nm.", ["The bolt torque is 25 Nm as per spec sheet."])
    assert g > 0.8


def test_groundedness_invented_number():
    from src.evaluator import groundedness
    # answer quotes a number that is NOT in the context -> hallucination, low number score
    g = groundedness("The torque is 999 Nm.", ["The bolt torque is 25 Nm as per spec sheet."])
    assert g < 0.6


def test_groundedness_no_context():
    from src.evaluator import groundedness
    assert groundedness("anything", []) == 0.0


def test_score_answer_composite_range():
    import numpy as np

    class _FakeEmbed:
        def encode(self, texts, normalize_embeddings=True):
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    scores = score_answer(
        answer="The bolt torque is 25 Nm.",
        reference="Torque specification is 25 Nm for the bolt.",
        context_chunks=["The bolt torque is 25 Nm as per spec sheet."],
        embed_model=_FakeEmbed(),
    )

    assert 0.0 <= scores["composite"] <= 1.0
    assert set(scores.keys()) == {"keyword_recall", "token_f1", "semantic_sim", "groundedness", "composite"}
    expected = round(
        0.35 * scores["keyword_recall"]
        + 0.25 * scores["token_f1"]
        + 0.25 * scores["semantic_sim"]
        + 0.15 * scores["groundedness"],
        4,
    )
    assert abs(scores["composite"] - expected) < 1e-4
