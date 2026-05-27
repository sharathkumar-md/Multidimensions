import pytest

from src.scorer import keyword_recall, score_answer, token_f1


class TestTokenF1:
    def test_identical_strings(self):
        assert token_f1("250 N/mm² static load", "250 N/mm² static load") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("alpha beta gamma", "delta epsilon zeta") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = token_f1("250 N/mm² dynamic load capacity", "The dynamic load is 250 N/mm²")
        assert 0.0 < score < 1.0

    def test_empty_expected(self):
        assert token_f1("", "some answer") == pytest.approx(0.0)

    def test_empty_actual(self):
        assert token_f1("expected answer here", "") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert token_f1("DU Bearing", "du bearing") == pytest.approx(1.0)

    def test_numeric_tokens_matched(self):
        score = token_f1("250 N/mm² 36000 psi", "250 N/mm² or 36 000 psi")
        assert score > 0.5


class TestKeywordRecall:
    def test_all_found(self):
        facts = ["250", "N/mm", "36000"]
        actual = "The static load is 250 N/mm² which equals 36000 psi"
        assert keyword_recall(facts, actual) == pytest.approx(1.0)

    def test_none_found(self):
        facts = ["HSG", "415", "60000"]
        actual = "The bearing does not have those specifications"
        assert keyword_recall(facts, actual) == pytest.approx(0.0)

    def test_partial_recall(self):
        facts = ["250", "N/mm", "36000"]
        actual = "The load is 250 N/mm²"
        score = keyword_recall(facts, actual)
        assert score == pytest.approx(2 / 3)

    def test_empty_facts_returns_one(self):
        assert keyword_recall([], "any answer") == pytest.approx(1.0)

    def test_case_insensitive(self):
        facts = ["Curiosity", "DU", "2011"]
        actual = "the curiosity rover used du bearings, launched in 2011"
        assert keyword_recall(facts, actual) == pytest.approx(1.0)

    def test_partial_string_match(self):
        facts = ["N/mm"]
        actual = "250 N/mm² static"
        assert keyword_recall(facts, actual) == pytest.approx(1.0)


class TestScoreAnswer:
    def test_perfect_answer(self):
        expected = "250 N/mm² static load capacity"
        key_facts = ["250", "N/mm"]
        actual = "The static load capacity is 250 N/mm²"
        scores = score_answer(expected, key_facts, actual)
        assert scores.keyword_recall == pytest.approx(1.0)
        assert 0.0 <= scores.composite <= 1.0
        assert 0.0 <= scores.semantic_similarity <= 1.0

    def test_wrong_answer_low_score(self):
        expected = "IT7 tolerance"
        key_facts = ["IT7"]
        actual = "Not found in the provided context."
        scores = score_answer(expected, key_facts, actual)
        assert scores.keyword_recall == pytest.approx(0.0)
        assert scores.composite < 0.5

    def test_composite_formula(self):
        expected = "HPMB bearings approved by Army Corps of Engineers"
        key_facts = ["HPMB", "Army Corps"]
        actual = "HPMB bearings have been approved by the USA Army Corps of Engineers"
        scores = score_answer(expected, key_facts, actual)
        expected_composite = 0.40 * scores.keyword_recall + 0.30 * scores.token_f1 + 0.30 * scores.semantic_similarity
        assert scores.composite == pytest.approx(expected_composite, abs=0.001)

    def test_scores_bounded(self):
        scores = score_answer("test answer here", ["test"], "completely different response")
        assert 0.0 <= scores.keyword_recall <= 1.0
        assert 0.0 <= scores.token_f1 <= 1.0
        assert -0.1 <= scores.semantic_similarity <= 1.1
        assert 0.0 <= scores.composite <= 1.1
