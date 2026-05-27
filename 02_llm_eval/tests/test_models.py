import json
from pathlib import Path

import pytest

from src.models import (
    AnswerScores,
    Difficulty,
    EvalReport,
    EvalResult,
    ModelResponse,
    ModelSummary,
    QAPair,
    QuestionCategory,
)

_QA_PATH = Path(__file__).parent.parent / "data" / "qa_set.json"


class TestQAPair:
    def test_parse_valid(self):
        qa = QAPair(
            id="q001",
            question="What is the load?",
            expected_answer="250 N/mm²",
            key_facts=["250", "N/mm"],
            source_doc="test.pdf",
            category=QuestionCategory.FACTUAL,
            difficulty=Difficulty.EASY,
        )
        assert qa.id == "q001"
        assert qa.category == QuestionCategory.FACTUAL

    def test_empty_key_facts_default(self):
        qa = QAPair(
            id="q002",
            question="Q",
            expected_answer="A",
            source_doc="doc.pdf",
            category=QuestionCategory.APPLICATION,
            difficulty=Difficulty.HARD,
        )
        assert qa.key_facts == []

    def test_invalid_category_raises(self):
        with pytest.raises(Exception):
            QAPair(
                id="q003",
                question="Q",
                expected_answer="A",
                source_doc="doc.pdf",
                category="unknown",
                difficulty=Difficulty.EASY,
            )


class TestQASet:
    def test_qa_file_exists(self):
        assert _QA_PATH.exists(), "qa_set.json not found"

    def test_qa_set_loads(self):
        with open(_QA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        assert "questions" in data
        assert len(data["questions"]) >= 20

    def test_all_questions_parse(self):
        with open(_QA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        pairs = [QAPair(**q) for q in data["questions"]]
        assert len(pairs) == len(data["questions"])

    def test_unique_ids(self):
        with open(_QA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        ids = [q["id"] for q in data["questions"]]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"

    def test_all_categories_present(self):
        with open(_QA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        cats = {q["category"] for q in data["questions"]}
        expected = {"factual", "comparative", "application", "reasoning"}
        assert expected.issubset(cats)

    def test_all_difficulties_present(self):
        with open(_QA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        diffs = {q["difficulty"] for q in data["questions"]}
        assert {"easy", "medium", "hard"}.issubset(diffs)

    def test_key_facts_nonempty_for_factual(self):
        with open(_QA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for q in data["questions"]:
            if q["category"] == "factual":
                assert q.get("key_facts"), f"{q['id']} factual question has no key_facts"


class TestAnswerScores:
    def test_composite_range(self):
        scores = AnswerScores(
            keyword_recall=0.8,
            token_f1=0.6,
            semantic_similarity=0.7,
            composite=0.72,
        )
        assert 0.0 <= scores.composite <= 1.0

    def test_all_fields_present(self):
        scores = AnswerScores(
            keyword_recall=1.0,
            token_f1=1.0,
            semantic_similarity=1.0,
            composite=1.0,
        )
        assert scores.keyword_recall == 1.0


class TestModelResponse:
    def test_parse(self):
        mr = ModelResponse(
            model_name="llama3.2:3b",
            question_id="q001",
            question="What load?",
            context_doc="test.pdf",
            answer="250 N/mm²",
            latency_ms=1234.5,
        )
        assert mr.prompt_tokens == 0
        assert mr.latency_ms == 1234.5
