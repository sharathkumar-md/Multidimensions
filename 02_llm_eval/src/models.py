from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    APPLICATION = "application"
    REASONING = "reasoning"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QAPair(BaseModel):
    id: str
    question: str
    expected_answer: str
    key_facts: list[str] = Field(default_factory=list)
    source_doc: str
    category: QuestionCategory
    difficulty: Difficulty


class ModelResponse(BaseModel):
    model_name: str
    question_id: str
    question: str
    context_doc: str
    answer: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AnswerScores(BaseModel):
    keyword_recall: float
    token_f1: float
    semantic_similarity: float
    composite: float


class EvalResult(BaseModel):
    response: ModelResponse
    expected_answer: str
    scores: AnswerScores


class ModelSummary(BaseModel):
    model_name: str
    total_questions: int
    avg_keyword_recall: float
    avg_token_f1: float
    avg_semantic_similarity: float
    avg_composite: float
    avg_latency_ms: float
    results_by_category: dict[str, float] = Field(default_factory=dict)
    results_by_difficulty: dict[str, float] = Field(default_factory=dict)


class EvalReport(BaseModel):
    timestamp: str
    models_evaluated: list[str]
    total_questions: int
    summaries: list[ModelSummary]
    detailed_results: list[EvalResult]
