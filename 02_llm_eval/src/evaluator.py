from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.context_loader import ContextLoader
from src.inference import run_inference
from src.models import EvalReport, EvalResult, ModelSummary, QAPair
from src.scorer import score_answer

_QA_PATH = Path(__file__).parent.parent / "data" / "qa_set.json"


class LLMEvaluator:
    def __init__(self) -> None:
        self._loader = ContextLoader()
        self._qa_pairs: list[QAPair] = self._load_qa_set()
        self._results_dir = settings.results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "LLMEvaluator ready | {n} questions | output={out}",
            n=len(self._qa_pairs),
            out=self._results_dir,
        )

    def _load_qa_set(self) -> list[QAPair]:
        with open(_QA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [QAPair(**q) for q in data["questions"]]

    def evaluate_model(self, model_name: str) -> list[EvalResult]:
        results: list[EvalResult] = []
        logger.info("=== Evaluating: {model} ===", model=model_name)

        for qa in self._qa_pairs:
            try:
                context = self._loader.get_context(qa.source_doc)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping {qid} — {err}", qid=qa.id, err=exc)
                continue

            response = run_inference(
                model_name=model_name,
                question_id=qa.id,
                question=qa.question,
                context=context,
                source_doc=qa.source_doc,
                ollama_host=settings.ollama_host,
            )
            scores = score_answer(qa.expected_answer, qa.key_facts, response.answer)
            results.append(
                EvalResult(
                    response=response,
                    expected_answer=qa.expected_answer,
                    scores=scores,
                )
            )
            logger.info(
                "{model} | {qid} | composite={c:.3f} | kw={k:.3f} | f1={f:.3f} | sem={s:.3f} | {lat:.0f}ms",
                model=model_name,
                qid=qa.id,
                c=scores.composite,
                k=scores.keyword_recall,
                f=scores.token_f1,
                s=scores.semantic_similarity,
                lat=response.latency_ms,
            )

        return results

    def _summarize(self, model_name: str, results: list[EvalResult]) -> ModelSummary:
        qa_map = {q.id: q for q in self._qa_pairs}
        by_cat: defaultdict[str, list[float]] = defaultdict(list)
        by_diff: defaultdict[str, list[float]] = defaultdict(list)

        for r in results:
            qa = qa_map.get(r.response.question_id)
            if qa:
                by_cat[qa.category.value].append(r.scores.composite)
                by_diff[qa.difficulty.value].append(r.scores.composite)

        n = len(results)
        if n == 0:
            return ModelSummary(
                model_name=model_name,
                total_questions=0,
                avg_keyword_recall=0.0,
                avg_token_f1=0.0,
                avg_semantic_similarity=0.0,
                avg_composite=0.0,
                avg_latency_ms=0.0,
                results_by_category={},
                results_by_difficulty={},
            )

        return ModelSummary(
            model_name=model_name,
            total_questions=n,
            avg_keyword_recall=round(sum(r.scores.keyword_recall for r in results) / n, 4),
            avg_token_f1=round(sum(r.scores.token_f1 for r in results) / n, 4),
            avg_semantic_similarity=round(
                sum(r.scores.semantic_similarity for r in results) / n, 4
            ),
            avg_composite=round(sum(r.scores.composite for r in results) / n, 4),
            avg_latency_ms=round(sum(r.response.latency_ms for r in results) / n, 1),
            results_by_category={
                cat: round(sum(s) / len(s), 4) for cat, s in by_cat.items()
            },
            results_by_difficulty={
                diff: round(sum(s) / len(s), 4) for diff, s in by_diff.items()
            },
        )

    def run(self, models: list[str] | None = None) -> EvalReport:
        models = models or settings.models_to_evaluate
        all_results: list[EvalResult] = []
        summaries: list[ModelSummary] = []

        for model in models:
            results = self.evaluate_model(model)
            if results:
                all_results.extend(results)
                summaries.append(self._summarize(model, results))

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            models_evaluated=models,
            total_questions=len(self._qa_pairs),
            summaries=summaries,
            detailed_results=all_results,
        )
        self._save_report(report)
        return report

    def _save_report(self, report: EvalReport) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self._results_dir / f"eval_{ts}.json"
        csv_path = self._results_dir / f"eval_{ts}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        qa_map = {q.id: q for q in self._qa_pairs}
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model", "question_id", "category", "difficulty",
                "keyword_recall", "token_f1", "semantic_sim", "composite",
                "latency_ms", "expected_answer", "actual_answer",
            ])
            for r in report.detailed_results:
                qa = qa_map.get(r.response.question_id)
                writer.writerow([
                    r.response.model_name,
                    r.response.question_id,
                    qa.category.value if qa else "",
                    qa.difficulty.value if qa else "",
                    r.scores.keyword_recall,
                    r.scores.token_f1,
                    r.scores.semantic_similarity,
                    r.scores.composite,
                    round(r.response.latency_ms),
                    r.expected_answer,
                    r.response.answer[:300],
                ])

        logger.info("Report saved → {j}", j=json_path)
        logger.info("CSV saved    → {c}", c=csv_path)

        self._print_summary(report)

    def _print_summary(self, report: EvalReport) -> None:
        header = f"\n{'Model':<30} {'Composite':>10} {'KW Recall':>10} {'Token F1':>10} {'Semantic':>10} {'Latency':>10}"
        separator = "-" * len(header)
        logger.info(header)
        logger.info(separator)
        for s in sorted(report.summaries, key=lambda x: x.avg_composite, reverse=True):
            logger.info(
                "{m:<30} {c:>10.3f} {k:>10.3f} {f:>10.3f} {s:>10.3f} {l:>9.0f}ms",
                m=s.model_name,
                c=s.avg_composite,
                k=s.avg_keyword_recall,
                f=s.avg_token_f1,
                s=s.avg_semantic_similarity,
                l=s.avg_latency_ms,
            )
