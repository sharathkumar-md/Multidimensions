"""
Entrypoint for the LLM evaluation pipeline.

Run from the 02_llm_eval/ directory:

    python run.py                           # evaluate all configured models
    python run.py llama3.2:3b               # evaluate a single model
    python run.py llama3.2:3b qwen2.5:7b    # evaluate specific models

Prerequisites:
    1. Ollama running: `ollama serve`
    2. Models pulled: `ollama pull llama3.2:3b` etc.
    3. OCR pipeline output in ../01_ocr/output/ (run Phase 1 first)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    import sys

    from loguru import logger

    from config.settings import settings
    from src.evaluator import LLMEvaluator

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(log_path, level="DEBUG", rotation="10 MB", retention="30 days")

    models = sys.argv[1:] if len(sys.argv) > 1 else None
    evaluator = LLMEvaluator()
    report = evaluator.run(models=models)

    print(f"\n{'='*60}")
    print(f"Evaluation complete — {len(report.summaries)} model(s) evaluated")
    print(f"{'='*60}")
    for s in sorted(report.summaries, key=lambda x: x.avg_composite, reverse=True):
        print(
            f"  {s.model_name:<28} composite={s.avg_composite:.3f}"
            f"  kw={s.avg_keyword_recall:.3f}"
            f"  f1={s.avg_token_f1:.3f}"
            f"  sem={s.avg_semantic_similarity:.3f}"
            f"  {s.avg_latency_ms:.0f}ms/q"
        )
    print(f"\nResults saved to: {settings.results_dir}/")


if __name__ == "__main__":
    main()
