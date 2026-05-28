"""
Local interactive query loop.

Usage:
    cd 03_rag
    python run.py --model Qwen/Qwen2.5-7B-Instruct [--build-index]

Flags:
    --build-index   Rebuild the index from OCR output before querying
    --model MODEL   HuggingFace model ID to load (defaults to first in settings)
    --no-hyde       Disable HyDE query rewriting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from src.generator import load_model
from src.pipeline import build_pipeline_index, query_once


def _setup_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(settings.log_file, level="DEBUG")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG interactive query")
    parser.add_argument("--model", default=settings.models_to_evaluate[0])
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--no-hyde", action="store_true")
    args = parser.parse_args()

    _setup_logging()

    if args.no_hyde:
        settings.hyde_enabled = False

    if args.build_index:
        logger.info("Building index...")
        build_pipeline_index()
        logger.info("Index built")

    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)

    print(f"\nRAG ready. Model: {args.model}")
    print("Type your question (or 'exit' to quit):\n")

    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question or question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break

        answer, retrieved = query_once(question, model, tokenizer, args.model)

        print(f"\nA: {answer}\n")
        print("Sources:")
        for r in retrieved:
            print(f"  [{r.rank}] {r.chunk.source_doc} p{r.chunk.page_num} (score={r.score:.3f})")
        print()


if __name__ == "__main__":
    main()
