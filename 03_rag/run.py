from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import settings
from src.generator import load_model
from src.pipeline import build_pipeline_index, query_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=settings.models_to_evaluate[0])
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--no-hyde", action="store_true")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(settings.log_file, level="DEBUG")

    if args.no_hyde:
        settings.hyde_enabled = False

    if args.build_index:
        build_pipeline_index()

    model, tokenizer = load_model(args.model)

    print(f"\nmodel: {args.model}  (exit to quit)\n")

    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in {"exit", "quit", "q"}:
            break

        answer, retrieved = query_once(question, model, tokenizer, args.model)

        print(f"\n{answer}\n")
        for r in retrieved:
            print(f"  [{r.rank}] {r.chunk.source_doc} p{r.chunk.page_num}  {r.score:.3f}")
        print()


if __name__ == "__main__":
    main()
