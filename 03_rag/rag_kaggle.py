from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from loguru import logger

HF_HOME = Path(os.environ.get("RAG_HF_HOME", "/kaggle/working/hf_cache"))
OCR_OUTPUT_DIR = Path(os.environ.get("RAG_OCR_OUTPUT_DIR", "/kaggle/input/multidimensions-ocr/ocr_output"))
QA_PATH = Path(os.environ.get("RAG_QA_PATH", "/kaggle/input/multidimensions-ocr/qa_set.json"))
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", str(Path(__file__).parent / "index")))
RESULTS_DIR = Path(os.environ.get("RAG_RESULTS_DIR", "/kaggle/working/results"))

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "hub")
HF_HOME.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(RESULTS_DIR / "rag_kaggle.log", level="DEBUG")

from config.settings import settings
from src.chunker import chunk_documents
from src.generator import delete_model_cache, load_model
from src.indexer import build_index
from src.pipeline import load_qa_set, run_eval


def _build_index_if_needed() -> None:
    if (INDEX_DIR / "chunks.json").exists():
        logger.info("index exists, skipping build")
        return
    chunks = chunk_documents(ocr_output_dir=OCR_OUTPUT_DIR)
    build_index(chunks, index_dir=INDEX_DIR)


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _eval_model(model_id: str, qa_set: list[dict], summary: list[dict]) -> None:
    logger.info(f"--- {model_id} ---")

    safe_name = model_id.replace("/", "__")
    results_path = RESULTS_DIR / f"{safe_name}.json"

    if results_path.exists():
        logger.info(f"already done, skipping")
        existing = json.loads(results_path.read_text())
        composites = [r.get("composite", 0) for r in existing if "composite" in r]
        avg = sum(composites) / len(composites) if composites else 0
        summary.append({"model": model_id, "avg_composite": round(avg, 4), "n_questions": len(existing)})
        return

    try:
        model, tokenizer = load_model(model_id)
        results = run_eval(
            qa_set=qa_set,
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            index_dir=INDEX_DIR,
            results_path=results_path,
        )
        composites = [r.get("composite", 0) for r in results if "composite" in r]
        avg = sum(composites) / len(composites) if composites else 0
        summary.append({"model": model_id, "avg_composite": round(avg, 4), "n_questions": len(results)})
        logger.info(f"avg_composite={avg:.4f}")

    except Exception as e:
        logger.error(f"{model_id} failed: {e}")
        summary.append({"model": model_id, "error": str(e)})

    finally:
        try:
            del model, tokenizer
        except NameError:
            pass
        _free_gpu()
        delete_model_cache(HF_HOME, model_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="single model to eval; omit to run all")
    args = parser.parse_args()

    _build_index_if_needed()

    if not QA_PATH.exists():
        raise FileNotFoundError(f"qa set not found: {QA_PATH}")

    # warm up embed + reranker into lru_cache before any LLM occupies disk
    from src.embed import get_embed_model
    from src.retriever import load_reranker
    get_embed_model()
    load_reranker()

    qa_set = load_qa_set(QA_PATH)
    summary: list[dict] = []

    models = [args.model] if args.model else settings.models_to_evaluate
    for model_id in models:
        _eval_model(model_id, qa_set, summary)

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("results:")
    for row in sorted(summary, key=lambda x: x.get("avg_composite", 0), reverse=True):
        if "error" in row:
            logger.info(f"  {row['model']}: ERROR - {row['error']}")
        else:
            logger.info(f"  {row['model']}: {row['avg_composite']:.4f} ({row['n_questions']} Qs)")


if __name__ == "__main__":
    main()
