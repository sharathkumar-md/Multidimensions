"""
Kaggle runner: batch eval across all models with GPU/storage management.

Usage in a Kaggle notebook cell:
    %run rag_kaggle.py

Expects:
- OCR output in /kaggle/input/multidimensions-ocr/ocr_output/ (or set RAG_OCR_OUTPUT_DIR)
- QA set at /kaggle/input/multidimensions-ocr/qa_set.json
- HF_TOKEN secret for Gemma models
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch
from loguru import logger

# ── Kaggle paths ─────────────────────────────────────────────────────────────
HF_HOME = Path("/kaggle/working/hf_cache")
OCR_OUTPUT_DIR = Path(os.environ.get(
    "RAG_OCR_OUTPUT_DIR",
    "/kaggle/input/multidimensions-ocr/ocr_output",
))
QA_PATH = Path(os.environ.get(
    "RAG_QA_PATH",
    "/kaggle/input/multidimensions-ocr/qa_set.json",
))
INDEX_DIR = Path("/kaggle/working/index")
RESULTS_DIR = Path("/kaggle/working/results")

os.environ["HF_HOME"] = str(HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "hub")
HF_HOME.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(RESULTS_DIR / "rag_kaggle.log", level="DEBUG")

from config.settings import settings
from src.chunker import chunk_documents
from src.generator import delete_model_cache, load_model
from src.indexer import build_index, load_index
from src.pipeline import load_qa_set, run_eval


def _build_index_if_needed() -> None:
    chunks_file = INDEX_DIR / "chunks.json"
    if chunks_file.exists():
        logger.info("Index already exists, skipping build")
        return
    logger.info("Building index from OCR output...")
    chunks = chunk_documents(ocr_output_dir=OCR_OUTPUT_DIR)
    build_index(chunks, index_dir=INDEX_DIR)


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("GPU memory freed")


def _eval_model(model_id: str, qa_set: list[dict], summary: list[dict]) -> None:
    logger.info(f"{'='*60}")
    logger.info(f"Evaluating: {model_id}")
    logger.info(f"{'='*60}")

    safe_name = model_id.replace("/", "__")
    results_path = RESULTS_DIR / f"{safe_name}.json"

    if results_path.exists():
        logger.info(f"Results already exist for {model_id}, skipping")
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
        logger.info(f"Done: {model_id} | avg_composite={avg:.4f}")

    except Exception as e:
        logger.error(f"Failed {model_id}: {e}")
        summary.append({"model": model_id, "error": str(e)})

    finally:
        try:
            del model, tokenizer
        except NameError:
            pass
        _free_gpu()
        delete_model_cache(HF_HOME)


def main() -> None:
    logger.info("RAG Kaggle eval started")
    _build_index_if_needed()

    if not QA_PATH.exists():
        logger.error(f"QA set not found: {QA_PATH}")
        raise FileNotFoundError(QA_PATH)

    qa_set = load_qa_set(QA_PATH)
    models = settings.models_to_evaluate
    logger.info(f"Models to evaluate: {models}")

    summary: list[dict] = []
    for model_id in models:
        _eval_model(model_id, qa_set, summary)

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("EVAL SUMMARY")
    for row in sorted(summary, key=lambda x: x.get("avg_composite", 0), reverse=True):
        if "error" in row:
            logger.info(f"  {row['model']}: ERROR — {row['error']}")
        else:
            logger.info(f"  {row['model']}: {row['avg_composite']:.4f} ({row['n_questions']} Qs)")
    logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
