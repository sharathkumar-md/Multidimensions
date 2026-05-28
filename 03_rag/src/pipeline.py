from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from config.settings import settings
from src.chunker import chunk_documents
from src.embed import get_embed_model
from src.evaluator import score_answer, _get_nli_model
from src.generator import build_prompt, generate, generate_raw
from src.indexer import build_index, load_index
from src.retriever import RetrievedChunk, load_reranker, retrieve


def load_qa_set(qa_path: Path) -> list[dict[str, str]]:
    """Load QA set from a JSON file with list of {question, answer} dicts."""
    with open(qa_path, encoding="utf-8") as f:
        qa = json.load(f)
    logger.info(f"Loaded {len(qa)} QA pairs from {qa_path}")
    return qa


def build_pipeline_index(ocr_output_dir: Path | None = None, index_dir: Path | None = None) -> None:
    chunks = chunk_documents(ocr_output_dir)
    build_index(chunks, index_dir)


def run_eval(
    qa_set: list[dict[str, str]],
    model,
    tokenizer,
    model_id: str,
    index_dir: Path | None = None,
    results_path: Path | None = None,
) -> list[dict[str, Any]]:
    collection, bm25, chunks = load_index(index_dir)
    reranker = load_reranker()
    embed_model = get_embed_model()
    nli_model = _get_nli_model()

    def hyde_fn(prompt: str, max_new_tokens: int = 120) -> str:
        return generate_raw(prompt, model, tokenizer, max_new_tokens=max_new_tokens)

    results: list[dict[str, Any]] = []

    for i, qa in enumerate(qa_set):
        question = qa["question"]
        reference = qa.get("answer", "")
        logger.info(f"[{i+1}/{len(qa_set)}] Q: {question[:80]}")

        retrieved = retrieve(
            query=question,
            collection=collection,
            bm25=bm25,
            chunks=chunks,
            reranker=reranker,
            generator_fn=hyde_fn if settings.hyde_enabled else None,
        )

        context_texts = [r.chunk.text for r in retrieved]
        answer = generate(question, context_texts, model, tokenizer, model_id)

        scores: dict = {}
        if reference:
            scores = score_answer(answer, reference, context_texts, embed_model, nli_model)

        record = {
            "question_id": f"q{i+1:03d}",
            "question": question,
            "answer": answer,
            "reference": reference,
            "sources": [
                {"chunk_id": r.chunk.chunk_id, "source_doc": r.chunk.source_doc,
                 "page_num": r.chunk.page_num, "score": r.score}
                for r in retrieved
            ],
            **scores,
        }
        results.append(record)
        logger.info(f"  composite={scores.get('composite', 'N/A')}")

        if results_path:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def query_once(
    question: str,
    model,
    tokenizer,
    model_id: str,
    index_dir: Path | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    collection, bm25, chunks = load_index(index_dir)
    reranker = load_reranker()

    def hyde_fn(prompt: str, max_new_tokens: int = 120) -> str:
        return generate_raw(prompt, model, tokenizer, max_new_tokens=max_new_tokens)

    retrieved = retrieve(
        query=question,
        collection=collection,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        generator_fn=hyde_fn if settings.hyde_enabled else None,
    )
    context_texts = [r.chunk.text for r in retrieved]
    answer = generate(question, context_texts, model, tokenizer, model_id)
    return answer, retrieved
