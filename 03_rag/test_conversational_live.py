"""Live (GPU) smoke test for the conversational awareness pipeline.

Runs scripted multi-turn conversations through the REAL model so you can eyeball
the three new behaviours end to end:

  1. router      — does each turn trigger a catalog lookup? (greetings should not)
  2. rewriter    — are follow-ups like "its stroke?" rewritten into standalone queries?
  3. simple_reply— do no-retrieval turns get a natural chit-chat answer?

The rolling history fed to the rewriter is built exactly like 04_demo/app.py
(_update_summary: last 2 turns). This is CPU-infeasible — run it on Colab/Kaggle:

    !python /content/MultiDimensions/03_rag/test_conversational_live.py --model Qwen/Qwen3-8B

Add --full to also run retrieval + grounded generation (requires the built index).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.conversational import needs_retrieval, rewrite_query, simple_reply


# scripted dialogues: each is a list of user turns. The expectations are hints
# printed alongside the actual decision so you can spot regressions at a glance.
CONVERSATIONS = [
    {
        "name": "greeting then product, then follow-up",
        "turns": [
            ("hi there",                         "NO  (greeting → no lookup)"),
            ("what linear motors do you sell?",  "YES (product question)"),
            ("what's its stroke length?",        "YES + rewrite 'its' → the linear motor"),
        ],
    },
    {
        "name": "thanks / small talk",
        "turns": [
            ("what can you help me with?",       "NO  (meta question about you)"),
            ("thanks, that's helpful",           "NO  (small talk)"),
        ],
    },
    {
        "name": "industry then 'the same'",
        "turns": [
            ("which products fit the packaging industry?", "YES (industry)"),
            ("and what about the same for pharma?",        "YES + rewrite 'the same'"),
        ],
    },
]


def _update_summary(summary: str, question: str, answer: str) -> str:
    """Mirror 04_demo/app.py _update_summary: keep the last 2 turns."""
    answer_short = answer.replace("\n", " ")[:150]
    line = f"• {question[:70]} → {answer_short}"
    lines = [l for l in summary.split("\n") if l.strip()]
    lines.append(line)
    return "\n".join(lines[-2:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--full", action="store_true",
                        help="also run retrieval + grounded generation (needs the index)")
    args = parser.parse_args()
    model_id = args.model

    from src.generator import load_model
    print(f"=== loading {model_id} ===", flush=True)
    model, tokenizer = load_model(model_id)

    ask_full = None
    if args.full:
        # lazy import so the router/rewrite path works even without an index
        from src.indexer import load_index
        from src.retriever import load_reranker, retrieve
        from src.generator import generate
        from config.settings import settings
        index_dir = Path(__file__).parent / "index"
        collection, bm25, chunks = load_index(index_dir)
        reranker = load_reranker()

        def ask_full(query: str) -> str:
            retrieved = retrieve(query=query, collection=collection, bm25=bm25,
                                 chunks=chunks, reranker=reranker, generator_fn=None)
            ctx = [r.chunk.text[:900] for r in retrieved]
            return generate(query, ctx, model, tokenizer, model_id)

    for convo in CONVERSATIONS:
        print(f"\n{'=' * 70}\nCONVERSATION: {convo['name']}\n{'=' * 70}", flush=True)
        summary = ""
        for question, expectation in convo["turns"]:
            print(f"\nUSER: {question}", flush=True)
            print(f"  expect: {expectation}", flush=True)

            route = needs_retrieval(question, model, tokenizer, model_id)
            print(f"  router      -> {'RETRIEVE' if route else 'no-lookup'}", flush=True)

            if not route:
                answer = simple_reply(question, summary, model, tokenizer, model_id)
                print(f"  reply       -> {answer}", flush=True)
            else:
                rewritten = rewrite_query(question, summary, model, tokenizer, model_id)
                changed = "(unchanged)" if rewritten == question else "(REWRITTEN)"
                print(f"  rewrite {changed} -> {rewritten}", flush=True)
                if ask_full is not None:
                    answer = ask_full(rewritten)
                    print(f"  answer      -> {answer[:300]}", flush=True)
                else:
                    answer = rewritten  # feed the query into history when not generating

            summary = _update_summary(summary, question, answer)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
