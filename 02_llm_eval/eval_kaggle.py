"""
LLM Evaluation — GGB Bearing Domain
Run this in Kaggle with GPU enabled (T4 x1).

Usage:
    python eval_kaggle.py

Steps it performs:
    1. Downloads 3 models from HuggingFace
    2. Runs each model on 25 domain questions from GGB bearing catalogs
    3. Scores answers (keyword recall + token F1 + semantic similarity)
    4. Saves results as CSV + JSON in /kaggle/working/
    5. Prints a leaderboard
"""
from __future__ import annotations

import csv
import gc
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Install dependencies ───────────────────────────────────────────────────────
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45.0",
    "accelerate>=0.30.0",
    "bitsandbytes>=0.43.0",
    "sentence-transformers>=2.7.0",
    "numpy",
], check=True)

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
QA_PATH   = REPO_ROOT / "02_llm_eval" / "data" / "qa_set.json"
OCR_DIR   = REPO_ROOT / "data" / "ocr_output"
OUT_DIR   = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT / "02_llm_eval" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Models to evaluate ────────────────────────────────────────────────────────
# All three are freely available on HuggingFace (no token required)
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",       # ~2 GB in 4-bit  — fast baseline
    "Qwen/Qwen2.5-7B-Instruct",        # ~4 GB in 4-bit  — best quality candidate
    "microsoft/Phi-3.5-mini-instruct", # ~2 GB in 4-bit  — strong technical reasoning
]

SYSTEM_PROMPT = (
    "You are a technical expert assistant for GGB bearing and bushing products.\n"
    "Answer questions precisely based ONLY on the provided document context.\n"
    "- Include specific numbers, units, and technical values exactly as they appear.\n"
    "- If the information is not in the context, say: \"Not found in the provided context.\"\n"
    "- Keep answers concise and factual."
)

_FIGURE_RE = re.compile(
    r"<!-- figure:.*?-->\n!\[.*?\]\(.*?\)\n\*Figure:.*?\*\n?",
    re.MULTILINE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_context_map() -> dict[str, str]:
    filename_to_hash: dict[str, str] = {}
    for mf in (OCR_DIR / "manifests").glob("*.json"):
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        fn = m.get("source_filename", "")
        if fn:
            filename_to_hash[fn] = mf.stem

    ctx: dict[str, str] = {}
    for fn, h in filename_to_hash.items():
        md = OCR_DIR / "markdown" / f"{h}.md"
        if md.exists():
            raw = md.read_text(encoding="utf-8")
            ctx[fn] = _FIGURE_RE.sub("", raw).strip()
    return ctx


def load_qa_pairs() -> list[dict]:
    with open(QA_PATH, encoding="utf-8") as f:
        return json.load(f)["questions"]


# ── Scoring ────────────────────────────────────────────────────────────────────
_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Sentence transformer loaded (all-MiniLM-L6-v2)")
    return _embed_model


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[\w.]+\b", text.lower()))


def token_f1(expected: str, actual: str) -> float:
    e, a = _tokenize(expected), _tokenize(actual)
    if not e or not a:
        return 0.0
    common = e & a
    p = len(common) / len(a)
    r = len(common) / len(e)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def keyword_recall(key_facts: list[str], actual: str) -> float:
    if not key_facts:
        return 1.0
    al = actual.lower()
    return sum(1 for kf in key_facts if kf.lower() in al) / len(key_facts)


def semantic_sim(expected: str, actual: str) -> float:
    embs = get_embed_model().encode([expected, actual], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def score_answer(expected: str, key_facts: list[str], actual: str) -> dict:
    kr  = keyword_recall(key_facts, actual)
    tf1 = token_f1(expected, actual)
    sem = semantic_sim(expected, actual)
    composite = 0.40 * kr + 0.30 * tf1 + 0.30 * sem
    return {
        "keyword_recall":      round(kr,        4),
        "token_f1":            round(tf1,       4),
        "semantic_similarity": round(sem,       4),
        "composite":           round(composite, 4),
    }


# ── Inference ──────────────────────────────────────────────────────────────────
def run_model(model_id: str, qa_pairs: list[dict], context_map: dict[str, str]) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  Model: {model_id}")
    print(f"{'='*60}")

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("  Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("  Downloading model (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    if DEVICE == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM used: {used:.2f} GB")

    results: list[dict] = []
    for i, qa in enumerate(qa_pairs):
        ctx = context_map.get(qa["source_doc"])
        if not ctx:
            print(f"  [{i+1:02d}/25] SKIP {qa['id']} — context not found")
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {qa['question']}"},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        scores = score_answer(qa["expected_answer"], qa.get("key_facts", []), answer)
        results.append({
            "model":    model_id,
            "id":       qa["id"],
            "category": qa["category"],
            "difficulty": qa["difficulty"],
            "question": qa["question"],
            "expected": qa["expected_answer"],
            "answer":   answer,
            "latency_ms": round(latency_ms),
            **scores,
        })
        print(
            f"  [{i+1:02d}/25] {qa['id']} | "
            f"composite={scores['composite']:.3f} | "
            f"kw={scores['keyword_recall']:.3f} | "
            f"{latency_ms:.0f}ms"
        )

    # Free GPU memory before loading the next model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    freed = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    print(f"  GPU freed | VRAM now: {freed:.2f} GB")

    return results


# ── Summary ────────────────────────────────────────────────────────────────────
def summarize(all_results: list[dict]) -> list[dict]:
    by_model: defaultdict[str, list] = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    summaries = []
    for model_id, rows in by_model.items():
        n = len(rows)
        by_cat:  defaultdict[str, list[float]] = defaultdict(list)
        by_diff: defaultdict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r["composite"])
            by_diff[r["difficulty"]].append(r["composite"])

        summaries.append({
            "model":          model_id,
            "n_questions":    n,
            "composite":      round(sum(r["composite"]           for r in rows) / n, 4),
            "keyword_recall": round(sum(r["keyword_recall"]      for r in rows) / n, 4),
            "token_f1":       round(sum(r["token_f1"]            for r in rows) / n, 4),
            "semantic_sim":   round(sum(r["semantic_similarity"]  for r in rows) / n, 4),
            "avg_latency_ms": round(sum(r["latency_ms"]          for r in rows) / n),
            "by_category":  {c: round(sum(s)/len(s), 3) for c, s in by_cat.items()},
            "by_difficulty": {d: round(sum(s)/len(s), 3) for d, s in by_diff.items()},
        })

    return sorted(summaries, key=lambda x: x["composite"], reverse=True)


def print_leaderboard(summaries: list[dict]) -> None:
    col = "{:<40} {:>10} {:>10} {:>10} {:>10} {:>10}"
    header = col.format("Model", "Composite", "KW Recall", "Token F1", "Semantic", "Latency")
    print(f"\n{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(col.format(
            s["model"].split("/")[-1],
            f"{s['composite']:.3f}",
            f"{s['keyword_recall']:.3f}",
            f"{s['token_f1']:.3f}",
            f"{s['semantic_sim']:.3f}",
            f"{s['avg_latency_ms']}ms",
        ))

    print("\n--- By category ---")
    for s in summaries:
        print(f"\n  {s['model'].split('/')[-1]}")
        for cat, val in sorted(s["by_category"].items()):
            print(f"    {cat:<15} {val:.3f}")

    print("\n--- By difficulty ---")
    for s in summaries:
        print(f"\n  {s['model'].split('/')[-1]}")
        for diff, val in sorted(s["by_difficulty"].items()):
            print(f"    {diff:<10} {val:.3f}")


def save_results(all_results: list[dict], summaries: list[dict]) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = OUT_DIR / f"eval_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "summaries": summaries, "results": all_results}, f, indent=2)
    print(f"\nJSON saved: {json_path}")

    csv_path = OUT_DIR / f"eval_{ts}.csv"
    fields = [
        "model", "id", "category", "difficulty",
        "composite", "keyword_recall", "token_f1", "semantic_similarity",
        "latency_ms", "question", "expected", "answer",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV saved:  {csv_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nLoading QA pairs and document contexts...")
    qa_pairs    = load_qa_pairs()
    context_map = load_context_map()
    print(f"  {len(qa_pairs)} questions | {len(context_map)} documents")

    print("\nPre-loading sentence transformer for scoring...")
    get_embed_model()

    all_results: list[dict] = []
    for model_id in MODELS:
        results = run_model(model_id, qa_pairs, context_map)
        all_results.extend(results)

    summaries = summarize(all_results)
    print_leaderboard(summaries)
    save_results(all_results, summaries)

    print(f"\nDone — {len(all_results)} answers scored across {len(MODELS)} models.")
    winner = summaries[0]["model"].split("/")[-1]
    print(f"Best model: {winner} (composite={summaries[0]['composite']:.3f})")


if __name__ == "__main__":
    main()
