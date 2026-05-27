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

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45.0", "accelerate>=0.30.0",
    "bitsandbytes>=0.43.0", "sentence-transformers>=2.7.0", "numpy",
], check=True)

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).parent.parent
QA_PATH = REPO / "02_llm_eval/data/qa_set.json"
OCR_DIR = REPO / "data/ocr_output"
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO / "02_llm_eval/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

SYSTEM_PROMPT = (
    "You are a technical expert on GGB bearing and bushing products. "
    "Answer only from the context provided. Be precise with numbers and units. "
    "If something isn't in the context, say: 'Not found in the provided context.'"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_embed = None


def load_docs():
    hash_map = {}
    for f in (OCR_DIR / "manifests").glob("*.json"):
        m = json.loads(f.read_text())
        if m.get("source_filename"):
            hash_map[m["source_filename"]] = f.stem

    fig_re = re.compile(r"<!-- figure:.*?-->\n!\[.*?\]\(.*?\)\n\*Figure:.*?\*\n?", re.M)
    docs = {}
    for fn, h in hash_map.items():
        p = OCR_DIR / "markdown" / f"{h}.md"
        if p.exists():
            docs[fn] = fig_re.sub("", p.read_text()).strip()
    return docs


def embed_model():
    global _embed
    if _embed is None:
        _embed = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed


def keyword_recall(facts, answer):
    if not facts:
        return 1.0
    a = answer.lower()
    return sum(1 for f in facts if f.lower() in a) / len(facts)


def token_f1(expected, answer):
    def tok(t):
        return set(re.findall(r"\b[\w.]+\b", t.lower()))
    e, a = tok(expected), tok(answer)
    if not e or not a:
        return 0.0
    c = e & a
    p, r = len(c) / len(a), len(c) / len(e)
    return 2 * p * r / (p + r) if p + r else 0.0


def sem_sim(expected, answer):
    vecs = embed_model().encode([expected, answer], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


def score(expected, facts, answer):
    kr = keyword_recall(facts, answer)
    f1 = token_f1(expected, answer)
    ss = sem_sim(expected, answer)
    return {
        "keyword_recall": round(kr, 4),
        "token_f1": round(f1, 4),
        "semantic_sim": round(ss, 4),
        "composite": round(0.4 * kr + 0.3 * f1 + 0.3 * ss, 4),
    }


def run_model(model_id, questions, docs):
    print(f"\n{model_id}")
    print("-" * len(model_id))

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, device_map="auto", dtype=torch.bfloat16
    )
    model.eval()

    results = []
    for i, q in enumerate(questions):
        ctx = docs.get(q["source_doc"])
        if not ctx:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {q['question']}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                temperature=1.0, pad_token_id=tokenizer.eos_token_id,
            )
        ms = (time.perf_counter() - t0) * 1000

        answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        s = score(q["expected_answer"], q.get("key_facts", []), answer)

        results.append({
            "model": model_id, "id": q["id"],
            "category": q["category"], "difficulty": q["difficulty"],
            "question": q["question"], "expected": q["expected_answer"],
            "answer": answer, "latency_ms": round(ms), **s,
        })
        print(f"  {q['id']}  composite={s['composite']:.3f}  kw={s['keyword_recall']:.3f}  {ms:.0f}ms")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def print_results(all_results):
    by_model = defaultdict(list)
    for r in all_results:
        by_model[r["model"]].append(r)

    rows = []
    for mid, res in by_model.items():
        n = len(res)
        rows.append({
            "model": mid.split("/")[-1],
            "composite": round(sum(r["composite"] for r in res) / n, 3),
            "kw": round(sum(r["keyword_recall"] for r in res) / n, 3),
            "f1": round(sum(r["token_f1"] for r in res) / n, 3),
            "sem": round(sum(r["semantic_sim"] for r in res) / n, 3),
            "ms": round(sum(r["latency_ms"] for r in res) / n),
        })
    rows.sort(key=lambda x: x["composite"], reverse=True)

    print(f"\n{'Model':<35} {'Composite':>10} {'KW':>8} {'F1':>8} {'Sem':>8} {'ms/q':>8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:<35} {r['composite']:>10.3f} {r['kw']:>8.3f} {r['f1']:>8.3f} {r['sem']:>8.3f} {r['ms']:>7}ms")

    return rows


def save(all_results, summary):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(OUT_DIR / f"eval_{ts}.json", "w") as f:
        json.dump({"ts": ts, "summary": summary, "results": all_results}, f, indent=2)

    with open(OUT_DIR / f"eval_{ts}.csv", "w", newline="") as f:
        fields = ["model", "id", "category", "difficulty", "composite",
                  "keyword_recall", "token_f1", "semantic_sim", "latency_ms",
                  "question", "expected", "answer"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    print(f"\nSaved to {OUT_DIR}")


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    questions = json.loads(QA_PATH.read_text())["questions"]
    docs = load_docs()
    print(f"{len(questions)} questions, {len(docs)} docs loaded")

    embed_model()  # warm up before first model loads

    all_results = []
    for mid in MODELS:
        all_results.extend(run_model(mid, questions, docs))

    summary = print_results(all_results)
    save(all_results, summary)


if __name__ == "__main__":
    main()
