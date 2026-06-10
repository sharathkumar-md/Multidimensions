"""Recompute groundedness + composite on already-generated answers, no GPU/model needed.

Each result record stores the answer and the retrieved chunk_ids; index/chunks.json maps
chunk_id -> text. We rebuild the exact context each answer saw, recompute groundedness,
and reuse the existing keyword_recall / token_f1 / semantic_sim (unaffected by the swap).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluator import groundedness

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
CHUNKS_FILE = HERE / "index" / "chunks.json"


def _load_chunk_text() -> dict[str, str]:
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return {c["chunk_id"]: c["text"] for c in chunks}


def rescore_file(path: Path, chunk_text: dict[str, str]) -> dict | None:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records or "answer" not in records[0]:
        return None

    composites = []
    for r in records:
        context = [chunk_text[s["chunk_id"]] for s in r.get("sources", []) if s["chunk_id"] in chunk_text]
        gr = round(groundedness(r["answer"], context), 4)
        r.pop("faithfulness", None)
        r["groundedness"] = gr
        kw, tf, ss = r.get("keyword_recall", 0), r.get("token_f1", 0), r.get("semantic_sim", 0)
        r["composite"] = round(0.35 * kw + 0.25 * tf + 0.25 * ss + 0.15 * gr, 4)
        composites.append(r["composite"])

    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    model = path.stem.replace("__", "/")
    avg = sum(composites) / len(composites)
    return {"model": model, "avg_composite": round(avg, 4), "n_questions": len(records)}


def main() -> None:
    chunk_text = _load_chunk_text()
    summary = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "summary.json":
            continue
        row = rescore_file(path, chunk_text)
        if row:
            summary.append(row)
            gr = json.loads(path.read_text(encoding="utf-8"))
            avg_gr = sum(r["groundedness"] for r in gr) / len(gr)
            print(f"{row['model']}: composite={row['avg_composite']:.4f} "
                  f"groundedness={avg_gr:.4f} (n={row['n_questions']})")

    summary.sort(key=lambda x: x["avg_composite"], reverse=True)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
