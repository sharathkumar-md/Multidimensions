"""
build_figure_index.py
---------------------
One-time script: reads 01_ocr manifests and builds figure_index.json in 04_demo/.

Run from the repo root:
    python 04_demo/build_figure_index.py

The index maps:
    source_filename → { doc_id, figures_base, by_page: { "3": ["doc_id_p3_f0.png", ...] } }
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_DIR / "01_ocr" / "output" / "manifests"
FIGURES_DIR   = REPO_DIR / "01_ocr" / "output" / "figures"
OUTPUT_PATH   = Path(__file__).resolve().parent / "figure_index.json"

# Filename pattern: <doc_id>_p<page>_f<idx>.png
_FIG_RE = re.compile(r"^(.+)_p(\d+)_f\d+\.png$")


def build() -> dict:
    index: dict[str, dict] = {}

    if not MANIFESTS_DIR.exists():
        print(f"[WARN] Manifests dir not found: {MANIFESTS_DIR}")
        return index

    for manifest_file in sorted(MANIFESTS_DIR.glob("*.json")):
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[SKIP] {manifest_file.name}: {exc}")
            continue

        doc_id   = manifest.get("doc_id", "")
        src_name = manifest.get("source_filename") or manifest.get("source_pdf", "")

        if not doc_id or not src_name:
            print(f"[SKIP] {manifest_file.name}: missing doc_id or source_filename")
            continue

        figures_base = FIGURES_DIR / doc_id
        if not figures_base.exists():
            print(f"[WARN] No figures dir for {src_name} ({doc_id[:8]}...)")
            continue

        # Group figures by page number, keep only files above size threshold (50 KB)
        by_page: dict[str, list[dict]] = {}
        for fig in sorted(figures_base.glob("*.png")):
            m = _FIG_RE.match(fig.name)
            if not m:
                continue
            size_kb = fig.stat().st_size / 1024
            if size_kb < 1:            # skip sub-pixel icons / dots (< 1 KB)
                continue
            page_str = m.group(2)
            by_page.setdefault(page_str, []).append({
                "filename": fig.name,
                "size_kb":  round(size_kb, 1),
            })

        # Sort each page's figures by size descending (largest = most likely product photo)
        for page in by_page:
            by_page[page].sort(key=lambda x: x["size_kb"], reverse=True)

        index[src_name] = {
            "doc_id":       doc_id,
            "figures_base": str(figures_base.relative_to(REPO_DIR)).replace("\\", "/"),
            "by_page":      by_page,
        }
        total = sum(len(v) for v in by_page.values())
        print(f"[OK]  {src_name:<55} {total:>3} usable figures across {len(by_page)} pages")

    return index


def _render_page_fallback(src_name: str, doc_id: str, figures_base: Path) -> dict[str, list[dict]]:
    """
    Render every page of a PDF as a 150 DPI PNG when native image extraction
    found nothing.  Works for PDFs that embed vector art or full-page composites
    that fitz.get_images() cannot surface.
    """
    import fitz  # PyMuPDF — already a dependency of 01_ocr

    # Search for the source PDF in known locations
    search_dirs = [
        REPO_DIR / "data" / "input",
        REPO_DIR / "Brand Resources",
    ]
    pdf_path = None
    for d in search_dirs:
        candidate = d / src_name
        if candidate.exists():
            pdf_path = candidate
            break

    if pdf_path is None:
        print(f"  [SKIP render] PDF not found for {src_name}")
        return {}

    figures_base.mkdir(parents=True, exist_ok=True)
    by_page: dict[str, list[dict]] = {}

    try:
        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
        for page_num in range(1, doc.page_count + 1):
            page = doc[page_num - 1]
            pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            fig_name = f"{doc_id}_p{page_num}_rendered.png"
            fig_path = figures_base / fig_name
            pix.save(str(fig_path))
            size_kb = fig_path.stat().st_size / 1024
            by_page[str(page_num)] = [{"filename": fig_name, "size_kb": round(size_kb, 1)}]
        doc.close()
        total = sum(len(v) for v in by_page.values())
        print(f"  [RENDER] {src_name:<52} {total:>3} pages rendered at 150 DPI")
    except Exception as exc:
        print(f"  [ERROR render] {src_name}: {exc}")

    return by_page


def build_with_fallback() -> dict:
    """Like build(), but renders pages for docs that have an empty figures dir."""
    index = build()

    # Find docs that have a figures dir but 0 usable figures, and try page render
    if not MANIFESTS_DIR.exists():
        return index

    for manifest_file in sorted(MANIFESTS_DIR.glob("*.json")):
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        doc_id   = manifest.get("doc_id", "")
        src_name = manifest.get("source_filename") or manifest.get("source_pdf", "")
        if not doc_id or not src_name:
            continue

        entry = index.get(src_name, {})
        if entry.get("by_page"):          # already has figures — skip
            continue

        figures_base = FIGURES_DIR / doc_id
        by_page = _render_page_fallback(src_name, doc_id, figures_base)
        if by_page:
            # BUG-014: store as relative path (same as the main build() path) so
            # product_images.py's REPO_DIR / figures_base join works correctly.
            try:
                fb_rel = str(figures_base.relative_to(REPO_DIR)).replace("\\", "/")
            except ValueError:
                fb_rel = str(figures_base).replace("\\", "/")
            index[src_name] = {
                "doc_id":       doc_id,
                "figures_base": fb_rel,
                "by_page":      by_page,
            }

    return index


def main() -> None:
    print(f"Scanning manifests in: {MANIFESTS_DIR}")
    print(f"Scanning figures  in : {FIGURES_DIR}\n")

    index = build_with_fallback()
    OUTPUT_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")

    total_docs = len(index)
    total_figs = sum(
        sum(len(v) for v in entry["by_page"].values())
        for entry in index.values()
    )
    print(f"\nDone. {total_docs} documents, {total_figs} usable figures.")
    print(f"Index written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
