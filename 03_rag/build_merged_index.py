"""Build one RAG index from multiple VLM OCR output dirs (original 18 + brand 6 = 24 docs).

Copies manifests + markdown from each source dir into a single merged dir (uuid filenames
don't collide), then chunks and indexes. Output overwrites 03_rag/index, which the demo reads.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

REPO = Path(__file__).parent.parent
SOURCES = [
    REPO / "data" / "ocr_output_vlm",
    REPO / "data" / "ocr_output_vlm_brand",
]
MERGED = Path(os.environ.get("RAG_MERGED_DIR", str(REPO / "data" / "ocr_output_vlm_all")))
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", str(Path(__file__).parent / "index")))


def _merge() -> None:
    if MERGED.exists():
        shutil.rmtree(MERGED)
    (MERGED / "manifests").mkdir(parents=True)
    (MERGED / "markdown").mkdir(parents=True)

    seen: set[str] = set()  # dedupe by source filename (same doc in two sets -> same chunk ids)
    total = 0
    for src in SOURCES:
        man = src / "manifests"
        if not man.exists():
            logger.warning(f"skip missing source: {src}")
            continue
        n = 0
        for f in sorted(man.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            name = d.get("source_filename", f.stem)
            if name in seen:
                logger.info(f"skip duplicate doc: {name} (from {src.name})")
                continue
            seen.add(name)
            shutil.copy2(f, MERGED / "manifests" / f.name)
            md_rel = (d.get("markdown_path") or "").replace("\\", "/")
            md_file = src / md_rel
            if md_file.exists():
                shutil.copy2(md_file, MERGED / "markdown" / md_file.name)
            n += 1
        total += n
        logger.info(f"merged {n} docs from {src.name}")
    logger.info(f"total merged docs: {total}")


def main() -> None:
    from src.chunker import chunk_documents
    from src.indexer import build_index

    _merge()
    chunks = chunk_documents(ocr_output_dir=MERGED)
    build_index(chunks, index_dir=INDEX_DIR)
    logger.info(f"merged index built at {INDEX_DIR}")


if __name__ == "__main__":
    main()
