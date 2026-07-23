"""
ingest.py
---------
Unified automated ingestion orchestrator for the MultiDimensions project.

Scans "data/input/" and "Brand Resources/" for PDF files. If any new or updated
PDF is detected:
  1. Runs standard OCR (01_ocr) to extract figures.
  2. Runs VLM OCR (01.1_ocr_vlm) to extract structured markdown text.
  3. Rebuilds the figure index (04_demo/figure_index.json) mapping pages to images.
  4. Rebuilds the RAG vector store index (03_rag).

Run from the repository root:
    python 03_rag/ingest.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from loguru import logger

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))  # allows `from shared.hashing import sha256_file`

from shared.hashing import sha256_file as compute_sha256  # CODE-009: single source of truth


# ── isolated import helpers ──────────────────────────────────────────────────

def import_ocr_pipeline():
    """Import 01_ocr modules in an isolated namespace to avoid collisions."""
    old_path = list(sys.path)
    old_modules = dict(sys.modules)
    try:
        for k in list(sys.modules.keys()):
            if k.startswith("config") or k.startswith("src"):
                sys.modules.pop(k, None)
                
        sys.path.insert(0, str(REPO_DIR / "01_ocr"))
        from config.settings import settings as ocr_settings
        from src.pipeline import OCRPipeline
        return ocr_settings, OCRPipeline
    finally:
        sys.path = old_path
        sys.modules.update(old_modules)
        for k in list(sys.modules.keys()):
            if k not in old_modules and (k.startswith("config") or k.startswith("src")):
                sys.modules.pop(k, None)


def import_vlm_pipeline():
    """Import 01.1_ocr_vlm modules in an isolated namespace to avoid collisions."""
    old_path = list(sys.path)
    old_modules = dict(sys.modules)
    try:
        for k in list(sys.modules.keys()):
            if k.startswith("config") or k.startswith("src"):
                sys.modules.pop(k, None)
                
        sys.path.insert(0, str(REPO_DIR / "01.1_ocr_vlm"))
        from config.settings import settings as vlm_settings
        from src.pipeline import run as vlm_run
        return vlm_settings, vlm_run
    finally:
        sys.path = old_path
        sys.modules.update(old_modules)
        for k in list(sys.modules.keys()):
            if k not in old_modules and (k.startswith("config") or k.startswith("src")):
                sys.modules.pop(k, None)


def import_rag_pipeline():
    """Import 03_rag modules in an isolated namespace to avoid collisions."""
    old_path = list(sys.path)
    old_modules = dict(sys.modules)
    try:
        for k in list(sys.modules.keys()):
            if k.startswith("config") or k.startswith("src"):
                sys.modules.pop(k, None)
                
        sys.path.insert(0, str(REPO_DIR / "03_rag"))
        from src.pipeline import build_pipeline_index
        return build_pipeline_index
    finally:
        sys.path = old_path
        sys.modules.update(old_modules)
        for k in list(sys.modules.keys()):
            if k not in old_modules and (k.startswith("config") or k.startswith("src")):
                sys.modules.pop(k, None)


def import_demo_modules():
    """Import 04_demo modules in an isolated namespace to avoid collisions."""
    old_path = list(sys.path)
    old_modules = dict(sys.modules)
    try:
        for k in list(sys.modules.keys()):
            if k.startswith("config") or k.startswith("src"):
                sys.modules.pop(k, None)
                
        sys.path.insert(0, str(REPO_DIR / "04_demo"))
        from build_figure_index import build_with_fallback, OUTPUT_PATH
        return build_with_fallback, OUTPUT_PATH
    finally:
        sys.path = old_path
        sys.modules.update(old_modules)
        for k in list(sys.modules.keys()):
            if k not in old_modules and (k.startswith("config") or k.startswith("src") or k.startswith("build_")):
                sys.modules.pop(k, None)


def import_captioning_module():
    """Import 01.2_vlm_captioning module in an isolated namespace."""
    old_path = list(sys.path)
    old_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(REPO_DIR / "01.2_vlm_captioning"))
        from caption_figures import run_captioning
        return run_captioning
    finally:
        sys.path = old_path
        sys.modules.update(old_modules)
        for k in list(sys.modules.keys()):
            if k not in old_modules and (k.startswith("config") or k.startswith("src") or k.startswith("caption_")):
                sys.modules.pop(k, None)

# CODE-009: compute_sha256 is now imported from shared/hashing.py above


def get_done_hashes_ocr() -> set[str]:
    """Get the set of SHA-256 hashes already processed by 01_ocr."""
    hashes = set()
    manifests_dir = REPO_DIR / "01_ocr" / "output" / "manifests"
    if manifests_dir.exists():
        for f in manifests_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if sha := data.get("sha256"):
                    hashes.add(sha)
            except Exception as e:
                logger.warning(f"Failed to read manifest {f}: {e}")
    return hashes


def get_done_hashes_vlm() -> set[str]:
    """Get the set of SHA-256 hashes already processed by 01.1_ocr_vlm."""
    hashes = set()
    manifests_dir = REPO_DIR / "data" / "ocr_output_vlm" / "manifests"
    if manifests_dir.exists():
        for f in manifests_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if sha := data.get("sha256"):
                    hashes.add(sha)
            except Exception as e:
                logger.warning(f"Failed to read manifest {f}: {e}")
    return hashes


# ── main process ─────────────────────────────────────────────────────────────

def main() -> None:
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}", colorize=True)
    logger.info("Starting MultiDimensions Unified Ingestion Pipeline...")

    # 1. Discover all PDF files from both data/input/ and Brand Resources/
    search_dirs = [
        REPO_DIR / "data" / "input",
        REPO_DIR / "Brand Resources",
    ]
    
    all_pdfs: list[Path] = []
    for d in search_dirs:
        if d.exists():
            all_pdfs.extend(d.rglob("*.pdf"))

            
    # Deduplicate by filename (if same file exists in both, keep first one found)
    seen_names = set()
    pdfs = []
    for p in all_pdfs:
        if p.name not in seen_names:
            seen_names.add(p.name)
            pdfs.append(p)

    if not pdfs:
        logger.warning("No PDF files found in data/input/ or Brand Resources/. Exiting.")
        return

    logger.info(f"Discovered {len(pdfs)} total PDF files.")

    # 2. Check which ones need normal OCR and/or VLM OCR
    done_ocr = get_done_hashes_ocr()
    done_vlm = get_done_hashes_vlm()

    to_run_ocr: list[Path] = []
    to_run_vlm: list[Path] = []

    for pdf in pdfs:
        sha = compute_sha256(pdf)
        if sha not in done_ocr:
            to_run_ocr.append(pdf)
        if sha not in done_vlm:
            to_run_vlm.append(pdf)

    # 3. Execute normal OCR if needed
    ocr_processed = False
    if to_run_ocr:
        logger.info(f"Running normal OCR on {len(to_run_ocr)} files for figure extraction...")
        ocr_settings, OCRPipeline = import_ocr_pipeline()
        ocr_settings.output_dir = REPO_DIR / "01_ocr" / "output"
        ocr_settings.max_workers = 1  # Force sequential to avoid Streamlit BrokenPipeError
        
        ocr_pipeline = OCRPipeline()
        ocr_pipeline.run(pdf_paths=to_run_ocr)
        ocr_processed = True
    else:
        logger.info("All PDFs already processed by standard OCR.")

    # 4. Execute VLM OCR if needed
    vlm_processed = False
    if to_run_vlm:
        logger.info(f"Running VLM OCR on {len(to_run_vlm)} files for structured text markdown...")
        vlm_settings, vlm_run = import_vlm_pipeline()
        vlm_settings.output_dir = REPO_DIR / "data" / "ocr_output_vlm"
        
        vlm_run(pdf_paths=to_run_vlm)
        vlm_processed = True
    else:
        logger.info("All PDFs already processed by VLM OCR.")

    # 5. Rebuild figure index if any OCR extraction occurred
    if ocr_processed or vlm_processed:
        logger.info("New OCR documents processed. Rebuilding figure index...")
        build_with_fallback, OUTPUT_PATH = import_demo_modules()
        index = build_with_fallback()
        OUTPUT_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
        logger.info(f"Figure index updated successfully at: {OUTPUT_PATH}")

        # 5.5 Run VLM Image Captioning
        logger.info("Running VLM to generate image captions...")
        run_captioning = import_captioning_module()
        run_captioning()

        # 6. Rebuild RAG Vector Store
        logger.info("Rebuilding RAG vector store index...")
        build_pipeline_index = import_rag_pipeline()
        build_pipeline_index(
            ocr_output_dir=REPO_DIR / "data" / "ocr_output_vlm",
            index_dir=REPO_DIR / "03_rag" / "index"
        )
        logger.info("RAG vector store index updated successfully.")
    else:
        # Check if figure index or RAG index are missing entirely and rebuild them if so
        fig_index_path = REPO_DIR / "04_demo" / "figure_index.json"
        rag_index_dir = REPO_DIR / "03_rag" / "index"
        
        if not fig_index_path.exists():
            logger.info("Figure index missing. Building...")
            build_with_fallback, OUTPUT_PATH = import_demo_modules()
            index = build_with_fallback()
            OUTPUT_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
            
        captions_path = REPO_DIR / "04_demo" / "figure_captions.json"
        need_rag_rebuild = False
        if not captions_path.exists():
            logger.info("Figure captions missing. Building...")
            run_captioning = import_captioning_module()
            run_captioning()
            need_rag_rebuild = True

        # BUG-003: check for Qdrant's chunks.json (not the removed ChromaDB artifact)
        if not (rag_index_dir / "chunks.json").exists() or need_rag_rebuild:
            logger.info("RAG vector store missing or needs update. Building...")
            build_pipeline_index = import_rag_pipeline()
            build_pipeline_index(
                ocr_output_dir=REPO_DIR / "data" / "ocr_output_vlm",
                index_dir=rag_index_dir
            )

    logger.info("MultiDimensions Ingestion Pipeline complete!")


if __name__ == "__main__":
    main()
