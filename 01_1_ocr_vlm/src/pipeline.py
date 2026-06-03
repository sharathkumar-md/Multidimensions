from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.pdf_to_images import pdf_to_images
from src.vlm_extractor import extract_page


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
        colorize=True,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_existing_hashes() -> set[str]:
    hashes: set[str] = set()
    if not settings.manifests_dir.exists():
        return hashes
    for f in settings.manifests_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if sha := d.get("sha256"):
                hashes.add(sha)
        except Exception:
            pass
    return hashes


def _process_pdf(pdf_path: Path) -> dict:
    doc_id = uuid.uuid4().hex
    t0 = time.perf_counter()
    sha256 = _sha256(pdf_path)

    logger.info(f"processing: {pdf_path.name}")
    pages = pdf_to_images(pdf_path, dpi=settings.page_dpi)
    logger.info(f"{pdf_path.name}: {len(pages)} pages")

    md_lines: list[str] = []
    for page_num, image in pages:
        logger.debug(f"  page {page_num}/{len(pages)}")
        page_md = extract_page(image, settings.model_id, settings.max_new_tokens)
        md_lines.append(f"## Page {page_num}\n\n{page_md}")

    markdown_content = "\n\n".join(md_lines)
    md_path = settings.markdown_dir / f"{doc_id}.md"
    md_path.write_text(markdown_content, encoding="utf-8")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    manifest = {
        "doc_id": doc_id,
        "source_filename": pdf_path.name,
        "source_path": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "sha256": sha256,
        "page_count": len(pages),
        "status": "completed",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "markdown_path": str(md_path.relative_to(settings.output_dir)),
        "processing_time_ms": elapsed_ms,
        "vlm_model": settings.model_id,
    }

    manifest_path = settings.manifests_dir / f"{doc_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"done: {pdf_path.name} in {elapsed_ms:.0f} ms")
    return manifest


def run(pdf_paths: list[Path] | None = None) -> None:
    _configure_logging()
    settings.ensure_dirs()

    if pdf_paths is None:
        pdf_paths = sorted(settings.input_dir.glob("*.pdf"))

    if not pdf_paths:
        logger.warning(f"no PDFs found in {settings.input_dir}")
        return

    existing_hashes = _load_existing_hashes() if settings.skip_duplicates else set()
    logger.info(f"found {len(pdf_paths)} PDF(s) | skip_duplicates={settings.skip_duplicates}")

    ok, failed = 0, 0
    for pdf_path in pdf_paths:
        if settings.skip_duplicates and _sha256(pdf_path) in existing_hashes:
            logger.info(f"skipping duplicate: {pdf_path.name}")
            continue
        try:
            _process_pdf(pdf_path)
            ok += 1
        except Exception as exc:
            logger.error(f"failed {pdf_path.name}: {exc}", exc_info=True)
            failed += 1

    logger.info(f"pipeline done — {ok} succeeded, {failed} failed")
