from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.pdf_to_images import pdf_to_images
from src.vlm_extractor import extract_page, load_model

# CODE-009: single source-of-truth for SHA-256 hashing; imported from shared/
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from shared.hashing import sha256_file as _sha256  # noqa: E402


def _done_hashes() -> set[str]:
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


def _process(pdf_path: Path, model, processor) -> None:
    doc_id = uuid.uuid4().hex
    t0 = time.perf_counter()

    logger.info(f"processing {pdf_path.name}")
    total_pages, pages_iter = pdf_to_images(pdf_path, dpi=settings.page_dpi)

    sections = []
    for page_num, img in pages_iter:
        logger.debug(f"  page {page_num}/{total_pages}")
        content = extract_page(img, model, processor, settings.max_new_tokens)
        sections.append(f"## Page {page_num}\n\n{content}")

    md_path = settings.markdown_dir / f"{doc_id}.md"
    md_path.write_text("\n\n".join(sections), encoding="utf-8")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    manifest = {
        "doc_id": doc_id,
        "source_filename": pdf_path.name,
        "source_path": str(pdf_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "sha256": _sha256(pdf_path),
        "page_count": total_pages,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "markdown_path": str(md_path.relative_to(settings.output_dir)),
        "processing_time_ms": elapsed_ms,
        "vlm_model": settings.model_id,
    }
    (settings.manifests_dir / f"{doc_id}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(f"done {pdf_path.name} in {elapsed_ms:.0f} ms")


def run(pdf_paths: list[Path] | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
               colorize=True)

    settings.ensure_dirs()
    pdf_paths = pdf_paths or sorted(settings.input_dir.rglob("*.pdf"))

    if not pdf_paths:
        logger.warning(f"no PDFs found in {settings.input_dir}")
        return

    done = _done_hashes() if settings.skip_duplicates else set()
    ok = failed = 0

    model = None
    processor = None

    for pdf in pdf_paths:
        if settings.skip_duplicates and _sha256(pdf) in done:
            logger.info(f"skipping {pdf.name} (already processed)")
            continue
            
        if model is None:
            model, processor = load_model(settings.model_id)
            
        try:
            _process(pdf, model, processor)
            ok += 1
        except Exception as e:
            logger.error(f"failed {pdf.name}: {e}", exc_info=True)
            failed += 1
            
    if model is not None:
        import torch
        del model, processor
        torch.cuda.empty_cache()

    logger.info(f"done — {ok} ok, {failed} failed")
