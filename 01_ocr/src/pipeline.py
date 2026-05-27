from __future__ import annotations

import json
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from loguru import logger

from config.settings import settings
from src.cleaner import clean_document
from src.models import (
    DocumentManifest,
    PageLayout,
    PipelineResult,
    ProcessingStatus,
    compute_sha256,
)
from src.pdf_processor import export_document_markdown, process_pdf


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
    if settings.log_file is not None:
        log_path = Path(settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        )
        logger.info("Log file: {path}", path=log_path.resolve())


def _load_existing_sha256s() -> set[str]:
    hashes: set[str] = set()
    for manifest_file in settings.manifests_dir.glob("*.json"):
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            if sha := data.get("sha256"):
                hashes.add(sha)
        except Exception as exc:
            logger.warning("Could not read manifest {f}: {err}", f=manifest_file.name, err=exc)
    logger.debug("Loaded {n} existing SHA-256 hashes from manifests", n=len(hashes))
    return hashes


def _write_markdown(doc_id: str, content: str) -> Path:
    md_path = settings.markdown_dir / f"{doc_id}.md"
    md_path.write_text(content, encoding="utf-8")
    logger.debug("Markdown written: {path}", path=md_path)
    return md_path


def _write_manifest(manifest: DocumentManifest) -> Path:
    manifest_path = settings.manifests_dir / f"{manifest.doc_id}.json"
    manifest_path.write_text(
        json.dumps(manifest.to_manifest_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.debug("Manifest written: {path}", path=manifest_path)
    return manifest_path


def _process_single_pdf(pdf_path: Path, existing_hashes: set[str]) -> DocumentManifest:
    doc_id = uuid.uuid4().hex
    t0 = time.perf_counter()

    manifest = DocumentManifest(
        doc_id=doc_id,
        source_filename=pdf_path.name,
        source_path=str(pdf_path),
        file_size_bytes=pdf_path.stat().st_size,
        sha256="",
        page_count=0,
        status=ProcessingStatus.PROCESSING,
    )

    try:
        sha256 = compute_sha256(pdf_path)
        manifest.sha256 = sha256

        if settings.skip_duplicates and sha256 in existing_hashes:
            logger.info("Skipping duplicate: {name} (sha256={sha})", name=pdf_path.name, sha=sha256[:12])
            manifest.status = ProcessingStatus.COMPLETED
            manifest.error_message = "Skipped: duplicate (sha256 already indexed)"
            return manifest

        logger.info("Processing: {name}", name=pdf_path.name)
        pages = process_pdf(pdf_path, doc_id)
        manifest.page_count = len(pages)
        pages = clean_document(pages)

        manifest.total_tables = sum(len(p.tables) for p in pages)
        manifest.total_figures = sum(len(p.figures) for p in pages)
        manifest.total_words = sum(p.word_count for p in pages)
        manifest.scanned_pages = sum(1 for p in pages if p.layout == PageLayout.SCANNED)
        manifest.digital_pages = sum(1 for p in pages if p.layout == PageLayout.DIGITAL)
        manifest.mixed_pages = sum(1 for p in pages if p.layout == PageLayout.MIXED)
        manifest.pages = pages

        logger.info(
            "{name}: {pages}p | {words} words | {tbl} tables | {fig} figures",
            name=pdf_path.name,
            pages=manifest.page_count,
            words=manifest.total_words,
            tbl=manifest.total_tables,
            fig=manifest.total_figures,
        )

        markdown_content = export_document_markdown(doc_id, pages)
        md_path = _write_markdown(doc_id, markdown_content)
        manifest.markdown_path = str(md_path.relative_to(settings.output_dir))
        manifest.figures_dir = str((settings.figures_dir / doc_id).relative_to(settings.output_dir))

        manifest.status = ProcessingStatus.COMPLETED
        manifest.completed_at = datetime.utcnow()
        manifest.processing_time_ms = (time.perf_counter() - t0) * 1000

        manifest_path = _write_manifest(manifest)
        manifest.manifest_path = str(manifest_path.relative_to(settings.output_dir))

        logger.info("Completed {name} in {ms:.0f} ms", name=pdf_path.name, ms=manifest.processing_time_ms)

    except Exception as exc:
        manifest.status = ProcessingStatus.FAILED
        manifest.error_message = str(exc)
        manifest.processing_time_ms = (time.perf_counter() - t0) * 1000
        logger.error("Failed to process {name}: {err}", name=pdf_path.name, err=exc, exc_info=True)
        # Always persist failure manifest so the doc is not retried blindly
        try:
            manifest_path = _write_manifest(manifest)
            manifest.manifest_path = str(manifest_path.relative_to(settings.output_dir))
        except Exception as write_exc:
            logger.warning("Could not write failure manifest: {err}", err=write_exc)

    return manifest


class OCRPipeline:
    def __init__(self) -> None:
        _configure_logging()
        settings.ensure_dirs()
        logger.info(
            "OCRPipeline initialised | input={inp} | output={out} | workers={w}",
            inp=settings.input_dir,
            out=settings.output_dir,
            w=settings.max_workers,
        )

    def discover_pdfs(self) -> list[Path]:
        pdfs = sorted(settings.input_dir.glob("*.pdf"))
        logger.info("Discovered {n} PDF file(s) in {dir}", n=len(pdfs), dir=settings.input_dir)
        for p in pdfs:
            logger.debug("  Found: {name} ({size} bytes)", name=p.name, size=p.stat().st_size)
        return pdfs

    def run(self, pdf_paths: list[Path] | None = None) -> PipelineResult:
        pipeline_t0 = time.perf_counter()

        if pdf_paths is None:
            pdf_paths = self.discover_pdfs()

        if not pdf_paths:
            logger.warning("No PDF files found. Exiting.")
            return PipelineResult(
                total_documents=0, successful=0, failed=0, partial=0,
                manifests=[], total_processing_time_ms=0.0,
            )

        existing_hashes = _load_existing_sha256s() if settings.skip_duplicates else set()
        manifests: list[DocumentManifest] = []

        if settings.max_workers == 1 or len(pdf_paths) == 1:
            logger.info("Processing {n} PDF(s) sequentially", n=len(pdf_paths))
            for pdf_path in pdf_paths:
                manifests.append(_process_single_pdf(pdf_path, existing_hashes))
        else:
            logger.info("Processing {n} PDF(s) with {w} workers", n=len(pdf_paths), w=settings.max_workers)
            with ProcessPoolExecutor(max_workers=settings.max_workers) as pool:
                futures = {pool.submit(_process_single_pdf, p, existing_hashes): p for p in pdf_paths}
                for future in as_completed(futures):
                    try:
                        manifests.append(future.result())
                    except Exception as exc:
                        logger.error("Worker exception for {name}: {err}", name=futures[future].name, err=exc, exc_info=True)

        successful = sum(1 for m in manifests if m.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for m in manifests if m.status == ProcessingStatus.FAILED)
        partial = sum(1 for m in manifests if m.status == ProcessingStatus.PARTIAL)
        total_ms = (time.perf_counter() - pipeline_t0) * 1000

        logger.info(
            "Pipeline complete in {ms:.0f} ms - {ok} succeeded, {fail} failed, {part} partial",
            ms=total_ms, ok=successful, fail=failed, part=partial,
        )
        return PipelineResult(
            total_documents=len(pdf_paths),
            successful=successful,
            failed=failed,
            partial=partial,
            manifests=manifests,
            total_processing_time_ms=total_ms,
        )
