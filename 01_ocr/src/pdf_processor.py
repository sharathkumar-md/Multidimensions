"""
PDF processing — hybrid two-path architecture.

Path A (digital pages): PyMuPDF — fast native text + pdfplumber tables.
Path B (scanned pages):  PyMuPDF render at 150 DPI -> RapidOCR.

Why two paths:
  Docling's C++ pre-process stage renders every page into full-res image arrays
  simultaneously, causing std::bad_alloc on PDFs with large embedded images.
  PyMuPDF renders at controlled DPI and RapidOCR reads the result directly.
"""
from __future__ import annotations

import time
from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger

from config.settings import settings
from src.models import (
    BoundingBox,
    ExtractedFigure,
    ExtractedTable,
    PageLayout,
    PageResult,
)

# A page is considered "scanned" if its text character density is below this threshold.
# Measured as: (len of extracted text) / (page area in pt^2)
_SCANNED_DENSITY_THRESHOLD = 0.002


# ── Page layout classification ────────────────────────────────────────────────

def _classify_page(page: fitz.Page) -> PageLayout:
    text = page.get_text("text")
    area = page.rect.width * page.rect.height
    density = len(text.strip()) / max(area, 1.0)
    has_images = len(page.get_images(full=False)) > 0
    is_text_sparse = density < _SCANNED_DENSITY_THRESHOLD

    if is_text_sparse and has_images:
        return PageLayout.SCANNED
    if has_images and not is_text_sparse:
        return PageLayout.MIXED
    return PageLayout.DIGITAL


# ── Path A: PyMuPDF native text extraction ────────────────────────────────────

def _extract_native_tables(pdf_path: Path, page_num_0: int, doc_id: str) -> list[ExtractedTable]:
    """Extract tables from a digital page using pdfplumber."""
    tables: list[ExtractedTable] = []
    if not settings.extract_tables:
        return tables
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_num_0]
            raw_tables = page.extract_tables()
            for idx, raw in enumerate(raw_tables or []):
                if not raw:
                    continue
                cell_count = sum(1 for row in raw for cell in row if cell)
                if cell_count < settings.min_table_cells:
                    continue
                headers = [str(c or "").strip() for c in (raw[0] or [])]
                md_rows: list[str] = []
                md_rows.append("| " + " | ".join(headers) + " |")
                md_rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in raw[1:]:
                    cells = [str(c or "").replace("\n", " ").strip() for c in row]
                    md_rows.append("| " + " | ".join(cells) + " |")
                tables.append(
                    ExtractedTable(
                        table_id=f"{doc_id}_p{page_num_0 + 1}_t{idx}",
                        page_num=page_num_0 + 1,
                        markdown="\n".join(md_rows),
                        row_count=len(raw) - 1,
                        col_count=len(headers),
                        headers=headers,
                        extraction_method="pdfplumber",
                    )
                )
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed on page {n}: {err}", n=page_num_0 + 1, err=exc)
    return tables


def _save_native_figure(
    xref: int,
    fig_idx: int,
    page_num: int,
    doc_id: str,
    figures_out_dir: Path,
    fitz_doc: fitz.Document,
) -> ExtractedFigure | None:
    """Extract and save an embedded image from a digital page."""
    if not settings.extract_figures:
        return None
    try:
        img_data = fitz_doc.extract_image(xref)
        if not img_data:
            return None
        figure_id = f"{doc_id}_p{page_num}_f{fig_idx}"
        ext = img_data.get("ext", "png")
        fig_path = figures_out_dir / f"{figure_id}.{ext}"
        fig_path.write_bytes(img_data["image"])
        return ExtractedFigure(
            figure_id=figure_id,
            page_num=page_num,
            caption="",
            image_path=str(fig_path.relative_to(settings.output_dir)),
            has_text=False,
        )
    except Exception as exc:
        logger.debug("Figure {idx} on page {n} could not be saved: {err}", idx=fig_idx, n=page_num, err=exc)
        return None


def _process_digital_page(
    fitz_doc: fitz.Document,
    fitz_page: fitz.Page,
    pdf_path: Path,
    page_num: int,
    doc_id: str,
    figures_out_dir: Path,
) -> PageResult:
    """Path A: extract text, tables, figures from a native digital page."""
    t0 = time.perf_counter()
    layout = _classify_page(fitz_page)
    raw_text = fitz_page.get_text("text").strip()
    word_count = len(raw_text.split())
    tables = _extract_native_tables(pdf_path, page_num - 1, doc_id)

    figures: list[ExtractedFigure] = []
    if settings.extract_figures:
        for fig_idx, img_info in enumerate(fitz_page.get_images(full=False)):
            fig = _save_native_figure(img_info[0], fig_idx, page_num, doc_id, figures_out_dir, fitz_doc)
            if fig:
                figures.append(fig)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug(
        "Page {n} [digital]: {words} words | {tbl} tables | {fig} figs | {ms:.0f}ms",
        n=page_num, words=word_count, tbl=len(tables), fig=len(figures), ms=elapsed_ms,
    )
    return PageResult(
        page_num=page_num,
        layout=layout,
        raw_text=raw_text,
        tables=tables,
        figures=figures,
        word_count=word_count,
        processing_time_ms=elapsed_ms,
    )


# ── Path B: PyMuPDF render -> RapidOCR for scanned pages ─────────────────────
# Bypasses Docling's C++ preprocessing entirely to avoid std::bad_alloc on
# PDFs that have large embedded images.

_rapidocr_engine = None


def _get_rapidocr():
    """Lazy-init RapidOCR engine (singleton — model loading is expensive)."""
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr import RapidOCR
        _rapidocr_engine = RapidOCR()
        logger.info("RapidOCR engine initialised")
    return _rapidocr_engine


def _process_scanned_page(
    fitz_doc: fitz.Document,
    page_num: int,
    doc_id: str,
    figures_out_dir: Path,
) -> PageResult:
    """
    Path B: render page at 150 DPI via PyMuPDF, then OCR with RapidOCR.
    150 DPI gives good OCR accuracy without excessive memory use.
    """
    t0 = time.perf_counter()
    try:
        fitz_page = fitz_doc[page_num - 1]
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = fitz_page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")
        del pix  # free immediately after conversion

        ocr = _get_rapidocr()
        ocr_output = ocr(img_bytes)

        lines: list[str] = []
        txts = getattr(ocr_output, "txts", None) or []
        for txt in txts:
            if txt and str(txt).strip():
                lines.append(str(txt).strip())

        raw_text = "\n".join(lines)
        word_count = len(raw_text.split())
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "Page {n} [scanned/RapidOCR]: {words} words | {ms:.0f}ms",
            n=page_num, words=word_count, ms=elapsed_ms,
        )
        return PageResult(
            page_num=page_num,
            layout=PageLayout.SCANNED,
            raw_text=raw_text,
            tables=[],
            figures=[],
            word_count=word_count,
            processing_time_ms=elapsed_ms,
        )
    except Exception as exc:
        logger.warning("OCR failed on page {n}: {err}", n=page_num, err=exc)
        return PageResult(
            page_num=page_num,
            layout=PageLayout.SCANNED,
            raw_text="",
            word_count=0,
            processing_time_ms=(time.perf_counter() - t0) * 1000,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def process_pdf(source_path: Path, doc_id: str) -> list[PageResult]:
    """
    Process a single PDF, routing each page through the appropriate path.

    Args:
        source_path: Path to the PDF file.
        doc_id:      Unique document identifier (used for naming output files).

    Returns:
        List of PageResult, one per page.
    """
    logger.info("Opening PDF: {name}", name=source_path.name)
    fitz_doc = fitz.open(str(source_path))
    page_count = fitz_doc.page_count
    logger.info("{name}: {n} pages detected", name=source_path.name, n=page_count)

    figures_out_dir = settings.figures_dir / doc_id
    figures_out_dir.mkdir(parents=True, exist_ok=True)

    page_results: list[PageResult] = []
    for page_num in range(1, page_count + 1):
        fitz_page = fitz_doc[page_num - 1]
        layout = _classify_page(fitz_page)
        logger.debug("Page {n}: layout={layout}", n=page_num, layout=layout.value)

        if layout == PageLayout.SCANNED:
            result = _process_scanned_page(fitz_doc, page_num, doc_id, figures_out_dir)
        else:
            result = _process_digital_page(fitz_doc, fitz_page, source_path, page_num, doc_id, figures_out_dir)

        page_results.append(result)

    fitz_doc.close()

    total_words = sum(p.word_count for p in page_results)
    scanned = sum(1 for p in page_results if p.layout == PageLayout.SCANNED)
    logger.info(
        "Completed {name}: {pages}p ({scanned} scanned) | {words} total words",
        name=source_path.name,
        pages=len(page_results),
        scanned=scanned,
        words=total_words,
    )
    return page_results


def export_document_markdown(doc_id: str, pages: list[PageResult]) -> str:
    """
    Assemble full document Markdown from per-page results.

    Structure per page:
      ## Page N
      <text>
      <!-- table: id -->  <markdown table>
      <!-- figure: id --> ![caption](path)
    """
    parts: list[str] = []
    for page in pages:
        parts.append(f"## Page {page.page_num}\n")
        if page.raw_text.strip():
            parts.append(page.raw_text.strip())
            parts.append("")
        for tbl in page.tables:
            parts.append(f"<!-- table: {tbl.table_id} -->")
            parts.append(tbl.markdown.strip())
            parts.append("")
        for fig in page.figures:
            caption = fig.caption or "(no caption)"
            parts.append(f"<!-- figure: {fig.figure_id} -->")
            parts.append(f"![{caption}]({fig.image_path})")
            parts.append(f"*Figure: {caption}*")
            parts.append("")

    markdown = "\n".join(parts)
    logger.debug("Assembled markdown for {doc_id}: {chars} chars", doc_id=doc_id, chars=len(markdown))
    return markdown
