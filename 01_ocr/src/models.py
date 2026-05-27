from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field


class PageLayout(str, Enum):
    DIGITAL = "digital"       # native PDF text, no OCR needed
    SCANNED = "scanned"       # full-page image, OCR required
    MIXED = "mixed"           # combination of both on one page


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"       # some pages failed, rest succeeded


class BoundingBox(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class ExtractedTable(BaseModel):
    table_id: str
    page_num: int
    bbox: BoundingBox | None = None
    markdown: str                          # table rendered as markdown
    row_count: int
    col_count: int
    headers: list[str] = Field(default_factory=list)
    extraction_method: str = "docling"     # docling | pdfplumber | table_transformer


class ExtractedFigure(BaseModel):
    figure_id: str
    page_num: int
    bbox: BoundingBox | None = None
    caption: str = ""
    image_path: str                        # relative path to saved PNG
    has_text: bool = False                 # True if OCR found text on this figure


class PageResult(BaseModel):
    page_num: int
    layout: PageLayout
    raw_text: str                          # cleaned text from this page
    tables: list[ExtractedTable] = Field(default_factory=list)
    figures: list[ExtractedFigure] = Field(default_factory=list)
    word_count: int = 0
    ocr_confidence: float | None = None    # 0.0–1.0, None if not OCR'd
    processing_time_ms: float = 0.0


class DocumentManifest(BaseModel):
    doc_id: str
    source_filename: str
    source_path: str
    file_size_bytes: int
    sha256: str
    page_count: int
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error_message: str | None = None

    # output paths
    markdown_path: str | None = None
    manifest_path: str | None = None
    figures_dir: str | None = None

    # stats
    total_tables: int = 0
    total_figures: int = 0
    total_words: int = 0
    scanned_pages: int = 0
    digital_pages: int = 0
    mixed_pages: int = 0
    processing_time_ms: float = 0.0

    # per-page results (populated during processing, excluded from manifest JSON)
    pages: list[PageResult] = Field(default_factory=list, exclude=True)

    @computed_field
    @property
    def has_scanned_content(self) -> bool:
        return self.scanned_pages > 0 or self.mixed_pages > 0

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialise to dict for the .json manifest file (excludes per-page raw text)."""
        return self.model_dump(
            exclude={"pages"},
            mode="json",
        )


class PipelineResult(BaseModel):
    """Top-level result returned by OCRPipeline.run()."""
    total_documents: int
    successful: int
    failed: int
    partial: int
    manifests: list[DocumentManifest]
    total_processing_time_ms: float
    run_at: datetime = Field(default_factory=datetime.utcnow)


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()
