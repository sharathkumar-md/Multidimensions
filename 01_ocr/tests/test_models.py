"""Tests for src/models.py"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from src.models import (
    BoundingBox,
    DocumentManifest,
    ExtractedTable,
    PageLayout,
    PageResult,
    ProcessingStatus,
    compute_sha256,
)


class TestBoundingBox:
    def test_valid_construction(self):
        bbox = BoundingBox(page=1, x0=0.0, y0=0.0, x1=100.0, y1=200.0)
        assert bbox.page == 1
        assert bbox.x1 == 100.0


class TestExtractedTable:
    def test_markdown_stored(self):
        tbl = ExtractedTable(
            table_id="doc1_p1_t0",
            page_num=1,
            markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            row_count=1,
            col_count=2,
        )
        assert "| A | B |" in tbl.markdown
        assert tbl.headers == []


class TestPageResult:
    def test_defaults(self):
        page = PageResult(page_num=1, layout=PageLayout.DIGITAL, raw_text="Hello world")
        assert page.tables == []
        assert page.figures == []
        assert page.ocr_confidence is None


class TestDocumentManifest:
    def _make_manifest(self) -> DocumentManifest:
        return DocumentManifest(
            doc_id="abc123",
            source_filename="test.pdf",
            source_path="/tmp/test.pdf",
            file_size_bytes=1024,
            sha256="deadbeef",
            page_count=5,
            scanned_pages=2,
            digital_pages=3,
        )

    def test_has_scanned_content_true(self):
        assert self._make_manifest().has_scanned_content is True

    def test_has_scanned_content_false(self):
        m = self._make_manifest()
        m.scanned_pages = 0
        m.mixed_pages = 0
        assert m.has_scanned_content is False

    def test_to_manifest_dict_excludes_pages(self):
        m = self._make_manifest()
        m.pages = [PageResult(page_num=1, layout=PageLayout.DIGITAL, raw_text="text")]
        assert "pages" not in m.to_manifest_dict()

    def test_to_manifest_dict_is_json_serialisable(self):
        json.dumps(self._make_manifest().to_manifest_dict(), default=str)

    def test_status_default_is_pending(self):
        assert self._make_manifest().status == ProcessingStatus.PENDING


class TestComputeSha256:
    def test_matches_known_hash(self, tmp_path):
        content = b"hello world"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        assert compute_sha256(f) == hashlib.sha256(content).hexdigest()

    def test_different_files_different_hashes(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content a")
        f2.write_bytes(b"content b")
        assert compute_sha256(f1) != compute_sha256(f2)
