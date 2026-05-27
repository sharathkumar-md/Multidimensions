import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.context_loader import ContextLoader


def _build_ocr_dir(tmp_path: Path, source_filename: str, md_content: str) -> Path:
    """Create a minimal OCR output directory structure in tmp_path."""
    doc_hash = "abc123def456"
    manifest_dir = tmp_path / "manifests"
    markdown_dir = tmp_path / "markdown"
    manifest_dir.mkdir()
    markdown_dir.mkdir()

    manifest = {"source_filename": source_filename, "total_pages": 2, "total_words": 100}
    (manifest_dir / f"{doc_hash}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    raw_md = (
        f"{md_content}\n"
        "<!-- figure: abc -->\n"
        "![(no caption)](fig.png)\n"
        "*Figure: (no caption)*\n"
    )
    (markdown_dir / f"{doc_hash}.md").write_text(raw_md, encoding="utf-8")
    return tmp_path


def _make_loader(tmp_path: Path, source_filename: str, md_content: str) -> ContextLoader:
    """Patch settings.ocr_output_dir at the module level so ContextLoader uses tmp_path."""
    _build_ocr_dir(tmp_path, source_filename, md_content)
    with patch("src.context_loader.settings") as mock_settings:
        mock_settings.ocr_output_dir = tmp_path
        loader = ContextLoader()
    return loader


class TestContextLoader:
    def test_loads_documents(self, tmp_path):
        loader = _make_loader(tmp_path, "test.pdf", "## Page 1\nBearing text")
        assert "test.pdf" in loader.list_documents()

    def test_get_context_returns_text(self, tmp_path):
        loader = _make_loader(tmp_path, "bearing.pdf", "## Page 1\nDU bearings 250 N/mm²")
        ctx = loader.get_context("bearing.pdf")
        assert "DU bearings" in ctx
        assert "250" in ctx

    def test_figure_references_stripped(self, tmp_path):
        loader = _make_loader(tmp_path, "fig_test.pdf", "## Page 1\nActual content here")
        ctx = loader.get_context("fig_test.pdf")
        assert "<!-- figure:" not in ctx
        assert "*Figure:" not in ctx

    def test_text_content_preserved(self, tmp_path):
        loader = _make_loader(tmp_path, "content.pdf", "## Page 1\nHPMB® bearings IT7 tolerance")
        ctx = loader.get_context("content.pdf")
        assert "HPMB" in ctx
        assert "IT7" in ctx

    def test_missing_doc_raises_key_error(self, tmp_path):
        loader = _make_loader(tmp_path, "present.pdf", "content")
        with pytest.raises(KeyError):
            loader.get_context("missing.pdf")

    def test_list_documents_sorted(self, tmp_path):
        loader = _make_loader(tmp_path, "alpha.pdf", "text")
        docs = loader.list_documents()
        assert docs == sorted(docs)

    def test_manifest_with_empty_filename_skipped(self, tmp_path):
        manifest_dir = tmp_path / "manifests"
        markdown_dir = tmp_path / "markdown"
        manifest_dir.mkdir()
        markdown_dir.mkdir()
        (manifest_dir / "nofn.json").write_text(
            json.dumps({"source_filename": "", "total_pages": 1}), encoding="utf-8"
        )
        with patch("src.context_loader.settings") as mock_settings:
            mock_settings.ocr_output_dir = tmp_path
            loader = ContextLoader()
        assert loader.list_documents() == []
