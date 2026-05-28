from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.chunker import Chunk, _page_num, _split_preserving_tables, chunk_documents


def test_chunk_dataclass_word_count():
    c = Chunk(chunk_id="x", doc_hash="abc", source_doc="doc.pdf", page_num=1, text="hello world foo")
    assert c.word_count == 3


def test_page_num_extraction():
    assert _page_num("## Page 12") == 12
    assert _page_num("## page 3 — Section") == 3
    assert _page_num("some other text") == 0


def test_split_no_tables():
    words = " ".join(f"word{i}" for i in range(200))
    chunks = _split_preserving_tables(words, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= 60  # allow slight overlap


def test_split_preserves_table_integrity():
    table = (
        "| Col1 | Col2 |\n"
        "|------|------|\n"
        "| A    | B    |\n"
        "| C    | D    |\n"
    )
    preamble = " ".join(f"word{i}" for i in range(100))
    text = preamble + "\n\n" + table
    chunks = _split_preserving_tables(text, chunk_size=50, chunk_overlap=5)
    # table rows should appear in the same chunk (not split mid-table)
    table_chunks = [c for c in chunks if "|" in c]
    for tc in table_chunks:
        rows = [ln for ln in tc.splitlines() if "|" in ln]
        # all rows of the same table block appear together
        assert len(rows) >= 2


def test_chunk_documents_uses_manifests(tmp_path):
    ocr_dir = tmp_path / "ocr_output"
    ocr_dir.mkdir()

    md_text = textwrap.dedent("""\
        ## Page 1
        This is the first page of the document with enough words to form a chunk.
        It contains technical specifications for a mechanical component.

        ## Page 2
        Second page has more details about the part dimensions and tolerances.
        Material: Steel. Hardness: 60 HRC. Weight: 2.5 kg.
    """)

    manifest = {
        "source_pdf": "test_doc.pdf",
        "output_markdown": "test_doc.md",
    }

    (ocr_dir / "test_doc.json").write_text(json.dumps(manifest), encoding="utf-8")
    (ocr_dir / "test_doc.md").write_text(md_text, encoding="utf-8")

    chunks = chunk_documents(ocr_output_dir=ocr_dir, chunk_size=30, chunk_overlap=5)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source_doc == "test_doc.pdf" for c in chunks)
    pages = {c.page_num for c in chunks}
    assert 1 in pages or 2 in pages


def test_chunk_documents_skips_missing_markdown(tmp_path):
    ocr_dir = tmp_path / "ocr_output"
    ocr_dir.mkdir()
    manifest = {"source_pdf": "ghost.pdf", "output_markdown": "ghost.md"}
    (ocr_dir / "ghost.json").write_text(json.dumps(manifest), encoding="utf-8")
    # ghost.md does NOT exist
    chunks = chunk_documents(ocr_output_dir=ocr_dir, chunk_size=50, chunk_overlap=5)
    assert chunks == []
