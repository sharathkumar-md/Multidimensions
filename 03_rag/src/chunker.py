from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from config.settings import settings


@dataclass
class Chunk:
    chunk_id: str
    doc_hash: str
    source_doc: str
    page_num: int
    text: str
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _page_num(header: str) -> int:
    m = re.search(r"##\s+Page\s+(\d+)", header, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _split_preserving_tables(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into token-sized chunks without breaking mid-table."""
    table_pattern = re.compile(r"(\|.+\|[\s\S]*?)(?=\n(?!\|)|\Z)")
    chunks: list[str] = []
    pos = 0
    buf: list[str] = []
    buf_tokens = 0

    segments: list[tuple[bool, str]] = []
    last = 0
    for m in table_pattern.finditer(text):
        if m.start() > last:
            segments.append((False, text[last : m.start()]))
        segments.append((True, m.group(0)))
        last = m.end()
    if last < len(text):
        segments.append((False, text[last:]))

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            chunks.append("".join(buf).strip())
            # keep overlap
            overlap_words: list[str] = []
            overlap_count = 0
            for seg in reversed(buf):
                words = seg.split()
                if overlap_count + len(words) <= chunk_overlap:
                    overlap_words = words + overlap_words
                    overlap_count += len(words)
                else:
                    need = chunk_overlap - overlap_count
                    overlap_words = words[-need:] + overlap_words
                    break
            buf = [" ".join(overlap_words)] if overlap_words else []
            buf_tokens = len(overlap_words)

    for is_table, seg in segments:
        seg_tokens = len(seg.split())
        if is_table:
            if buf_tokens + seg_tokens > chunk_size and buf:
                flush()
            buf.append(seg)
            buf_tokens += seg_tokens
            if buf_tokens >= chunk_size:
                flush()
        else:
            words = seg.split()
            i = 0
            while i < len(words):
                take = min(chunk_size - buf_tokens, len(words) - i)
                buf.append(" ".join(words[i : i + take]))
                buf_tokens += take
                i += take
                if buf_tokens >= chunk_size:
                    flush()

    flush()
    return [c for c in chunks if c]


def _split_section(section_text: str, page: int, doc_hash: str, source: str,
                   chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    raw_chunks = _split_preserving_tables(section_text, chunk_size, chunk_overlap)
    result: list[Chunk] = []
    for idx, text in enumerate(raw_chunks):
        if not text.strip():
            continue
        cid = f"{doc_hash}_{page}_{idx}"
        result.append(Chunk(chunk_id=cid, doc_hash=doc_hash, source_doc=source,
                            page_num=page, text=text))
    return result


def chunk_documents(
    ocr_output_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    ocr_output_dir = ocr_output_dir or settings.ocr_output_dir
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    ocr_output_dir = Path(ocr_output_dir)
    manifests = sorted(ocr_output_dir.glob("*.json"))
    logger.info(f"Found {len(manifests)} manifests in {ocr_output_dir}")

    all_chunks: list[Chunk] = []

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        md_file = ocr_output_dir / manifest["output_markdown"]
        if not md_file.exists():
            logger.warning(f"Markdown file missing: {md_file}")
            continue

        md_text = md_file.read_text(encoding="utf-8")
        doc_hash = _hash(manifest.get("source_pdf", md_file.name))
        source_doc = manifest.get("source_pdf", md_file.stem)

        # Split on ## Page N headers
        page_pattern = re.compile(r"(##\s+Page\s+\d+[^\n]*)", re.IGNORECASE)
        parts = page_pattern.split(md_text)

        # parts alternates: [pre_text, header, section, header, section, ...]
        current_page = 0
        current_text = parts[0]

        i = 1
        while i < len(parts):
            header = parts[i]
            section = parts[i + 1] if i + 1 < len(parts) else ""
            # flush previous section
            if current_text.strip():
                chunks = _split_section(current_text, current_page, doc_hash, source_doc,
                                        chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
            current_page = _page_num(header)
            current_text = section
            i += 2

        # flush last section
        if current_text.strip():
            chunks = _split_section(current_text, current_page, doc_hash, source_doc,
                                    chunk_size, chunk_overlap)
            all_chunks.extend(chunks)

        logger.debug(f"{source_doc}: {len([c for c in all_chunks if c.doc_hash == doc_hash])} chunks")

    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks
