from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
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


def _is_toc_chunk(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    toc_like = sum(1 for w in words if re.match(r"^\d+(\.\d+)*\.?$", w))
    return toc_like / len(words) > 0.4


def _clean_markdown(text: str) -> str:
    # strip ```markdown ... ``` fences added by VLM output
    text = re.sub(r"```markdown\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\(figures[^\)]+\)", "", text)
    text = re.sub(r"\*Figure:.*?\*", "", text)
    text = re.sub(r"\b[a-f0-9]{16,}_p\d+_[ft]\d+\b", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_num(header: str) -> int:
    m = re.search(r"##\s+Page\s+(\d+)", header, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _split_sentences(text: str) -> list[str]:
    # Split on sentence boundaries and paragraph breaks, keep tables whole
    table_pattern = re.compile(r"(\|.+\|[\s\S]*?)(?=\n(?!\|)|\Z)")
    segments: list[tuple[bool, str]] = []
    last = 0
    for m in table_pattern.finditer(text):
        if m.start() > last:
            segments.append((False, text[last:m.start()]))
        segments.append((True, m.group(0)))
        last = m.end()
    if last < len(text):
        segments.append((False, text[last:]))

    sentences: list[str] = []
    for is_table, seg in segments:
        if is_table:
            if seg.strip():
                sentences.append(seg.strip())
        else:
            # split on sentence-ending punctuation or double newline
            parts = re.split(r"(?<=[.!?])\s+|\n\n+", seg)
            for p in parts:
                p = p.strip()
                if p:
                    sentences.append(p)
    return sentences


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _semantic_split(
    sentences: list[str],
    embed_fn,
    breakpoint_percentile: int = 80,
    min_chunk_words: int = 30,
    max_chunk_words: int = 400,
) -> list[str]:
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    # embed with context window of 1 neighbour on each side
    combined = []
    for i, s in enumerate(sentences):
        window = sentences[max(0, i - 1): i + 2]
        combined.append(" ".join(window))

    vecs = np.array(embed_fn(combined))

    # similarity between consecutive combined embeddings
    sims = [_cosine_sim(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]

    # breakpoints = positions where similarity falls below threshold
    threshold = float(np.percentile(sims, 100 - breakpoint_percentile))
    breakpoints = {i + 1 for i, s in enumerate(sims) if s < threshold}

    # build chunks from sentence groups
    raw_chunks: list[str] = []
    buf: list[str] = []
    for i, sent in enumerate(sentences):
        if i in breakpoints and buf:
            raw_chunks.append(" ".join(buf))
            buf = []
        buf.append(sent)
    if buf:
        raw_chunks.append(" ".join(buf))

    # merge chunks that are too small, split ones that are too large
    final: list[str] = []
    carry = ""
    for chunk in raw_chunks:
        text = (carry + " " + chunk).strip() if carry else chunk
        words = text.split()
        if len(words) < min_chunk_words:
            carry = text
        elif len(words) > max_chunk_words:
            # hard split at max_chunk_words
            while len(words) > max_chunk_words:
                final.append(" ".join(words[:max_chunk_words]))
                words = words[max_chunk_words:]
            carry = " ".join(words) if words else ""
        else:
            final.append(text)
            carry = ""
    if carry.strip():
        if final:
            final[-1] = final[-1] + " " + carry
        else:
            final.append(carry)

    return [c for c in final if c.strip() and not _is_toc_chunk(c)]


def _split_section(
    section_text: str,
    page: int,
    doc_hash: str,
    source: str,
    embed_fn,
) -> list[Chunk]:
    sentences = _split_sentences(section_text)
    texts = _semantic_split(sentences, embed_fn)
    return [
        Chunk(
            chunk_id=f"{doc_hash}_{page}_{idx}",
            doc_hash=doc_hash,
            source_doc=source,
            page_num=page,
            text=text,
        )
        for idx, text in enumerate(texts)
        if text.strip()
    ]


def chunk_documents(
    ocr_output_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    from src.embed import embed_texts

    ocr_output_dir = Path(ocr_output_dir or settings.ocr_output_dir)

    manifest_dir = ocr_output_dir / "manifests"
    manifests = sorted(manifest_dir.glob("*.json")) if manifest_dir.exists() else sorted(ocr_output_dir.glob("*.json"))
    logger.info(f"{len(manifests)} manifests in {ocr_output_dir}")

    all_chunks: list[Chunk] = []

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        md_rel = manifest.get("markdown_path") or manifest.get("output_markdown", "")
        md_file = (ocr_output_dir / md_rel.replace("\\", "/")).resolve()
        if not md_file.exists():
            logger.warning(f"missing markdown: {md_file}")
            continue

        md_text = _clean_markdown(md_file.read_text(encoding="utf-8"))
        source_doc = manifest.get("source_filename") or manifest.get("source_pdf", md_file.stem)
        doc_hash = _hash(source_doc)

        page_pattern = re.compile(r"(##\s+Page\s+\d+[^\n]*)", re.IGNORECASE)
        parts = page_pattern.split(md_text)

        current_page = 0
        current_text = parts[0]

        i = 1
        while i < len(parts):
            header = parts[i]
            section = parts[i + 1] if i + 1 < len(parts) else ""
            if current_text.strip():
                all_chunks.extend(_split_section(current_text, current_page, doc_hash, source_doc, embed_texts))
            current_page = _page_num(header)
            current_text = section
            i += 2

        if current_text.strip():
            all_chunks.extend(_split_section(current_text, current_page, doc_hash, source_doc, embed_texts))

        n = len([c for c in all_chunks if c.doc_hash == doc_hash])
        logger.debug(f"{source_doc}: {n} chunks")

    logger.info(f"total chunks: {len(all_chunks)}")
    return all_chunks
