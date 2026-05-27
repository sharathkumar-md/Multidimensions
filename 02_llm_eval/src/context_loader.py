from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from config.settings import settings

_FIGURE_PATTERN = re.compile(
    r"<!-- figure:.*?-->\n!\[.*?\]\(.*?\)\n\*Figure:.*?\*\n?",
    re.MULTILINE,
)


class ContextLoader:
    """Loads OCR-produced markdown files and maps them by source PDF filename."""

    def __init__(self) -> None:
        self._manifest_dir = settings.ocr_output_dir / "manifests"
        self._markdown_dir = settings.ocr_output_dir / "markdown"
        self._filename_to_hash: dict[str, str] = {}
        self._hash_to_content: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        for manifest_path in self._manifest_dir.glob("*.json"):
            doc_hash = manifest_path.stem
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            filename = manifest.get("source_filename", "")
            if filename:
                self._filename_to_hash[filename] = doc_hash

        for md_path in self._markdown_dir.glob("*.md"):
            doc_hash = md_path.stem
            raw = md_path.read_text(encoding="utf-8")
            self._hash_to_content[doc_hash] = _FIGURE_PATTERN.sub("", raw).strip()

        logger.info(
            "ContextLoader: {n} documents loaded",
            n=len(self._filename_to_hash),
        )

    def get_context(self, source_filename: str) -> str:
        doc_hash = self._filename_to_hash.get(source_filename)
        if not doc_hash:
            raise KeyError(f"No OCR output found for: {source_filename}")
        content = self._hash_to_content.get(doc_hash, "")
        if not content:
            raise ValueError(f"Empty content for: {source_filename}")
        return content

    def list_documents(self) -> list[str]:
        return sorted(self._filename_to_hash.keys())
