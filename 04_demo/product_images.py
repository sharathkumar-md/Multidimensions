from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).parent
REPO_DIR = DEMO_DIR.parent
DEFAULT_CATALOG_PATH = DEMO_DIR / "product_images.json"


@dataclass(frozen=True)
class ProductImage:
    title: str
    image_path: str
    source_doc: str = ""
    matched_alias: str = ""


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    demo_relative = (base_dir / path).resolve()
    if demo_relative.exists():
        return demo_relative

    return (REPO_DIR / path).resolve()


def _load_catalog(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    base_dir: Path = DEMO_DIR,
) -> list[tuple[list[str], ProductImage]]:
    if not catalog_path.exists():
        return []

    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries: list[tuple[list[str], ProductImage]] = []

    for product_name, raw in data.items():
        if not isinstance(raw, dict):
            continue

        image_path = str(raw.get("image_path", "")).strip()
        if not image_path:
            continue

        resolved = _resolve_path(image_path, base_dir)
        if not resolved.exists():
            continue

        aliases = [product_name, *raw.get("aliases", [])]
        normalized_aliases = [_normalize(str(alias)) for alias in aliases]
        normalized_aliases = [alias for alias in normalized_aliases if alias]
        if not normalized_aliases:
            continue

        title = str(raw.get("title") or product_name).strip()
        image = ProductImage(
            title=title,
            image_path=str(resolved),
            source_doc=str(raw.get("source_doc", "")).strip(),
        )
        entries.append((normalized_aliases, image))

    return entries


def _chunk_text(retrieved_item: Any) -> str:
    chunk = getattr(retrieved_item, "chunk", None)
    return str(getattr(chunk, "text", "") or "")


def resolve_product_image(
    question: str,
    answer: str,
    retrieved: list[Any],
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    base_dir: Path = DEMO_DIR,
) -> dict[str, str] | None:
    catalog = _load_catalog(catalog_path=catalog_path, base_dir=base_dir)
    if not catalog:
        return None

    search_texts = [_chunk_text(item) for item in retrieved]
    search_texts.extend([answer, question])

    for text in search_texts:
        normalized_text = _normalize(text)
        if not normalized_text:
            continue

        padded_text = f" {normalized_text} "
        for aliases, image in catalog:
            for alias in aliases:
                if f" {alias} " in padded_text:
                    return {
                        "title": image.title,
                        "image_path": image.image_path,
                        "source_doc": image.source_doc,
                        "matched_alias": alias,
                    }

    return None
