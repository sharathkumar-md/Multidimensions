from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[2] / "04_demo"))

from product_images import resolve_product_image


# ── helpers ──────────────────────────────────────────────────────────────────

def _retrieved(text: str, source_doc: str = "", page_num: int = 0):
    """Build a minimal retrieved-result stub."""
    return SimpleNamespace(
        chunk=SimpleNamespace(text=text, source_doc=source_doc, page_num=page_num)
    )


def _catalog(tmp_path: Path, data: dict) -> Path:
    catalog_path = tmp_path / "product_images.json"
    catalog_path.write_text(json.dumps(data), encoding="utf-8")
    return catalog_path


def _asset(tmp_path: Path, name: str = "product.png") -> str:
    """Write a stub image file large enough to pass the 50 KB filter."""
    path = tmp_path / name
    path.write_bytes(b"x" * (60 * 1024))   # 60 KB
    return path.name


# ── curated-catalog tests ────────────────────────────────────────────────────

def test_resolves_product_from_exact_question(tmp_path):
    image_name = _asset(tmp_path, "washdown.png")
    catalog_path = _catalog(tmp_path, {
        "washdown electro pack": {
            "title": "Washdown Electro Pack",
            "image_path": image_name,
            "source_doc": "Food & Beverage.pdf",
        }
    })

    result = resolve_product_image(
        question="Tell me about the Washdown Electro Pack.",
        answer="",
        retrieved=[],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert result is not None
    assert result["from_index"] is False
    image = result["images"][0]
    assert image["title"] == "Washdown Electro Pack"
    assert image["source_doc"] == "Food & Beverage.pdf"
    assert image["image_path"].endswith("washdown.png")


def test_resolves_alias_case_insensitively(tmp_path):
    image_name = _asset(tmp_path, "epw.png")
    catalog_path = _catalog(tmp_path, {
        "washdown electro pack": {
            "title": "Washdown Electro Pack",
            "image_path": image_name,
            "aliases": ["EPW"],
        }
    })

    result = resolve_product_image(
        question="Can I pitch epw for washdown areas?",
        answer="",
        retrieved=[],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert result is not None
    assert result["images"][0]["matched_alias"] == "epw"


def test_prefers_highest_ranked_retrieved_chunk(tmp_path):
    first_image = _asset(tmp_path, "first.png")
    second_image = _asset(tmp_path, "second.png")
    catalog_path = _catalog(tmp_path, {
        "first product": {
            "title": "First Product",
            "image_path": first_image,
        },
        "second product": {
            "title": "Second Product",
            "image_path": second_image,
        },
    })

    result = resolve_product_image(
        question="Compare first product and second product.",
        answer="The answer mentions first product later.",
        retrieved=[
            _retrieved("The best source chunk is about second product."),
            _retrieved("A lower-ranked chunk is about first product."),
        ],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert result is not None
    assert result["images"][0]["title"] == "Second Product"


def test_returns_none_for_generic_question(tmp_path):
    """No product mentioned → guard returns None regardless of catalog."""
    image_name = _asset(tmp_path)
    catalog_path = _catalog(tmp_path, {
        "known product": {
            "title": "Known Product",
            "image_path": image_name,
        }
    })

    result = resolve_product_image(
        question="Hello there",
        answer="Hi, how can I help?",
        retrieved=[],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert result is None


def test_returns_none_when_image_file_is_missing(tmp_path):
    catalog_path = _catalog(tmp_path, {
        "known product": {
            "title": "Known Product",
            "image_path": "missing.png",
        }
    })

    result = resolve_product_image(
        question="Tell me about known product",
        answer="",
        retrieved=[],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    # curated match skipped (file missing); figure index also empty → None
    assert result is None
