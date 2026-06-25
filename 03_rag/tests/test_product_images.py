from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[2] / "04_demo"))

from product_images import resolve_product_image


def _retrieved(text: str):
    return SimpleNamespace(chunk=SimpleNamespace(text=text))


def _catalog(tmp_path: Path, data: dict) -> Path:
    catalog_path = tmp_path / "product_images.json"
    catalog_path.write_text(json.dumps(data), encoding="utf-8")
    return catalog_path


def _asset(tmp_path: Path, name: str = "product.png") -> str:
    path = tmp_path / name
    path.write_bytes(b"placeholder")
    return path.name


def test_resolves_product_from_exact_question(tmp_path):
    image_name = _asset(tmp_path, "washdown.png")
    catalog_path = _catalog(tmp_path, {
        "washdown electro pack": {
            "title": "Washdown Electro Pack",
            "image_path": image_name,
            "source_doc": "Food & Beverage.pdf",
        }
    })

    image = resolve_product_image(
        question="Tell me about the Washdown Electro Pack.",
        answer="",
        retrieved=[],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert image is not None
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

    image = resolve_product_image(
        question="Can I pitch epw for washdown areas?",
        answer="",
        retrieved=[],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert image is not None
    assert image["matched_alias"] == "epw"


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

    image = resolve_product_image(
        question="Compare first product and second product.",
        answer="The answer mentions first product later.",
        retrieved=[
            _retrieved("The best source chunk is about second product."),
            _retrieved("A lower-ranked chunk is about first product."),
        ],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert image is not None
    assert image["title"] == "Second Product"


def test_returns_none_when_no_catalog_match(tmp_path):
    image_name = _asset(tmp_path)
    catalog_path = _catalog(tmp_path, {
        "known product": {
            "title": "Known Product",
            "image_path": image_name,
        }
    })

    image = resolve_product_image(
        question="Hello there",
        answer="Hi, how can I help?",
        retrieved=[],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert image is None


def test_returns_none_when_image_file_is_missing(tmp_path):
    catalog_path = _catalog(tmp_path, {
        "known product": {
            "title": "Known Product",
            "image_path": "missing.png",
        }
    })

    image = resolve_product_image(
        question="Tell me about known product",
        answer="",
        retrieved=[],
        catalog_path=catalog_path,
        base_dir=tmp_path,
    )

    assert image is None
