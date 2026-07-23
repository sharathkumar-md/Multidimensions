from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DEMO_DIR = Path(__file__).parent
REPO_DIR = DEMO_DIR.parent
DEFAULT_CATALOG_PATH    = DEMO_DIR / "product_images.json"
DEFAULT_FIG_INDEX_PATH  = DEMO_DIR / "figure_index.json"

# Maximum images returned in a gallery
MAX_GALLERY_IMAGES = 20

# Minimum file size to consider a figure a real product photo (bytes)
MIN_FIGURE_BYTES = 1 * 1024   # skip sub-pixel icons / dots (< 1 KB)

# Regex heuristic: model numbers like "P01", "EM", "EPW", "SF-300", "ABZ-510"
_MODEL_NUMBER_RE = re.compile(r"\b([A-Z]{1,4}[-/]?\d{2,}[A-Z0-9\-]*|[A-Z]{2,6}\d*)\b")


# ── data model ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProductImage:
    title: str
    image_path: str
    source_doc: str = ""
    matched_alias: str = ""


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _chunk_text(retrieved_item: Any) -> str:
    chunk = getattr(retrieved_item, "chunk", None)
    return str(getattr(chunk, "text", "") or "")


def _chunk_meta(retrieved_item: Any) -> tuple[str, int]:
    """Return (source_doc, page_num) from a retrieved result."""
    chunk = getattr(retrieved_item, "chunk", None)
    if chunk is None:
        return "", 0
    return str(getattr(chunk, "source_doc", "") or ""), int(getattr(chunk, "page_num", 0) or 0)


# ── curated catalog (product_images.json) ────────────────────────────────────

@lru_cache(maxsize=4)
def _load_catalog(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    base_dir: Path = DEMO_DIR,
) -> list[tuple[list[str], ProductImage]]:
    """Load and parse the curated product catalog.  Cached for the process lifetime
    since catalog_path is a static file that only changes on manual edits.
    """
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
        normalized_aliases = [_normalize(str(a)) for a in aliases if _normalize(str(a))]
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


def _match_curated(
    texts: list[str],
    catalog: list[tuple[list[str], ProductImage]],
) -> dict | None:
    """Return first curated catalog match across all texts."""
    for text in texts:
        normalized = _normalize(text)
        if not normalized:
            continue
        padded = f" {normalized} "
        for aliases, image in catalog:
            for alias in aliases:
                if f" {alias} " in padded:
                    return {
                        "title":         image.title,
                        "image_path":    image.image_path,
                        "source_doc":    image.source_doc,
                        "matched_alias": alias,
                    }
    return None


# ── figure index (01_ocr/output/figures/) ────────────────────────────────────

# PERF-001: mtime-based cache so we don't re-read 91 KB on every chat message
_fig_index_cache: dict = {"path": None, "mtime": None, "data": None}


def _load_figure_index(
    index_path: Path = DEFAULT_FIG_INDEX_PATH,
) -> dict:
    global _fig_index_cache
    manifests_dir = REPO_DIR / "01_ocr" / "output" / "manifests"
    rebuild_needed = False

    if not index_path.exists():
        rebuild_needed = True
    elif manifests_dir.exists():
        index_mtime = index_path.stat().st_mtime
        for manifest_file in manifests_dir.glob("*.json"):
            if manifest_file.stat().st_mtime > index_mtime:
                rebuild_needed = True
                break

    if rebuild_needed:
        try:
            import sys
            if str(DEMO_DIR) not in sys.path:
                sys.path.insert(0, str(DEMO_DIR))
            from build_figure_index import build_with_fallback

            index = build_with_fallback()
            index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
            _fig_index_cache = {"path": index_path, "mtime": index_path.stat().st_mtime, "data": index}
            return index
        except Exception as e:
            logger.warning(f"Failed to build figure index: {e}")
            if index_path.exists():
                try:
                    data = json.loads(index_path.read_text(encoding="utf-8"))
                    _fig_index_cache = {"path": index_path, "mtime": index_path.stat().st_mtime, "data": data}
                    return data
                except Exception as e:
                    logger.error(f"Failed to load existing index after build failure: {e}")
            return {}

    # Check module-level cache before hitting disk
    try:
        current_mtime = index_path.stat().st_mtime
    except OSError:
        return {}

    if (
        _fig_index_cache["path"] == index_path
        and _fig_index_cache["mtime"] == current_mtime
        and _fig_index_cache["data"] is not None
    ):
        return _fig_index_cache["data"]

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        _fig_index_cache = {"path": index_path, "mtime": current_mtime, "data": data}
        return data
    except Exception:
        return {}


def _product_mentioned(
    question: str,
    answer: str,
    retrieved: list[Any],
    catalog: list[tuple[list[str], ProductImage]] | None = None,
) -> bool:
    """
    Returns True if the query/answer appears to be about a specific product.
    Three signals checked in order:
      1. A curated catalog alias is present in the question or answer
      2. A model-number-like token is present (e.g. P01, EPW, ABZ-510)
      3. Any retrieved chunk contains substantive product-specific keywords
    """
    combined = f"{question} {answer}"

    # Signal 1: curated catalog alias match (fastest, most precise)
    if catalog:
        norm_combined = _normalize(combined)
        padded = f" {norm_combined} "
        for aliases, _ in catalog:
            for alias in aliases:
                if f" {alias} " in padded:
                    return True

    # Signal 2: model number pattern in question or answer
    if _MODEL_NUMBER_RE.search(combined):
        return True

    # Signal 3: product-domain vocabulary in retrieved chunks
    product_terms = re.compile(
        r"\b(clutch|brake|encoder|bearing|bushing|headset|motor|actuator|sensor"
        r"|shaft|module|spring|tension|capping|torque|linear|rotary|pneumatic"
        r"|hydraulic|servo|gear|coupling|drive|spindle)\b",
        re.IGNORECASE,
    )
    for item in retrieved:
        if product_terms.search(_chunk_text(item)):
            return True

    return False



def _figures_for_chunk(
    source_doc: str,
    page_num: int,
    fig_index: dict,
) -> list[str]:
    """Return absolute image paths for figures on a given page, sorted by size desc."""
    entry = fig_index.get(source_doc)
    if not entry:
        return []
    by_page: dict[str, list[dict]] = entry.get("by_page", {})
    
    # Path logic: if figures_base is absolute (from old scripts), it ignores REPO_DIR.
    # If it is relative (from updated scripts), it joins to REPO_DIR correctly.
    figures_base = REPO_DIR / Path(entry.get("figures_base", ""))

    page_figs = by_page.get(str(page_num), [])
    paths = []
    for fig in page_figs:                      # already sorted size-desc by build script
        full_path = figures_base / fig["filename"]
        if full_path.exists():
            paths.append(str(full_path))
    return paths


def _find_figures_by_filenames(filenames: list[str], fig_index: dict) -> list[dict]:
    gallery = []
    seen = set()
    for doc_name, doc_data in fig_index.items():
        figures_base = REPO_DIR / Path(doc_data.get("figures_base", ""))
        for page_num, figs in doc_data.get("by_page", {}).items():
            for fig in figs:
                fname = fig["filename"]
                if fname in filenames and fname not in seen:
                    full_path = figures_base / fname
                    if full_path.exists():
                        seen.add(fname)
                        gallery.append({
                            "image_path": str(full_path),
                            "title": Path(fname).stem.replace("_", " ").title(),
                            "source_doc": doc_name,
                        })
    return gallery


# ── public API ───────────────────────────────────────────────────────────────

def resolve_product_image(
    question: str,
    answer: str,
    retrieved: list[Any],
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    fig_index_path: Path = DEFAULT_FIG_INDEX_PATH,
    base_dir: Path = DEMO_DIR,
) -> dict | None:
    """
    Returns None if no product is mentioned.
    Otherwise returns a dict with keys:
        images      – list of dicts [{image_path, title, source_doc}, ...]  (1-3 items)
        from_index  – True if images came from the figure index (not curated catalog)
    """
    # ── guard: only show images when a product is actually mentioned ──────────
    catalog = _load_catalog(catalog_path=catalog_path, base_dir=base_dir)
    if not _product_mentioned(question, answer, retrieved, catalog=catalog):
        return None

    # ── layer 1: curated catalog (high-confidence, keyword match) ─────────────
    search_texts = [_chunk_text(item) for item in retrieved] + [answer, question]
    curated = _match_curated(search_texts, catalog)
    if curated:
        return {
            "images":     [curated],
            "from_index": False,
        }

    # ── layer 2: AI Display Tags ──────────────────────────────
    display_tags = re.findall(r"<DISPLAY:\s*([^>]+)>", answer)
    if display_tags:
        fig_index = _load_figure_index(fig_index_path)
        if fig_index:
            gallery = _find_figures_by_filenames(display_tags, fig_index)
            if gallery:
                return {
                    "images": gallery[:MAX_GALLERY_IMAGES],
                    "from_index": True,
                }

    return None
