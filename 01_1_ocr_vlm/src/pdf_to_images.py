from __future__ import annotations

from pathlib import Path

import fitz  # pymupdf
from PIL import Image


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[tuple[int, Image.Image]]:
    """Return list of (1-based page_num, PIL Image) for every page in the PDF."""
    doc = fitz.open(str(pdf_path))
    pages: list[tuple[int, Image.Image]] = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append((i, img))
    doc.close()
    return pages
