from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fitz
from PIL import Image


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> tuple[int, Iterator[tuple[int, Image.Image]]]:
    """Yield (page_num, PIL.Image) pairs for every page of a PDF.

    The fitz.Document is opened inside the generator and closed in a
    try/finally block so the handle is always released — even if the caller
    breaks out early or an exception is raised mid-iteration.
    """
    # Peek at page count without keeping the document open at the outer scope.
    with fitz.open(str(pdf_path)) as _peek:
        total_pages = len(_peek)

    def _generator() -> Iterator[tuple[int, Image.Image]]:
        doc = fitz.open(str(pdf_path))
        try:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            for i, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                yield i, img
        finally:
            doc.close()

    return total_pages, _generator()
