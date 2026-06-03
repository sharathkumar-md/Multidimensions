from pathlib import Path

import fitz
from PIL import Image


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[tuple[int, Image.Image]]:
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append((i, img))
    doc.close()
    return pages
