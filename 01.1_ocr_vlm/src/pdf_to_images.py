from pathlib import Path

import fitz
from PIL import Image

from collections.abc import Iterator

def pdf_to_images(pdf_path: Path, dpi: int = 150) -> tuple[int, Iterator[tuple[int, Image.Image]]]:
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    
    def _generator():
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            yield i, img
        doc.close()
        
    return total_pages, _generator()
