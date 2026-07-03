"""
shared/hashing.py
-----------------
Single source-of-truth for SHA-256 file hashing.

Previously this logic was copy-pasted in three places:
  - 03_rag/ingest.py  (compute_sha256)
  - 04_demo/app.py    (inline in _check_and_run_ingest)
  - 01.1_ocr_vlm/src/pipeline.py  (_sha256)

CODE-009: Centralised here so future changes only need to be made in one place.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of *path* using a 64 KB streaming read."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
