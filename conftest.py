"""Root conftest.py — ensures each sub-module's own package root is on sys.path
BEFORE any of its src.* imports run.  This fixes the namespace collision where
running pytest from the repo root caused 02_llm_eval/tests to import
01_ocr/src/models.py instead of 02_llm_eval/src/models.py.
"""
import sys
from pathlib import Path

# Repo root (this file's directory)
REPO = Path(__file__).parent

# Shared utilities directory (for shared.hashing, etc.)
SHARED = REPO / "shared"
if str(SHARED.parent) not in sys.path:
    sys.path.insert(0, str(REPO))
