"""
Pytest configuration for 01_ocr pipeline.
Adds the pipeline root to sys.path so `src` and `config` are importable
regardless of the directory pytest is invoked from.
"""
import sys
from pathlib import Path

# Insert 01_ocr/ at the front of sys.path
sys.path.insert(0, str(Path(__file__).parent))
