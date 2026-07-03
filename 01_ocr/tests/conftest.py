"""conftest.py for 01_ocr/tests — ensures src.* resolves to 01_ocr/src/."""
import sys
from pathlib import Path

_MODULE_ROOT = str(Path(__file__).parent.parent)
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)
