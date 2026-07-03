import sys
from pathlib import Path
import numpy as np

# Insert THIS module's root at position 0 so src.* resolves to 02_llm_eval/src/
# not to 01_ocr/src/ (which would happen when pytest is run from the repo root).
_MODULE_ROOT = str(Path(__file__).parent.parent)
if _MODULE_ROOT in sys.path:
    sys.path.remove(_MODULE_ROOT)
sys.path.insert(0, _MODULE_ROOT)


# Mock sentence_transformers before they are imported by src/scorer.py
class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass
    def encode(self, texts, *args, **kwargs):
        n = len(texts) if isinstance(texts, (list, tuple)) else 1
        return np.ones((n, 384), dtype=np.float32) / np.sqrt(384)

class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass
    def predict(self, pairs, *args, **kwargs):
        if isinstance(pairs, (list, tuple)):
            if len(pairs) > 0 and isinstance(pairs[0], (list, tuple)):
                return np.ones(len(pairs), dtype=np.float32) * 0.9
            return 0.9
        return 0.9

import sentence_transformers
sentence_transformers.SentenceTransformer = FakeSentenceTransformer
sentence_transformers.CrossEncoder = FakeCrossEncoder

