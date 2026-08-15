import sys
import os
from pathlib import Path
import numpy as np

# Set test environment variables BEFORE importing settings
os.environ.setdefault("RAG_AUTH_ENABLED", "false")
os.environ.setdefault("RAG_KEYCLOAK_CLIENT_SECRET", "test-secret-for-testing-only")
os.environ.setdefault("RAG_AUTH_SESSION_KEY", "test-session-key-for-testing-only-32chars")
os.environ.setdefault("RAG_KEYCLOAK_SERVER_URL", "https://test-keycloak.example.com")

sys.path.insert(0, str(Path(__file__).parent))

# Mock sentence_transformers
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

