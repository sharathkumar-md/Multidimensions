from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_RAG_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _RAG_DIR / ".env"

class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="RAG_",
        extra="ignore",
    )

    # Core Directories
    ocr_output_dir: Path = Field(default=_RAG_DIR.parent / "data" / "ocr_output")
    index_dir: Path = Field(default=_RAG_DIR / "index")
    results_dir: Path = Field(default=_RAG_DIR / "results")
    web_cache_dir: Path = Field(default=_RAG_DIR / ".cache" / "web_search")

    # RAG Parameters
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=100)

    embed_model: str = Field(default="BAAI/bge-large-en-v1.5")
    top_k_dense: int = Field(default=30)
    top_k_sparse: int = Field(default=30)
    top_k_rerank: int = Field(default=8)
    hyde_enabled: bool = Field(default=True)
    web_search_enabled: bool = Field(default=True)

    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Logging & Audit
    log_level: str = Field(default="INFO")
    log_file: Path = Field(default=_RAG_DIR / "logs" / "rag.log")
    audit_log_file: Path = Field(default=_RAG_DIR / "logs" / "audit.log")

    # Security & Auth (To be used in Phase 3/4)
    admin_password_hash: str = Field(default="")
    auth_cookie_key: str = Field(default="multidimensions_auth_secret_key_123")

settings = RAGSettings()
