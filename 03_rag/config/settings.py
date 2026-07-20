"""
Centralized configuration for the MultiDimensions RAG pipeline.

All settings are driven by environment variables (RAG_ prefix) or a .env file,
following the Twelve-Factor App methodology. No secrets have hardcoded defaults
— missing required secrets at runtime raise a clear ValidationError.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_RAG_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _RAG_DIR / ".env"


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="RAG_",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # ── Core Directories ────────────────────────────────────────────────────────
    ocr_output_dir: Path = Field(
        default=_RAG_DIR.parent / "data" / "ocr_output",
        description="Directory containing OCR output markdown files.",
    )
    index_dir: Path = Field(
        default=_RAG_DIR / "index",
        description="Directory for the Qdrant on-disk vector index.",
    )
    results_dir: Path = Field(
        default=_RAG_DIR / "results",
        description="Directory for evaluation results.",
    )
    web_cache_dir: Path = Field(
        default=_RAG_DIR / ".cache" / "web_search",
        description="Directory for DuckDuckGo search result cache.",
    )

    # ── Generator Model ─────────────────────────────────────────────────────────
    generator_model_id: str = Field(
        default="Qwen/Qwen3-8B",
        description="HuggingFace model ID for the answer-generation LLM.",
    )

    # ── RAG Parameters ───────────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)

    embed_model: str = Field(default="BAAI/bge-large-en-v1.5")
    qdrant_collection_name: str = Field(
        default="rag_chunks",
        description="Qdrant collection name. Override to support multiple tenants.",
    )
    top_k_dense: int = Field(default=30, ge=1, le=200)
    top_k_sparse: int = Field(default=30, ge=1, le=200)
    top_k_rerank: int = Field(default=8, ge=1, le=50)
    hyde_enabled: bool = Field(default=True)
    web_search_enabled: bool = Field(default=True)
    web_search_max_results: int = Field(default=15, ge=1, le=50)
    web_snippet_max_chars: int = Field(
        default=500,
        description="Max characters per web snippet before injection into the LLM prompt. "
                    "Limits prompt-injection surface area.",
    )

    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(
        default=20,
        description="Maximum LLM queries per user per minute. 0 = disabled.",
    )

    # ── Logging & Audit ──────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_file: Path = Field(default=_RAG_DIR / "logs" / "rag.log")
    audit_log_file: Path = Field(default=_RAG_DIR / "logs" / "audit.log")

    # ── Security & Auth (Keycloak OIDC) ─────────────────────────────────────────
    auth_enabled: bool = Field(
        default=False,
        description="Set True in production. False bypasses auth for local development.",
    )
    keycloak_server_url: str = Field(default="https://keycloak.example.com")
    keycloak_realm: str = Field(default="master")
    keycloak_client_id: str = Field(default="rag-bot")
    # No default — must be explicitly set in production environment
    keycloak_client_secret: Optional[str] = Field(default=None)
    # Redirect URI that Keycloak should send users back to after login
    app_base_url: str = Field(
        default="http://localhost:8501",
        description="Base URL of this Streamlit app (used for OAuth redirect URI).",
    )
    # Cookie / session signing key — must be overridden in production
    auth_session_key: Optional[str] = Field(default=None)

    # ── Validators ───────────────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v.upper()

    @model_validator(mode="after")
    def validate_auth_config(self) -> "RAGSettings":
        """Prevent silent misconfigurations in production auth mode."""
        if self.auth_enabled:
            if not self.keycloak_client_secret:
                raise ValueError(
                    "RAG_KEYCLOAK_CLIENT_SECRET must be set when RAG_AUTH_ENABLED=true. "
                    "Never rely on a default secret in production."
                )
            if not self.auth_session_key:
                raise ValueError(
                    "RAG_AUTH_SESSION_KEY must be set when RAG_AUTH_ENABLED=true. "
                    "Use a randomly generated 32+ character secret string."
                )
            if "example.com" in self.keycloak_server_url:
                raise ValueError(
                    "RAG_KEYCLOAK_SERVER_URL still points to the placeholder URL. "
                    "Set a real Keycloak server URL."
                )
        return self

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist. Call at startup."""
        for directory in [
            self.index_dir,
            self.results_dir,
            self.web_cache_dir,
            self.log_file.parent,
            self.audit_log_file.parent,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


settings = RAGSettings()
