"""
API-layer settings — separate from the RAG pipeline settings.
Reads from 05_api/.env (API_ prefix).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_API_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _API_DIR / ".env"


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="API_",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info")

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        description="Comma-separated allowed origins.",
    )

    session_store_path: Path = Field(
        default_factory=lambda: Path(".sessions.json"),
        description="Path to the local file-backed session store (deprecated, use database_url instead).",
    )
    database_url: str = Field(
        default="sqlite+aiosqlite:///.sessions.db",
        description="Connection string for the enterprise database (PostgreSQL/SQLite).",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ── Auth ──────────────────────────────────────────────────────────────────
    auth_enabled: bool = Field(
        default=False,
        description="False bypasses JWT validation for local development.",
    )
    keycloak_server_url: str = Field(default="https://keycloak.example.com")
    keycloak_realm: str = Field(default="multidimensions")
    keycloak_client_id: str = Field(default="rag-sales-bot")

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = Field(default=20, ge=0)

    # ── Session store ─────────────────────────────────────────────────────────
    session_store_path: Path = Field(default=_API_DIR / ".sessions.json")

    # ── Upload ────────────────────────────────────────────────────────────────
    upload_dir: Path = Field(default=_API_DIR.parent / "data" / "uploads")
    max_upload_size_mb: int = Field(default=50, ge=1)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.lower()

    def ensure_directories(self) -> None:
        """Create necessary directories at startup."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.session_store_path.parent.mkdir(parents=True, exist_ok=True)


api_settings = APISettings()
