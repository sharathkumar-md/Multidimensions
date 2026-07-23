"""Shared Pydantic models for request / response bodies."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserInfo(BaseModel):
    sub: str                     # Keycloak subject (unique user ID)
    email: str
    name: str = ""
    roles: list[str] = Field(default_factory=list)

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles or "rag-admin" in self.roles


# ── Route ──────────────────────────────────────────────────────────────────────

class RouteType(str, Enum):
    LOCAL = "LOCAL"
    WEB = "WEB"
    NONE = "NONE"


# ── Source ────────────────────────────────────────────────────────────────────

class Source(BaseModel):
    source_doc: str
    page_num: int = 0
    snippet: str = ""


class ProductImage(BaseModel):
    image_path: str
    title: str
    source_doc: str


# ── Messages ─────────────────────────────────────────────────────────────────

class Message(BaseModel):
    id: str
    role: str                    # "user" | "assistant"
    content: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sources: list[Source] = Field(default_factory=list)
    product_images: list[ProductImage] = Field(default_factory=list)
    route: Optional[RouteType] = None


# ── Sessions ──────────────────────────────────────────────────────────────────

class Session(BaseModel):
    id: str
    title: str = "New conversation"
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0


class CreateSessionRequest(BaseModel):
    title: str = "New conversation"


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    question: str = Field(..., min_length=1, max_length=4000)


class StreamToken(BaseModel):
    """One SSE event payload — either a text token or the final done event."""
    token: Optional[str] = None
    done: bool = False
    sources: list[Source] = Field(default_factory=list)
    product_images: list[ProductImage] = Field(default_factory=list)
    route: Optional[RouteType] = None
    error: Optional[str] = None


# ── Admin ─────────────────────────────────────────────────────────────────────

class IndexStats(BaseModel):
    n_chunks: int
    n_docs: int
    last_updated: Optional[datetime]
    gpu_available: bool


class IngestionStatus(BaseModel):
    running: bool
    progress: float = 0.0        # 0.0 – 1.0
    current_file: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    filename: str
    size_bytes: int
    message: str = "Upload successful. Ingestion will start automatically."


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    rag_loaded: bool = False
    gpu_available: bool = False
