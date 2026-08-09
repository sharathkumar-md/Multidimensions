"""Shared Pydantic models for request / response bodies."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, AliasChoices


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    sources: list[Source] = Field(default_factory=list)
    product_images: list[ProductImage] = Field(default_factory=list)
    route: Optional[RouteType] = None


# ── Sessions ──────────────────────────────────────────────────────────────────

class Session(BaseModel):
    id: str
    title: str = "New conversation"
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_count: int = 0


class CreateSessionRequest(BaseModel):
    title: str = "New conversation"


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        validation_alias=AliasChoices("session_id", "sessionId"),
        serialization_alias="session_id",
        description="Active session ID",
    )
    # Issue #6 fix: 4000 chars (~1000 tokens) caused OOM via HyDE double-pass.
    # 1000 chars (~250 tokens) is ample for any real sales query.
    question: str = Field(..., min_length=1, max_length=1000)
    # Fix 001: web_search toggle wired end-to-end
    web_search: bool = Field(default=False, description="Route query through web retrieval")
    # Conversation history for multi-turn context (optional, loaded from session if not provided)
    history: list[Message] = Field(default_factory=list, description="Previous messages in the conversation")


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
