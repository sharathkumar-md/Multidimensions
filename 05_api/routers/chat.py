"""
Chat router — SSE streaming endpoint.

POST /api/chat
  Body: { session_id, question }
  Response: text/event-stream
  Each event: data: <JSON>\n\n
  Final event: data: {"done":true,"sources":[...],"route":"LOCAL"|"WEB"|"NONE"}
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger

from auth import get_current_user
from models import ChatRequest, Message, RouteType, Source, UserInfo
from rate_limit import rate_limited
from rag_service import stream_answer
from session_store import store

router = APIRouter(prefix="/api", tags=["Chat"])


async def _event_stream(session_id: str, question: str, user: UserInfo):
    """
    Internal async generator that:
    1. Saves the user message to the store
    2. Streams tokens from the RAG service
    3. Accumulates full response
    4. Saves the assistant message with sources on completion
    """
    # Persist user message
    user_msg = Message(
        id=str(uuid.uuid4()),
        role="user",
        content=question,
        created_at=datetime.utcnow(),
    )
    await store.append_message(session_id, user_msg)

    # Auto-title the session from first question (truncated)
    session = await store.get_session(session_id, user.sub)
    if session and session.message_count <= 2:
        title = question[:60] + ("…" if len(question) > 60 else "")
        await store.update_title(session_id, title)

    # Stream from RAG
    full_content = ""
    final_sources = []
    final_images = []
    final_route = "NONE"

    try:
        async for raw in stream_answer(question):
            yield f"data: {raw}\n\n"
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if "token" in payload:
                full_content += payload["token"]

            if payload.get("done"):
                final_sources = payload.get("sources", [])
                final_images = payload.get("product_images", [])
                final_route = payload.get("route", "NONE")

    except Exception as exc:
        logger.error(f"Stream error for session {session_id}: {exc}")
        error_payload = json.dumps({"error": str(exc), "done": True})
        yield f"data: {error_payload}\n\n"

    finally:
        # Persist assistant message
        if full_content:
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=full_content,
                created_at=datetime.utcnow(),
                sources=[Source(**s) for s in final_sources],
                route=RouteType(final_route) if final_route in RouteType._value2member_map_ else None,
            )
            await store.append_message(session_id, assistant_msg)
            logger.info(
                f"Completed chat | session={session_id} | "
                f"route={final_route} | tokens={len(full_content)} | user={user.email}"
            )


@router.post("/chat", summary="Send a message and stream the AI response")
async def chat(
    body: ChatRequest,
    user: UserInfo = Depends(rate_limited),
) -> StreamingResponse:
    """
    Streams the AI response as Server-Sent Events (SSE).

    - Each `data:` line is a JSON object with either `token` or `done=true`.
    - The final event includes `sources`, `product_images`, and `route`.
    - Rate-limited per user (configurable via API_RATE_LIMIT_PER_MINUTE).
    """
    # Validate session ownership
    session = await store.get_session(body.session_id, user.sub)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )

    logger.info(f"Chat | session={body.session_id} | user={user.email} | q={body.question[:80]}")

    return StreamingResponse(
        _event_stream(body.session_id, body.question, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
