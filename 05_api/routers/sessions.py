"""Sessions router — CRUD for conversation history."""
from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from auth import get_current_user
from models import CreateSessionRequest, Message, Session, UserInfo
from session_store import store

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.get("", response_model=list[Session], summary="List all sessions for the current user")
async def list_sessions(user: UserInfo = Depends(get_current_user)) -> list[Session]:
    sessions = await store.get_sessions(user.sub)
    logger.debug(f"list_sessions: {len(sessions)} for {user.email}")
    return sessions


@router.post("", response_model=Session, status_code=status.HTTP_201_CREATED, summary="Create a new session")
async def create_session(
    body: CreateSessionRequest,
    user: UserInfo = Depends(get_current_user),
) -> Session:
    session = await store.create_session(user_id=user.sub, title=body.title)
    logger.info(f"Created session {session.id} for {user.email}")
    return session


@router.get("/{session_id}", response_model=Session, summary="Get a single session")
async def get_session(
    session_id: str,
    user: UserInfo = Depends(get_current_user),
) -> Session:
    session = await store.get_session(session_id, user.sub)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return session


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a session")
async def delete_session(
    session_id: str,
    user: UserInfo = Depends(get_current_user),
) -> None:
    deleted = await store.delete_session(session_id, user.sub)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    logger.info(f"Deleted session {session_id} by {user.email}")


@router.get("/{session_id}/messages", response_model=list[Message], summary="Get all messages in a session")
async def get_messages(
    session_id: str,
    user: UserInfo = Depends(get_current_user),
) -> list[Message]:
    session = await store.get_session(session_id, user.sub)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return await store.get_messages(session_id, user.sub)
