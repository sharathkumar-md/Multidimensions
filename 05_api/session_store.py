"""
Enterprise Session store — PostgreSQL / SQLite via SQLAlchemy (Async)
Replaces the old file-backed JSON persistence.
"""
from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from models import Message, Session
from database import AsyncSessionLocal, engine, Base
from models_db import DBSession, DBMessage

class DatabaseSessionStore:
    """Async SQLAlchemy session store."""

    async def initialize(self):
        # Create tables if they don't exist
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("DatabaseSessionStore initialized and tables created.")

    async def create_session(self, user_id: str, title: str = "New conversation") -> Session:
        async with AsyncSessionLocal() as db:
            db_session = DBSession(
                id=str(uuid.uuid4()),
                title=title,
                user_id=user_id,
            )
            db.add(db_session)
            await db.commit()
            await db.refresh(db_session)
            logger.info(f"Created session {db_session.id} for user {user_id}")
            return Session(
                id=db_session.id,
                title=db_session.title,
                user_id=db_session.user_id,
                created_at=db_session.created_at.isoformat(),
                updated_at=db_session.updated_at.isoformat(),
                message_count=db_session.message_count,
            )

    async def get_sessions(self, user_id: str) -> List[Session]:
        async with AsyncSessionLocal() as db:
            stmt = select(DBSession).where(DBSession.user_id == user_id).order_by(DBSession.updated_at.desc())
            result = await db.execute(stmt)
            sessions = result.scalars().all()
            return [
                Session(
                    id=s.id,
                    title=s.title,
                    user_id=s.user_id,
                    created_at=s.created_at.isoformat(),
                    updated_at=s.updated_at.isoformat(),
                    message_count=s.message_count,
                ) for s in sessions
            ]

    async def get_session(self, session_id: str, user_id: str) -> Optional[Session]:
        async with AsyncSessionLocal() as db:
            stmt = select(DBSession).where(DBSession.id == session_id, DBSession.user_id == user_id)
            result = await db.execute(stmt)
            s = result.scalars().first()
            if not s:
                return None
            return Session(
                id=s.id,
                title=s.title,
                user_id=s.user_id,
                created_at=s.created_at.isoformat(),
                updated_at=s.updated_at.isoformat(),
                message_count=s.message_count,
            )

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        async with AsyncSessionLocal() as db:
            stmt = delete(DBSession).where(DBSession.id == session_id, DBSession.user_id == user_id)
            result = await db.execute(stmt)
            if result.rowcount > 0:
                await db.commit()
                logger.info(f"Deleted session {session_id}")
                return True
            return False

    async def update_title(self, session_id: str, title: str) -> None:
        async with AsyncSessionLocal() as db:
            stmt = update(DBSession).where(DBSession.id == session_id).values(title=title, updated_at=datetime.utcnow())
            await db.execute(stmt)
            await db.commit()

    async def get_messages(self, session_id: str, user_id: str) -> List[Message]:
        # Validate session ownership first
        if not await self.get_session(session_id, user_id):
            return []
            
        async with AsyncSessionLocal() as db:
            stmt = select(DBMessage).where(DBMessage.session_id == session_id).order_by(DBMessage.created_at.asc())
            result = await db.execute(stmt)
            messages = result.scalars().all()
            return [
                Message(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    sources=m.sources if m.sources else [],
                    product_images=m.product_images if m.product_images else [],
                    route=m.route,
                    created_at=m.created_at.isoformat(),
                ) for m in messages
            ]

    async def append_message(self, session_id: str, message: Message) -> None:
        async with AsyncSessionLocal() as db:
            db_msg = DBMessage(
                id=message.id,
                session_id=session_id,
                role=message.role,
                content=message.content,
                sources=[s.model_dump() for s in message.sources],
                product_images=[i.model_dump() for i in message.product_images],
                route=message.route.value if message.route else None,
            )
            db.add(db_msg)
            
            # Update session message_count and updated_at
            stmt = update(DBSession).where(DBSession.id == session_id).values(
                message_count=DBSession.message_count + 1,
                updated_at=datetime.utcnow()
            )
            await db.execute(stmt)
            await db.commit()


# Singleton
store = DatabaseSessionStore()
