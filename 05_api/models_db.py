from datetime import datetime, timezone
import uuid

from sqlalchemy import Column, ForeignKey, String, DateTime, Integer, JSON
from sqlalchemy.orm import relationship

from database import Base


def _utcnow():
    """Timezone-aware UTC now. Replaces deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


class DBSession(Base):
    __tablename__ = 'sessions'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    title = Column(String, default="New conversation")
    created_at = Column(DateTime(timezone=True), default=_utcnow)  # Issue #12 fix
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)  # Issue #12 fix
    message_count = Column(Integer, default=0)

    # Issue #11 fix: back-reference enables cascade delete of messages when session is deleted
    messages = relationship("DBMessage", back_populates="session", cascade="all, delete-orphan")


class DBMessage(Base):
    __tablename__ = 'messages'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    # Issue #11 fix: proper FK constraint — orphaned messages are now impossible
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    sources = Column(JSON, nullable=True)
    product_images = Column(JSON, nullable=True)
    route = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow)  # Issue #12 fix

    session = relationship("DBSession", back_populates="messages")
