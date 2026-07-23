from sqlalchemy import Column, String, DateTime, Integer, JSON
from datetime import datetime
import uuid

from database import Base

class DBSession(Base):
    __tablename__ = 'sessions'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, nullable=False)
    title = Column(String, default="New conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    message_count = Column(Integer, default=0)

class DBMessage(Base):
    __tablename__ = 'messages'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    sources = Column(JSON, nullable=True)
    product_images = Column(JSON, nullable=True)
    route = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
