"""Test configuration for 05_api tests."""
import os
import sys
from pathlib import Path

import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine

# Add 03_rag to sys.path for imports
_PROJECT_ROOT = Path(__file__).parent.parent
_RAG_DIR = _PROJECT_ROOT / "03_rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

# Set test environment variables BEFORE importing database/main
os.environ.setdefault("API_AUTH_ENABLED", "false")
os.environ.setdefault("RAG_AUTH_ENABLED", "false")
os.environ.setdefault("API_RATE_LIMIT_PER_MINUTE", "0")
os.environ.setdefault("API_DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from database import Base  # noqa: E402

# Override the database engine for tests
_test_engine = create_async_engine(
    "sqlite+aiosqlite:///:memory:",
    connect_args={"check_same_thread": False}
)

# Patch the database module's engine
import database  # noqa: E402

database.engine = _test_engine
database.AsyncSessionLocal.configure(bind=_test_engine)


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create database tables for each test."""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
