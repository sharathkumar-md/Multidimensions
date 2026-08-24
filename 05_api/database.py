from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from api_config import api_settings
from urllib.parse import quote_plus

if api_settings.database_password:
    DATABASE_URL = (
        "postgresql+asyncpg://"
        f"keycloak_app:{quote_plus(api_settings.database_password)}"
        "@/keycloak"
        "?host=/cloudsql/sales-assistant-multid:us-central1:multidimensions-db"
    )
else:
    DATABASE_URL = api_settings.database_url


engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
