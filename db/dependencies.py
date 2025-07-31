# db/dependencies.py
import os

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from core.config import settings

load_dotenv()
# 비동기 DB 엔진 생성
async_db_uri = (
    f"mysql+aiomysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
    f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB_NAME}"
)
engine = create_async_engine(async_db_uri, pool_recycle=3600)

# 비동기 세션 메이커
AsyncSessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db() -> AsyncSession:
    """
    API 의존성 주입(Dependency Injection)을 통해
    각 API 요청마다 독립적인 비동기 DB 세션을 생성하고 제공합니다.
    """
    async with AsyncSessionLocal() as session:
        yield session