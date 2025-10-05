from collections.abc import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import get_settings

settings = get_settings()
database_url = settings.database_url
engine_kwargs: dict[str, object] = {"future": True}

database_backend = make_url(database_url).get_backend_name()

if database_backend == "sqlite":
    engine_kwargs["connect_args"] = {"timeout": 30}

engine = create_async_engine(database_url, **engine_kwargs)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base for SQLAlchemy models."""

    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide a scoped asynchronous database session."""

    async with SessionLocal() as session:
        yield session


if database_backend == "sqlite":

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=30000;")
        cursor.close()
