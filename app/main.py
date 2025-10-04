from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import health, users
from .config import get_settings
from .database import Base, engine

settings = get_settings()

API_DESCRIPTION = """Async FastAPI service that manages users, extended profile data, and follow relationships."""

TAGS_METADATA = [
    {
        "name": "health",
        "description": "Liveness and readiness probes for infrastructure monitoring.",
    },
    {
        "name": "users",
        "description": "CRUD operations for user accounts, profile records, and follow relationships.",
    },
]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Ensure database schema is created on startup."""

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""

    app = FastAPI(
        title=settings.app_name,
        description=API_DESCRIPTION,
        version="1.0.0",
        contact={
            "name": "nups-api maintainers",
            "url": "https://github.com/tgim4253/nups-api",
        },
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        swagger_ui_parameters={"docExpansion": "list"},
    )

    app.include_router(health.router)
    app.include_router(users.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=True,
    )
