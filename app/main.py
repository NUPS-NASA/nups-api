from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import datasets, health, projects, repositories, sessions, stats, users
from .config import get_settings
from .database import Base, engine
from .docs import (
    API_DESCRIPTION,
    TERMS_OF_SERVICE_URL,
    get_contact_info,
    get_license_info,
    get_servers,
    get_swagger_ui_parameters,
    get_tags_metadata,
)

settings = get_settings()


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
        contact=get_contact_info(),
        license_info=get_license_info(),
        terms_of_service=TERMS_OF_SERVICE_URL,
        servers=get_servers(),
        openapi_tags=get_tags_metadata(),
        lifespan=lifespan,
        swagger_ui_parameters=get_swagger_ui_parameters(),
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(users.router, prefix="/api")
    app.include_router(repositories.router, prefix="/api")
    app.include_router(projects.router, prefix="/api")
    app.include_router(datasets.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")
    app.include_router(stats.router, prefix="/api")

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
