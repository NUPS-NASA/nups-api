"""Health and readiness routes."""

from fastapi import APIRouter

from ...config import get_settings

router = APIRouter(tags=["health"])


@router.get(
    "/",
    summary="Service health check",
    response_description="Current service health details.",
)
async def read_root() -> dict[str, str | int]:
    """Expose basic runtime metadata for uptime monitoring."""

    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "port": settings.app_port,
        "database": settings.database_url,
    }


__all__ = ["router"]
