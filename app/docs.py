"""Centralised OpenAPI/Swagger documentation helpers."""

from __future__ import annotations

from typing import Any

API_DESCRIPTION = """FastAPI service for the nups platform delivering user management, collaborative projects, uploads, and processing telemetry.\n\n## Domain highlights\n- **Users & profiles** — manage accounts, social graphs, and profile metadata.\n- **Repositories & datasets** — organise uploads, dataset versions, and per-item assets.\n- **Projects & pins** — coordinate contributors, link uploads, and curate pinned work.\n- **Sessions & candidates** — inspect processing runs, pipeline steps, and review detection outcomes.\n\n## Operational notes\n- Every route is mounted beneath the `/api` prefix.\n- Write endpoints persist changes immediately via SQLAlchemy + async sessions.\n- Authentication is omitted in this sample service; integrate your preferred provider before production use."""

CONTACT_INFO = {
    "name": "nups-api maintainers",
    "url": "https://github.com/tgim4253/nups-api",
}

LICENSE_INFO = {
    "name": "MIT License",
    "url": "https://opensource.org/licenses/MIT",
}

TERMS_OF_SERVICE_URL = "https://github.com/tgim4253/nups-api#readme"

SERVERS = [
    {
        "url": "http://localhost:8000",
        "description": "Local development",
    }
]

TAGS_METADATA = [
    {
        "name": "health",
        "description": "Liveness and readiness probes used by deploy environments.",
    },
    {
        "name": "users",
        "description": "Create accounts, manage profiles, and control follow relationships.",
    },
    {
        "name": "repositories",
        "description": "CRUD for uploads plus search, filtering, and ownership views.",
    },
    {
        "name": "stars",
        "description": "Star and unstar repositories; inspect starred collections per user.",
    },
    {
        "name": "projects",
        "description": "Collaborative project registry, membership management, and linked uploads.",
    },
    {
        "name": "pins",
        "description": "Pin, reorder, or remove projects showcased on user profiles.",
    },
    {
        "name": "datasets",
        "description": "Versioned dataset catalogue associated with repositories.",
    },
    {
        "name": "data",
        "description": "Fine-grained data artefacts contained within dataset versions.",
    },
    {
        "name": "uploads",
        "description": "Stage FITS files, commit uploads, and trigger processing sessions.",
    },
    {
        "name": "sessions",
        "description": "Processing session history and pipeline step inspection.",
    },
    {
        "name": "candidates",
        "description": "Candidate detection review including verification workflow.",
    },
    {
        "name": "stats",
        "description": "User contribution metrics, counters, and time-series data.",
    },
]

SWAGGER_UI_PARAMETERS = {
    "docExpansion": "list",
    "defaultModelsExpandDepth": 0,
    "tryItOutEnabled": True,
}


def get_tags_metadata() -> list[dict[str, Any]]:
    """Return a copy of tag metadata to prevent accidental mutation."""

    return [tag.copy() for tag in TAGS_METADATA]


def get_swagger_ui_parameters() -> dict[str, Any]:
    """Return Swagger UI options applied to the FastAPI app."""

    return SWAGGER_UI_PARAMETERS.copy()


def get_contact_info() -> dict[str, str]:
    """Contact metadata for the FastAPI application."""

    return CONTACT_INFO.copy()


def get_license_info() -> dict[str, str]:
    """License metadata for the FastAPI application."""

    return LICENSE_INFO.copy()


def get_servers() -> list[dict[str, str]]:
    """List of server entries advertised via OpenAPI."""

    return [server.copy() for server in SERVERS]
