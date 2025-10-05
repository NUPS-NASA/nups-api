"""Centralised OpenAPI/Swagger documentation helpers."""

from __future__ import annotations

from typing import Any

API_DESCRIPTION = """REST API that powers the nups sample platform. It is built with FastAPI and ships with async SQLAlchemy models, JWT based authentication helpers, and file-handling utilities used by the upload pipeline.\n\n## 주요 기능 (Key capabilities)\n- **Health** — readiness and version metadata used by deployment probes.\n- **Users & auth** — register accounts, manage profiles, and issue/refresh access tokens.\n- **Repositories & uploads** — organise repository metadata, staged uploads, and committed assets.\n- **Datasets & data artefacts** — publish dataset versions and inspect per-item entries.\n- **Projects & members** — curate project spaces, manage membership, and maintain pinned showcases.\n- **Sessions & candidates** — review processing sessions and evaluate detection candidates.\n- **Stats** — surface contribution metrics and aggregate counters.\n\n## Operational notes\n- All endpoints are namespaced beneath the `/api` prefix.\n- Write operations persist immediately via SQLAlchemy async sessions.\n- Default server configuration listens on port 4000 (override with the `APP_PORT` setting)."""

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
        "url": "http://localhost:4000",
        "description": "Local development (default FastAPI settings)",
    },
    {
        "url": "http://localhost:8000",
        "description": "Alternate development port (e.g. uvicorn default)",
    },
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
        "description": "Project catalogue with CRUD operations, membership management, and search helpers.",
    },
    {
        "name": "pins",
        "description": "Manage a user's pinned project showcase including ordering and removal.",
    },
    {
        "name": "datasets",
        "description": "Dataset registry for versioned releases associated with repositories.",
    },
    {
        "name": "data",
        "description": "Access granular data artefacts stored inside dataset versions.",
    },
    {
        "name": "uploads",
        "description": "Handle staged files, commit uploads, and manage storage lifecycle events.",
    },
    {
        "name": "pipeline",
        "description": "Inspect the canonical analysis pipeline and step metadata.",
    },
    {
        "name": "sessions",
        "description": "Inspect processing sessions, pipeline steps, and derived outcomes.",
    },
    {
        "name": "candidates",
        "description": "Review detection candidates surfaced from processing sessions.",
    },
    {
        "name": "stats",
        "description": "Retrieve user contribution statistics, counters, and aggregates.",
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
