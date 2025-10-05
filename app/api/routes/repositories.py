"""Repository, star, and related endpoints."""

from collections.abc import Sequence

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import Select, func, or_, select

from ... import models, schemas
from ..dependencies import DBSession

router = APIRouter(tags=["repositories"])


async def _get_user_or_404(user_id: int, db: DBSession) -> models.User:
    """Return the user or raise a 404."""

    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


async def _get_repository_or_404(repository_id: int, db: DBSession) -> models.Repository:
    """Return the repository or raise a 404."""

    repository = await db.get(models.Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
    return repository


async def _load_latest_sessions(repo_ids: Sequence[int], db: DBSession) -> dict[int, models.Session]:
    """Return the most recent session per repository."""

    if not repo_ids:
        return {}

    subq = (
        select(
            models.Session.repository_id,
            func.max(models.Session.id).label("max_id"),
        )
        .where(models.Session.repository_id.in_(repo_ids))
        .group_by(models.Session.repository_id)
        .subquery()
    )

    session_result = await db.scalars(
        select(models.Session).join(subq, models.Session.id == subq.c.max_id)
    )
    sessions = session_result.unique().all()
    return {session.repository_id: session for session in sessions}


def _base_repository_query() -> Select:
    """Return a selectable for repositories ordered by identifier."""

    return select(models.Repository).order_by(models.Repository.id)


@router.get(
    "/repositories",
    response_model=list[schemas.RepositoryRead],
    summary="List repositories (uploads)",
)
async def list_repositories(
    db: DBSession,
    owner_id: int | None = Query(default=None, description="Filter by owner identifier."),
    q: str | None = Query(default=None, description="Free-text search on name/description."),
    starred_by: int | None = Query(default=None, description="Filter by starring user."),
    include_session: bool = Query(False, description="Include latest session summary."),
) -> list[schemas.RepositoryRead]:
    """Return repositories applying optional filters."""

    query = _base_repository_query()

    if owner_id is not None:
        query = query.where(models.Repository.user_id == owner_id)

    if q:
        like_pattern = f"%{q}%"
        query = query.where(
            or_(
                models.Repository.name.ilike(like_pattern),
                models.Repository.description.ilike(like_pattern),
            )
        )

    starred_repo_ids: set[int] = set()

    if starred_by is not None:
        query = query.join(
            models.StarredRepository,
            models.Repository.id == models.StarredRepository.repository_id,
        ).where(models.StarredRepository.user_id == starred_by)

    result = await db.scalars(query)
    repositories = result.unique().all()

    if starred_by is not None:
        starred_repo_ids = {repo.id for repo in repositories}

    sessions_map: dict[int, models.Session] = {}
    if include_session:
        sessions_map = await _load_latest_sessions([repo.id for repo in repositories], db)

    for repo in repositories:
        setattr(repo, "starred", repo.id in starred_repo_ids)
        setattr(repo, "session", sessions_map.get(repo.id))

    return repositories


@router.post(
    "/repositories",
    response_model=schemas.RepositoryRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a repository (upload)",
)
async def create_repository(payload: schemas.RepositoryCreate, db: DBSession) -> schemas.RepositoryRead:
    """Create a new repository for the provided owner."""

    await _get_user_or_404(payload.user_id, db)

    existing = await db.scalar(
        select(models.Repository).where(
            models.Repository.user_id == payload.user_id,
            models.Repository.name == payload.name,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Repository name already exists for user")

    repository = models.Repository(
        user_id=payload.user_id,
        name=payload.name,
        description=payload.description,
    )
    db.add(repository)
    await db.commit()
    await db.refresh(repository)

    setattr(repository, "starred", False)
    setattr(repository, "session", None)
    return repository


@router.get(
    "/users/{user_id}/repositories",
    response_model=list[schemas.RepositoryRead],
    summary="List a user's uploads",
)
async def list_user_repositories(
    user_id: int,
    db: DBSession,
    include_session: bool = Query(True, description="Include latest session summary."),
) -> list[schemas.RepositoryRead]:
    """Return repositories owned by the specified user."""

    await _get_user_or_404(user_id, db)

    result = await db.scalars(
        _base_repository_query().where(models.Repository.user_id == user_id)
    )
    repositories = result.all()

    sessions_map: dict[int, models.Session] = {}
    if include_session:
        sessions_map = await _load_latest_sessions([repo.id for repo in repositories], db)

    for repo in repositories:
        setattr(repo, "starred", False)
        setattr(repo, "session", sessions_map.get(repo.id))

    return repositories


@router.get(
    "/repositories/{repository_id}",
    response_model=schemas.RepositoryRead,
    summary="Get repository",
)
async def get_repository(repository_id: int, db: DBSession) -> schemas.RepositoryRead:
    """Return a repository by identifier."""

    repository = await _get_repository_or_404(repository_id, db)
    setattr(repository, "starred", False)
    setattr(repository, "session", None)
    return repository


@router.put(
    "/repositories/{repository_id}",
    response_model=schemas.RepositoryRead,
    summary="Update repository",
)
async def update_repository(
    repository_id: int, payload: schemas.RepositoryUpdate, db: DBSession
) -> schemas.RepositoryRead:
    """Update repository attributes."""

    repository = await _get_repository_or_404(repository_id, db)

    updates = payload.model_dump(exclude_unset=True)
    if "name" in updates and updates["name"] is not None:
        repository.name = updates["name"]
    if "description" in updates:
        repository.description = updates["description"]

    await db.commit()
    await db.refresh(repository)

    setattr(repository, "starred", False)
    setattr(repository, "session", None)
    return repository


@router.delete(
    "/repositories/{repository_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete repository",
)
async def delete_repository(repository_id: int, db: DBSession) -> Response:
    """Delete a repository."""

    repository = await _get_repository_or_404(repository_id, db)
    await db.delete(repository)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put(
    "/repositories/{repository_id}/star",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Star a repository",
    tags=["stars"],
)
async def star_repository(repository_id: int, db: DBSession, user_id: int = Query(..., description="User identifier performing the star.")) -> Response:
    """Add a star association for the repository."""

    await _get_repository_or_404(repository_id, db)
    await _get_user_or_404(user_id, db)

    existing = await db.scalar(
        select(models.StarredRepository).where(
            models.StarredRepository.repository_id == repository_id,
            models.StarredRepository.user_id == user_id,
        )
    )
    if existing is not None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    star = models.StarredRepository(
        repository_id=repository_id,
        user_id=user_id,
    )
    db.add(star)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/repositories/{repository_id}/star",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unstar a repository",
    tags=["stars"],
)
async def unstar_repository(
    repository_id: int, db: DBSession, user_id: int = Query(..., description="User identifier undoing the star."),
) -> Response:
    """Remove a star association for the repository."""

    await _get_repository_or_404(repository_id, db)
    await _get_user_or_404(user_id, db)

    star = await db.scalar(
        select(models.StarredRepository).where(
            models.StarredRepository.repository_id == repository_id,
            models.StarredRepository.user_id == user_id,
        )
    )
    if star is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found or not starred")

    await db.delete(star)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/users/{user_id}/stars",
    response_model=list[schemas.StarRead],
    summary="List starred repositories",
    tags=["stars"],
)
async def list_user_stars(user_id: int, db: DBSession) -> list[schemas.StarRead]:
    """Return repositories starred by the user."""

    await _get_user_or_404(user_id, db)

    result = await db.scalars(
        select(models.StarredRepository)
        .where(models.StarredRepository.user_id == user_id)
        .order_by(models.StarredRepository.starred_at.desc())
    )
    return result.all()


__all__ = ["router"]
