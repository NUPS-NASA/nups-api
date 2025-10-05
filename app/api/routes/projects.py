"""Project, membership, and pinned project endpoints."""

from collections.abc import Sequence

from fastapi import APIRouter, HTTPException, Query, Response, status
from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import selectinload

from ... import models, schemas
from ..dependencies import DBSession

router = APIRouter(tags=["projects"])

_ALLOWED_ROLES = {"pm", "member"}


async def _get_project_or_404(project_id: int, db: DBSession) -> models.Project:
    project = await db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


async def _get_user_or_404(user_id: int, db: DBSession) -> models.User:
    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


async def _attach_project_metadata(projects: Sequence[models.Project], db: DBSession) -> None:
    if not projects:
        return

    project_ids = [project.id for project in projects]
    counts_result = await db.execute(
        select(
            models.ProjectUser.project_id,
            func.count(models.ProjectUser.user_id).label("member_count"),
        ).where(models.ProjectUser.project_id.in_(project_ids)).group_by(models.ProjectUser.project_id)
    )
    counts_map = {row.project_id: row.member_count for row in counts_result}

    for project in projects:
        setattr(project, "members_count", counts_map.get(project.id, 0))
        # Tags are not backed by storage yet; surface an empty list for the schema.
        setattr(project, "tags", [])


def _base_project_query() -> Select:
    return select(models.Project).order_by(models.Project.id)


@router.get(
    "/projects",
    response_model=list[schemas.ProjectRead],
    summary="List projects",
)
async def list_projects(
    db: DBSession,
    member_id: int | None = Query(None, description="Filter by user membership"),
    q: str | None = Query(None, description="Free-text search"),
) -> list[schemas.ProjectRead]:
    query = _base_project_query()

    if member_id is not None:
        query = query.join(
            models.ProjectUser,
            models.Project.id == models.ProjectUser.project_id,
        ).where(models.ProjectUser.user_id == member_id)

    if q:
        like_pattern = f"%{q}%"
        query = query.where(
            or_(
                models.Project.name.ilike(like_pattern),
                models.Project.description.ilike(like_pattern),
            )
        )

    result = await db.scalars(query)
    projects = result.unique().all()

    await _attach_project_metadata(projects, db)
    return projects


@router.post(
    "/projects",
    response_model=schemas.ProjectRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a project",
)
async def create_project(payload: schemas.ProjectCreate, db: DBSession) -> schemas.ProjectRead:
    project = models.Project(**payload.model_dump())
    db.add(project)
    await db.commit()
    await db.refresh(project)

    await _attach_project_metadata([project], db)
    return project


@router.get(
    "/projects/{project_id}",
    response_model=schemas.ProjectRead,
    summary="Get project",
)
async def get_project(project_id: int, db: DBSession) -> schemas.ProjectRead:
    project = await _get_project_or_404(project_id, db)
    await _attach_project_metadata([project], db)
    return project


@router.put(
    "/projects/{project_id}",
    response_model=schemas.ProjectRead,
    summary="Update project",
)
async def update_project(
    project_id: int, payload: schemas.ProjectUpdate, db: DBSession
) -> schemas.ProjectRead:
    project = await _get_project_or_404(project_id, db)

    updates = payload.model_dump(exclude_unset=True)
    for field, value in updates.items():
        setattr(project, field, value)

    await db.commit()
    await db.refresh(project)
    await _attach_project_metadata([project], db)
    return project


@router.delete(
    "/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete project",
)
async def delete_project(project_id: int, db: DBSession) -> Response:
    project = await _get_project_or_404(project_id, db)
    await db.delete(project)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/projects/{project_id}/members",
    response_model=list[schemas.ProjectMemberRead],
    summary="List project members",
)
async def list_project_members(project_id: int, db: DBSession) -> list[schemas.ProjectMemberRead]:
    await _get_project_or_404(project_id, db)

    result = await db.scalars(
        select(models.ProjectUser)
        .options(selectinload(models.ProjectUser.user).selectinload(models.User.profile))
        .where(models.ProjectUser.project_id == project_id)
        .order_by(models.ProjectUser.joined_at)
    )
    return result.all()


@router.post(
    "/projects/{project_id}/members",
    response_model=schemas.ProjectMemberRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a project member",
)
async def add_project_member(
    project_id: int, payload: schemas.ProjectMemberCreate, db: DBSession
) -> schemas.ProjectMemberRead:
    await _get_project_or_404(project_id, db)
    user = await _get_user_or_404(payload.user_id, db)

    if payload.role not in _ALLOWED_ROLES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid role")

    existing = await db.scalar(
        select(models.ProjectUser).where(
            models.ProjectUser.project_id == project_id,
            models.ProjectUser.user_id == payload.user_id,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Already a member")

    membership = models.ProjectUser(
        project_id=project_id,
        user_id=payload.user_id,
        role=payload.role,
    )
    db.add(membership)
    await db.commit()
    await db.refresh(membership)
    await db.refresh(user, attribute_names=["profile"])
    membership.user = user

    return membership


@router.patch(
    "/projects/{project_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Update member role",
)
async def update_project_member(
    project_id: int, user_id: int, payload: schemas.ProjectMemberUpdate, db: DBSession
) -> Response:
    await _get_project_or_404(project_id, db)

    if payload.role not in _ALLOWED_ROLES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid role")

    membership = await db.scalar(
        select(models.ProjectUser).where(
            models.ProjectUser.project_id == project_id,
            models.ProjectUser.user_id == user_id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    membership.role = payload.role
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/projects/{project_id}/members/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a project member",
)
async def remove_project_member(project_id: int, user_id: int, db: DBSession) -> Response:
    await _get_project_or_404(project_id, db)

    membership = await db.scalar(
        select(models.ProjectUser).where(
            models.ProjectUser.project_id == project_id,
            models.ProjectUser.user_id == user_id,
        )
    )
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    await db.delete(membership)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/projects/{project_id}/repositories",
    response_model=list[schemas.RepositoryRead],
    summary="List repositories in a project",
)
async def list_project_repositories(project_id: int, db: DBSession) -> list[schemas.RepositoryRead]:
    await _get_project_or_404(project_id, db)

    result = await db.scalars(
        select(models.Repository)
        .join(models.ProjectRepository)
        .where(models.ProjectRepository.project_id == project_id)
        .order_by(models.Repository.id)
    )
    repositories = result.all()

    for repo in repositories:
        setattr(repo, "starred", False)
        setattr(repo, "session", None)

    return repositories


@router.post(
    "/projects/{project_id}/repositories",
    status_code=status.HTTP_201_CREATED,
    summary="Attach a repository to project",
)
async def attach_repository_to_project(
    project_id: int, payload: schemas.ProjectRepositoryLinkCreate, db: DBSession
) -> Response:
    await _get_project_or_404(project_id, db)
    await _get_repository_or_404(payload.repository_id, db)
    await _get_user_or_404(payload.uploaded_by, db)

    existing = await db.scalar(
        select(models.ProjectRepository).where(
            models.ProjectRepository.project_id == project_id,
            models.ProjectRepository.repository_id == payload.repository_id,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Already linked")

    link = models.ProjectRepository(
        project_id=project_id,
        repository_id=payload.repository_id,
        uploaded_by=payload.uploaded_by,
    )
    db.add(link)
    await db.commit()
    return Response(status_code=status.HTTP_201_CREATED)


@router.delete(
    "/projects/{project_id}/repositories/{repository_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Detach a repository from project",
)
async def detach_repository_from_project(
    project_id: int, repository_id: int, db: DBSession
) -> Response:
    await _get_project_or_404(project_id, db)

    link = await db.scalar(
        select(models.ProjectRepository).where(
            models.ProjectRepository.project_id == project_id,
            models.ProjectRepository.repository_id == repository_id,
        )
    )
    if link is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Link not found")

    await db.delete(link)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _get_repository_or_404(repository_id: int, db: DBSession) -> models.Repository:
    repository = await db.get(models.Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
    return repository


# -----------------------
# Pinned project endpoints
# -----------------------


@router.get(
    "/users/{user_id}/pinned-projects",
    response_model=list[schemas.PinRead],
    tags=["pins"],
    summary="List pinned projects",
)
async def list_pinned_projects(user_id: int, db: DBSession) -> list[schemas.PinRead]:
    await _get_user_or_404(user_id, db)

    result = await db.scalars(
        select(models.PinnedProject)
        .options(selectinload(models.PinnedProject.project))
        .where(models.PinnedProject.user_id == user_id)
        .order_by(
            models.PinnedProject.position.is_(None),
            models.PinnedProject.position,
            models.PinnedProject.pinned_at,
        )
    )
    pins = result.all()

    projects = [pin.project for pin in pins if pin.project is not None]
    await _attach_project_metadata(projects, db)

    return pins


@router.post(
    "/users/{user_id}/pinned-projects",
    response_model=schemas.PinRead,
    status_code=status.HTTP_201_CREATED,
    tags=["pins"],
    summary="Pin a project",
)
async def pin_project(user_id: int, payload: schemas.PinCreate, db: DBSession) -> schemas.PinRead:
    await _get_user_or_404(user_id, db)
    project = await _get_project_or_404(payload.project_id, db)

    existing = await db.scalar(
        select(models.PinnedProject).where(
            models.PinnedProject.user_id == user_id,
            models.PinnedProject.project_id == payload.project_id,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Already pinned")

    position = payload.position
    if position is None:
        max_position = await db.scalar(
            select(func.max(models.PinnedProject.position)).where(models.PinnedProject.user_id == user_id)
        )
        position = (max_position or 0) + 1

    pin = models.PinnedProject(
        user_id=user_id,
        project_id=payload.project_id,
        position=position,
    )
    db.add(pin)
    await db.commit()
    await db.refresh(pin)
    await _attach_project_metadata([project], db)
    pin.project = project

    return pin


@router.patch(
    "/users/{user_id}/pinned-projects",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["pins"],
    summary="Reorder pinned projects",
)
async def reorder_pins(user_id: int, payload: schemas.PinReorder, db: DBSession) -> Response:
    await _get_user_or_404(user_id, db)

    pins_result = await db.scalars(
        select(models.PinnedProject).where(models.PinnedProject.user_id == user_id)
    )
    existing = {pin.project_id: pin for pin in pins_result.all()}

    if set(existing.keys()) != set(payload.project_ids):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Project id set mismatch")

    for index, project_id in enumerate(payload.project_ids, start=1):
        existing[project_id].position = index

    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/users/{user_id}/pinned-projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["pins"],
    summary="Unpin a project",
)
async def unpin_project(user_id: int, project_id: int, db: DBSession) -> Response:
    await _get_user_or_404(user_id, db)

    pin = await db.scalar(
        select(models.PinnedProject).where(
            models.PinnedProject.user_id == user_id,
            models.PinnedProject.project_id == project_id,
        )
    )
    if pin is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pin not found")

    await db.delete(pin)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
