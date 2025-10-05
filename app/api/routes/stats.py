"""User statistics and contribution endpoints."""

from datetime import date

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from ... import models, schemas
from ..dependencies import DBSession

router = APIRouter(tags=["stats"])


async def _get_user_or_404(user_id: int, db: DBSession) -> models.User:
    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get(
    "/users/{user_id}/stats",
    response_model=schemas.UserStatsRead,
    summary="Profile counters",
)
async def get_user_stats(user_id: int, db: DBSession) -> schemas.UserStatsRead:
    await _get_user_or_404(user_id, db)

    projects_count = await db.scalar(
        select(func.count()).select_from(models.ProjectUser).where(models.ProjectUser.user_id == user_id)
    )
    uploads_count = await db.scalar(
        select(func.count()).select_from(models.Repository).where(models.Repository.user_id == user_id)
    )
    followers_count = await db.scalar(
        select(func.count()).select_from(models.Follow).where(models.Follow.following_id == user_id)
    )
    following_count = await db.scalar(
        select(func.count()).select_from(models.Follow).where(models.Follow.follower_id == user_id)
    )

    return schemas.UserStatsRead(
        projects=projects_count or 0,
        uploads=uploads_count or 0,
        followers=followers_count or 0,
        following=following_count or 0,
    )


@router.get(
    "/users/{user_id}/contributions",
    response_model=schemas.ContributionRead,
    summary="Contribution heatmap data",
)
async def get_user_contributions(
    user_id: int,
    db: DBSession,
    from_date: date | None = Query(None, alias="from"),
    to_date: date | None = Query(None, alias="to"),
    include_sky_points: bool = Query(False),
) -> schemas.ContributionRead:
    await _get_user_or_404(user_id, db)

    query = (
        select(
            func.date(models.Repository.created_at).label("bucket_date"),
            func.count(models.Repository.id).label("bucket_count"),
        )
        .where(models.Repository.user_id == user_id)
        .group_by("bucket_date")
        .order_by("bucket_date")
    )

    if from_date is not None:
        query = query.where(func.date(models.Repository.created_at) >= from_date)
    if to_date is not None:
        query = query.where(func.date(models.Repository.created_at) <= to_date)

    rows = await db.execute(query)
    buckets: list[schemas.ContributionBucket] = []
    for row in rows:
        bucket_date = row.bucket_date
        if isinstance(bucket_date, str):
            bucket_date = date.fromisoformat(bucket_date)
        buckets.append(
            schemas.ContributionBucket(date=bucket_date, count=row.bucket_count)
        )

    sky_points = [] if include_sky_points else None

    return schemas.ContributionRead(buckets=buckets, sky_points=sky_points)


__all__ = ["router"]
