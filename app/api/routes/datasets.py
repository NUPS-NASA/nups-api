"""Dataset and data item endpoints."""

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select

from ... import models, schemas
from ..dependencies import DBSession

router = APIRouter(tags=["datasets"])


async def _get_repository_or_404(repository_id: int, db: DBSession) -> models.Repository:
    repository = await db.get(models.Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
    return repository


async def _get_dataset_or_404(dataset_id: int, db: DBSession) -> models.Dataset:
    dataset = await db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return dataset


@router.get(
    "/datasets",
    response_model=list[schemas.DatasetRead],
    summary="List datasets",
)
async def list_datasets(
    db: DBSession,
    repository_id: int = Query(..., description="Repository identifier"),
) -> list[schemas.DatasetRead]:
    await _get_repository_or_404(repository_id, db)

    result = await db.scalars(
        select(models.Dataset)
        .where(models.Dataset.repository_id == repository_id)
        .order_by(models.Dataset.version.desc())
    )
    return result.all()


@router.post(
    "/datasets",
    response_model=schemas.DatasetRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create dataset version",
)
async def create_dataset(payload: schemas.DatasetCreate, db: DBSession) -> schemas.DatasetRead:
    await _get_repository_or_404(payload.repository_id, db)

    existing = await db.scalar(
        select(models.Dataset).where(
            models.Dataset.repository_id == payload.repository_id,
            models.Dataset.version == payload.version,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset version already exists")

    dataset = models.Dataset(**payload.model_dump())
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


@router.get(
    "/data",
    response_model=list[schemas.DataRead],
    summary="List data items",
    tags=["data"],
)
async def list_data(
    db: DBSession,
    dataset_id: int = Query(..., description="Dataset identifier"),
) -> list[schemas.DataRead]:
    await _get_dataset_or_404(dataset_id, db)

    result = await db.scalars(
        select(models.Data)
        .where(models.Data.dataset_id == dataset_id)
        .order_by(models.Data.created_at.desc())
    )
    return result.all()


@router.post(
    "/data",
    response_model=schemas.DataRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a data item",
    tags=["data"],
)
async def create_data(payload: schemas.DataCreate, db: DBSession) -> schemas.DataRead:
    await _get_dataset_or_404(payload.dataset_id, db)

    existing = await db.scalar(
        select(models.Data).where(models.Data.hash == payload.hash)
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Data hash already exists")

    data_item = models.Data(**payload.model_dump())
    db.add(data_item)
    await db.commit()
    await db.refresh(data_item)
    return data_item


__all__ = ["router"]
