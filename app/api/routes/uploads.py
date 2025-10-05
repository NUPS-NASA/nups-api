"""Endpoints handling the staged upload workflow for FITS files."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlalchemy import func, select

from ... import models, schemas
from ...config import get_settings
from ...database import SessionLocal
from ..dependencies import DBSession

router = APIRouter(tags=["uploads"])

settings = get_settings()
TEMP_DIR = Path(settings.storage_tmp_dir).expanduser().resolve()
DATA_DIR = Path(settings.storage_data_dir).expanduser().resolve()
TEMP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

_BLANK_PREVIEW_PNG = bytes(
    [
        0x89,
        0x50,
        0x4E,
        0x47,
        0x0D,
        0x0A,
        0x1A,
        0x0A,
        0x00,
        0x00,
        0x00,
        0x0D,
        0x49,
        0x48,
        0x44,
        0x52,
        0x00,
        0x00,
        0x00,
        0x01,
        0x00,
        0x00,
        0x00,
        0x01,
        0x08,
        0x06,
        0x00,
        0x00,
        0x00,
        0x1F,
        0x15,
        0xC4,
        0x89,
        0x00,
        0x00,
        0x00,
        0x0A,
        0x49,
        0x44,
        0x41,
        0x54,
        0x78,
        0x9C,
        0x63,
        0xF8,
        0x0F,
        0x04,
        0x00,
        0x09,
        0xFB,
        0x03,
        0xFD,
        0xA7,
        0x8F,
        0x19,
        0x61,
        0x00,
        0x00,
        0x00,
        0x00,
        0x49,
        0x45,
        0x4E,
        0x44,
        0xAE,
        0x42,
        0x60,
        0x82,
    ]
)


async def _save_upload_file(upload: UploadFile, destination: Path) -> int:
    """Persist the uploaded file to disk and return its size in bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0

    with destination.open("wb") as buffer:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            buffer.write(chunk)

    await upload.close()
    return size


def _write_preview_image(destination: Path) -> None:
    """Write a placeholder preview image to the destination path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_BLANK_PREVIEW_PNG)


def _load_temp_metadata(temp_dir: Path) -> dict | None:
    """Load metadata stored alongside the staged upload, if available."""

    metadata_path = temp_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _ensure_path_within(path: Path, parent: Path) -> None:
    """Ensure that the path is located within the parent directory."""

    try:
        path.relative_to(parent)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid temporary path") from exc


def _unique_target(base_path: Path, suffix: str) -> Path:
    """Return a path guaranteed not to clash with an existing file."""

    candidate = base_path
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    parent = candidate.parent
    counter = 1
    while True:
        renamed = parent / f"{stem}_{counter}{suffix}"
        if not renamed.exists():
            return renamed
        counter += 1


def _start_session_worker(session_ids: list[int]) -> None:
    """Launch a background worker that simulates long-running session processing."""

    if not session_ids:
        return

    def _runner() -> None:
        async def _process() -> None:
            async with SessionLocal() as session:
                for session_id in session_ids:
                    session_obj = await session.get(models.Session, session_id)
                    if session_obj is None:
                        continue

                    session_obj.status = "running"
                    session_obj.current_step = "processing"
                    session_obj.started_at = datetime.now(tz=timezone.utc)
                    session_obj.progress = 0
                    await session.commit()

                    await asyncio.sleep(10)

                    session_obj.progress = 100
                    session_obj.status = "completed"
                    session_obj.current_step = "completed"
                    session_obj.finished_at = datetime.now(tz=timezone.utc)
                    await session.commit()

        asyncio.run(_process())

    threading.Thread(target=_runner, daemon=True).start()


async def _get_user_or_404(user_id: int, db: DBSession) -> models.User:
    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.post(
    "/uploads/prepare",
    response_model=list[schemas.TempUploadItem],
    summary="Stage FITS files for review",
)
async def stage_uploads(files: list[UploadFile] = File(...)) -> list[schemas.TempUploadItem]:
    """Store uploaded FITS files in a temporary location and return metadata."""

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    staged_items: list[schemas.TempUploadItem] = []

    for upload in files:
        temp_id = uuid.uuid4().hex
        item_dir = TEMP_DIR / temp_id
        original_name = upload.filename or f"{temp_id}.fits"
        fits_path = item_dir / original_name

        size_bytes = await _save_upload_file(upload, fits_path)
        preview_path = item_dir / "preview.png"
        _write_preview_image(preview_path)

        metadata = {
            "temp_id": temp_id,
            "original_filename": original_name,
            "size_bytes": size_bytes,
            "content_type": upload.content_type,
        }

        metadata_path = item_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        staged_items.append(
            schemas.TempUploadItem(
                temp_id=temp_id,
                filename=original_name,
                size_bytes=size_bytes,
                content_type=upload.content_type,
                fits_temp_path=str(fits_path),
                image_temp_path=str(preview_path),
                fits_data_json=metadata,
                metadata_json=metadata,
            )
        )

    return staged_items


@router.post(
    "/uploads/commit",
    response_model=schemas.UploadCommitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Commit staged FITS files into a repository",
)
async def commit_uploads(payload: schemas.UploadCommitRequest, db: DBSession) -> schemas.UploadCommitResponse:
    """Persist staged uploads by creating repository metadata and launching sessions."""

    if not payload.items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No staged items provided")

    await _get_user_or_404(payload.user_id, db)

    existing = await db.scalar(
        select(models.Repository).where(
            models.Repository.user_id == payload.user_id,
            models.Repository.name == payload.repository_name,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Repository name already exists for user")

    repository = models.Repository(
        user_id=payload.user_id,
        name=payload.repository_name,
        description=payload.repository_description,
    )
    db.add(repository)
    await db.flush()

    max_version = await db.scalar(
        select(func.max(models.Dataset.version)).where(models.Dataset.repository_id == repository.id)
    )
    dataset_version = (max_version or 0) + 1

    dataset = models.Dataset(
        repository_id=repository.id,
        version=dataset_version,
        captured_at=payload.captured_at,
    )
    db.add(dataset)
    await db.flush()

    committed_data: list[models.Data] = []

    for item in payload.items:
        fits_temp_path = Path(item.fits_temp_path).expanduser().resolve()
        _ensure_path_within(fits_temp_path, TEMP_DIR)

        if not fits_temp_path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Temporary FITS file not found for {item.temp_id}")

        temp_item_dir = fits_temp_path.parent
        image_temp_path: Path | None = None

        if item.image_temp_path:
            candidate_image = Path(item.image_temp_path).expanduser().resolve()
            _ensure_path_within(candidate_image, TEMP_DIR)
            if candidate_image.exists():
                image_temp_path = candidate_image

        metadata = item.metadata_json or _load_temp_metadata(temp_item_dir) or {}
        fits_metadata = item.fits_data_json or metadata or {}

        original_name = metadata.get("original_filename") if isinstance(metadata, dict) else None
        if not original_name:
            original_name = fits_temp_path.name

        target_dir = DATA_DIR / f"repository_{repository.id}" / f"dataset_{dataset.version}"
        target_dir.mkdir(parents=True, exist_ok=True)

        fits_dest = target_dir / original_name
        fits_dest = _unique_target(fits_dest, fits_dest.suffix)
        fits_temp_path.replace(fits_dest)

        preview_dest: Path | None = None
        if image_temp_path is not None and image_temp_path.exists():
            preview_dest = target_dir / image_temp_path.name
            preview_dest = _unique_target(preview_dest, preview_dest.suffix)
            image_temp_path.replace(preview_dest)

        metadata_path = temp_item_dir / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink(missing_ok=True)
        shutil.rmtree(temp_item_dir, ignore_errors=True)

        hasher = sha256()
        with fits_dest.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                if not chunk:
                    break
                hasher.update(chunk)

        data_record = models.Data(
            dataset_id=dataset.id,
            hash=hasher.hexdigest(),
            fits_original_path=str(fits_dest),
            fits_image_path=str(preview_dest) if preview_dest else None,
            fits_data_json=fits_metadata,
            metadata_json=metadata,
        )
        db.add(data_record)
        committed_data.append(data_record)

    await db.flush()

    sessions: list[models.Session] = []
    for data_item in committed_data:
        session_model = models.Session(
            run_id=uuid.uuid4(),
            repository_id=repository.id,
            dataset_id=dataset.id,
            data_id=data_item.id,
            data_version=dataset.version,
            current_step=None,
            status="queued",
            progress=0,
        )
        db.add(session_model)
        sessions.append(session_model)

    await db.flush()
    session_ids = [session_model.id for session_model in sessions]

    await db.commit()

    await db.refresh(repository)
    await db.refresh(dataset)
    for data_item in committed_data:
        await db.refresh(data_item)
    for session_model in sessions:
        await db.refresh(session_model)

    setattr(repository, "starred", False)
    setattr(repository, "session", None)

    _start_session_worker(session_ids)

    return schemas.UploadCommitResponse(
        repository=repository,
        dataset=dataset,
        data=committed_data,
        sessions=sessions,
    )


__all__ = ["router"]

