"""Endpoints handling the staged upload workflow for FITS files."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

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
# 영어 주석으로 작성
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

def _write_preview_image_from_fits(fits_path: Path, destination: Path) -> None:
    """Generate a preview PNG from a FITS file with colorbar and labels."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Read FITS file
        with fits.open(fits_path) as hdul:
            data = hdul[0].data

        if data is None:
            raise ValueError("Empty FITS data")

        # Replace NaN and infinities
        data = np.nan_to_num(data)

        # Contrast scaling using percentiles
        vmin, vmax = np.percentile(data, [1, 99])

        # Plot setup
        plt.figure(figsize=(6, 6))
        img = plt.imshow(
            data,
            cmap="gray",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.title(f"{fits_path.name} Image")

        # Add colorbar
        cbar = plt.colorbar(img)
        cbar.set_label("Pixel Value")

        # Save as PNG
        plt.savefig(destination, bbox_inches="tight", pad_inches=0.1)
        plt.close()

    except Exception as e:
        # fallback to blank placeholder if error occurs
        destination.write_bytes(_BLANK_PREVIEW_PNG)

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


def _hash_file(path: Path) -> str:
    """Return the SHA256 hex digest for the provided file path."""

    hasher = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


async def _simulate_step_runtime(seconds: int) -> None:
    """Sleep in one-second increments to emulate long-running work."""

    for _ in range(seconds):
        await asyncio.sleep(1)


async def _step_ingest() -> None:
    await _simulate_step_runtime(10)


async def _step_calibration() -> None:
    await _simulate_step_runtime(10)


async def _step_registration() -> None:
    await _simulate_step_runtime(10)


async def _step_photometry() -> None:
    await _simulate_step_runtime(10)


async def _step_classification() -> None:
    await _simulate_step_runtime(10)


async def _step_reporting() -> None:
    await _simulate_step_runtime(10)


SIMULATED_PIPELINE_STEPS: list[tuple[str, Callable[[], Awaitable[None]]]] = [
    ("ingest", _step_ingest),
    ("calibration", _step_calibration),
    ("registration", _step_registration),
    ("photometry", _step_photometry),
    ("classification", _step_classification),
    ("reporting", _step_reporting),
]

_ACTIVE_SESSION_LOCK = threading.Lock()
_ACTIVE_SESSION_WORKERS: set[int] = set()

PREPROCESS_CATEGORY_KEYS = ("dark", "bias", "flat")


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
    """Launch background workers for the provided session IDs."""

    for session_id in session_ids:
        _start_single_session_worker(session_id)


def _start_single_session_worker(session_id: int) -> None:
    """Launch a background worker for a single session if it's not already running."""

    with _ACTIVE_SESSION_LOCK:
        if session_id in _ACTIVE_SESSION_WORKERS:
            return
        _ACTIVE_SESSION_WORKERS.add(session_id)

    def _runner(selected_session_id: int) -> None:
        async def _process() -> None:
            try:
                async with SessionLocal() as session:
                    session_obj = await session.get(models.Session, selected_session_id)
                    if session_obj is None:
                        return

                    if not SIMULATED_PIPELINE_STEPS:
                        session_obj.status = "completed"
                        session_obj.progress = 100
                        now = datetime.now(tz=timezone.utc)
                        session_obj.started_at = session_obj.started_at or now
                        session_obj.finished_at = now
                        session_obj.current_step = "completed"
                        await session.commit()
                        return

                    session_obj.status = "running"
                    session_obj.current_step = SIMULATED_PIPELINE_STEPS[0][0]
                    session_obj.started_at = session_obj.started_at or datetime.now(tz=timezone.utc)
                    session_obj.progress = 0
                    await session.commit()

                    result = await session.scalars(
                        select(models.PipelineStep)
                        .where(models.PipelineStep.run_id == session_obj.run_id)
                    )
                    existing_steps = {step.step_name: step for step in result}

                    pipeline_steps: list[models.PipelineStep] = []
                    for step_name, _ in SIMULATED_PIPELINE_STEPS:
                        step_record = existing_steps.get(step_name)
                        if step_record is None:
                            step_record = models.PipelineStep(
                                run_id=session_obj.run_id,
                                step_name=step_name,
                                status="queued",
                                progress=0,
                            )
                            session.add(step_record)
                        else:
                            step_record.status = "queued"
                            step_record.progress = 0
                            step_record.started_at = None
                            step_record.finished_at = None
                        pipeline_steps.append(step_record)

                    await session.commit()

                    total_steps = len(SIMULATED_PIPELINE_STEPS)

                    for index, (step_name, step_runner) in enumerate(SIMULATED_PIPELINE_STEPS):
                        step_record = pipeline_steps[index]
                        step_record.status = "running"
                        step_record.started_at = datetime.now(tz=timezone.utc)
                        step_record.progress = 0

                        session_obj.current_step = step_name
                        session_obj.status = "running"
                        session_obj.progress = int((index / total_steps) * 100)
                        await session.commit()

                        await step_runner()

                        step_record.status = "completed"
                        step_record.finished_at = datetime.now(tz=timezone.utc)
                        step_record.progress = 100

                        session_obj.progress = int(((index + 1) / total_steps) * 100)
                        await session.commit()

                    session_obj.status = "completed"
                    session_obj.current_step = "completed"
                    session_obj.finished_at = datetime.now(tz=timezone.utc)
                    session_obj.progress = 100
                    await session.commit()
            finally:
                with _ACTIVE_SESSION_LOCK:
                    _ACTIVE_SESSION_WORKERS.discard(selected_session_id)

        asyncio.run(_process())

    threading.Thread(target=_runner, args=(session_id,), daemon=True).start()


@router.get(
    "/uploads/temp/{temp_id}/preview",
    response_class=FileResponse,
    summary="Return the generated preview image for a staged upload item.",
)
async def get_temp_preview(temp_id: str) -> Response:
    """Serve the preview PNG associated with a staged FITS upload."""

    item_dir = (TEMP_DIR / temp_id).expanduser().resolve()
    _ensure_path_within(item_dir, TEMP_DIR)

    preview_path = item_dir / "preview.png"

    if preview_path.exists():
        return FileResponse(preview_path, media_type="image/png")

    return Response(content=_BLANK_PREVIEW_PNG, media_type="image/png")


async def _get_user_or_404(user_id: int, db: DBSession) -> models.User:
    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.post(
    "/uploads/prepare",
    response_model=schemas.StageUploadsResponse,
    summary="Stage FITS files for review",
)
async def stage_uploads(
    files: list[UploadFile] = File(...),
    dark_files: list[UploadFile] | None = File(default=None),
    bias_files: list[UploadFile] | None = File(default=None),
    flat_files: list[UploadFile] | None = File(default=None),
) -> schemas.StageUploadsResponse:
    """Store uploaded FITS files in a temporary location and return metadata."""

    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    staged_items: list[schemas.TempUploadItem] = []
    staged_preprocess: dict[str, list[schemas.TempPreprocessItem]] = defaultdict(list)

    for upload in files:
        temp_id = uuid.uuid4().hex
        item_dir = TEMP_DIR / temp_id
        original_name = upload.filename or f"{temp_id}.fits"
        fits_path = item_dir / original_name

        size_bytes = await _save_upload_file(upload, fits_path)
        preview_path = item_dir / "preview.png"
        _write_preview_image_from_fits(fits_path, preview_path)

        try:
            with fits.open(fits_path) as hdul:
                header = dict(hdul[0].header)
        except Exception:
            header = {}

        metadata = {
            "temp_id": temp_id,
            "tmp_fits": str(fits_path),
            "tmp_png": str(preview_path),
            "fits_header": header,
            "original_filename": original_name,
            "data_kind": "observation",
        }

        metadata_path = item_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        staged_items.append(
            schemas.TempUploadItem(
                temp_id=temp_id,
                filename=original_name,
                size_bytes=size_bytes,
                content_type=upload.content_type,
                tmp_fits=str(fits_path),
                tmp_png=str(preview_path),
                fits_header=header,
                metadata_json=metadata,
            )
        )

    preprocess_inputs = {
        "dark": dark_files or [],
        "bias": bias_files or [],
        "flat": flat_files or [],
    }

    for category, uploads in preprocess_inputs.items():
        for upload in uploads:
            temp_id = uuid.uuid4().hex
            item_dir = TEMP_DIR / temp_id
            original_name = upload.filename or f"{category}_{temp_id}.fits"
            fits_path = item_dir / original_name

            size_bytes = await _save_upload_file(upload, fits_path)
            preview_path = item_dir / "preview.png"
            _write_preview_image_from_fits(fits_path, preview_path)

            try:
                with fits.open(fits_path) as hdul:
                    header = dict(hdul[0].header)
            except Exception:
                header = {}

            metadata = {
                "temp_id": temp_id,
                "tmp_fits": str(fits_path),
                "tmp_png": str(preview_path),
                "fits_header": header,
                "original_filename": original_name,
                "data_kind": "preprocess",
                "preprocess_category": category,
            }

            metadata_path = item_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

            staged_preprocess[category].append(
                schemas.TempPreprocessItem(
                    temp_id=temp_id,
                    category=category,
                    filename=original_name,
                    size_bytes=size_bytes,
                    temp_path=str(fits_path),
                    tmp_png=str(preview_path),
                    metadata_json=metadata,
                )
            )

    return schemas.StageUploadsResponse(items=staged_items, preprocess=dict(staged_preprocess))


@router.post(
    "/uploads/commit",
    response_model=schemas.UploadCommitResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Commit staged FITS files into a repository",
)
async def commit_uploads(payload: schemas.UploadCommitRequest, db: DBSession) -> schemas.UploadCommitResponse:
    """Persist staged uploads by creating repository metadata and launching sessions."""

    # ---- Basic validation ----
    if not payload.items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No staged items provided")

    await _get_user_or_404(payload.user_id, db)

    # ---- Check duplicate repository ----
    existing = await db.scalar(
        select(models.Repository).where(
            models.Repository.user_id == payload.user_id,
            models.Repository.name == payload.repository_name,
        )
    )
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Repository name already exists for user")

    # ---- Create new repository and dataset ----
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

    # ---- Prepare FITS items ----
    pending_items: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

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

        metadata_source = item.metadata_json or _load_temp_metadata(temp_item_dir) or {}
        if not isinstance(metadata_source, dict):
            metadata_source = {}
        metadata = dict(metadata_source)
        metadata.setdefault("data_kind", "observation")
        metadata.setdefault("original_filename", fits_temp_path.name)

        fits_metadata_source = item.fits_data_json or metadata_source or {}
        if not isinstance(fits_metadata_source, dict):
            fits_metadata_source = {}
        fits_metadata = dict(fits_metadata_source)

        original_name = item.metadata_json.get("original_filename") if item.metadata_json else None
        if not original_name:
            original_name = metadata.get("original_filename", fits_temp_path.name)

        metadata["original_filename"] = original_name

        data_hash = _hash_file(fits_temp_path)

        if data_hash in seen_hashes:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate FITS content detected in request for item {item.temp_id}",
            )
        seen_hashes.add(data_hash)

        existing_data = await db.scalar(
            select(models.Data).where(models.Data.hash == data_hash)
        )

        if existing_data is not None:
            pending_items.append(
                {
                    "status": "existing",
                    "data": existing_data,
                    "temp_item_dir": temp_item_dir,
                    "fits_temp_path": fits_temp_path,
                    "image_temp_path": image_temp_path,
                }
            )
            continue

        pending_items.append(
            {
                "status": "new",
                "temp_item_dir": temp_item_dir,
                "fits_temp_path": fits_temp_path,
                "image_temp_path": image_temp_path,
                "metadata": metadata,
                "fits_metadata": fits_metadata,
                "original_name": original_name,
                "hash": data_hash,
            }
        )

    preprocess_pending: list[dict[str, Any]] = []

    for preprocess_item in payload.preprocess_items or []:
        if preprocess_item.category not in PREPROCESS_CATEGORY_KEYS:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported preprocessing category: {preprocess_item.category}",
            )

        preprocess_temp_path = Path(preprocess_item.temp_path).expanduser().resolve()
        _ensure_path_within(preprocess_temp_path, TEMP_DIR)

        if not preprocess_temp_path.exists():
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Temporary preprocessing file not found for {preprocess_item.temp_id}",
            )

        temp_item_dir = preprocess_temp_path.parent

        metadata_source = preprocess_item.metadata_json or _load_temp_metadata(temp_item_dir) or {}
        if not isinstance(metadata_source, dict):
            metadata_source = {}

        metadata = dict(metadata_source)
        metadata.setdefault("data_kind", "preprocess")
        metadata.setdefault("preprocess_category", preprocess_item.category)
        metadata.setdefault("temp_id", preprocess_item.temp_id)

        fits_metadata_source = metadata_source.get("fits_header") if isinstance(metadata_source, dict) else None
        if preprocess_item.metadata_json and "fits_header" in preprocess_item.metadata_json:
            fits_metadata_source = preprocess_item.metadata_json["fits_header"]
        fits_metadata = dict(fits_metadata_source) if isinstance(fits_metadata_source, dict) else None

        original_name = (
            preprocess_item.original_name
            or metadata.get("original_filename")
            or preprocess_temp_path.name
        )
        metadata["original_filename"] = original_name

        preview_temp_path: Path | None = None
        preview_candidate = metadata.get("tmp_png")
        if isinstance(preview_candidate, str):
            try:
                preview_path = Path(preview_candidate).expanduser().resolve()
                _ensure_path_within(preview_path, TEMP_DIR)
            except HTTPException:
                preview_temp_path = None
            else:
                preview_temp_path = preview_path if preview_path.exists() else None
        metadata["tmp_png"] = str(preview_temp_path) if preview_temp_path else None

        data_hash = _hash_file(preprocess_temp_path)

        if data_hash in seen_hashes:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate file content detected in preprocessing payload for {preprocess_item.temp_id}",
            )
        seen_hashes.add(data_hash)

        existing_data = await db.scalar(
            select(models.Data).where(models.Data.hash == data_hash)
        )

        pending_entry = {
            "category": preprocess_item.category,
            "temp_item_dir": temp_item_dir,
            "preprocess_temp_path": preprocess_temp_path,
            "preview_temp_path": preview_temp_path,
        }

        if existing_data is not None:
            pending_entry.update(
                {
                    "status": "existing",
                    "data": existing_data,
                }
            )
            preprocess_pending.append(pending_entry)
            continue

        pending_entry.update(
            {
                "status": "new",
                "metadata": metadata,
                "fits_metadata": fits_metadata,
                "original_name": original_name,
                "hash": data_hash,
            }
        )
        preprocess_pending.append(pending_entry)

    # ---- Move FITS files into permanent storage ----
    target_dir = DATA_DIR / f"repository_{repository.id}" / f"dataset_{dataset.version}"
    target_dir.mkdir(parents=True, exist_ok=True)

    committed_data: list[models.Data] = []
    committed_preprocess: list[models.Data] = []

    for pending in pending_items:
        item_status = pending["status"]
        temp_item_dir = pending["temp_item_dir"]
        fits_temp_path = pending["fits_temp_path"]
        image_temp_path = pending.get("image_temp_path")

        if item_status == "existing":
            data_record = pending["data"]
            committed_data.append(data_record)

            # cleanup temp
            if fits_temp_path.exists():
                fits_temp_path.unlink(missing_ok=True)
            if image_temp_path is not None and image_temp_path.exists():
                image_temp_path.unlink(missing_ok=True)
            shutil.rmtree(temp_item_dir, ignore_errors=True)
            continue

        # ---- New data ----
        if not fits_temp_path.exists():
            await db.rollback()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Temporary FITS file missing during commit")

        metadata = pending["metadata"]
        fits_metadata = pending["fits_metadata"]
        original_name = pending["original_name"]

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

        # ---- Create new Data record ----
        metadata = dict(metadata)
        metadata.setdefault("data_kind", "observation")
        metadata["original_filename"] = original_name
        metadata.pop("tmp_fits", None)
        metadata.pop("tmp_png", None)
        metadata["stored_fits_path"] = str(fits_dest)
        if preview_dest is not None:
            metadata["stored_preview_path"] = str(preview_dest)

        if isinstance(fits_metadata, dict):
            fits_metadata = dict(fits_metadata)

        data_record = models.Data(
            hash=pending["hash"],
            fits_original_path=str(fits_dest),
            fits_image_path=str(preview_dest) if preview_dest else None,
            fits_data_json=fits_metadata,
            metadata_json=metadata,
        )
        db.add(data_record)
        committed_data.append(data_record)

    calibration_root = target_dir / "calibration"

    for pending in preprocess_pending:
        temp_item_dir = pending["temp_item_dir"]
        preprocess_temp_path = pending["preprocess_temp_path"]
        preview_temp_path = pending.get("preview_temp_path")
        category = pending["category"]

        if pending["status"] == "existing":
            data_record = pending["data"]
            committed_preprocess.append(data_record)

            if preprocess_temp_path.exists():
                preprocess_temp_path.unlink(missing_ok=True)
            if preview_temp_path is not None and preview_temp_path.exists():
                preview_temp_path.unlink(missing_ok=True)
            shutil.rmtree(temp_item_dir, ignore_errors=True)
            continue

        if not preprocess_temp_path.exists():
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Temporary preprocessing file missing during commit",
            )

        metadata = pending["metadata"]
        fits_metadata = pending["fits_metadata"]
        original_name = pending["original_name"]

        category_dir = calibration_root / category
        category_dir.mkdir(parents=True, exist_ok=True)

        fits_dest = category_dir / original_name
        fits_dest = _unique_target(fits_dest, fits_dest.suffix)
        preprocess_temp_path.replace(fits_dest)

        preview_dest: Path | None = None
        if preview_temp_path is not None and preview_temp_path.exists():
            preview_dest = category_dir / preview_temp_path.name
            preview_dest = _unique_target(preview_dest, preview_dest.suffix)
            preview_temp_path.replace(preview_dest)

        metadata_path = temp_item_dir / "metadata.json"
        if metadata_path.exists():
            metadata_path.unlink(missing_ok=True)
        shutil.rmtree(temp_item_dir, ignore_errors=True)

        metadata = dict(metadata)
        metadata.setdefault("data_kind", "preprocess")
        metadata["preprocess_category"] = category
        metadata["original_filename"] = original_name
        metadata.pop("tmp_fits", None)
        metadata["stored_fits_path"] = str(fits_dest)
        if preview_dest is not None:
            metadata["stored_preview_path"] = str(preview_dest)
        metadata.pop("tmp_png", None)

        data_record = models.Data(
            hash=pending["hash"],
            fits_original_path=str(fits_dest),
            fits_image_path=str(preview_dest) if preview_dest else None,
            fits_data_json=fits_metadata,
            metadata_json=metadata,
        )
        db.add(data_record)
        committed_preprocess.append(data_record)

    # ---- Flush before creating relationships ----
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        message = str(getattr(exc, "orig", exc))
        if "data.hash" in message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Commit failed because a FITS file already exists in another dataset.",
            ) from exc
        if "dataset_data" in message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Commit failed because the dataset already references one of the FITS files.",
            ) from exc
        raise

    # ---- Create processing sessions ----
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

    # ---- Refresh objects ----
    await db.refresh(repository)
    await db.refresh(dataset)
    for data_item in committed_data:
        await db.refresh(data_item)
        setattr(data_item, "dataset_id", dataset.id)
    for data_item in committed_preprocess:
        await db.refresh(data_item)
        setattr(data_item, "dataset_id", dataset.id)
    for session_model in sessions:
        await db.refresh(session_model)

    setattr(repository, "starred", False)
    setattr(repository, "session", None)

    # ---- Start background worker ----
    if session_ids:
        _start_session_worker(session_ids)

    # ---- Return final response ----
    return schemas.UploadCommitResponse(
        repository=repository,
        dataset=dataset,
        data=committed_data,
        preprocess_data=committed_preprocess,
        sessions=sessions,
    )


__all__ = ["router"]
