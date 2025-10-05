"""Pipeline orchestration for dataset analysis sessions.

This module coordinates the astrophotometry pipeline using the
algorithms provided in ``core``.  Each step stores summary data in the
``pipeline_steps`` table so the frontend can inspect progress and
results sequentially.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .. import models
from ..config import get_settings
from core.PrepareFits import (
    calibrate_frame,
    detect_stars,
    load_fits_data,
    measure_frame_photometry,
    read_time_from_header,
    save_detection_preview,
    extract_exptime,
    median_combine,
)
from core.FWHM import get_header_airmass, estimate_frame_fwhm
from core.Detrending import (
    detrend_by_covariates,
    pick_comps_rms_aware_general,
    weighted_reference,
)
from core.PrepareFits import align_to_reference

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared state passed between pipeline steps."""

    session: models.Session
    dataset: models.Dataset
    analysis_dir: Path
    light_files: list[Path]
    bias_files: list[Path]
    dark_files: list[Path]
    flat_files: list[Path]
    master_bias: np.ndarray | None = None
    master_dark: dict[float, np.ndarray] = field(default_factory=dict)
    master_flat: np.ndarray | None = None
    reference_image: np.ndarray | None = None
    reference_header: Any | None = None
    reference_stars: np.ndarray | None = None
    photometry_matrix: np.ndarray | None = None
    times: np.ndarray | None = None
    airmass: np.ndarray | None = None
    fwhm: np.ndarray | None = None
    sky: np.ndarray | None = None
    raw_lightcurve: np.ndarray | None = None
    raw_lightcurves: dict[int, np.ndarray] = field(default_factory=dict)
    detrended_flux: np.ndarray | None = None
    detrended_lightcurves: dict[int, np.ndarray] = field(default_factory=dict)
    target_index: int | None = None
    comparison_indices: list[int] = field(default_factory=list)


@dataclass
class StepDefinition:
    name: str
    runner: Callable[[PipelineContext, AsyncSession], Awaitable[dict[str, Any]]]


def _ensure_numpy_array(data: Sequence[float] | np.ndarray | None, length: int) -> np.ndarray:
    if data is None:
        return np.full(length, np.nan, dtype=float)
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 0:
        arr = np.full(length, float(arr), dtype=float)
    if arr.shape[0] > length:
        return arr[:length]
    if arr.shape[0] < length:
        pad = np.full(length - arr.shape[0], np.nan, dtype=float)
        return np.concatenate([arr, pad])
    return arr


def _median_combine_from_files(files: Sequence[Path]) -> tuple[np.ndarray | None, Any | None]:
    if not files:
        return None, None
    existing = [str(path) for path in files if path.is_file()]
    if not existing:
        return None, None
    master, hdr = median_combine(existing)
    return master, hdr


def _build_master_dark(files: Sequence[Path]) -> dict[float, np.ndarray]:
    if not files:
        return {}
    by_exposure: dict[float, list[np.ndarray]] = {}
    for path in files:
        if not path.is_file():
            continue
        data, hdr = load_fits_data(str(path))
        expt = extract_exptime(hdr)
        if expt is None:
            continue
        by_exposure.setdefault(float(expt), []).append(data.astype(float))
    combined: dict[float, np.ndarray] = {}
    for expt, stack in by_exposure.items():
        if not stack:
            continue
        combined[expt] = np.nanmedian(np.stack(stack, axis=0), axis=0)
    return combined


def _build_master_flat(
    files: Sequence[Path],
    master_bias: np.ndarray | None,
    dark_dict: dict[float, np.ndarray],
) -> np.ndarray | None:
    if not files:
        return None
    calibrated_stack: list[np.ndarray] = []
    for path in files:
        if not path.is_file():
            continue
        data, hdr = load_fits_data(str(path))
        calibrated = data.astype(float)
        if master_bias is not None:
            calibrated = calibrated - master_bias
        if dark_dict:
            expt = extract_exptime(hdr)
            if expt is not None and dark_dict:
                nearest = min(dark_dict.keys(), key=lambda value: abs(value - expt))
                scale = expt / nearest if nearest else 1.0
                calibrated = calibrated - dark_dict[nearest] * scale
        calibrated_stack.append(calibrated)
    if not calibrated_stack:
        return None
    flat = np.nanmedian(np.stack(calibrated_stack, axis=0), axis=0)
    finite = np.isfinite(flat)
    if np.any(finite):
        median = float(np.nanmedian(flat[finite]))
        if np.isfinite(median) and median != 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                flat = flat / median
    return flat


def _downsample_image(image: np.ndarray, max_size: int = 128) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1])

    height, width = arr.shape
    row_step = max(1, int(math.ceil(height / max_size)))
    col_step = max(1, int(math.ceil(width / max_size)))
    return arr[::row_step, ::col_step]


def _image_snapshot_payload(image: np.ndarray, max_size: int = 128) -> dict[str, Any]:
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1])

    preview_array = _downsample_image(arr, max_size=max_size)
    preview: list[list[float | None]] = [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in preview_array
    ]

    finite = np.isfinite(arr)
    if np.any(finite):
        min_val = float(np.nanmin(arr[finite]))
        max_val = float(np.nanmax(arr[finite]))
        median_val = float(np.nanmedian(arr[finite]))
        mean_val = float(np.nanmean(arr[finite]))
    else:
        min_val = max_val = median_val = mean_val = None

    return {
        "shape": list(arr.shape),
        "preview": preview,
        "stats": {
            "min": min_val,
            "max": max_val,
            "median": median_val,
            "mean": mean_val,
        },
    }


def _save_full_frame(path: Path, image: np.ndarray) -> str | None:
    try:
        np.savez_compressed(path, image=np.asarray(image, dtype=float))
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to store calibration frame at %s: %s", path, exc)
        return None
    return str(path)


def _build_calibration_snapshot(
    raw_frame: np.ndarray,
    calibrated_frame: np.ndarray,
    analysis_dir: Path,
    frame_stem: str,
) -> dict[str, Any]:
    before_payload = _image_snapshot_payload(raw_frame)
    after_payload = _image_snapshot_payload(calibrated_frame)

    before_path = analysis_dir / f"{frame_stem}_raw_frame.npz"
    after_path = analysis_dir / f"{frame_stem}_calibrated_frame.npz"

    before_payload["npz_path"] = _save_full_frame(before_path, raw_frame)
    after_payload["npz_path"] = _save_full_frame(after_path, calibrated_frame)

    return {"before": before_payload, "after": after_payload}


def _summary_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.floating):
        return float(value)
    return float(value)


async def _step_prepare_calibration(ctx: PipelineContext, db: AsyncSession) -> dict[str, Any]:
    master_bias, _ = _median_combine_from_files(ctx.bias_files)
    master_dark = _build_master_dark(ctx.dark_files)
    master_flat = _build_master_flat(ctx.flat_files, master_bias, master_dark)

    ctx.master_bias = master_bias
    ctx.master_dark = master_dark
    ctx.master_flat = master_flat

    summary = {
        "light_frames": len(ctx.light_files),
        "bias_frames": len(ctx.bias_files),
        "dark_frames": len(ctx.dark_files),
        "flat_frames": len(ctx.flat_files),
        "has_master_bias": master_bias is not None,
        "dark_exposures": sorted(master_dark.keys()),
        "has_master_flat": master_flat is not None,
    }
    return summary


async def _step_detect_reference(ctx: PipelineContext, db: AsyncSession) -> dict[str, Any]:
    if not ctx.light_files:
        raise RuntimeError("No light frames available for reference detection")

    first_path = ctx.light_files[0]
    image, header = load_fits_data(str(first_path))
    calibrated = calibrate_frame(
        image,
        header,
        master_bias=ctx.master_bias,
        dark_dict=ctx.master_dark,
        flat_norm=ctx.master_flat,
    )
    ctx.reference_image = calibrated
    ctx.reference_header = header

    calibration_snapshot = _build_calibration_snapshot(
        image,
        calibrated,
        ctx.analysis_dir,
        first_path.stem,
    )

    stars = detect_stars(calibrated)
    ctx.reference_stars = stars

    preview_path = ctx.analysis_dir / "detected_stars_preview.png"
    try:
        save_detection_preview(calibrated, stars[:, :2], str(preview_path))
        preview = str(preview_path)
    except Exception as exc:  # pragma: no cover - visualization is best-effort
        logger.warning("Failed to save detection preview: %s", exc)
        preview = None

    summary = {
        "reference_frame": first_path.name,
        "detected_stars": int(stars.shape[0]),
        "preview_path": preview,
        "calibration_snapshot": calibration_snapshot,
    }
    return summary


async def _step_photometry(ctx: PipelineContext, db: AsyncSession) -> dict[str, Any]:
    if ctx.reference_image is None or ctx.reference_stars is None:
        raise RuntimeError("Reference image and stars must be prepared before photometry")

    xy_positions = ctx.reference_stars[:, :2]
    rows: list[np.ndarray] = []
    times: list[float] = []
    airmass: list[float] = []
    fwhm: list[float] = []
    sky: list[float] = []
    skipped = 0

    for index, path in enumerate(ctx.light_files):
        data, header = load_fits_data(str(path))
        calibrated = calibrate_frame(
            data,
            header,
            master_bias=ctx.master_bias,
            dark_dict=ctx.master_dark,
            flat_norm=ctx.master_flat,
        )

        aligned = calibrated
        if index > 0:
            try:
                aligned, _ = align_to_reference(calibrated, ctx.reference_image)
            except Exception as exc:
                logger.warning("Alignment failed for %s: %s", path.name, exc)
                skipped += 1
                continue

        fluxes = measure_frame_photometry(aligned, xy_positions)
        rows.append(fluxes)

        time_value = read_time_from_header(header)
        times.append(float(time_value) if np.isfinite(time_value) else float(index))

        airmass.append(float(get_header_airmass(header)) if header is not None else np.nan)
        fwhm.append(float(estimate_frame_fwhm(aligned, xy_positions)))
        finite = np.isfinite(aligned)
        sky.append(float(np.nanmedian(aligned[finite])) if np.any(finite) else np.nan)

    if not rows:
        raise RuntimeError("Photometry failed for all frames")

    matrix = np.vstack(rows)
    ctx.photometry_matrix = matrix
    ctx.times = np.asarray(times, dtype=float)
    ctx.airmass = np.asarray(airmass, dtype=float)
    ctx.fwhm = np.asarray(fwhm, dtype=float)
    ctx.sky = np.asarray(sky, dtype=float)

    summary = {
        "processed_frames": len(rows),
        "skipped_frames": skipped,
        "stars_measured": int(matrix.shape[1]),
        "median_airmass": _summary_float(np.nanmedian(ctx.airmass)),
        "median_fwhm": _summary_float(np.nanmedian(ctx.fwhm)),
        "median_sky": _summary_float(np.nanmedian(ctx.sky)),
    }
    return summary


async def _step_lightcurve(ctx: PipelineContext, db: AsyncSession) -> dict[str, Any]:
    if ctx.photometry_matrix is None:
        raise RuntimeError("Photometry matrix missing before lightcurve step")

    matrix = ctx.photometry_matrix
    median_flux = np.nanmedian(matrix, axis=0)
    if not np.any(np.isfinite(median_flux)):
        raise RuntimeError("Unable to determine target star (all medians are NaN)")

    target_index = int(np.nanargmax(median_flux))
    ctx.target_index = target_index

    xy = (
        ctx.reference_stars[:, :2]
        if ctx.reference_stars is not None
        else np.zeros((matrix.shape[1], 2))
    comp_ids = pick_comps_rms_aware_general(
        target_index,
        matrix,
        median_flux,
        xy,
        bright_tol=0.5,
        k=max(min(20, matrix.shape[1] - 1) if matrix.shape[1] > 1 else 0, 3),
    )

    per_star_payload: dict[str, dict[str, Any]] = {}
    ctx.raw_lightcurves = {}
    ctx.raw_lightcurve = None
    ctx.comparison_indices = []
    stored = 0
    skipped = 0
    target_weights: np.ndarray | None = None

    for star_index in range(matrix.shape[1]):
        comp_ids = pick_comps_rms_aware_general(
            star_index,
            matrix,
            median_flux,
            xy,
            bright_tol=0.3,
            k=min(20, matrix.shape[1] - 1) if matrix.shape[1] > 1 else 0,
        )
        if len(comp_ids) < 2:
            sorted_indices = np.argsort(-median_flux)
            comp_ids = [int(idx) for idx in sorted_indices if idx != star_index][:5]

        comp_ids = [int(idx) for idx in comp_ids if idx != star_index]
        if not comp_ids:
            if star_index == target_index:
                raise RuntimeError("Not enough comparison stars identified for lightcurve")
            skipped += 1
            continue

        reference, weights = weighted_reference(matrix[:, comp_ids])
        star_series = matrix[:, star_index]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_rel = star_series / reference
        finite = np.isfinite(raw_rel)
        median = np.nanmedian(raw_rel[finite]) if np.any(finite) else np.nan
        if np.isfinite(median) and median not in (0.0, np.nan):
            raw_rel = raw_rel / median

        ctx.raw_lightcurves[star_index] = raw_rel
        per_star_payload[str(star_index)] = {
            "comparison_indices": comp_ids,
            "comparison_weights": weights.astype(float).tolist(),
            "raw_relative_flux": raw_rel.astype(float).tolist(),
        }
        stored += 1

        if star_index == target_index:
            ctx.comparison_indices = comp_ids
            ctx.raw_lightcurve = raw_rel
            target_weights = weights

    if target_index not in ctx.raw_lightcurves or ctx.raw_lightcurve is None:
        raise RuntimeError("Target star lightcurve could not be computed")

    times = (
        ctx.times
        if ctx.times is not None
        else np.arange(ctx.raw_lightcurves[target_index].shape[0], dtype=float)
    )
    if (
        times.shape[0] != ctx.raw_lightcurves[target_index].shape[0]
        or np.any(~np.isfinite(times))
    ):
        times = np.arange(ctx.raw_lightcurves[target_index].shape[0], dtype=float)
    ctx.times = times

    target_weights = (
        target_weights if target_weights is not None else np.zeros(0, dtype=float)
    )

    payload = {
        "target_index": target_index,
        "comparison_indices": ctx.comparison_indices,
        "times": times.astype(float).tolist(),
        "raw_relative_flux": ctx.raw_lightcurve.astype(float).tolist(),
        "comparison_weights": target_weights.astype(float).tolist(),
        "stars": per_star_payload,
    }

    record = await db.scalar(
        select(models.Lightcurve).where(models.Lightcurve.session_id == ctx.session.id)
    )
    if record is None:
        record = models.Lightcurve(session_id=ctx.session.id, data=payload)
        db.add(record)
    else:
        record.data = payload
    await db.flush()

    summary = {
        "target_index": target_index,
        "comparison_count": len(ctx.comparison_indices),
        "lightcurve_points": len(ctx.raw_lightcurve),
        "stars_saved": stored,
        "stars_skipped": skipped,
    }
    return summary


async def _step_detrend(ctx: PipelineContext, db: AsyncSession) -> dict[str, Any]:
    if not ctx.raw_lightcurves or ctx.times is None:
        raise RuntimeError("Raw lightcurves missing before detrending step")
    if ctx.raw_lightcurve is None:
        raise RuntimeError("Target raw lightcurve missing before detrending step")

    first_series = next(iter(ctx.raw_lightcurves.values()))
    length = int(first_series.shape[0])

    covariates: list[np.ndarray] = []
    cov_names: list[str] = []
    for name, data in ("airmass", ctx.airmass), ("fwhm", ctx.fwhm), ("sky", ctx.sky):
        arr = _ensure_numpy_array(data, length)
        if np.any(np.isfinite(arr)):
            covariates.append(arr)
            cov_names.append(name)

    per_star_payload: dict[str, dict[str, Any]] = {}
    detrended_lightcurves: dict[int, np.ndarray] = {}

    target_baseline: np.ndarray | None = None
    target_corrected: np.ndarray | None = None
    target_good: np.ndarray | None = None

    for star_index, raw_series in ctx.raw_lightcurves.items():
        series = _ensure_numpy_array(raw_series, length)
        if covariates:
            baseline, corrected, good = detrend_by_covariates(series, covariates)
        else:
            baseline = np.ones_like(series)
            corrected = series.copy()
            good = np.isfinite(series)

        detrended_lightcurves[star_index] = corrected
        per_star_payload[str(star_index)] = {
            "raw_relative_flux": series.astype(float).tolist(),
            "detrended_flux": corrected.astype(float).tolist(),
            "baseline": baseline.astype(float).tolist(),
            "good_mask": good.astype(bool).tolist(),
        }

        if star_index == ctx.target_index:
            target_baseline = baseline
            target_corrected = corrected
            target_good = good

    if target_corrected is None or target_baseline is None or target_good is None:
        raise RuntimeError("Target star detrending failed")

    ctx.detrended_flux = target_corrected
    ctx.detrended_lightcurves = detrended_lightcurves

    payload = {
        "times": ctx.times.astype(float).tolist(),
        "raw_relative_flux": ctx.raw_lightcurve.astype(float).tolist()
        if ctx.raw_lightcurve is not None
        else [],
        "detrended_flux": target_corrected.astype(float).tolist(),
        "baseline": target_baseline.astype(float).tolist(),
        "covariate_names": cov_names,
        "good_mask": target_good.astype(bool).tolist(),
        "stars": per_star_payload,
    }

    record = await db.scalar(
        select(models.Denoise).where(models.Denoise.session_id == ctx.session.id)
    )
    if record is None:
        record = models.Denoise(session_id=ctx.session.id, data=payload)
        db.add(record)
    else:
        record.data = payload
    await db.flush()

    summary = {
        "covariates_used": cov_names,
        "points": len(target_corrected),
        "stars_detrended": len(per_star_payload),
    }
    return summary


async def _step_candidate(ctx: PipelineContext, db: AsyncSession) -> dict[str, Any]:
    if ctx.detrended_flux is None or ctx.times is None:
        raise RuntimeError("Detrended flux missing before candidate extraction")

    flux = ctx.detrended_flux
    times = ctx.times
    finite = np.isfinite(flux)
    if not np.any(finite):
        raise RuntimeError("Detrended flux contains no finite values")

    min_index = int(np.nanargmin(flux))
    min_time = float(times[min_index]) if np.isfinite(times[min_index]) else float(min_index)
    median_flux = float(np.nanmedian(flux[finite])) if np.any(finite) else float("nan")
    min_flux = float(flux[min_index])
    depth = median_flux - min_flux if np.isfinite(median_flux) else None

    payload = {
        "target_index": ctx.target_index,
        "comparison_indices": ctx.comparison_indices,
        "event_time": min_time,
        "event_index": min_index,
        "median_flux": median_flux,
        "min_flux": min_flux,
        "depth": depth,
    }

    record = await db.scalar(
        select(models.Candidate).where(models.Candidate.session_id == ctx.session.id)
    )
    if record is None:
        record = models.Candidate(session_id=ctx.session.id, data=payload)
        db.add(record)
    else:
        record.data = payload
    await db.flush()

    summary = {
        "event_index": min_index,
        "event_time": min_time,
        "depth": depth,
    }
    return summary


PIPELINE_STEPS: list[StepDefinition] = [
    StepDefinition("prepare-calibration", _step_prepare_calibration),
    StepDefinition("detect-reference", _step_detect_reference),
    StepDefinition("photometry", _step_photometry),
    StepDefinition("lightcurve", _step_lightcurve),
    StepDefinition("detrend", _step_detrend),
    StepDefinition("candidate", _step_candidate),
]


class SessionPipelineRunner:
    """Coordinate pipeline execution for a session."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def _load_context(self, session_obj: models.Session) -> PipelineContext:
        dataset = await self.db.scalar(
            select(models.Dataset)
            .options(
                selectinload(models.Dataset.data_items),
                selectinload(models.Dataset.preprocess_items).selectinload(
                    models.DatasetPreprocessData.data
                ),
            )
            .where(models.Dataset.id == session_obj.dataset_id)
        )
        if dataset is None:
            raise RuntimeError("Dataset not found for session")

        light_files = [
            Path(item.fits_original_path)
            for item in dataset.data_items
            if item.fits_original_path
        ]
        if not light_files:
            raise RuntimeError("Dataset has no associated light frames")

        preprocess_map: dict[str, list[Path]] = {"bias": [], "dark": [], "flat": []}
        for link in dataset.preprocess_items:
            data = link.data
            if data is None:
                continue
            preprocess_map.setdefault(link.category, []).append(Path(data.fits_original_path))

        settings = get_settings()
        analysis_root = (
            Path(settings.storage_data_dir)
            / f"repository_{session_obj.repository_id}"
            / f"dataset_{dataset.version}"
            / "analysis"
        )
        analysis_root.mkdir(parents=True, exist_ok=True)

        return PipelineContext(
            session=session_obj,
            dataset=dataset,
            analysis_dir=analysis_root,
            light_files=sorted(light_files),
            bias_files=sorted(preprocess_map.get("bias", [])),
            dark_files=sorted(preprocess_map.get("dark", [])),
            flat_files=sorted(preprocess_map.get("flat", [])),
        )

    async def _get_or_create_step(self, run_id, step_name: str) -> models.PipelineStep:
        step = await self.db.scalar(
            select(models.PipelineStep)
            .where(
                models.PipelineStep.run_id == run_id,
                models.PipelineStep.step_name == step_name,
            )
            .limit(1)
        )
        if step is None:
            step = models.PipelineStep(
                run_id=run_id,
                step_name=step_name,
                status="queued",
                progress=0,
            )
            self.db.add(step)
            await self.db.flush()
        return step

    async def run(self, session_id: int) -> None:
        session_obj = await self.db.get(models.Session, session_id)
        if session_obj is None:
            logger.warning("Session %s no longer exists", session_id)
            return

        total_steps = len(PIPELINE_STEPS)

        now = datetime.now(tz=timezone.utc)
        session_obj.started_at = session_obj.started_at or now

        try:
            context = await self._load_context(session_obj)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("Failed to prepare pipeline context for session %s", session_id)
            session_obj.status = "failed"
            session_obj.current_step = PIPELINE_STEPS[0].name if PIPELINE_STEPS else None
            session_obj.progress = session_obj.progress or 0
            session_obj.finished_at = datetime.now(tz=timezone.utc)
            await self.db.commit()
            return

        session_obj.status = "running"
        session_obj.current_step = None
        session_obj.progress = 0
        await self.db.commit()

        for index, definition in enumerate(PIPELINE_STEPS):
            step_record = await self._get_or_create_step(session_obj.run_id, definition.name)
            step_record.status = "running"
            step_record.started_at = datetime.now(tz=timezone.utc)
            step_record.progress = 0
            step_record.log = None

            session_obj.current_step = definition.name
            session_obj.status = "running"
            session_obj.progress = int((index / total_steps) * 100)

            await self.db.commit()

            try:
                result = await definition.runner(context, self.db)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.exception("Pipeline step %s failed", definition.name)
                step_record.status = "failed"
                step_record.finished_at = datetime.now(tz=timezone.utc)
                step_record.log = f"{type(exc).__name__}: {exc}"
                await self.db.commit()

                session_obj.status = "failed"
                session_obj.finished_at = datetime.now(tz=timezone.utc)
                await self.db.commit()
                return

            step_record.status = "completed"
            step_record.finished_at = datetime.now(tz=timezone.utc)
            step_record.progress = 100
            step_record.data = result
            await self.db.commit()

            session_obj.progress = int(((index + 1) / total_steps) * 100)
            await self.db.commit()

        session_obj.status = "completed"
        session_obj.current_step = "completed"
        session_obj.finished_at = datetime.now(tz=timezone.utc)
        session_obj.progress = 100
        await self.db.commit()


__all__ = ["SessionPipelineRunner", "PIPELINE_STEPS"]
