"""Declarative pipeline configuration backed by core analysis steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Mapping

from . import (
    calibration,
    classification,
    denoise,
    ingest,
    lightcurve,
    registration,
    reporting,
)


@dataclass(frozen=True)
class PipelineStepDefinition:
    """Metadata describing a single pipeline step."""

    name: str
    title: str
    description: str
    runner: Callable[[], Awaitable[None]]


DEFAULT_PIPELINE: tuple[PipelineStepDefinition, ...] = (
    PipelineStepDefinition(
        name=ingest.STEP_NAME,
        title=ingest.STEP_TITLE,
        description=ingest.STEP_DESCRIPTION,
        runner=ingest.run_step,
    ),
    PipelineStepDefinition(
        name=calibration.STEP_NAME,
        title=calibration.STEP_TITLE,
        description=calibration.STEP_DESCRIPTION,
        runner=calibration.run_step,
    ),
    PipelineStepDefinition(
        name=registration.STEP_NAME,
        title=registration.STEP_TITLE,
        description=registration.STEP_DESCRIPTION,
        runner=registration.run_step,
    ),
    PipelineStepDefinition(
        name=lightcurve.STEP_NAME,
        title=lightcurve.STEP_TITLE,
        description=lightcurve.STEP_DESCRIPTION,
        runner=lightcurve.run_step,
    ),
    PipelineStepDefinition(
        name=denoise.STEP_NAME,
        title=denoise.STEP_TITLE,
        description=denoise.STEP_DESCRIPTION,
        runner=denoise.run_step,
    ),
    PipelineStepDefinition(
        name=classification.STEP_NAME,
        title=classification.STEP_TITLE,
        description=classification.STEP_DESCRIPTION,
        runner=classification.run_step,
    ),
    PipelineStepDefinition(
        name=reporting.STEP_NAME,
        title=reporting.STEP_TITLE,
        description=reporting.STEP_DESCRIPTION,
        runner=reporting.run_step,
    ),
)

_STEP_LOOKUP: Mapping[str, PipelineStepDefinition] = {
    step.name: step for step in DEFAULT_PIPELINE
}


def get_default_pipeline() -> tuple[PipelineStepDefinition, ...]:
    """Return the immutable default pipeline definition."""

    return DEFAULT_PIPELINE


def get_step(name: str) -> PipelineStepDefinition | None:
    """Return metadata for a named step if it exists."""

    return _STEP_LOOKUP.get(name)


def iter_step_names() -> Iterable[str]:
    """Iterate over canonical step names in execution order."""

    return (step.name for step in DEFAULT_PIPELINE)


__all__ = [
    "PipelineStepDefinition",
    "DEFAULT_PIPELINE",
    "get_default_pipeline",
    "get_step",
    "iter_step_names",
]
