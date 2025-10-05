"""Simulated ingest step for the analysis pipeline."""

from __future__ import annotations

from .runtime import bind_runtime

STEP_NAME = "ingest"
STEP_TITLE = "Ingest raw exposures"
STEP_DESCRIPTION = "Pull uploaded FITS files into the processing workspace."

run_step = bind_runtime(1.5)
