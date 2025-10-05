"""Simulated lightcurve extraction step."""

from __future__ import annotations

from .runtime import bind_runtime

STEP_NAME = "lightcurve"
STEP_TITLE = "Extract lightcurve"
STEP_DESCRIPTION = "Generate time-series photometry for the aligned exposures."

run_step = bind_runtime(1.8)
