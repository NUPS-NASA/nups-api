"""Simulated denoising step for extracted lightcurves."""

from __future__ import annotations

from .runtime import bind_runtime

STEP_NAME = "denoise"
STEP_TITLE = "Denoise lightcurve"
STEP_DESCRIPTION = "Reduce noise using frequency-domain filtering techniques."

run_step = bind_runtime(1.0)
