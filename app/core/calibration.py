"""Simulated calibration step applying dark/bias/flat corrections."""

from __future__ import annotations

from .runtime import bind_runtime

STEP_NAME = "calibration"
STEP_TITLE = "Calibrate exposures"
STEP_DESCRIPTION = "Apply dark, bias, and flat-field corrections to raw frames."

run_step = bind_runtime(2.0)
