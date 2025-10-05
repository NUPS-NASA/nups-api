"""Simulated registration step for aligning frames."""

from __future__ import annotations

from .runtime import bind_runtime

STEP_NAME = "registration"
STEP_TITLE = "Register exposures"
STEP_DESCRIPTION = "Align frames against reference stars to stabilise the stack."

run_step = bind_runtime(1.2)
