"""Simulated classification step for candidate events."""

from __future__ import annotations

from .runtime import bind_runtime

STEP_NAME = "classification"
STEP_TITLE = "Classify events"
STEP_DESCRIPTION = "Score extracted signals to prioritise follow-up."

run_step = bind_runtime(1.4)
