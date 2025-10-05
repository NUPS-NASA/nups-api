"""Simulated reporting step producing summary artefacts."""

from __future__ import annotations

from .runtime import bind_runtime

STEP_NAME = "reporting"
STEP_TITLE = "Compile report"
STEP_DESCRIPTION = "Generate session-level reports and notify subscribers."

run_step = bind_runtime(0.8)
