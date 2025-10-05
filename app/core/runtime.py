"""Runtime helpers for simulated pipeline steps."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable


async def simulate_runtime(seconds: float = 1.0) -> None:
    """Sleep in small increments to emulate work without blocking tests."""

    # Sleep using small intervals to keep control responsive while keeping
    # the total runtime deterministic for tests.
    remaining = float(seconds)
    interval = 0.1
    while remaining > 0:
        await asyncio.sleep(min(interval, remaining))
        remaining -= interval


def bind_runtime(seconds: float) -> Callable[[], Awaitable[None]]:
    """Return a coroutine factory that simulates work for ``seconds`` seconds."""

    async def _runner() -> None:
        await simulate_runtime(seconds)

    return _runner
