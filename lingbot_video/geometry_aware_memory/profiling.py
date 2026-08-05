from __future__ import annotations

import contextlib
import contextvars
import time
from collections.abc import Iterator

import torch


ProfileTimings = dict[str, float]
_ACTIVE_TIMINGS: contextvars.ContextVar[ProfileTimings | None] = (
    contextvars.ContextVar("gim_profile_timings", default=None)
)


def synchronize_device(device: torch.device) -> None:
    """Finish queued CUDA work so wall-clock stage boundaries are meaningful."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def profiling_scope(timings: ProfileTimings | None) -> Iterator[None]:
    """Expose one rank-local timing sink through DDP's forward wrapper."""

    token = _ACTIVE_TIMINGS.set(timings)
    try:
        yield
    finally:
        _ACTIVE_TIMINGS.reset(token)


@contextlib.contextmanager
def synchronized_stage(
    timings: ProfileTimings | None,
    name: str,
    device: torch.device,
) -> Iterator[None]:
    """Accumulate synchronized wall time for one diagnostic stage.

    This deliberately serializes CUDA execution. It belongs only in the
    temporary profiling branch and must not be used for throughput training.
    """

    active_timings = _ACTIVE_TIMINGS.get()
    if active_timings is not None:
        timings = active_timings
    if timings is None:
        yield
        return
    synchronize_device(device)
    started_at = time.perf_counter()
    try:
        yield
    finally:
        synchronize_device(device)
        timings[name] = timings.get(name, 0.0) + (
            time.perf_counter() - started_at
        )
