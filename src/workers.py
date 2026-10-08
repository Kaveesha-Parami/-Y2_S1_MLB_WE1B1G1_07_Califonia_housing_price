"""Bounded background inference executor. No Streamlit APIs inside workers."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import Any, Callable


class QueueFullError(RuntimeError):
    """Background worker and queue capacity reached."""


class BoundedWorkers:
    """Limit both running jobs and jobs waiting for a worker slot.

    Executor itself has an unbounded queue; the semaphore places a hard cap on
    submitted-but-not-completed futures, preventing an accidental memory surge.
    """

    def __init__(self, max_workers: int = 2, max_inflight: int = 4) -> None:
        if max_workers < 1 or max_inflight < max_workers:
            raise ValueError('Worker limits must be positive and inflight >= workers.')
        self._slots = BoundedSemaphore(max_inflight)
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix='housing_inference')

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future:
        if not self._slots.acquire(blocking=False):
            raise QueueFullError('All background inference slots are busy. Retry shortly.')
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except Exception:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        return future
