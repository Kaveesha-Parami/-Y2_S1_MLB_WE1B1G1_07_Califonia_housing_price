import threading
import pytest
from src.workers import BoundedWorkers, QueueFullError


def test_job_queue_is_bounded_and_releases_capacity():
    event = threading.Event()
    pool = BoundedWorkers(max_workers=1, max_inflight=1)
    def wait():
        event.wait(timeout=2)
        return 42
    first = pool.submit(wait)
    try:
        with pytest.raises(QueueFullError):
            pool.submit(lambda: 'should never run')
    finally:
        event.set()
    assert first.result(timeout=3) == 42
    assert pool.submit(lambda: 7).result(timeout=3) == 7
