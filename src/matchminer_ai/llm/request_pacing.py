"""Process-local dispatch pacing; contains no prompts, credentials or responses."""

import random
import time
from threading import Condition, Lock

from matchminer_ai.cancellation import check_cancelled


class RequestPacer:
    """Space starts across clients; pause queued requests after capacity failures."""

    def __init__(self):
        self.condition = Condition()
        self.next_start = 0.0

    def wait(self, interval):
        started = time.monotonic()
        with self.condition:
            while self.next_start > time.monotonic():
                check_cancelled()
                self.condition.wait(min(0.1, self.next_start - time.monotonic()))
            check_cancelled()
            self.next_start = time.monotonic() + interval
        return time.monotonic() - started

    def defer(self, delay):
        with self.condition:
            self.next_start = max(self.next_start, time.monotonic() + delay)
            self.condition.notify_all()


_lock = Lock()
_pacers = {}


def endpoint_pacer(base_url, model):
    # Model-specific: two hosted models can share the same API base URL.
    key = (base_url.rstrip("/"), model)
    with _lock:
        return _pacers.setdefault(key, RequestPacer())


def capacity_delay(attempt, initial, maximum, retry_after=None):
    """Equal-jitter exponential backoff, with bounded server-provided delay."""
    ceiling = min(maximum, initial * 2 ** min(attempt + 1, 20))
    delay = random.uniform(ceiling / 2, ceiling)
    if retry_after is not None:
        delay = max(delay, min(retry_after, maximum))
    return delay
