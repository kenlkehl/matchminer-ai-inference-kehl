"""Bounded per-endpoint concurrency control, independent of response content."""

from dataclasses import dataclass
import math
import re
import time


@dataclass
class AdaptiveConcurrency:
    maximum: int
    initial: int = 32
    minimum: int = 1
    success_threshold: int = 32
    increase_step: int = 32
    adjustment_interval: float = 10.0
    backoff_factor: float = 0.5

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (
            self.maximum, self.initial, self.minimum,
            self.success_threshold, self.increase_step,
        )):
            raise ValueError("Adaptive concurrency counts must be positive integers")
        if not 0 < self.backoff_factor < 1 or self.adjustment_interval <= 0:
            raise ValueError("Invalid adaptive concurrency adjustment settings")
        self.limit = min(self.maximum, max(self.minimum, self.initial))
        self.successes = 0
        self.last_adjustment = float('-inf')
        self.growth_blocked = False
        self.reason = "initial"

    def resize(self, maximum):
        self.maximum = maximum
        self.limit = min(self.limit, maximum)

    def observe(self, *, successful=False, pressure=False, saturated=False, now=None):
        now = time.monotonic() if now is None else now
        if pressure:
            self.successes = 0
            if (self.reason != "capacity_backoff"
                    or now - self.last_adjustment >= self.adjustment_interval):
                self.limit = max(min(self.minimum, self.maximum),
                                 math.floor(self.limit * self.backoff_factor))
                self.last_adjustment, self.reason = now, "capacity_backoff"
        elif successful:
            self.successes += 1
            if (saturated and not self.growth_blocked
                    and self.successes >= self.success_threshold
                    and now - self.last_adjustment >= self.adjustment_interval):
                self.limit = min(self.maximum, max(self.limit * 2,
                                                  self.limit + self.increase_step))
                self.successes = 0
                self.last_adjustment, self.reason = now, "successful_saturated_window"
        return self.limit


def vllm_load(text):
    """Read aggregate load only; never retain request/prompt metric labels."""
    aliases = {
        "num_requests_running": "running", "num_requests_waiting": "waiting",
        "kv_cache_usage_perc": "kv_cache_usage", "gpu_cache_usage_perc": "kv_cache_usage",
        "num_preemptions_total": "preemptions",
    }
    values = {}
    for line in text.splitlines():
        match = re.match(r"vllm:(\w+)(?:\{[^}]*\})?\s+(\S+)", line)
        if not match or match[1] not in aliases:
            continue
        try:
            value = float(match[2])
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        key = aliases[match[1]]
        values[key] = (max(values.get(key, 0), value) if key == "kv_cache_usage"
                       else values.get(key, 0) + value)
    return values
