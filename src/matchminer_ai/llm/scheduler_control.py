"""Adaptive per-server concurrency for the teacher pool (llm.teacher_pool), driven by vLLM /metrics.

Moved unchanged from the training repository's adaptive_controller.py (2026-11-06).
"""
import time
from datetime import datetime, timezone
HARD_CEILING = 1024


def apply_tuning(controllers, ceilings, document, applied_revision):
    """Apply a validated operator command once; retain automatic pressure backoff."""
    revision = document.get('revision')
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError('Tuning requires a nonempty revision')
    if revision == applied_revision:
        return None
    changes = document.get('endpoints')
    if not isinstance(changes, dict) or not changes:
        raise ValueError('Tuning requires endpoint settings')
    for name, settings in changes.items():
        if name not in controllers or not isinstance(settings, dict):
            raise ValueError('Unknown endpoint or invalid settings: ' + str(name))
        cap, upper = settings.get('cap'), settings.get('upper')
        if (type(cap) is not int or type(upper) is not int or
                not 32 <= cap <= upper <= min(ceilings[name], HARD_CEILING)):
            raise ValueError('Invalid tuning cap/upper for ' + name)
    events = []
    for name, settings in changes.items():
        controller = controllers[name]
        old_cap, old_upper = controller.cap, controller.upper
        controller.cap, controller.upper = settings['cap'], settings['upper']
        controller.samples = []
        controller.throughput_samples = []
        controller.previous_cap = controller.previous_rate = None
        controller.pressure_streak = 0
        controller.draining = controller.draining or controller.cap < old_cap
        events.append(dict(endpoint=name, old_cap=old_cap, old_upper=old_upper,
                           new_cap=controller.cap, new_upper=controller.upper))
    return dict(at=now(), revision=revision, changes=events)


def now():
    return datetime.now(timezone.utc).isoformat()

def read_metrics(session, endpoint):
    response = session.get(endpoint.removesuffix("/v1") + "/metrics", timeout=15)
    response.raise_for_status()
    return parse_metrics(response.text)


def parse_metrics(text):
    """Share scheduler metric parsing between sync supervisors and the async pool."""
    names = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:kv_cache_usage_perc": "cache",
        "vllm:num_preemptions_total": "preemptions",
        "vllm:generation_tokens_total": "tokens",
    }
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name not in names:
            continue
        key, value = names[name], float(line.rsplit(" ", 1)[1])
        values[key] = max(values.get(key, 0), value) if key == "cache" else values.get(key, 0) + value
    if set(values) != set(names.values()):
        raise ValueError("Required scheduler metrics are unavailable; retain current concurrency")
    return {"time": time.time(), **values}

class Controller:
    """Increase on sustained headroom; drain on pressure or clear throughput regression."""
    def __init__(self, initial, ceiling):
        self.cap = initial
        self.upper = ceiling
        self.samples = []
        self.throughput_samples = []
        self.previous_cap = None
        self.previous_rate = None
        self.pressure_streak = 0
        self.draining = False

    def reset_observations(self):
        """Discard incomparable measurements without lowering concurrency limits."""
        self.samples = []
        self.throughput_samples = []
        self.previous_cap = self.previous_rate = None
        self.pressure_streak = 0

    def observe(self, sample):
        if self.draining:
            if sample["running"] + sample["waiting"] > self.cap:
                return None
            self.draining = False
            self.reset_observations()
        if self.samples:
            previous = self.samples[-1]
            if (sample['time'] <= previous['time'] or sample['time'] - previous['time'] > 60
                    or sample['tokens'] < previous['tokens']
                    or sample['preemptions'] < previous['preemptions']):
                self.reset_observations()
        self.samples.append(sample)
        first = self.samples[0]
        supplied = sample['running'] + sample['waiting'] >= self.cap * .70
        if supplied:
            self.throughput_samples.append(sample)
        else:
            # Client outages, sparse work, and end-of-stage draining say nothing
            # about GPU capacity. Neither comparison window may contain these.
            self.throughput_samples = []
            self.previous_cap = self.previous_rate = None
        pressure = (sample["cache"] >= .90 or sample["waiting"] > max(8, self.cap * .20)
                    or sample["preemptions"] > first["preemptions"])
        self.pressure_streak = self.pressure_streak + 1 if pressure else 0
        elapsed = sample["time"] - first["time"]
        critical = sample["cache"] >= .90 or sample["preemptions"] > first["preemptions"]
        recent = self.samples[-3:]
        peak_running = max(s["running"] for s in recent)
        stable_running = (len(recent) == 3 and peak_running > 0 and
                          peak_running - min(s["running"] for s in recent) <= max(8, peak_running * .05))
        # A cold server may queue while prefill gradually admits requests. Do not
        # mistake that ramp for the scheduler's steady-state sequence ceiling.
        pressure_ready = self.pressure_streak >= 3 and (
            critical or (elapsed >= 180 and stable_running))
        throughput_elapsed = (sample['time'] - self.throughput_samples[0]['time']
                              if self.throughput_samples else 0)
        throughput_ready = len(self.throughput_samples) >= 3 and throughput_elapsed >= 180
        if not pressure_ready and not throughput_ready:
            # Keep pressure observations bounded during a prolonged idle period.
            if elapsed >= 180 and not pressure:
                self.samples = recent
            return None
        rate = ((sample['tokens'] - self.throughput_samples[0]['tokens']) / throughput_elapsed
                if throughput_ready else None)
        new, reason = self.cap, "hold"
        if pressure_ready:
            # Round backoffs to multiples of 8, not 32: on a small server a 32-step
            # turns the 25% cut into more than half (72 would fall to 32, not 48).
            if sample["cache"] < .90 and sample["preemptions"] == first["preemptions"] and 0 < sample["running"] < self.cap:
                new = max(32, min(self.cap, (int(peak_running) + 7) // 8 * 8))
            else:
                new = max(32, int(self.cap * .75) // 8 * 8)
            self.upper = min(self.upper, new)
            reason = "scheduler queue/cache pressure or preemption"
        elif (throughput_ready and self.previous_rate is not None and rate < self.previous_rate * .90
              and self.previous_cap is not None and self.cap > self.previous_cap):
            new = self.previous_cap
            self.upper = min(self.upper, new)
            reason = "generation throughput regressed after increasing concurrency"
        elif (throughput_ready and all(s["waiting"] <= max(2, self.cap * .05) and s["cache"] < .80
                  and s["running"] >= self.cap * .70 for s in recent)
              and self.cap < self.upper):
            new = min(self.upper, max(self.cap + 32, int(self.cap * 1.5) // 32 * 32))
            reason = "sustained scheduler and cache headroom"
        new = min(self.upper, new)
        event = {"at": now(), "old_cap": self.cap, "new_cap": new,
                 "generation_tokens_per_second": rate, "reason": reason, "metrics": sample}
        self.previous_cap = self.cap if new > self.cap else None
        self.previous_rate = rate if new > self.cap else None
        self.draining = new < self.cap
        self.cap = new
        self.samples = [sample]
        self.throughput_samples = []
        self.pressure_streak = 0
        return event
