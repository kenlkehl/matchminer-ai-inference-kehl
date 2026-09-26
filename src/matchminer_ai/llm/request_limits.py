"""Process-wide HTTP concurrency limits shared by guideline workers."""

from collections import Counter
from contextlib import contextmanager
from threading import Condition

_condition = Condition()
_states: dict[str, dict] = {}


@contextmanager
def endpoint_slot(base_url, limit):
    """Hold a slot through the complete response stream, including error paths.

    Calls sharing an endpoint share the cap even across separate disease clients.
    If simultaneous callers specify different limits, respect the smallest cap.
    Independent Python processes have independent caps.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("max_concurrent_requests must be a positive integer")
    key = base_url.rstrip("/")
    with _condition:
        state = _states.setdefault(key, {"active": 0, "limits": Counter()})
        state["limits"][limit] += 1
        while state["active"] >= min(state["limits"]):
            _condition.wait()
        state["active"] += 1
    try:
        yield
    finally:
        with _condition:
            state["active"] -= 1
            state["limits"][limit] -= 1
            if not state["limits"][limit]:
                del state["limits"][limit]
            if not state["limits"]:
                del _states[key]
            _condition.notify_all()
