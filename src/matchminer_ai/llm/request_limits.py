"""Process-wide HTTP concurrency limits shared by guideline workers."""

from collections import Counter
from contextlib import contextmanager
from threading import Condition

from matchminer_ai.cancellation import check_cancelled

_condition = Condition()
_states: dict[tuple[str, str], dict] = {}


@contextmanager
def endpoint_slot(base_url, limit, *, pool="generation"):
    """Hold a slot through the complete response stream, including error paths.

    Calls sharing an endpoint and pool share the cap across disease clients.
    Preparation requests use a separate bounded pool so token counting need not
    wait for long generation streams to finish.
    If simultaneous callers specify different limits, respect the smallest cap.
    Independent Python processes have independent caps.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("max_concurrent_requests must be a positive integer")
    if pool not in {"generation", "preparation"}:
        raise ValueError("Request pool must be generation or preparation")
    key = (base_url.rstrip("/"), pool)
    with _condition:
        state = _states.setdefault(key, {"active": 0, "limits": Counter()})
        state["limits"][limit] += 1
        acquired = False
        try:
            while state["active"] >= min(state["limits"]):
                check_cancelled()
                _condition.wait(0.1)
            check_cancelled()
            state["active"] += 1
            acquired = True
        finally:
            if not acquired:
                state["limits"][limit] -= 1
                if not state["limits"][limit]:
                    del state["limits"][limit]
                if not state["limits"]:
                    del _states[key]
                _condition.notify_all()
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
