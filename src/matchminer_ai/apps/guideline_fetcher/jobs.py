"""Bounded in-memory jobs, isolated by browser session and expired after one hour."""

import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .services import InputError
from matchminer_ai.cancellation import CancellationToken, InferenceCancelled, cancellation_scope


class BusyError(ValueError):
    pass


class Jobs:
    def __init__(self, ttl=3600):
        self.ttl = ttl
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guideline-fetcher")
        self.items = {}
        self.controls = {}

    def _clean(self):
        for key, job in list(self.items.items()):
            if job["status"] in {"complete", "failed", "cancelled"} and time.time() - job["updated"] > self.ttl:
                del self.items[key]
                self.controls.pop(key, None)

    def submit(self, owner, kind, fn, *, restore=None):
        with self.lock:
            self._clean()
            active = [j for j in self.items.values() if j["status"] in {"queued", "running", "cancelling"}]
            if len(active) >= 4 or any(j["owner"] == owner for j in active):
                raise BusyError(
                    "A job is already running. Wait for it to finish before starting another."
                )
            key = secrets.token_urlsafe(24)
            self.items[key] = dict(
                id=key,
                owner=owner,
                kind=kind,
                status="queued",
                progress="Waiting for the inference worker",
                updated=time.time(),
                result=None,
                restore=restore,
            )
            token = CancellationToken()
            self.controls[key] = {"token": token, "future": None}

        def progress(text):
            token.check()
            with self.lock:
                if key in self.items:
                    self.items[key].update(progress=text, updated=time.time())

        def run():
            try:
                with cancellation_scope(token):
                    with self.lock:
                        token.check()
                        self.items[key]["status"] = "running"
                    result = fn(progress)
                    token.check()
                with self.lock:
                    token.check()
                    self.items[key].update(
                        status="complete",
                        progress="Ready for review",
                        result=result,
                        updated=time.time(),
                    )
            except InferenceCancelled:
                with self.lock:
                    self.items[key].update(status="cancelled", progress="Stopped. No further inference requests will be sent.", updated=time.time())
            except Exception as exc:
                # Do not serialize provider errors, filenames or patient text to logs/UI.
                error = (
                    str(exc)
                    if isinstance(exc, InputError)
                    else (
                        f"{type(exc).__name__}: inference could not finish. Check model availability and server configuration, then retry."
                    )
                )
                with self.lock:
                    cancelled = token.event.is_set()
                    self.items[key].update(status="cancelled" if cancelled else "failed", progress="Stopped." if cancelled else error, updated=time.time())

        with self.lock:
            if token.event.is_set():
                self.items[key].update(status="cancelled", progress="Stopped.", updated=time.time())
            else:
                self.controls[key]["future"] = self.pool.submit(run)
        return key

    def cancel(self, owner, key):
        with self.lock:
            self._clean()
            job = self.items.get(key)
            if not job or job["owner"] != owner:
                return None
            if job["status"] in {"queued", "running", "cancelling"}:
                control = self.controls[key]
                control["token"].cancel()
                future = control["future"]
                queued = future is not None and future.cancel()
                job.update(status="cancelled" if queued else "cancelling", progress="Stopped." if queued else "Stopping local workers…", updated=time.time())
            return self.get(owner, key)

    def get(self, owner, key):
        with self.lock:
            self._clean()
            job = self.items.get(key)
            if not job or job["owner"] != owner:
                return None
            return {k: v for k, v in job.items() if k != "owner"}

    def clear(self, owner):
        with self.lock:
            if any(
                j["owner"] == owner and j["status"] in {"queued", "running", "cancelling"}
                for j in self.items.values()
            ):
                raise BusyError("Wait for the current job to finish before clearing the workspace.")
            removed = [k for k, j in self.items.items() if j["owner"] == owner]
            for key in removed:
                del self.items[key]
                self.controls.pop(key, None)

    def latest(self, owner, *, exclude_prefix=None):
        with self.lock:
            self._clean()
            mine = [j for j in self.items.values() if j["owner"] == owner
                    and not (exclude_prefix and j["kind"].startswith(exclude_prefix))]
            if not mine:
                return None
            return self.get(owner, max(mine, key=lambda j: j["updated"])["id"])

    def shutdown(self):
        with self.lock:
            for control in self.controls.values():
                control["token"].cancel()
                if control["future"] is not None:
                    control["future"].cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)
