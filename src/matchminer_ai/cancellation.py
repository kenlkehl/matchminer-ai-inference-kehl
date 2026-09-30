"""Opt-in, context-local cancellation for patient inference jobs.

Cancellation stops local work and further dispatch. An HTTP request already
accepted by a remote server may finish there; its result is discarded locally.
"""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from threading import Event, Thread
import time


class InferenceCancelled(BaseException):
    """Control flow, deliberately not caught as a model/validation failure."""


class CancellationToken:
    def __init__(self):
        self.event = Event()

    def cancel(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise InferenceCancelled("Inference stopped.")


_token = ContextVar("matchminer_inference_cancellation", default=None)


@contextmanager
def cancellation_scope(token):
    handle = _token.set(token)
    try:
        token.check()
        yield
    finally:
        _token.reset(handle)


def check_cancelled():
    token = _token.get()
    if token is not None:
        token.check()


def cancel_sleep(seconds):
    token = _token.get()
    if token is None:
        time.sleep(seconds)
    else:
        token.check()
        token.event.wait(seconds)
        token.check()


def submit_cancellable(pool, fn, *args, **kwargs):
    """Carry the caller's cancellation scope into an existing thread pool."""
    check_cancelled()
    return pool.submit(copy_context().run, fn, *args, **kwargs)


def run_cancellable(fn):
    """Stop waiting on blocking HTTP promptly, retaining its capacity slot.

    Only scoped calls use a daemon transport thread. That thread owns the HTTP
    response and concurrency slot until the response closes or times out, so
    cancelled requests cannot silently exceed the endpoint's in-flight cap.
    """
    token = _token.get()
    if token is None:
        return fn()
    token.check()
    done = Event()
    outcome = []
    context = copy_context()

    def run():
        try:
            value = context.run(fn)
            if not token.event.is_set():
                outcome.append((True, value))
        except BaseException as exc:
            if not token.event.is_set():
                outcome.append((False, exc))
        finally:
            done.set()

    Thread(target=run, name="mmai-http", daemon=True).start()
    while not done.wait(0.1):
        token.check()
    token.check()
    ok, value = outcome[0]
    if not ok:
        raise value
    return value


async def await_cancellable(awaitable):
    token = _token.get()
    if token is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            token.check()
            done, _ = await asyncio.wait({task}, timeout=0.1)
            if done:
                token.check()
                return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
