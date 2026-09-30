"""Cancellation must escape retries and release local agents, not falsify answers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from threading import Event, Timer
import time

import pytest

from matchminer_ai.cancellation import (
    CancellationToken,
    InferenceCancelled,
    await_cancellable,
    cancel_sleep,
    cancellation_scope,
    submit_cancellable,
)
from matchminer_ai.llm.request_limits import endpoint_slot, _states
from matchminer_ai.llm.request_pacing import RequestPacer
from matchminer_ai.llm.structured import StructuredClient, StructuredConfig
from matchminer_ai.patients import NoteSearchLimits
from matchminer_ai.patients._note_repl import NoteREPL


def test_scope_carries_to_questions_and_does_not_leak():
    token = CancellationToken()
    with ThreadPoolExecutor(2) as pool:
        with cancellation_scope(token):
            futures = [submit_cancellable(pool, cancel_sleep, 30) for _ in range(2)]
            token.cancel()
            for future in futures:
                with pytest.raises(InferenceCancelled):
                    future.result(1)
        assert pool.submit(cancel_sleep, 0).result(1) is None


def test_cancelled_capacity_wait_unregisters_limit():
    base = "http://capacity-cancel.synthetic.invalid/v1"
    token = CancellationToken()
    with endpoint_slot(base, 1), ThreadPoolExecutor(1) as pool:

        def wait():
            with endpoint_slot(base, 1):
                pytest.fail("Cancelled waiter acquired capacity")

        with cancellation_scope(token):
            future = submit_cancellable(pool, wait)
        time.sleep(0.02)
        token.cancel()
        with pytest.raises(InferenceCancelled):
            future.result(1)
    assert (base, "generation") not in _states


def test_cancelled_cooldown_does_not_dispatch_or_reserve_next_start():
    pacer, token = RequestPacer(), CancellationToken()
    pacer.defer(30)
    original = pacer.next_start
    with ThreadPoolExecutor(1) as pool:
        with cancellation_scope(token):
            future = submit_cancellable(pool, pacer.wait, 2)
        token.cancel()
        with pytest.raises(InferenceCancelled):
            future.result(1)
    assert pacer.next_start == original


def test_blocking_http_stops_waiting_without_retry_or_releasing_wire_capacity(
    monkeypatch,
):
    started, release, closed = Event(), Event(), Event()
    base = "http://http-cancel.synthetic.invalid/v1"
    calls = []

    class Response(io.BytesIO):
        def close(self):
            closed.set()
            super().close()

    def serve(*args, **kwargs):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return Response(
            b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'
        )

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", serve)
    client = StructuredClient(
        StructuredConfig(
            base_url=base,
            model="synthetic",
            stream=False,
            tokenizer_mode="bytes",
            max_concurrent_requests=1,
        ),
        None,
    )
    token = CancellationToken()
    with ThreadPoolExecutor(1) as pool:
        with cancellation_scope(token):
            future = submit_cancellable(
                pool,
                client.complete,
                "test",
                [{"role": "user", "content": "Synthetic"}],
                {},
                lambda _: None,
            )
        try:
            assert started.wait(2)
            token.cancel()
            with pytest.raises(InferenceCancelled):
                future.result(1)
            assert _states[(base, "generation")]["active"] == 1
            assert len(calls) == 1
        finally:
            release.set()
            assert closed.wait(2)
    deadline = time.monotonic() + 2
    while (base, "generation") in _states and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (base, "generation") not in _states


def test_cancel_python_cell_kills_worker():
    token = CancellationToken()
    with (
        cancellation_scope(token),
        NoteREPL(
            "Fabricated notes.", NoteSearchLimits(cell_timeout_seconds=30)
        ) as worker,
    ):
        timer = Timer(0.2, token.cancel)
        timer.start()
        try:
            started = time.monotonic()
            with pytest.raises(InferenceCancelled):
                worker.execute("while True: pass")
            assert time.monotonic() - started < 2
            assert worker.process.poll() is not None
        finally:
            timer.cancel()


def test_async_remote_generation_is_cancelled_and_closed():
    token, closed = CancellationToken(), Event()

    async def request():
        try:
            await asyncio.sleep(30)
        finally:
            closed.set()

    async def run():
        timer = Timer(0.1, token.cancel)
        timer.start()
        try:
            with cancellation_scope(token), pytest.raises(InferenceCancelled):
                await await_cancellable(request())
        finally:
            timer.cancel()

    asyncio.run(run())
    assert closed.is_set()
