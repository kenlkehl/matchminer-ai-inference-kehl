"""Preparation stays responsive when generation streams occupy their full cap."""

import io
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from urllib.error import URLError

import pytest

from matchminer_ai.llm.request_limits import endpoint_slot
from matchminer_ai.llm.structured import (
    EndpointError,
    StructuredClient,
    StructuredConfig,
)


def test_preparation_proceeds_during_stream_without_bypassing_generation_cap(
    monkeypatch,
):
    entered = Event()
    second_entered = Event()
    release = Event()
    openings = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            entered.set()
            assert release.wait(5)
            yield b'data: {"choices":[{"index":0,"delta":{"content":"{}"},"finish_reason":"stop"}]}\n'
            yield b"data: [DONE]\n"

    def respond(request, timeout):
        if request.full_url.endswith("/chat/completions"):
            openings.append(request.full_url)
            if len(openings) == 2:
                second_entered.set()
            return Stream()
        body = json.loads(request.data) if request.data else None
        if request.full_url.endswith("/tokenize"):
            assert body["chat_template_kwargs"]["enable_thinking"] is True
            value = {"count": 73, "max_model_len": 262144}
        else:
            assert request.full_url.endswith("/models")
            value = {"data": [{"id": "synthetic-model", "max_model_len": 262144}]}
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    client = StructuredClient(
        StructuredConfig(
            base_url="http://preparation.synthetic.invalid/v1",
            model="synthetic-model",
            max_concurrent_requests=1,
        ),
        None,
    )
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(client._http, "/chat/completions", {"stream": True})
        try:
            assert entered.wait(2)
            count = pool.submit(
                client.count_tokens, [{"role": "user", "content": "Synthetic input"}]
            )
            assert count.result(timeout=2) == 73
            assert (
                pool.submit(client.discover).result(timeout=2)["id"]
                == "synthetic-model"
            )
            second = pool.submit(client._http, "/chat/completions", {"stream": True})
            assert not second_entered.wait(0.1)
            assert len(openings) == 1
        finally:
            release.set()
        assert first.result(timeout=2)["choices"][0]["finish_reason"] == "stop"
        assert second.result(timeout=2)["choices"][0]["finish_reason"] == "stop"


def test_preparation_cap_shared_across_clients_and_released_after_error(monkeypatch):
    lock = Lock()
    full = Event()
    over_cap = Event()
    release = Event()
    active = 0
    peak = 0
    calls = 0

    def respond(request, timeout):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            number = calls
            peak = max(peak, active)
            if active == 4:
                full.set()
            if active > 4:
                over_cap.set()
        try:
            assert release.wait(5)
            if number == 1:
                raise URLError("synthetic failure")
            return io.BytesIO(b'{"count":73,"max_model_len":262144}')
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    config = StructuredConfig(base_url="http://bounded.synthetic.invalid/v1")
    clients = [StructuredClient(config, None) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(c.count_tokens, [{"role": "user", "content": "Synthetic"}])
            for c in clients
        ]
        try:
            assert full.wait(2)
            assert not over_cap.wait(0.1)
        finally:
            release.set()
        errors = 0
        for future in futures:
            try:
                assert future.result(timeout=2) == 73
            except EndpointError:
                errors += 1
    assert errors == 1 and peak == 4 and active == 0
    # A released preparation slot remains reusable after the failed request.
    with endpoint_slot(config.base_url, 1, pool="preparation"):
        pass


def test_unknown_request_pool_is_rejected():
    with pytest.raises(ValueError, match="Request pool"):
        with endpoint_slot("http://synthetic.invalid/v1", 1, pool="unbounded"):
            pass
