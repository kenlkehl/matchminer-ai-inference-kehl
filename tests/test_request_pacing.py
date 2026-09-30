"""Pacing spaces network dispatches without serializing question workers."""

import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
from threading import Event, Lock
from urllib.error import HTTPError

import pytest

from matchminer_ai.llm.request_pacing import capacity_delay, endpoint_pacer
from matchminer_ai.llm.structured import (
    EndpointError,
    StructuredClient,
    StructuredConfig,
    _retry_after_seconds,
    resolve_structured_config,
)
from matchminer_ai.patients.note_search_qa import NoteSearchLLMConfig, _resolve_llm


def response():
    return io.BytesIO(
        b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}'
    )


def test_clients_space_dispatch_but_keep_responses_concurrent(monkeypatch):
    lock, full, release = Lock(), Event(), Event()
    starts = []
    active, peak = 0, 0

    def serve(request, timeout):
        nonlocal active, peak
        with lock:
            starts.append(time.monotonic())
            active += 1
            peak = max(peak, active)
            if active == 3:
                full.set()
        try:
            assert release.wait(3)
            return response()
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", serve)
    config = StructuredConfig(
        base_url="http://paced-concurrent.synthetic.invalid/v1",
        model="synthetic",
        max_concurrent_requests=3,
        request_start_interval_seconds=0.025,
    )
    clients = [StructuredClient(config, None) for _ in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(c._http, "/chat/completions", {}) for c in clients]
        try:
            assert full.wait(3)
        finally:
            release.set()
        values = [j.result(3) for j in jobs]
    assert peak == 3
    assert all(b - a >= 0.020 for a, b in zip(starts, starts[1:]))
    assert max(v["transport"]["dispatch_wait_seconds"] for v in values) >= 0.035


def test_cooldown_does_not_block_preparation_or_a_different_model(monkeypatch):
    base = "http://isolated-pacing.synthetic.invalid/v1"
    endpoint_pacer(base, "busy-model").defer(1.0)
    monkeypatch.setattr(
        "matchminer_ai.llm.structured.urlopen", lambda *a, **k: response()
    )
    busy = StructuredClient(
        StructuredConfig(
            base_url=base, model="busy-model", request_start_interval_seconds=0.1
        ),
        None,
    )
    other = StructuredClient(
        StructuredConfig(
            base_url=base, model="other-model", request_start_interval_seconds=0.1
        ),
        None,
    )
    started = time.monotonic()
    busy._http("/models")
    busy._http("/tokenize", {})
    other._http("/chat/completions", {})
    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_capacity_retry_preserves_prompt_and_honors_server_delay(
    monkeypatch, tmp_path, status, checkpointed
):
    starts, bodies = [], []

    def serve(request, timeout):
        starts.append(time.monotonic())
        bodies.append(json.loads(request.data))
        if len(starts) == 1:
            raise HTTPError(
                request.full_url,
                status,
                "private provider body",
                {"Retry-After": ".03"},
                None,
            )
        return response()

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", serve)
    monkeypatch.setattr(
        "matchminer_ai.llm.request_pacing.random.uniform", lambda a, b: a
    )
    config = StructuredConfig(
        base_url=f"http://retry-{status}-{checkpointed}.synthetic.invalid/v1",
        model="synthetic",
        stream=False,
        tokenizer_mode="bytes",
        max_tokens=2048,
        capacity_retry_initial_seconds=0.005,
        capacity_retry_max_seconds=0.1,
    )
    client = StructuredClient(config, tmp_path if checkpointed else None)
    messages = [{"role": "user", "content": "Fabricated record; return JSON."}]
    assert client.complete("test", messages, {"type": "object"}, lambda v: None) == {}
    assert len(starts) == 2 and starts[1] - starts[0] >= 0.025
    assert bodies[1] == bodies[0]  # No validation feedback for a rejected request.
    assert "private provider body" not in "".join(
        p.read_text() for p in tmp_path.rglob("*.json")
    )


def test_capacity_retries_remain_bounded(monkeypatch):
    calls = []

    def reject(request, timeout):
        calls.append(request)
        raise HTTPError(request.full_url, 429, "private", {}, None)

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", reject)
    client = StructuredClient(
        StructuredConfig(
            base_url="http://retry-budget.synthetic.invalid/v1",
            model="synthetic",
            tokenizer_mode="bytes",
            stream=False,
            attempts=3,
            capacity_retry_initial_seconds=0.001,
            capacity_retry_max_seconds=0.002,
        ),
        None,
    )
    with pytest.raises(EndpointError):
        client.complete(
            "test", [{"role": "user", "content": "Synthetic"}], {}, lambda v: None
        )
    assert len(calls) == 3


def test_backoff_jitter_and_retry_after_are_bounded(monkeypatch):
    ranges = []
    monkeypatch.setattr(
        "matchminer_ai.llm.request_pacing.random.uniform",
        lambda a, b: ranges.append((a, b)) or a,
    )
    assert [capacity_delay(i, 15, 60) for i in range(4)] == [15, 30, 30, 30]
    assert ranges == [(15, 30), (30, 60), (30, 60), (30, 60)]
    assert capacity_delay(0, 15, 60, 45) == 45
    assert capacity_delay(0, 15, 60, 3600) == 60


def test_retry_after_only_retains_valid_delay(monkeypatch):
    monkeypatch.setattr("matchminer_ai.llm.structured.time.time", lambda: 1000)
    future = format_datetime(datetime.fromtimestamp(1045, tz=timezone.utc), usegmt=True)
    assert _retry_after_seconds(future) == 45
    assert _retry_after_seconds("12") == 12
    assert _retry_after_seconds("-5") == 0
    assert all(
        _retry_after_seconds(v) is None
        for v in [None, "private invalid header", "nan", "inf"]
    )


def test_note_search_preserves_dispatch_policy_without_sending_it_to_model():
    config = NoteSearchLLMConfig(
        model="synthetic",
        base_url="http://policy.synthetic.invalid/v1",
        request_start_interval_seconds=2,
        capacity_retry_initial_seconds=15,
        capacity_retry_max_seconds=60,
    )
    resolved = _resolve_llm(config)
    assert resolved.request_start_interval_seconds == 2
    assert resolved.capacity_retry_initial_seconds == 15
    assert resolved.capacity_retry_max_seconds == 60
    assert not any("seconds" in k for k in resolved.request_params)
    assert not any("seconds" in k for k in resolved.extra_body)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_pacing_rejected_before_http(value):
    with pytest.raises(ValueError):
        resolve_structured_config(
            {
                "server_urls": ["http://invalid-policy.synthetic.invalid/v1"],
                "model_name": "synthetic",
                "context_window": 262144,
                "request_start_interval_seconds": value,
            },
            cache_dir=None,
        )


def test_disabled_policy_preserves_old_public_config_identity():
    public = StructuredConfig().public_dict()
    assert not any(k.endswith("seconds") for k in public)


def test_malformed_chat_envelope_still_uses_bounded_validation_retries(monkeypatch):
    calls = []

    def invalid(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b"[]")

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", invalid)
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda seconds: None)
    client = StructuredClient(
        StructuredConfig(
            base_url="http://malformed-pacing.synthetic.invalid/v1",
            model="synthetic",
            tokenizer_mode="bytes",
            stream=False,
            attempts=2,
        ),
        None,
    )
    with pytest.raises(EndpointError, match="bounded retries"):
        client.complete(
            "test", [{"role": "user", "content": "Synthetic"}], {}, lambda v: None
        )
    assert len(calls) == 2
