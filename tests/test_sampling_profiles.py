"""Vendor defaults, explicit overrides, actual wire requests and shared capacity."""

import copy
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config
from matchminer_ai.llm.remote_inference import build_remote_request_config
from matchminer_ai.llm.request_limits import endpoint_slot
from matchminer_ai.llm.sampling import vendor_defaults
from matchminer_ai.llm.structured import StructuredClient, resolve_structured_config


@pytest.mark.parametrize(
    "model,k",
    [
        ("google/gemma-4-31B-it", 64),
        ("nvidia/Gemma-4-31B-IT-NVFP4", 64),
        ("google/gemma-4-26B-A4B-it", 64),
        ("Qwen/Qwen3.8-27B", 20),
        ("Inferact/Qwen3.8-Flash-Next-NVFP4", 20),
    ],
)
def test_defaults_follow_model_for_local_and_remote(model, k):
    config = load_default_preset()
    config.trial["local"]["model_name"] = model
    config.trial["remote"]["model_name"] = model
    for remote in (False, True):
        config.remote["enabled"] = remote
        runtime = build_llm_runtime_config("trial", config.trial, config=config)
        params, extra = (
            build_remote_request_config(runtime)
            if remote
            else (runtime["sampling_params"], {})
        )
        assert params["temperature"] == 1.0
        assert params["top_p"] == 0.95
        assert extra.get("top_k", params.get("top_k")) == k
        template = runtime["chat_template_kwargs"]
        assert template["enable_thinking"] is True
        if k == 20:
            assert template["reasoning_effort"] == "xhigh"
            if remote:
                assert params["reasoning_effort"] == "xhigh"
        else:
            assert "reasoning_effort" not in template
            assert "reasoning_effort" not in params


def test_explicit_overrides_and_non_thinking_mode():
    config = load_default_preset()
    config.remote["enabled"] = True
    stage = config.trial
    stage["remote"]["model_name"] = "Qwen/Qwen3.8-27B"
    stage["remote"]["request_params"].update(temperature=0.4, reasoning_effort="low")
    stage["remote"]["extra_body"]["top_k"] = 12
    before = copy.deepcopy(stage)
    result = build_llm_runtime_config("trial", stage, config=config)
    assert stage == before
    params, extra = build_remote_request_config(result)
    assert params["temperature"] == 0.4 and extra["top_k"] == 12
    assert extra["chat_template_kwargs"]["reasoning_effort"] == "low"
    stage["remote"]["extra_body"]["chat_template_kwargs"]["reasoning_effort"] = "medium"
    with pytest.raises(ValueError, match="must agree"):
        build_llm_runtime_config("trial", stage, config=config)
    defaults, template = vendor_defaults(
        "Qwen/Qwen3.8-Flash-Next", template={"enable_thinking": False}
    )
    assert defaults["temperature"] == 0.7 and defaults["top_p"] == 0.8
    assert defaults["presence_penalty"] == 1.5
    assert "reasoning_effort" not in template
    assert vendor_defaults("unrecognized/model") == ({}, {})
    with pytest.raises(ValueError, match="reasoning_effort"):
        vendor_defaults("Qwen/Qwen3.8-27B", reasoning_effort="high")


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-4B", "unsloth/Qwen3.5-4B-GGUF"])
def test_qwen35_general_task_defaults_follow_thinking_without_graded_effort(model, thinking):
    config = load_default_preset()
    stage = config.patient
    stage["remote"]["model_name"] = model
    stage["local"]["model_name"] = model
    stage["local"]["chat_template_kwargs"] = {"enable_thinking": thinking}
    stage["remote"]["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    expected = {
        "temperature": 1.0 if thinking else 0.7,
        "top_p": 0.95 if thinking else 0.8,
        "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
    }
    before = copy.deepcopy(stage)
    for remote in (False, True):
        config.remote["enabled"] = remote
        runtime = build_llm_runtime_config("patient", stage, config=config)
        params, extra = (build_remote_request_config(runtime) if remote
                         else (runtime["sampling_params"], {}))
        assert all({**params, **extra}[key] == value for key, value in expected.items())
        assert runtime["chat_template_kwargs"] == {"enable_thinking": thinking}
        assert "reasoning_effort" not in params
        assert "reasoning_effort" not in extra
    assert stage == before
    assert vendor_defaults("opaque-alias", profile="qwen3.5", template={"enable_thinking": thinking}) == (
        expected, {"enable_thinking": thinking}
    )


def test_discovered_model_controls_structured_requests_and_tokenizer(
    monkeypatch, tmp_path
):
    seen = []

    def respond(req, timeout):
        body = json.loads(req.data) if req.data else None
        seen.append((req.full_url, body))
        if req.full_url.endswith("/models"):
            value = {
                "data": [
                    {"id": "Inferact/Qwen3.8-Flash-Next-NVFP4", "max_model_len": 262144}
                ]
            }
        elif req.full_url.endswith("/tokenize"):
            value = {"count": 58, "max_model_len": 262144}
        else:
            value = {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]
            }
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    config = load_default_preset()
    config.remote["enabled"] = True
    runtime = build_llm_runtime_config("guideline", config.guideline, config=config)
    resolved, _metadata = resolve_structured_config(runtime, cache_dir=tmp_path)
    assert resolved.top_k == 20 and resolved.max_tokens == 100000
    assert resolved.context_window == 262144
    assert resolved.request_params["reasoning_effort"] == "xhigh"
    client = StructuredClient(resolved, tmp_path)
    client.count_tokens([{"role": "user", "content": "Synthetic input"}])
    assert seen[-1][1]["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    assert resolved.max_concurrent_requests == 32


def test_shared_endpoint_limit_releases_slots_after_failure():
    lock = threading.Lock()
    active = 0
    peak = 0
    barrier = threading.Barrier(8)

    def run(i):
        nonlocal active, peak
        barrier.wait()
        try:
            with endpoint_slot("http://synthetic.invalid/v1", 2):
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.01)
                with lock:
                    active -= 1
                if i == 0:
                    raise RuntimeError("synthetic error")
        except RuntimeError:
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))
    assert peak == 2 and active == 0
    with endpoint_slot("http://synthetic.invalid/v1", 1):
        pass
