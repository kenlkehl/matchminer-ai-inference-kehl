"""Reflection Beam discovery, defaults, overrides and vLLM request wiring."""
import copy
import io
import json

import httpx
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config
from matchminer_ai.llm.model_profiles import (
    REFLECTION_BEAM, configure_served_model, resolve_model_profile,
)
from matchminer_ai.llm.reasoning import resolve_reasoning_parser, resolve_tool_call_parser
from matchminer_ai.llm.sampling import remote_sampling, vendor_defaults
from matchminer_ai.llm.structured import StructuredClient, resolve_structured_config
from matchminer_ai.llm.vllm_server import build_vllm_server_command

MODEL = "reflection/Beam-1.0-501B-A23B-NVFP4"
EXPECTED = dict(temperature=0.7, top_p=0.99, top_k=-1, min_p=0.0,
                presence_penalty=0.0, repetition_penalty=1.0)


@pytest.mark.parametrize("model", [
    MODEL, "reflection/Beam-1.0-501B-A23B", "reflection/Beam-1.0-501B-A23B-FP8",
    "other/Beam-1.0-501B-A23B-GGUF", "/weights/Beam-1.0-501B-A23B-NVFP4",
    "Beam-1.0-501B-A23B", "reflection-beam", "BEAM",
])
def test_beam_family_uses_published_sampling_and_parsers(model):
    assert resolve_model_profile(model) is REFLECTION_BEAM
    assert resolve_reasoning_parser(model) == "beam"
    assert resolve_tool_call_parser(model) == "beam"
    defaults, template = vendor_defaults(model)
    assert defaults == EXPECTED
    assert template == {"reasoning_effort": "medium"}


@pytest.mark.parametrize("model", [
    "beam-search", "acme/sunbeam-7b", "reflection/Reflection-70B",
    "reflection/Beam-2.0-501B", "reflection/Beam-1.0-501B-A23B-DFlash2",
])
def test_unrelated_models_and_drafter_do_not_get_beam_defaults(model):
    assert resolve_model_profile(model) is None
    assert vendor_defaults(model) == ({}, {})


@pytest.mark.parametrize("profile", ["beam", "reflection-beam"])
def test_explicit_profile_overrides_and_opt_out(profile):
    runtime = {
        "model_name": "opaque-alias", "sampling_profile": profile,
        "remote": {
            "request_params": {"temperature": 0.0, "reasoning_effort": "low", "max_tokens": 123},
            "extra_body": {"top_k": 7, "min_p": 0.1,
                           "chat_template_kwargs": {"allow_missing_thinking": True}},
        },
    }
    original = copy.deepcopy(runtime)
    params, extra = remote_sampling(runtime)
    assert params == dict(temperature=0.0, top_p=0.99, presence_penalty=0.0,
                         reasoning_effort="low", max_tokens=123)
    assert extra == dict(top_k=7, min_p=0.1, repetition_penalty=1.0,
                         chat_template_kwargs={"allow_missing_thinking": True, "reasoning_effort": "low"})
    assert runtime == original
    assert vendor_defaults(MODEL, profile="none") == ({}, {})
    assert resolve_reasoning_parser(MODEL, "custom") == "custom"
    assert resolve_tool_call_parser(MODEL, "none") is None


def test_endpoint_discovery_and_local_server_command(monkeypatch):
    monkeypatch.setattr("matchminer_ai.llm.model_profiles.httpx.get", lambda url, **kw:
        httpx.Response(200, json={"data": [{"id": MODEL}]}, request=httpx.Request("GET", url)))
    config = load_default_preset()
    budget = config.trial["remote"]["request_params"]["max_tokens"]
    config.trial["remote"]["extra_body"].update(top_k=64, repetition_penalty=1.2)
    model, profile = configure_served_model(config, ["test:8000"], sections=["trial"])
    assert model == MODEL and profile is REFLECTION_BEAM
    runtime = build_llm_runtime_config("trial", config.trial, config=config)
    params, extra = remote_sampling(runtime)
    assert all({**params, **extra}[key] == value for key, value in EXPECTED.items())
    assert params["max_tokens"] == budget
    assert extra["chat_template_kwargs"]["reasoning_effort"] == "medium"
    config.trial["local"]["model_name"] = MODEL
    command = build_vllm_server_command(config=config, task="trial").command
    assert command[command.index("--reasoning-parser") + 1] == "beam"
    assert command[command.index("--tool-call-parser") + 1] == "beam"
    assert "--enable-auto-tool-choice" in command


def test_structured_discovery_sends_beam_sampling_on_wire(monkeypatch):
    seen = []

    def respond(req, timeout):
        body = json.loads(req.data) if req.data else None
        seen.append((req.full_url, body))
        if req.full_url.endswith("/models"):
            value = {"data": [{"id": MODEL, "max_model_len": 262144}]}
        elif req.full_url.endswith("/tokenize"):
            value = {"count": 30, "max_model_len": 262144}
        else:
            value = {"choices": [{"message": {"content": "{}", "reasoning_content": "Synthetic reasoning"}, "finish_reason": "stop"}]}
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=["http://test:8000/v1"])
    config.guideline["remote"]["model_name"] = ""
    config.guideline["remote"]["request_params"] = {"max_tokens": 1000}
    config.guideline["remote"]["extra_body"] = {}
    config.guideline["stream"] = False
    runtime = build_llm_runtime_config("guideline", config.guideline, config=config)
    resolved, metadata = resolve_structured_config(runtime, cache_dir=None)
    assert metadata["discovery"]["id"] == MODEL
    assert resolved.thinking == "default"
    StructuredClient(resolved, None).complete(
        "beam-test", [{"role": "user", "content": "Synthetic input"}],
        {"type": "object"}, lambda result: None,
    )
    body = next(body for url, body in seen if url.endswith("/chat/completions"))
    assert body["model"] == MODEL and body["max_tokens"] == 1000
    assert all(body[key] == value for key, value in EXPECTED.items())
    assert body["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("temperature", [None, 0.0])
def test_local_generation_defaults_preserve_explicit_overrides(temperature):
    config = load_default_preset()
    config.remote["enabled"] = False
    config.trial["local"].update(model_name=MODEL, chat_template_kwargs={}, generation={"max_tokens": 123})
    if temperature is not None:
        config.trial["local"]["generation"]["temperature"] = temperature
    runtime = build_llm_runtime_config("trial", config.trial, config=config)
    assert runtime["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert runtime["sampling_params"] == {**EXPECTED, "max_tokens": 123,
                                          "temperature": 0.7 if temperature is None else temperature}


@pytest.mark.parametrize("effort", ["low", "max"])
def test_explicit_beam_effort_survives_defaults(effort):
    _, template = vendor_defaults(MODEL, template={"reasoning_effort": effort})
    assert template == {"reasoning_effort": effort}
    params, extra = remote_sampling({
        "model_name": MODEL, "sampling_profile": "auto",
        "remote": {"extra_body": {"chat_template_kwargs": template}},
    })
    assert extra["chat_template_kwargs"] == template
    assert "reasoning_effort" not in params


def test_beam_rejects_conflicting_api_and_template_efforts():
    with pytest.raises(ValueError, match="must agree"):
        remote_sampling({"model_name": MODEL, "sampling_profile": "auto", "remote": {
            "request_params": {"reasoning_effort": "low"},
            "extra_body": {"chat_template_kwargs": {"reasoning_effort": "max"}},
        }})
