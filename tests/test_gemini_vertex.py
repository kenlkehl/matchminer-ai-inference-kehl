"""Hosted Gemini wire settings and ADC for the structured workup client."""

import io
import json
from dataclasses import replace

import pytest

from matchminer_ai.llm.remote_auth import prepare_messages_for_provider
from matchminer_ai.llm.remote_inference import build_remote_request_config
from matchminer_ai.llm.structured import (
    StructuredClient,
    StructuredConfig,
    resolve_structured_config,
)

ENDPOINT = "https://aiplatform.googleapis.com/v1/projects/profile-notes/locations/global/endpoints/openapi"
MODEL = "google/gemini-3.8-flash"


def runtime(**overrides):
    return {
        "provider": "google_agent_platform", "google_project_id": "profile-notes",
        "model_name": MODEL, "server_urls": [ENDPOINT], "context_window": 1048576,
        "tokenizer_mode": "bytes", "stream": False, "sampling_profile": "auto",
        "reasoning_effort": "xhigh", "max_concurrent_requests": 1,
        "remote": {"request_params": {"max_tokens": 65536, "temperature": 1,
                                      "top_p": .95, "presence_penalty": 1},
                   "extra_body": {"top_k": 64, "chat_template_kwargs": {"enable_thinking": True}}},
        **overrides,
    }


@pytest.mark.parametrize("effort,expected", [("xhigh", "high"), ("high", "high"),
                                             ("medium", "medium"), ("low", "low")])
def test_gemini_manages_sampling_and_uses_supported_reasoning(effort, expected):
    params, extra = build_remote_request_config(runtime(reasoning_effort=effort))
    assert params == {"max_tokens": 65536, "reasoning_effort": expected}
    assert extra == {}


def test_gemini_preserves_system_instructions():
    messages = [{"role": "system", "content": "Follow the schema."},
                {"role": "user", "content": "Fabricated note."}]
    assert prepare_messages_for_provider(messages, runtime()) == messages


@pytest.mark.parametrize("stream", [False, True])
def test_structured_wire_uses_fresh_adc_and_no_vllm_parameters(monkeypatch, stream):
    tokens = iter(["first-token", "second-token"])
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-key")
    auth_calls, requests = [], []

    def token(config):
        auth_calls.append(config)
        return next(tokens)

    def respond(request, timeout):
        requests.append(request)
        if stream:
            return io.BytesIO(b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":null}]}\n\n'
                              b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                              b'data: [DONE]\n\n')
        return io.BytesIO(b'{"choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}')

    monkeypatch.setattr("matchminer_ai.llm.structured.remote_bearer_token", token)
    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    config, _ = resolve_structured_config(runtime(stream=stream), cache_dir=None)
    client = StructuredClient(config, None)
    for index in range(2):
        assert client.complete(str(index), [{"role": "user", "content": "Return JSON."}],
                               {"type": "object"}, lambda data: data) == {}
    assert [r.get_header("Authorization") for r in requests] == ["Bearer first-token", "Bearer second-token"]
    assert all(r.full_url == ENDPOINT + "/chat/completions" for r in requests)
    for request in requests:
        body = json.loads(request.data)
        assert body["reasoning_effort"] == "high"
        assert body["response_format"]["type"] == "json_schema"
        assert body["max_tokens"] == 65536
        assert not {"temperature", "top_p", "top_k", "chat_template_kwargs"} & body.keys()
    assert auth_calls == [{"provider": "google_agent_platform", "google_project_id": "profile-notes"}] * 2


@pytest.mark.parametrize("url", ["http://aiplatform.googleapis.com/v1", "https://untrusted.test/v1",
                                 "https://aiplatform.googleapis.com.untrusted.test/v1"])
def test_structured_adc_rejects_non_google_destination_before_auth(monkeypatch, url):
    def fail(*args):
        pytest.fail("Credentials must not be obtained for an untrusted host")

    monkeypatch.setattr("matchminer_ai.llm.structured.remote_bearer_token", fail)
    config, _ = resolve_structured_config(runtime(), cache_dir=None)
    client = StructuredClient(replace(config, base_url=url), None)
    with pytest.raises(ValueError, match="HTTPS Agent Platform"):
        client._http("/chat/completions", {})


def test_local_catalog_config_identity_does_not_change():
    config = StructuredConfig(base_url="http://server/v1", model="Qwen/Qwen3.8-27B")
    assert "provider" not in config.public_dict()
    assert "google_project_id" not in config.public_dict()


def test_hosted_patient_prompt_budget_uses_bytes_not_proxy_token_count(monkeypatch):
    from types import SimpleNamespace

    from matchminer_ai.patients import prompt_builder as module

    messages = [{"role": "user", "content": "é" * 200}]
    class ProxyTokenizer:
        def apply_chat_template(self, **kwargs):
            return "proxy rendering"

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=[1])

    monkeypatch.setattr(module, "_worker_tokenizer", ProxyTokenizer())
    monkeypatch.setattr(module, "get_serial_patient_prompt", lambda **kwargs: messages)
    monkeypatch.setattr(module, "_worker_config", {
        "prompt_files": {"primer": "unused", "question": "unused"},
        "max_model_len": 1000, "prompt_margin_tokens": 0, "model_name": MODEL,
        "tokenizer_mode": "bytes", "sampling_params": {"max_tokens": 900},
    })
    item = module.PromptWorkItem(0, None, "", "", "Fabricated note")
    assert module.build_prompt_worker(item).max_tokens == 1000 - (64 + 32 + 400) - 256
    module._worker_config["max_model_len"] = 700
    with pytest.raises(ValueError, match="context budget"):
        module.build_prompt_worker(item)
