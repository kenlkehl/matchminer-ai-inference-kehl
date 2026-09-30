"""Gemma MaaS preserves supported thinking/sampling and the provider chat contract."""

import copy
import io
import json

import pytest

from matchminer_ai.llm.remote_inference import build_remote_request_config
from matchminer_ai.llm.structured import StructuredClient, resolve_structured_config

ENDPOINT = "https://aiplatform.googleapis.com/v1/projects/profile-notes/locations/global/endpoints/openapi"
MODEL = "google/gemma-4-26b-a4b-it-maas"


def runtime(thinking=True, stream=False):
    return {
        "provider": "google_agent_platform",
        "google_project_id": "profile-notes",
        "server_urls": [ENDPOINT],
        "model_name": MODEL,
        "context_window": 262144,
        "tokenizer_mode": "bytes",
        "sampling_profile": "auto",
        "stream": stream,
        "remote": {
            "request_params": {"max_tokens": 65536},
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": thinking,
                    "preserve_thinking": False,
                },
                "min_p": 0.0,
                "repetition_penalty": 1.0,
            },
        },
    }


@pytest.mark.parametrize("thinking", [True, False])
def test_gemma_maas_retains_only_supported_extensions(thinking):
    source = runtime(thinking)
    before = copy.deepcopy(source)
    params, extra = build_remote_request_config(source)
    assert params == {"max_tokens": 65536, "temperature": 1.0, "top_p": 0.95}
    assert extra == {"top_k": 64, "chat_template_kwargs": {"enable_thinking": thinking}}
    assert source == before


@pytest.mark.parametrize("stream", [False, True])
def test_gemma_structured_wire_folds_instructions_and_enables_thinking(
    monkeypatch, stream
):
    requests = []
    monkeypatch.setattr(
        "matchminer_ai.llm.structured.remote_bearer_token", lambda config: "test-token"
    )

    def respond(request, timeout):
        requests.append(request)
        if stream:
            return io.BytesIO(
                b'data: {"choices":[{"delta":{"reasoning_content":"private reasoning"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":null}]}\n\n'
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                b"data: [DONE]\n\n"
            )
        return io.BytesIO(
            b'{"choices":[{"message":{"content":"{}","reasoning_content":"private reasoning"},"finish_reason":"stop"}]}'
        )

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    config, _ = resolve_structured_config(runtime(stream=stream), cache_dir=None)
    messages = [
        {"role": "system", "content": "Follow the schema."},
        {"role": "user", "content": "Fabricated notes only."},
    ]
    original = copy.deepcopy(messages)
    client = StructuredClient(config, None)
    assert (
        client.complete("test", messages, {"type": "object"}, lambda value: value) == {}
    )
    assert messages == original
    body = json.loads(requests[0].data)
    assert body["messages"] == [
        {
            "role": "user",
            "content": "Instructions:\nFollow the schema.\n\nRequest:\nFabricated notes only.",
        }
    ]
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["temperature"] == 1.0 and body["top_p"] == 0.95 and body["top_k"] == 64
    assert body["response_format"]["type"] == "json_schema"
    assert body["max_tokens"] == 65536 and "reasoning_effort" not in body
    assert requests[0].get_header("Authorization") == "Bearer test-token"
