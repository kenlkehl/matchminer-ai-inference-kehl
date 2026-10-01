"""Compression keeps source rows intact and sends bounded plain-text requests."""

import io
import json
from copy import deepcopy
from threading import Barrier, Lock, get_ident
from urllib.error import HTTPError

import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.cancellation import (
    CancellationToken,
    InferenceCancelled,
    cancellation_scope,
)
from matchminer_ai.llm.structured import EndpointError
from matchminer_ai.patients import (
    NoteCompressionError,
    compress_patient_note,
    compress_patient_notes,
    compression,
)


def reply(text="Compressed fabricated note.", *, finish="stop", **message):
    return {
        "choices": [
            {
                "message": {"content": text, **message},
                "finish_reason": finish,
            }
        ]
    }


@pytest.fixture
def endpoint(monkeypatch):
    config = load_default_preset()
    config.remote.update(
        enabled=True,
        server_urls=["http://compression.synthetic.invalid/v1"],
        context_window=16384,
        tokenizer_mode="bytes",
        stream=False,
        max_retries=1,
        max_concurrent_requests=2,
        api_key_env="SYNTHETIC_COMPRESSION_KEY",
    )
    config.patient["remote"] = {
        "model_name": "synthetic-model",
        "request_params": {"max_tokens": 2048, "temperature": 0.25},
        "extra_body": {},
    }
    calls = []
    responder = [lambda request, body: reply()]

    def serve(request, *, timeout):
        body = json.loads(request.data) if request.data else None
        calls.append((request, body, timeout))
        value = responder[0](request, body)
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", serve)
    return config, calls, responder


def test_single_note_normalizes_before_dispatch_and_returns_only_final_text(endpoint):
    config, calls, responder = endpoint
    responder[0] = lambda *args: reply(
        "  60F; stable.  ", reasoning_content="Private model explanation."
    )
    text = " \t60F.\n\n  Stable.\r\n\tNo\u00a0\u2003fever. "
    assert compress_patient_note(text, config=config) == "60F; stable."
    request, body, timeout = calls[0]
    assert (
        request.full_url == "http://compression.synthetic.invalid/v1/chat/completions"
    )
    assert body["messages"][1] == {"role": "user", "content": "60F. Stable. No fever."}
    assert body["messages"][0] == {
        "role": "system",
        "content": (
            "Compress this note to extreme but lossless token density while "
            "remaining understandable. Return only the compressed note, with "
            "no explanatory text, commentary, or preamble."
            " Use common clinical abbreviations, but if you invent abbreviations, "
            "define them at first use."
        ),
    }
    assert "response_format" not in body
    assert body["model"] == "synthetic-model"
    assert body["temperature"] == 0.25
    assert body["max_tokens"] == 2048
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert timeout == config.remote["request_timeout"]


def test_requests_overlap_with_bound_and_restore_rows_and_metadata(endpoint):
    config, calls, responder = endpoint
    source = pd.DataFrame(
        {
            "patient_id": [
                "fabricated-a",
                "fabricated-b",
                "fabricated-a",
                "fabricated-b",
            ],
            "note_date": ["2026-02-02", "2026-01-01", None, "2026-02-01"],
            "note_type": ["visit", "scan", "lab", "visit"],
            "note_text": ["Note A.", "Note B.", "Note C.", "Note D."],
        },
        index=[7, 7, 2, 0],
    )
    original, original_config = source.copy(deep=True), deepcopy(config)
    barrier, lock = Barrier(2), Lock()
    state = {"active": 0, "peak": 0}

    def answer(request, body):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            # Serial execution deadlocks at this barrier and fails the test.
            barrier.wait(timeout=3)
            return reply(body["messages"][1]["content"] + " compressed")
        finally:
            with lock:
                state["active"] -= 1

    responder[0] = answer
    progress, threads = [], []

    def report(completed, total):
        progress.append((completed, total))
        threads.append(get_ident())

    result = compress_patient_notes(source, config=config, progress_callback=report)
    assert state["peak"] == 2
    assert len(calls) == 4
    assert result["compressed_note_text"].tolist() == [
        text + " compressed" for text in source["note_text"]
    ]
    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_frame_equal(result.drop(columns="compressed_note_text"), original)
    assert config == original_config
    assert progress == [(i, 4) for i in range(5)]
    assert set(threads) == {get_ident()}
    assert all(
        body["messages"][1]["content"] in source["note_text"].tolist()
        for _, body, _ in calls
    )
    assert all(
        body["chat_template_kwargs"]["enable_thinking"] is False
        and body["messages"][0]["content"].startswith("Compress this note")
        for _, body, _ in calls
    )


@pytest.mark.parametrize(
    "model", ["synthetic-model", "google/gemma-4-31B-it", "Qwen/Qwen3.8-27B"]
)
def test_default_off_overrides_inherited_effort_without_mutating_config(
    endpoint, model
):
    config, calls, _ = endpoint
    config.patient["remote"].update(
        model_name=model,
        request_params={"max_tokens": 2048, "reasoning_effort": "xhigh"},
        extra_body={
            "reasoning_effort": "xhigh",
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "medium",
            },
        },
    )
    original = deepcopy(config)
    compress_patient_note("Fabricated note.", config=config)
    body = calls[0][1]
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert "reasoning_effort" not in body
    assert "reasoning_effort" not in body["chat_template_kwargs"]
    assert config == original
    if model.startswith("Qwen/"):
        # Sampling defaults must resolve after choosing non-thinking mode.
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.8
        assert body["presence_penalty"] == 1.5


def test_explicit_thinking_on_changes_request_without_changing_prompt(endpoint):
    config, calls, _ = endpoint
    config.patient["remote"].update(
        model_name="Qwen/Qwen3.8-27B",
        request_params={"max_tokens": 2048, "reasoning_effort": "medium"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    original = deepcopy(config)
    compress_patient_note("Fabricated note.", config=config)
    compress_patient_note("Fabricated note.", config=config, thinking="on")
    default_body, body = calls[0][1], calls[1][1]
    assert default_body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["messages"] == default_body["messages"]
    assert body["messages"][0]["content"].startswith("Compress this note")
    assert body["chat_template_kwargs"]["enable_thinking"] is True
    assert body["chat_template_kwargs"]["reasoning_effort"] == "medium"
    assert body["reasoning_effort"] == "medium"
    assert body["temperature"] == 1.0
    assert config == original


@pytest.mark.parametrize("thinking", [None, True, "default", "OFF"])
def test_invalid_thinking_is_rejected_before_dispatch(endpoint, thinking):
    config, calls, _ = endpoint
    with pytest.raises(ValueError, match="thinking must be"):
        compress_patient_note("Fabricated note.", config=config, thinking=thinking)
    assert calls == []


@pytest.mark.parametrize("model", ["google/gemini-3.8-pro", "google/gemini-3.8-flash"])
def test_default_off_omits_unsupported_switch_and_keeps_inherited_effort(
    endpoint, model, monkeypatch
):
    config, calls, responder = endpoint
    config.remote["provider"] = "google_agent_platform"
    config.remote["server_urls"] = [
        "https://aiplatform.googleapis.com/v1/projects/fabricated/locations/global/endpoints/openapi"
    ]
    config.patient["remote"]["model_name"] = model
    config.patient["remote"]["request_params"]["reasoning_effort"] = "medium"
    monkeypatch.setattr(
        "matchminer_ai.llm.structured.remote_bearer_token", lambda _: "fabricated-token"
    )
    original = deepcopy(config)
    responder[0] = lambda *args: reply(
        "CT done.", reasoning_content="Omitted reasoning."
    )
    result = compress_patient_notes(
        pd.DataFrame({"note_text": ["CT completed."]}), config=config
    )
    assert result.compressed_note_text.tolist() == ["CT done."]
    assert result.attrs["note_compression"] == {
        "thinking_requested": "off",
        "thinking": "default",
    }
    assert len(calls) == 1
    assert "chat_template_kwargs" not in calls[0][1]
    assert calls[0][1]["reasoning_effort"] == "medium"
    assert config == original


@pytest.mark.parametrize("route", ["/tokenize", "/chat/completions"])
def test_endpoint_rejecting_switch_retries_once_without_it(endpoint, route):
    config, calls, responder = endpoint
    config.remote["tokenizer_mode"] = "endpoint"
    config.patient["remote"]["request_params"]["reasoning_effort"] = "medium"
    config.patient["remote"]["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": True, "custom": "kept"}
    }
    original = deepcopy(config)
    rejected = []

    def respond(request, body):
        if (
            request.full_url.endswith(route)
            and body.get("chat_template_kwargs", {}).get("enable_thinking") is False
        ):
            rejected.append(body)
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(
                    b'{"error": {"message": "enable_thinking=false is not supported"}}'
                ),
            )
        if request.full_url.endswith("/tokenize"):
            return {"count": 100}
        return reply("CT done.", reasoning_content="Discarded reasoning.")

    responder[0] = respond
    result = compress_patient_notes(
        pd.DataFrame({"note_text": ["CT completed."]}), config=config
    )
    assert result.compressed_note_text.tolist() == ["CT done."]
    assert len(rejected) == 1
    assert calls[-1][1]["chat_template_kwargs"] == {"custom": "kept"}
    assert calls[-1][1]["reasoning_effort"] == "medium"
    assert result.attrs["note_compression"]["thinking"] == "default"
    assert all("compression.synthetic.invalid" in r.full_url for r, _, _ in calls)
    assert config == original


@pytest.mark.parametrize(
    "status, message",
    [
        (400, "Invalid max_tokens"),
        (401, "enable_thinking unsupported"),
        (503, "enable_thinking unsupported"),
    ],
)
def test_other_endpoint_failures_do_not_trigger_thinking_fallback(
    endpoint, status, message
):
    config, calls, responder = endpoint

    def fail(*args):
        raise EndpointError(message, http_status=status)

    responder[0] = fail
    with pytest.raises(NoteCompressionError):
        compress_patient_note("Fabricated note.", config=config)
    assert len(calls) == 1
    assert calls[0][1]["chat_template_kwargs"]["enable_thinking"] is False


def test_unsupported_switch_fallback_is_bounded_and_not_applied_to_explicit_on(
    endpoint,
):
    config, calls, responder = endpoint

    def fail(request, body):
        raise HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            {},
            io.BytesIO(b'{"error": {"message": "enable_thinking is unsupported"}}'),
        )

    responder[0] = fail
    with pytest.raises(NoteCompressionError):
        compress_patient_note("Fabricated note.", config=config)
    assert len(calls) == 2
    assert "chat_template_kwargs" not in calls[1][1]
    calls.clear()
    with pytest.raises(NoteCompressionError):
        compress_patient_note("Fabricated note.", config=config, thinking="on")
    assert len(calls) == 1


def test_gemma_maas_off_survives_provider_filtering_and_prompt_folding(
    endpoint, monkeypatch
):
    config, calls, _ = endpoint
    config.remote["provider"] = "google_agent_platform"
    config.remote["server_urls"] = [
        "https://aiplatform.googleapis.com/v1/projects/fabricated/locations/global/endpoints/openapi"
    ]
    config.patient["remote"]["model_name"] = "google/gemma-4-26b-a4b-it-maas"
    monkeypatch.setattr(
        "matchminer_ai.llm.structured.remote_bearer_token", lambda _: "fabricated-token"
    )
    compress_patient_note("Fabricated note.", config=config)
    body = calls[0][1]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert "Instructions:\nCompress this note" in body["messages"][0]["content"]


def test_empty_and_blank_notes_do_not_call_endpoint(endpoint):
    config, calls, _ = endpoint
    progress = []
    source = pd.DataFrame({"note_text": ["", " \n\t "]}, index=[2, 1])
    result = compress_patient_notes(
        source, config=config, progress_callback=lambda *p: progress.append(p)
    )
    assert result["compressed_note_text"].tolist() == ["", ""]
    assert progress == [(0, 2), (2, 2)]
    assert compress_patient_note("", config=config) == ""
    assert compress_patient_notes(source.iloc[:0], config=config).empty
    assert calls == []


@pytest.mark.parametrize("value", [None, 7, float("nan")])
def test_rejects_nonstring_input_before_any_dispatch(endpoint, value):
    config, calls, _ = endpoint
    with pytest.raises(TypeError, match="must be a string"):
        compress_patient_notes(
            pd.DataFrame({"note_text": ["Valid.", value]}), config=config
        )
    assert calls == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_rejects_invalid_concurrency(endpoint, limit):
    config, calls, _ = endpoint
    with pytest.raises(ValueError, match="positive integer"):
        compress_patient_notes(
            pd.DataFrame({"note_text": ["Fabricated."]}),
            config=config,
            max_concurrent_requests=limit,
        )
    assert calls == []


def test_requires_endpoint_and_canonical_input_without_overwriting(endpoint):
    config, calls, _ = endpoint
    with pytest.raises(TypeError, match="DataFrame"):
        compress_patient_notes(["Fabricated."], config=config)
    with pytest.raises(ValueError, match="note_text"):
        compress_patient_notes(pd.DataFrame({"text": ["Fabricated."]}), config=config)
    with pytest.raises(ValueError, match="unique columns"):
        compress_patient_notes(
            pd.DataFrame([["A", "B"]], columns=["note_text", "note_text"]),
            config=config,
        )
    with pytest.raises(ValueError, match="already contains"):
        compress_patient_notes(
            pd.DataFrame({"note_text": ["A"], "compressed_note_text": ["B"]}),
            config=config,
        )
    config.remote["enabled"] = False
    with pytest.raises(ValueError, match="remote LLM endpoint"):
        compress_patient_note("Fabricated.", config=config)
    assert calls == []


@pytest.mark.parametrize(
    "response",
    [
        reply(""),
        reply("Partial note.", finish="length"),
        reply(None),
        reply("Rejected", refusal="Refused"),
        reply("<think>Explanation</think>60F; stable."),
        reply("Here is the compressed note: 60F; stable."),
        {},
    ],
)
def test_never_returns_incomplete_refused_malformed_or_explanatory_text(
    endpoint, response
):
    config, _, responder = endpoint
    responder[0] = lambda *args: response
    with pytest.raises(NoteCompressionError, match="bounded endpoint retries"):
        compress_patient_note("Fabricated note.", config=config)


def test_bad_output_retries_with_only_output_contract_feedback(endpoint, monkeypatch):
    config, calls, responder = endpoint
    config.remote["max_retries"] = 2
    monkeypatch.setattr(compression, "cancel_sleep", lambda _: None)
    responses = iter([reply("Sure, here is a summary."), reply("60F; stable.")])
    responder[0] = lambda *args: next(responses)
    assert compress_patient_note(" 60F.\n\nStable. ", config=config) == "60F; stable."
    assert len(calls) == 2
    assert calls[1][1]["messages"][:2] == calls[0][1]["messages"]
    assert calls[1][1]["messages"][2] == {
        "role": "user",
        "content": (
            "Return the complete compressed note only. No explanations, "
            "commentary, preamble, or reasoning."
        ),
    }


def test_transport_error_never_becomes_note_or_content_in_logs(endpoint, caplog):
    config, _, responder = endpoint

    def fail(*args):
        raise EndpointError("Fabricated private provider echo.")

    responder[0] = fail
    with pytest.raises(NoteCompressionError) as exc:
        compress_patient_note("Fabricated private patient note.", config=config)
    assert "private" not in str(exc.value)
    assert "private" not in caplog.text
    assert exc.value.__suppress_context__


def test_rejects_oversized_note_without_truncation(endpoint):
    config, calls, _ = endpoint
    config.remote.update(context_window=512, safety_tokens=32)
    config.patient["remote"]["request_params"]["max_tokens"] = 128
    with pytest.raises(ValueError, match="prompt exceeds"):
        compress_patient_note("X" * 2000, config=config)
    assert calls == []


def test_discovers_once_and_uses_configured_auth_and_model(endpoint, monkeypatch):
    config, calls, responder = endpoint
    config.remote.pop("context_window")
    config.patient["remote"]["model_name"] = ""
    monkeypatch.setenv("SYNTHETIC_COMPRESSION_KEY", "fabricated-key")

    def answer(request, body):
        if request.full_url.endswith("/models"):
            return {"data": [{"id": "discovered-model", "max_model_len": 16384}]}
        return reply("Fabricated.")

    responder[0] = answer
    result = compress_patient_notes(
        pd.DataFrame(
            {"note_text": ["First fabricated note.", "Second fabricated note."]}
        ),
        config=config,
    )
    assert result["compressed_note_text"].tolist() == ["Fabricated."] * 2
    assert sum(request.full_url.endswith("/models") for request, _, _ in calls) == 1
    assert all(
        body["model"] == "discovered-model"
        for request, body, _ in calls
        if body is not None
    )
    assert all(
        body["chat_template_kwargs"]["enable_thinking"] is False
        for _, body, _ in calls
        if body is not None
    )
    assert all(
        request.get_header("Authorization") == "Bearer fabricated-key"
        for request, _, _ in calls
    )


def test_cancellation_propagates_into_note_workers_without_retry(endpoint):
    config, calls, responder = endpoint
    token = CancellationToken()

    def answer(*args):
        token.cancel()
        return reply("Discard this late result.")

    responder[0] = answer
    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        compress_patient_note("Fabricated note.", config=config)
    assert len(calls) == 1


def test_streamed_response_discards_separate_reasoning(endpoint, monkeypatch):
    config, _, _ = endpoint
    config.remote["stream"] = True
    bodies = []

    def serve(request, *, timeout):
        body = json.loads(request.data)
        bodies.append(body)
        events = [
            {"delta": {"reasoning_content": "Explanation omitted."}},
            {"delta": {"content": "60F; "}},
            {"delta": {"content": "stable."}, "finish_reason": "stop"},
        ]
        data = (
            "".join(
                "data: " + json.dumps({"choices": [{"index": 0, **e}]}) + "\n\n"
                for e in events
            )
            + "data: [DONE]\n\n"
        )
        return io.BytesIO(data.encode())

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", serve)
    assert compress_patient_note("60F. Stable.", config=config) == "60F; stable."
    assert bodies[0]["stream"] is True
    assert "response_format" not in bodies[0]
