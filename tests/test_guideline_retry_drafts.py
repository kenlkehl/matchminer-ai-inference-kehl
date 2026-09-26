"""Retry complete synthetic drafts without weakening source/field validation."""

import copy
import json
from dataclasses import replace

import pytest

from matchminer_ai.llm.structured import (
    EndpointError,
    StructuredClient,
    StructuredConfig,
)
from matchminer_ai.trials._guideline_schema import FIELDS
from matchminer_ai.trials._guideline_specificity import validate_decision_field_batch


def response(value, *, finish="stop", raw=None):
    return {"choices": [{"finish_reason": finish, "message": {
        "content": json.dumps(value) if raw is None else raw,
        "reasoning_content": "Do not replay this private reasoning trace.",
    }}]}


def validate(value):
    if value.get("logic") != "(A OR B) AND C":
        raise ValueError("Group complete concepts without dropping any condition")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda _: None)
    return StructuredClient(
        StructuredConfig(model="synthetic", attempts=2, tokenizer_mode="bytes"),
        tmp_path,
    )


MESSAGES = [{"role": "user", "content": "Synthetic source: C applies to A or B."}]
SCHEMA = {"type": "object"}
BAD = {"logic": "A OR B AND C", "preserved": "Synthetic option"}
GOOD = {**BAD, "logic": "(A OR B) AND C"}


def test_retry_revises_complete_draft_and_preserves_source_and_budget(client, monkeypatch):
    calls = []

    def generate(endpoint, body, **kwargs):
        calls.append(copy.deepcopy(body))
        return response(BAD if len(calls) == 1 else GOOD)

    monkeypatch.setattr(client, "_http", generate)
    assert client.complete("job", MESSAGES, SCHEMA, validate) == GOOD
    retry = calls[1]
    assert retry["messages"][:-2] == MESSAGES
    assert retry["messages"][-2]["role"] == "assistant"
    assert json.loads(retry["messages"][-2]["content"]) == BAD
    assert "private reasoning trace" not in json.dumps(retry["messages"])
    assert "COMPLETE revised JSON" in retry["messages"][-1]["content"]
    assert retry["max_tokens"] == calls[0]["max_tokens"] == 100000
    assert calls[0]["messages"] == MESSAGES


def test_resume_reuses_latest_rejected_draft_and_keeps_original_attempt(client, monkeypatch):
    client.config = replace(client.config, attempts=1)
    monkeypatch.setattr(client, "_http", lambda *a, **k: response(BAD))
    with pytest.raises(EndpointError):
        client.complete("job", MESSAGES, SCHEMA, validate)
    first = next(client.cache_dir.glob("*/attempt-1.json"))
    saved = first.read_bytes()
    resumed = StructuredClient(client.config, client.cache_dir)
    calls = []

    def generate(endpoint, body, **kwargs):
        calls.append(body)
        assert json.loads(body["messages"][-2]["content"]) == BAD
        return response(GOOD)

    monkeypatch.setattr(resumed, "_http", generate)
    assert resumed.complete("job", MESSAGES, SCHEMA, validate) == GOOD
    assert len(calls) == 1
    assert first.read_bytes() == saved
    assert first.with_name("attempt-2.json").exists()
    monkeypatch.setattr(resumed, "_http", lambda *a, **k: pytest.fail("No regeneration"))
    assert resumed.complete("job", MESSAGES, SCHEMA, validate) == GOOD


def test_later_retry_carries_forward_the_latest_partial_correction(client, monkeypatch):
    client.config = replace(client.config, attempts=3)
    drafts = [BAD, {**GOOD, "second_problem": True}, GOOD]
    calls = []

    def validate_both(value):
        validate(value)
        if value.get("second_problem"):
            raise ValueError("Correct the second problem without regressing the first")

    def generate(endpoint, body, **kwargs):
        calls.append(body)
        return response(drafts[len(calls) - 1])

    monkeypatch.setattr(client, "_http", generate)
    assert client.complete("job", MESSAGES, SCHEMA, validate_both) == GOOD
    assert json.loads(calls[2]["messages"][-2]["content"]) == drafts[1]
    assert calls[2]["messages"][:-2] == MESSAGES


def test_oversized_draft_is_omitted_without_truncating_source_or_output(client, monkeypatch):
    client.config = replace(client.config, context_window=1800, max_tokens=128, safety_tokens=64)
    calls = []

    def generate(endpoint, body, **kwargs):
        calls.append(body)
        return response({**BAD, "large": "x" * 5000} if len(calls) == 1 else GOOD)

    monkeypatch.setattr(client, "_http", generate)
    assert client.complete("job", MESSAGES, SCHEMA, validate) == GOOD
    assert calls[1]["messages"][:-1] == MESSAGES
    assert all(m["role"] != "assistant" for m in calls[1]["messages"])
    assert calls[1]["max_tokens"] == 128


@pytest.mark.parametrize("finish, raw", [("length", None), ("stop", "{incomplete")])
def test_incomplete_or_malformed_response_is_never_used_as_draft(client, monkeypatch, finish, raw):
    calls = []

    def generate(endpoint, body, **kwargs):
        calls.append(body)
        return response(BAD, finish=finish, raw=raw) if len(calls) == 1 else response(GOOD)

    monkeypatch.setattr(client, "_http", generate)
    assert client.complete("job", MESSAGES, SCHEMA, validate) == GOOD
    assert all(m["role"] != "assistant" for m in calls[1]["messages"])


def test_batch_feedback_names_each_rejected_candidate_without_rewriting():
    states = [
        {"name": name, "space": {field: "NA" for field in FIELDS}}
        for name in ("Synthetic A", "Synthetic B")
    ]
    states[0]["space"]["cancer_burden_allowed"] = "A OR B AND C"
    states[1]["space"]["cancer_burden_allowed"] = "after first-line treatment"
    original = copy.deepcopy(states)
    with pytest.raises(ValueError) as error:
        validate_decision_field_batch(states, "Fictional disease")
    assert "Synthetic A" in str(error.value)
    assert "Synthetic B" in str(error.value)
    assert "mixes AND and OR" in str(error.value)
    assert "contains treatment-line criteria" in str(error.value)
    assert states == original
