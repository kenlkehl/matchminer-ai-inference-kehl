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
from matchminer_ai.trials._guideline_specificity import (
    validate_decision_field_batch,
    validate_decision_fields,
)


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


def test_catalog_retry_retains_full_findings_and_recent_repeated_errors(client, monkeypatch):
    from matchminer_ai.trials._guideline_generation import Client
    from matchminer_ai.trials._guideline_canonical import CATALOG

    catalog_client = Client(client.config, client.cache_dir)
    errors = ["First field error", "Old coverage error", "Another field error", "Later error", "Newest issue", "First field error"]
    rendered = catalog_client.retry_feedback_history(CATALOG, errors)
    assert "First field error" in rendered
    assert "Old coverage error" not in rendered
    assert "Newest issue" in rendered
    long_error = "Complete synthetic findings. " * 200 + "Final omitted population restriction."
    assert long_error in catalog_client.retry_feedback(CATALOG, long_error)
    assert "Final omitted population restriction." in catalog_client.retry_messages(
        MESSAGES, CATALOG, [long_error], json.dumps(BAD),
    )[-1]["content"]


def test_catalog_validation_lists_all_bad_fields_without_mutating_draft(tmp_path):
    from matchminer_ai.trials._guideline_canonical import validate_catalog
    from matchminer_ai.trials._guideline_sources import load_guideline
    from test_guideline_extraction import catalog, make_library

    guideline = load_guideline(make_library(tmp_path), "fictional")
    value = catalog()
    value["states"].append(copy.deepcopy(value["states"][0]))
    value["states"][0]["name"] = "Synthetic first population"
    value["states"][1]["name"] = "Synthetic second population"
    for item in value["states"]:
        item["space"]["cancer_burden_allowed"] = "Extent A OR extent B AND extent C"
    original = copy.deepcopy(value)
    with pytest.raises(ValueError) as error:
        validate_catalog(value, guideline.pages, "Fictional disease")
    assert "Synthetic first population" in str(error.value)
    assert "Synthetic second population" in str(error.value)
    assert value == original


@pytest.mark.parametrize("field,text", [
    ("cancer_burden_allowed", "Extent A AND size less than or equal to 2 cm AND negative margins"),
    ("prior_treatment_required", "Prior initial therapy AND relapse at or after 2 years after therapy"),
])
def test_resume_revalidates_numeric_comparator_without_regenerating(client, monkeypatch, field, text):
    value = {"name": "Synthetic extent", "space": dict.fromkeys(FIELDS, "NA")}
    value["space"][field] = text
    client.config = replace(client.config, attempts=1)
    monkeypatch.setattr(client, "_http", lambda *args, **kwargs: response(value))

    def old_validator(draft):
        raise ValueError("Old validator mistook the numeric comparator for mixed logic")

    with pytest.raises(EndpointError):
        client.complete("numeric-comparator", MESSAGES, SCHEMA, old_validator)
    raw = next(client.cache_dir.glob("*/attempt-1.json"))
    original = raw.read_bytes()
    resumed = StructuredClient(client.config, client.cache_dir)
    monkeypatch.setattr(resumed, "_http", lambda *a, **k: pytest.fail("No regeneration"))
    assert resumed.complete(
        "numeric-comparator", MESSAGES, SCHEMA,
        lambda draft: validate_decision_fields(draft, "Fictional guideline"),
    ) == value
    assert raw.read_bytes() == original
    assert not raw.with_name("attempt-2.json").exists()


def test_catalog_resume_retains_all_coverage_findings_after_a_grouping_error(
    client, monkeypatch,
):
    from matchminer_ai.trials._guideline_canonical import CATALOG
    from matchminer_ai.trials._guideline_completeness import require_coverage
    from matchminer_ai.trials._guideline_generation import Client
    from test_guideline_extraction import catalog

    findings = [
        {
            "input_name": f"Synthetic distinct population {index}",
            "status": "missing",
            "matched_population_names": [],
            "reason": (
                "Preserve both bounds on prior therapy in the original setting. " * 4
                + f"Complete restriction for population {index}."
            ),
        }
        for index in range(7)
    ]
    drafts = [catalog() for _ in range(3)]
    drafts[1]["states"][0]["space"]["cancer_burden_allowed"] = "A OR B AND C"
    drafts[2]["states"][0]["space"]["cancer_burden_allowed"] = "(A OR B) AND C"

    def validate_catalog_draft(value):
        validate_decision_field_batch(value["states"], "Fictional guideline")
        if value == drafts[0]:
            require_coverage({"reviews": findings})

    calls = []

    def generate(endpoint, body, **kwargs):
        calls.append(copy.deepcopy(body))
        return response(drafts[len(calls) - 1])

    initial = Client(client.config, client.cache_dir)
    monkeypatch.setattr(initial, "_http", generate)
    with pytest.raises(EndpointError):
        initial.complete("catalog-sequence", MESSAGES, CATALOG, validate_catalog_draft)
    raw_paths = list(client.cache_dir.glob("*/attempt-*.json"))
    original_raw = {path: path.read_bytes() for path in raw_paths}
    resumed = Client(replace(client.config, attempts=1), client.cache_dir)
    monkeypatch.setattr(resumed, "_http", generate)
    assert resumed.complete(
        "catalog-sequence", MESSAGES, CATALOG, validate_catalog_draft,
    ) == drafts[2]
    feedback = calls[2]["messages"][-1]["content"]
    for finding in findings:
        assert finding["input_name"] in feedback
        assert finding["reason"] in feedback
    assert "mixes AND and OR" in feedback
    assert calls[2]["messages"][:-2] == MESSAGES
    assert calls[2]["max_tokens"] == 100000
    assert all(path.read_bytes() == raw for path, raw in original_raw.items())


@pytest.mark.parametrize("text", [
    "Extent A AND one or more tumors >=1 cm AND extent B",
    "Extent A AND two or fewer lesions",
    "Extent A AND 3 or more sites",
    "Extent A AND (one or more nodes OR extent B)",
])
def test_count_predicates_do_not_require_population_grouping(text):
    value = {"name": "Synthetic count", "space": dict.fromkeys(FIELDS, "NA")}
    value["space"]["cancer_burden_allowed"] = text
    original = copy.deepcopy(value)
    validate_decision_fields(value, "Fictional guideline")
    assert value == original


@pytest.mark.parametrize("text", [
    "Extent A AND one or more tumors OR extent B",
    "Extent A AND (two or fewer lesions OR extent B AND extent C)",
    "Grade one OR more advanced disease AND extent A",
    "Grade 2 OR more aggressive disease AND extent A",
])
def test_count_exceptions_preserve_real_population_alternatives(text):
    value = {"name": "Synthetic count", "space": dict.fromkeys(FIELDS, "NA")}
    value["space"]["cancer_burden_allowed"] = text
    original = copy.deepcopy(value)
    with pytest.raises(ValueError, match="mixes AND and OR"):
        validate_decision_fields(value, "Fictional guideline")
    assert value == original
