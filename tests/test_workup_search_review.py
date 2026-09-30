"""Synthetic integration checks of the real worker, reviewer and request budgets."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.llm.structured import (
    EndpointError,
    StructuredClient,
    StructuredConfig,
)
from matchminer_ai.patients import (
    WorkupSearchReviewConfig,
    review_patient_workup_with_note_search,
)
from matchminer_ai.patients import workup_search as workup
from matchminer_ai.patients import workup_search_review as review


def assessment(status="completed", *, more=False):
    unknown = status in {"unclear", "not_documented"}
    return {
        "status": "unknown" if unknown else "answered",
        "answer": {
            "status": status,
            "applicability": "uncertain",
            "bottom_line": "Fabricated assessment.",
        },
        "limitations": ["No sufficient documentation located."] if unknown else [],
        "needs_more_evidence": more,
    }


@pytest.fixture
def harness(monkeypatch):
    review._VOCABULARY.clear()
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=["http://fabricated.invalid/v1"])
    resolved = StructuredConfig(
        base_url="http://fabricated.invalid/v1",
        model="test-model",
        max_tokens=1024,
        context_window=32768,
        tokenizer_mode="bytes",
        attempts=1,
    )
    monkeypatch.setattr(
        workup, "resolve_structured_config", lambda *a, **k: (resolved, {})
    )
    calls = []

    def install(
        handler,
        *,
        code="search('CT ordered', context=0)",
        initial="planned",
        flag=False,
    ):
        def http(self, endpoint, body=None, **kwargs):
            assert endpoint == "/chat/completions"
            payload = json.loads(body["messages"][1]["content"])
            calls.append(payload)
            if "last_cell_result" in payload:
                value = assessment(initial)
                value.pop("needs_more_evidence")
                value.update(
                    action="python" if payload["search_only"] else "final",
                    code=[code] if payload["search_only"] else [],
                    memory="",
                )
                if "needs_review" in kwargs["output_schema"]["properties"]:
                    value["needs_review"] = flag
            else:
                value = handler(payload)
            return {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": json.dumps(value)}}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }

        monkeypatch.setattr(StructuredClient, "_http", http)
        return calls

    return config, install, calls


def run(config, notes, **kwargs):
    return review_patient_workup_with_note_search(
        notes,
        [{"name": "CT", "conditions": "If indicated"}],
        config=config,
        population_context="Fabricated population",
        **kwargs,
    )


def test_unseen_completion_corrects_plan_and_keeps_original_dates(harness):
    config, install, calls = harness

    def handler(payload):
        if "unsuccessful_terms" in payload:
            assert "provisional_assessment" not in payload
            return {"terms": ["CT"]}
        assert payload["provisional_assessment"]["answer"]["status"] == "planned"
        assert [e["quote"] for e in payload["source_excerpts"]] == [
            "CT ordered.",
            "CT completed.",
        ]
        return assessment()

    install(handler)
    notes = pd.DataFrame(
        [
            {"note_text": "CT ordered.", "note_date": "2026-01-01"},
            {"note_text": "CT completed.", "note_date": "2026-02-01"},
        ]
    )
    result = run(config, notes)
    row = result["assessments"][0]
    assert row["status"] == "completed" and row["review_status"] == "answered"
    assert [e["note_number"] for e in row["evidence"]] == [1, 2]
    assert row["evidence"][1]["note_date"].startswith("2026-02-01")
    audit = row["metadata"]["followup_review"]
    assert len(audit["rounds"]) == 1 and audit["rounds"][0]["changed"]
    assert audit["coverage"]["unreviewed_stored_positions"] == 0
    assert audit["full_record"] is None and audit["requests"] == 2
    assert (
        row["metadata"]["requests"] == result["metadata"]["requests"] == len(calls) == 4
    )
    assert row["metadata"]["prompt_tokens"] == 400


def test_zero_match_alternative_recovers_and_cache_is_patient_free(harness):
    config, install, calls = harness

    def handler(payload):
        if "unsuccessful_terms" in payload:
            if payload["unsuccessful_terms"] is None:
                return {"terms": ["CT"]}
            assert payload["unsuccessful_terms"] == ["CT"]
            return {"terms": ["ct", "computed tomography", "COMPUTED-TOMOGRAPHY"]}
        assert "computed tomography" in payload["source_excerpts"][0]["quote"]
        return assessment()

    install(handler, code="search('CT', context=0)", initial="not_documented")
    first = run(config, "PatientALPHA: computed tomography was performed.")
    second = run(config, "PatientBETA: computed tomography was performed.")
    for result in (first, second):
        row = result["assessments"][0]
        assert row["status"] == "completed"
        audit = row["metadata"]["followup_review"]["zero_match_research"]
        assert audit["initial_terms"] == ["CT"]
        assert audit["alternative_terms"] == ["computed tomography"]
        assert audit["match_count_after_research"] == 1
    plans = [c for c in calls if "unsuccessful_terms" in c]
    assert len(plans) == 2
    assert "PatientALPHA" not in json.dumps(plans) and "PatientBETA" not in json.dumps(
        plans
    )
    assert "PatientALPHA" not in json.dumps(review._VOCABULARY)
    assert "PatientALPHA" not in json.dumps(second)


def test_each_phase_receives_its_own_role_and_response_contract(harness, monkeypatch):
    config, install, _ = harness
    install(lambda p: {"terms": ["CT"]} if "unsuccessful_terms" in p else assessment())
    installed = StructuredClient._http
    captured = []

    def inspect(self, endpoint, body=None, **kwargs):
        captured.append((body["messages"], kwargs["output_schema"]))
        return installed(self, endpoint, body, **kwargs)

    monkeypatch.setattr(StructuredClient, "_http", inspect)
    result = run(config, "PatientALPHA: CT ordered. Later CT completed.")
    assert result["assessments"][0]["status"] == "completed"
    assert len(captured) == 4
    for messages, schema in captured:
        prompt = messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        assert "{assessment_contract}" not in prompt
        if "unsuccessful_terms" in payload:
            assert set(payload["workup_item"]) == {
                "recommendation",
                "guideline_population",
            }
            assert "task" not in payload["workup_item"]
            assert "PatientALPHA" not in json.dumps(messages)
            assert prompt.startswith("Cancer guidelines recommend tests")
            assert "Python will search the patient's record" in prompt
            assert set(schema["required"]) == {"terms"}
        elif "source_excerpts" in payload:
            assert prompt.startswith("Cancer guidelines recommend tests")
            assert "follow-up evidence reviewer" in prompt
            assert "How to carry out your role" not in prompt
            assert "For a Python action" not in prompt
            assert "Python reference" not in prompt
            assert set(schema["required"]) == {
                "status",
                "answer",
                "limitations",
                "needs_more_evidence",
            }
            example, _ = json.JSONDecoder().raw_decode(
                prompt.split("ordered chest CT:\n", 1)[1]
            )
            assert set(example) == set(schema["required"])
        else:
            assert "initial record-search and answering agent" in prompt
            assert schema["properties"]["answer"]["type"] == "object"


def test_zero_matches_never_turn_into_explicit_nonperformance(harness):
    config, install, calls = harness
    install(
        lambda p: {"terms": ["CT"] if p["unsuccessful_terms"] is None else []},
        code="search('CT')",
        initial="not_documented",
    )
    row = run(config, "Unrelated fabricated record.")["assessments"][0]
    assert row["status"] == "not_documented" and row["review_status"] == "unknown"
    assert row["evidence"] == []
    assert row["metadata"]["followup_review"]["full_record"] is None
    assert len(calls) == 4


def test_model_flag_expands_context_and_then_runs_serial_full_record(harness):
    config, install, calls = harness

    def handler(payload):
        if "unsuccessful_terms" in payload:
            return {"terms": ["CT"]}
        full = payload["review_reasons"][0].startswith("Serial")
        return assessment("completed" if full else "planned", more=not full)

    install(handler, flag=True)
    limits = WorkupSearchReviewConfig(review_context_chars=8, max_evidence_chars=400)
    text = "α " + "padding " * 20 + "CT ordered. " + "context " * 20
    row = run(config, text, review=limits)["assessments"][0]
    reviews = [p for p in calls if "source_excerpts" in p]
    assert len(reviews[1]["source_excerpts"][0]["quote"]) > len(
        reviews[0]["source_excerpts"][0]["quote"]
    )
    assert row["status"] == "completed"
    audit = row["metadata"]["followup_review"]
    assert len(audit["rounds"]) == 2 and audit["full_record"]["complete"]
    history, _ = workup._history(workup._notes(text))
    for evidence in row["evidence"]:
        assert history[evidence["start"] : evidence["end"]] == evidence["quote"]
        assert evidence["note_date"] is None


@pytest.mark.parametrize("failure", ["validation", "endpoint"])
def test_failed_review_preserves_validated_answer_and_reports_failure(harness, failure):
    config, install, _ = harness

    def handler(payload):
        if "unsuccessful_terms" in payload:
            return {"terms": ["CT"]}
        if failure == "endpoint":
            raise EndpointError("private synthetic content")
        return {"invalid": "private synthetic content"}

    install(handler, flag=True)
    row = run(
        config,
        "CT ordered.",
        review=WorkupSearchReviewConfig(max_full_record_fallback_items=0),
    )["assessments"][0]
    assert row["status"] == "planned"
    audit = row["metadata"]["followup_review"]
    assert audit["status"] == "incomplete" and audit["unresolved_or_coverage_limited"]
    assert audit["failures"][0]["kind"] == (
        "invalid_response" if failure == "validation" else "endpoint_failure"
    )
    assert "last validated" in row["limitations"][-1]
    assert "private synthetic" not in json.dumps(row)


def test_invalid_alternative_vocabulary_uses_bounded_fallback(harness):
    config, install, _ = harness

    def handler(payload):
        if "unsuccessful_terms" in payload:
            return (
                {"terms": ["CT"]}
                if payload["unsuccessful_terms"] is None
                else {"bad": True}
            )
        return assessment()

    install(handler, code="search('CT')", initial="not_documented")
    row = run(config, "Computed tomography completed.")["assessments"][0]
    assert row["status"] == "completed"
    assert row["metadata"]["followup_review"]["full_record"]["complete"]


def test_full_record_fallback_is_limited_to_items_in_input_order(harness):
    config, install, _ = harness

    def handler(payload):
        if "unsuccessful_terms" in payload:
            return {"terms": ["CT"]}
        return assessment(
            "completed"
            if payload["review_reasons"][0].startswith("Serial")
            else "planned",
            more=True,
        )

    install(handler, flag=True)
    result = review_patient_workup_with_note_search(
        "CT ordered.",
        [{"name": "A"}, {"name": "B"}, {"name": "C"}],
        config=config,
        review=WorkupSearchReviewConfig(
            max_review_passes=1, max_full_record_fallback_items=1
        ),
    )
    assert [a["status"] for a in result["assessments"]] == [
        "completed",
        "planned",
        "planned",
    ]
    assert [
        a["metadata"]["followup_review"]["full_record"] is not None
        for a in result["assessments"]
    ] == [True, False, False]
    assert result["metadata"]["incomplete_review_items"] == 3


@pytest.mark.parametrize("budget", ["calls", "chunks"])
def test_budget_exhaustion_keeps_prior_assessment_and_marks_incomplete(harness, budget):
    config, install, calls = harness

    def handler(payload):
        return (
            {"terms": ["CT"]}
            if "unsuccessful_terms" in payload
            else assessment("planned", more=True)
        )

    install(handler, flag=True)
    limits = WorkupSearchReviewConfig(
        max_review_passes=1,
        max_evidence_chars=80,
        max_full_record_chunks=1,
        max_calls_per_item=2 if budget == "calls" else 12,
    )
    row = run(config, "CT ordered. " + "x" * 500, review=limits)["assessments"][0]
    audit = row["metadata"]["followup_review"]
    assert row["status"] == "planned" and not audit["full_record"]["complete"]
    assert audit["failures"][-1]["kind"] == (
        "request_budget" if budget == "calls" else "full_record_chunk_budget"
    )
    assert audit["requests"] <= limits.max_calls_per_item
    assert len(calls) == 2 + audit["requests"]


def test_disabling_followup_retains_initial_request_path(harness):
    config, install, calls = harness

    def unexpected(_):
        pytest.fail("No review/vocabulary requests expected")

    install(unexpected)
    row = run(
        config, "CT ordered.", review=WorkupSearchReviewConfig(max_review_passes=0)
    )["assessments"][0]
    assert row["status"] == "planned" and len(calls) == 2
    assert "followup_review" not in row["metadata"]


def test_literal_search_is_escaped_unicode_aware_and_bounded():
    history = "α CT CTish computed_tomography computed-tomography [x] " * 600
    found = review.literal_matches(history, ["ct", "CT", "computed tomography", "[x]"])
    assert found["match_count"] == 2400 and found["positions_omitted"]
    assert len(found["positions"]) == 512
    assert all(
        history[a:b] in {"CT", "computed_tomography", "computed-tomography", "[x]"}
        for a, b in found["positions"]
    )


def test_limits_and_header_only_matches(harness):
    config, install, _ = harness
    install(
        lambda p: {"terms": ["Note"] if p["unsuccessful_terms"] is None else []},
        code="search('missing')",
        initial="not_documented",
    )
    row = run(config, "Fabricated unrelated record.")["assessments"][0]
    assert row["metadata"]["followup_review"]["coverage"]["match_count"] == 0
    for changes in (
        {"max_calls_per_item": 0},
        {"max_review_passes": -1},
        {"retry_zero_match_missing": 1},
    ):
        with pytest.raises(ValueError):
            replace(WorkupSearchReviewConfig(), **changes)


def test_large_fallback_splits_to_fit_without_losing_source_text(harness):
    config, install, calls = harness

    def handler(payload):
        if "unsuccessful_terms" in payload:
            return {"terms": ["CT"]}
        full = payload["review_reasons"][0].startswith("Serial")
        return assessment("completed" if full else "planned", more=not full)

    install(handler, flag=True)
    text = "CT ordered. " + "fabricated " * 5500
    row = run(
        config,
        text,
        review=WorkupSearchReviewConfig(max_review_passes=1, max_evidence_chars=40000),
    )["assessments"][0]
    audit = row["metadata"]["followup_review"]
    assert audit["full_record"]["complete"] and audit["full_record"]["chunks"] > 1
    fallback = [
        p for p in calls if p.get("review_reasons", [""])[0].startswith("Serial")
    ]
    assert all(len(p["source_excerpts"][0]["quote"]) < 40000 for p in fallback)
    _, spans = workup._history(workup._notes(text))
    cursor = spans[0]["start"]
    for payload in fallback:
        excerpt = payload["source_excerpts"][0]
        assert excerpt["start"] <= cursor < excerpt["end"]
        cursor = excerpt["end"]
    assert cursor == spans[0]["end"]
    assert audit["status"] == "complete"


def test_concurrent_identical_items_share_one_patient_free_plan(harness):
    config, install, calls = harness
    install(lambda p: {"terms": ["CT"]}, code="read(0, len(history))")
    result = review_patient_workup_with_note_search(
        "CT ordered.", [{"name": "CT"}] * 6, config=config
    )
    assert len(result["assessments"]) == 6
    assert len([p for p in calls if "unsuccessful_terms" in p]) == 1
    assert result["metadata"]["requests"] == 13
    initial_answers = [p for p in calls if p.get("last_cell_result") is not None]
    assert all(
        "start" in p["last_cell_result"]["source_excerpts"][0] for p in initial_answers
    )
