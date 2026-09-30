"""Fabricated workup notes; use the real REPL and stub only endpoint generation."""

from dataclasses import replace
import json
from threading import Barrier

import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.llm.structured import EndpointError, StructuredConfig
from matchminer_ai.patients import (
    NoteSearchLimits,
    WorkupSearchReviewConfig,
    review_patient_workup_with_note_search,
)
from matchminer_ai.patients import note_search_qa as qa, workup_search as workup


@pytest.fixture
def harness(monkeypatch):
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=["http://fabricated.invalid/v1"])
    resolved = StructuredConfig(
        base_url="http://fabricated.invalid/v1",
        model="Qwen/Qwen3.8-27B-FP8",
        max_tokens=2048,
        context_window=32768,
        tokenizer_mode="bytes",
        request_params={"reasoning_effort": "xhigh"},
    )
    monkeypatch.setattr(
        workup, "resolve_structured_config", lambda *a, **k: (resolved, {})
    )
    return config, resolved


def test_dated_evidence_conditions_order_and_structured_answer(harness, monkeypatch):
    config, _ = harness
    calls = []
    source = pd.DataFrame(
        [
            {"note_text": "CT completed. No ECG result.", "note_date": "2026-02-01"},
            {"note_text": "CT ordered.", "note_date": "2026-01-01"},
        ]
    )
    history, _ = workup._history(workup._notes(source))
    recommendations = [
        {"name": "CT", "conditions": "If indicated", "category": "test"},
        {"name": "ECG"},
    ]

    def complete(self, job, messages, schema, validator):
        calls.append(messages)
        assert schema["properties"]["answer"]["type"] == "object"
        p = json.loads(messages[1]["content"])
        question = json.loads(p["question"])
        assert question["guideline_population"] == "Fabricated population"
        assert question["recommendation"] in [
            {"name": "CT", "conditions": "If indicated"},
            {"name": "ECG", "conditions": ""},
        ]
        value = dict(
            action="python",
            code=["search('CT|ECG')"],
            memory="",
            status="unknown",
            answer={},
            limitations=[],
        )
        if p["last_cell_result"] is not None:
            found = question["recommendation"]["name"] == "CT"
            value.update(
                action="final",
                code=[],
                status="answered" if found else "unknown",
                answer={
                    "status": "completed" if found else "not_documented",
                    "applicability": "uncertain",
                    "bottom_line": "Synthetic assessment",
                },
                limitations=[] if found else ["No ECG documentation located."],
            )
        validator(value)
        return value

    monkeypatch.setattr(qa._MeasuredClient, "complete", complete)
    progress = []
    result = review_patient_workup_with_note_search(
        source,
        recommendations,
        config=config,
        population_context="Fabricated population",
        review=WorkupSearchReviewConfig(max_review_passes=0),
        progress_callback=progress.append,
        limits=NoteSearchLimits(max_cells=2),
    )
    rows = result["assessments"]
    assert [r["name"] for r in rows] == ["CT", "ECG"]
    assert rows[0]["recommendation"] == recommendations[0]
    assert [e["note_number"] for e in rows[0]["evidence"]] == [1, 2]
    assert rows[0]["evidence"][1]["note_date"].startswith("2026-02-01")
    assert [e["quote"] for e in rows[0]["evidence"]] == [
        "CT ordered.",
        "CT completed. No ECG result.",
    ]
    for row in rows:
        assert row["metadata"]["evidence_selection"] == "automatic_reviewed_excerpts"
    assert result["metadata"]["evidence_selection"] == "automatic_reviewed_excerpts"
    assert (
        rows[1]["status"] == "not_documented" and rows[1]["review_status"] == "unknown"
    )
    assert rows[1]["limitations"] == ["No ECG documentation located."]
    assert result["metadata"]["scope"] == "searched_excerpts"
    assert result["metadata"]["preserve_thinking"] is False
    assert len([p for p in progress if "items finished" in p]) == 2
    assert all([m["role"] for m in call] == ["system", "user"] for call in calls)
    assert all(history not in call[1]["content"] for call in calls)


def test_item_failure_is_not_missing_documentation(harness, monkeypatch):
    config, _ = harness

    def fail(*args):
        raise EndpointError("private content must not escape")

    monkeypatch.setattr(qa._MeasuredClient, "complete", fail)
    result = review_patient_workup_with_note_search(
        "Fabricated text", [{"name": "CT"}], config=config
    )
    row = result["assessments"][0]
    assert row["status"] == row["review_status"] == "error"
    assert result["metadata"]["failed_items"] == 1
    assert "private content" not in json.dumps(result)


def test_default_workup_schedules_more_than_four_questions_together(
    harness, monkeypatch
):
    config, _ = harness
    simultaneous = Barrier(6)

    def complete(self, job, messages, schema, validator):
        payload = json.loads(messages[1]["content"])
        searching = payload["last_cell_result"] is None
        if searching:
            simultaneous.wait(timeout=10)
        value = {
            "action": "python" if searching else "final",
            "code": ["search('missing')"] if searching else [],
            "memory": "",
            "status": "unknown",
            "answer": {
                "status": "not_documented",
                "applicability": "uncertain",
                "bottom_line": "No matching documentation located.",
            },
            "limitations": ["Search may miss evidence."],
        }
        validator(value)
        return value

    monkeypatch.setattr(qa._MeasuredClient, "complete", complete)
    result = review_patient_workup_with_note_search(
        "Fabricated unrelated note.",
        [{"name": f"Workup {i}"} for i in range(6)],
        review=WorkupSearchReviewConfig(max_review_passes=0),
        config=config,
    )
    assert len(result["assessments"]) == 6
    assert result["metadata"]["max_parallel_questions"] == 6
    assert all(a["metadata"]["max_calls"] == 16 for a in result["assessments"])


def test_validator_rejects_headers_cross_note_quotes_and_unsupported_findings():
    history, spans = workup._history(
        workup._notes(
            pd.DataFrame(
                [
                    {"note_text": "Planned.", "note_date": None},
                    {"note_text": "Performed.", "note_date": "2026-01-01"},
                ]
            )
        )
    )
    contract = workup._answer_format(spans)
    base = dict(
        status="unknown",
        answer={
            "status": "not_documented",
            "applicability": "uncertain",
            "bottom_line": "No evidence.",
        },
        evidence=[],
    )
    contract.validate(base)
    for start, end in [(0, 5), (spans[0]["end"] - 2, spans[1]["start"] + 2)]:
        with pytest.raises(ValueError, match="inside one original note"):
            contract.validate(
                {
                    **base,
                    "evidence": [
                        {"start": start, "end": end, "quote": history[start:end]}
                    ],
                }
            )
    for changes in [
        {"status": "completed"},
        {"status": "not_done"},
        {"applicability": "not_applicable"},
    ]:
        with pytest.raises(ValueError, match="require patient-note evidence"):
            contract.validate({**base, "answer": {**base["answer"], **changes}})
    assert spans[1]["note_date"] is None  # unavailable dates sort last, never inferred


@pytest.mark.parametrize("provider", ["openai", "google_agent_platform"])
def test_config_preserves_endpoint_provider_profile_and_no_reasoning_replay(
    harness, monkeypatch, provider
):
    config, original = harness
    resolved = replace(
        original, provider=provider, google_project_id="fabricated-project"
    )
    monkeypatch.setattr(
        workup, "resolve_structured_config", lambda *a, **k: (resolved, {})
    )
    captured = []

    def batch(patients, *, llm, **kw):
        captured.append(qa._resolve_llm(llm))
        return {
            "patients": [
                {
                    "answers": [
                        {
                            "status": "error",
                            "answer": None,
                            "evidence": [],
                            "limitations": [],
                            "metadata": {"requests": 0},
                        }
                    ]
                }
            ],
            "metadata": {},
            "notice": "test",
        }

    monkeypatch.setattr(workup, "_run_question_batch", batch)
    review_patient_workup_with_note_search(
        "Fabricated text", [{"name": "CT"}], config=config
    )
    llm = captured[0]
    assert llm.base_url == resolved.base_url and llm.provider == provider
    assert llm.google_project_id == "fabricated-project"
    assert llm.request_params["reasoning_effort"] == "xhigh"
    if provider == "openai":
        assert llm.extra_body["chat_template_kwargs"]["preserve_thinking"] is False
        assert llm.thinking == "on"
    else:
        assert "chat_template_kwargs" not in llm.extra_body
