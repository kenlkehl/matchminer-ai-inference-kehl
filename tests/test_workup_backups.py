"""Fabricated notes and mocked HTTP; exercise actual retries and isolated REPLs."""

import copy
import json
from collections import Counter

import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.llm import structured
from matchminer_ai.patients import (
    NoteSearchLimits,
    WorkupSearchReviewConfig,
    review_patient_workup,
    review_patient_workup_with_note_search,
    workup,
    workup_search,
)
from matchminer_ai.patients import workup_search_review as followup


@pytest.fixture
def harness(monkeypatch):
    primary = load_default_preset()
    primary.remote.update(enabled=True, server_urls=["http://primary.invalid/v1"])
    primary.patient["remote"]["model_name"] = "primary"
    primary.patient.update(chunk_size=20, chunk_overlap=0)
    backup = copy.deepcopy(primary)
    backup.remote["server_urls"] = ["http://backup.invalid/v1"]
    backup.patient["remote"]["model_name"] = "backup"
    discoveries, calls = [], []

    def resolve(runtime, **kwargs):
        model = runtime["model_name"]
        discoveries.append(model)
        return structured.StructuredConfig(
            model=model,
            base_url=f"http://{model}.invalid/v1",
            tokenizer_mode="bytes",
            max_tokens=2048,
            context_window=32768,
            attempts=runtime["max_retries"],
        ), {}

    monkeypatch.setattr(workup, "resolve_structured_config", resolve)
    monkeypatch.setattr(workup_search, "resolve_structured_config", resolve)
    monkeypatch.setattr(structured, "cancel_sleep", lambda *_: None)
    followup._VOCABULARY.clear()

    def install(handler):
        def http(self, endpoint, body=None, **kwargs):
            payload = json.loads(body["messages"][1]["content"])
            calls.append((body["model"], payload))
            result = handler(body["model"], payload)
            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": result
                            if isinstance(result, str)
                            else json.dumps(result)
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }

        monkeypatch.setattr(structured.StructuredClient, "_http", http)

    return primary, backup, discoveries, calls, install


def action(payload, code="search('CT|ECG')", *, review=False):
    value = {
        "action": "python",
        "code": [code],
        "memory": "",
        "status": "unknown",
        "answer": {},
        "limitations": [],
    }
    if not payload["search_only"]:
        value.update(
            action="final",
            code=[],
            status="answered",
            answer={
                "status": "completed",
                "applicability": "uncertain",
                "bottom_line": "Fabricated documentation located.",
            },
        )
    if review:
        value["needs_review"] = False
    return value


NOTES = pd.DataFrame(
    [{"note_text": "CT completed. ECG completed.", "note_date": "2026-01-01"}]
)
NO_FOLLOWUP = WorkupSearchReviewConfig(max_review_passes=0)


def test_python_error_backup_retries_only_failed_item_and_preserves_dates(harness):
    primary, backup, discoveries, calls, install = harness

    def handler(model, payload):
        item = json.loads(payload["question"])["recommendation"]["name"]
        if model == "primary" and item != "CT":
            return action({"search_only": True}, code="undefined_function()")
        return action(payload)

    install(handler)
    result = review_patient_workup_with_note_search(
        NOTES,
        [{"name": "CT"}, {"name": "ECG"}, {"name": "Biopsy"}],
        config=primary,
        backup_config=backup,
        review=NO_FOLLOWUP,
        max_parallel_questions=2,
    )
    ct, ecg, biopsy = result["assessments"]
    assert ct["status"] == ecg["status"] == "completed"
    assert not ct["metadata"].get("backup_used")
    assert ecg["metadata"]["backup_used"] is True
    assert ecg["metadata"]["model"] == "backup"
    assert ecg["metadata"]["max_consecutive_cell_errors"] == 3
    assert ecg["metadata"]["cells"] == 4
    assert ecg["metadata"]["requests"] == 5
    assert ecg["metadata"]["max_calls"] == 32
    assert ecg["metadata"]["backup_events"][0]["reason"] == "consecutive_python_errors"
    assert all(
        e["note_date"] == workup._notes(NOTES)[0]["note_date"] for e in ecg["evidence"]
    )
    assert Counter(m for m, _ in calls) == {"primary": 8, "backup": 4}
    assert discoveries == ["primary", "backup"]
    assert result["metadata"]["backup_items"] == 2
    assert biopsy["metadata"]["backup_used"] and biopsy["status"] == "completed"


def test_successful_python_cell_resets_streak_and_backup_stays_lazy(harness):
    primary, backup, discoveries, calls, install = harness

    def handler(model, payload):
        # Two errors separated by successful cells must not add up to a switch.
        step = int(payload["memory"] or "0")
        value = (
            action(payload)
            if step == 4
            else action(
                {"search_only": True},
                code="undefined_function()" if step % 2 == 0 else "search('CT')",
            )
        )
        value["memory"] = str(step + 1)
        return value

    install(handler)
    result = review_patient_workup_with_note_search(
        NOTES,
        [{"name": "CT"}],
        config=primary,
        backup_config=backup,
        max_consecutive_failures=2,
        review=NO_FOLLOWUP,
        limits=NoteSearchLimits(max_cells=4, max_calls=6),
    )
    assert result["assessments"][0]["status"] == "completed"
    assert discoveries == ["primary"]
    assert {m for m, _ in calls} == {"primary"}


@pytest.mark.parametrize("backup_fails", [False, True])
def test_invalid_responses_switch_after_threshold_and_backup_is_bounded(
    harness, backup_fails
):
    primary, backup, discoveries, calls, install = harness
    install(
        lambda model, p: (
            "invalid JSON" if model == "primary" or backup_fails else action(p)
        )
    )
    result = review_patient_workup_with_note_search(
        NOTES,
        [{"name": "CT"}],
        config=primary,
        backup_config=backup,
        max_consecutive_failures=2,
        review=NO_FOLLOWUP,
        limits=NoteSearchLimits(max_cells=4, max_calls=6),
    )
    item = result["assessments"][0]
    assert item["status"] == ("error" if backup_fails else "completed")
    assert item["metadata"]["backup_used"]
    assert Counter(m for m, _ in calls) == {
        "primary": 2,
        "backup": 3 if backup_fails else 2,
    }
    assert discoveries == ["primary", "backup"]
    assert item["metadata"]["requests"] == len(calls)
    assert item["metadata"]["max_calls"] == 12


def test_no_backup_retains_existing_error_and_never_resolves_backup(harness):
    primary, _, discoveries, calls, install = harness
    install(lambda *_: "invalid JSON")
    result = review_patient_workup_with_note_search(
        NOTES,
        [{"name": "CT"}],
        config=primary,
        review=NO_FOLLOWUP,
    )
    assert result["assessments"][0]["status"] == "error"
    assert discoveries == ["primary"] and len(calls) == 3


def test_full_note_backup_preserves_validated_prior_packet(harness, monkeypatch):
    primary, backup, discoveries, calls, install = harness

    class Tokenizer:
        is_fast = True

        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer()
    )
    notes = pd.DataFrame(
        [
            {"note_text": "CT ordered.", "note_date": "2026-01-01"},
            {"note_text": "CT completed.", "note_date": "2026-02-01"},
        ]
    )

    def handler(model, payload):
        note = payload["raw_note_fragments"][0]
        if note["note_number"] == 2 and model == "primary":
            return "invalid JSON"
        if model == "backup":
            assert payload["prior_assessments"][0]["status"] == "planned"
            assert "evidence" not in payload["prior_assessments"][0]
        return {
            "assessments": [
                {
                    "name": "CT",
                    "status": "planned" if note["note_number"] == 1 else "completed",
                    "applicability": "uncertain",
                    "bottom_line": "Fabricated finding.",
                }
            ]
        }

    install(handler)
    result = review_patient_workup(
        notes,
        [{"name": "CT"}],
        config=primary,
        backup_config=backup,
        max_consecutive_failures=2,
    )
    assert Counter(m for m, _ in calls) == {"primary": 3, "backup": 1}
    assert result["metadata"]["backup_used"] and result["metadata"]["model"] == "backup"
    assert result["metadata"]["requests"] == 4
    assert discoveries == ["primary", "backup"]
    assert result["assessments"][0]["status"] == "completed"
    assert [e["quote"] for e in result["assessments"][0]["evidence"]] == [
        "CT ordered.", "CT completed."
    ]
    assert [e["note_date"] for e in result["assessments"][0]["evidence"]] == [
        n["note_date"] for n in workup._notes(notes)
    ]


def test_followup_switch_uses_backup_for_remaining_review(harness):
    primary, backup, discoveries, calls, install = harness
    notes = pd.DataFrame(
        [
            {"note_text": "CT ordered.", "note_date": "2026-01-01"},
            {"note_text": "CT completed.", "note_date": "2026-02-01"},
        ]
    )

    def handler(model, payload):
        if "last_cell_result" in payload:
            value = action(payload, code="search('CT ordered', context=0)", review=True)
            if value["action"] == "final":
                value["answer"]["status"] = "planned"
            return value
        if model == "primary":
            return "invalid JSON"
        if "unsuccessful_terms" in payload:
            return {"terms": ["CT"]}
        return {
            "status": "answered",
            "answer": {
                "status": "completed",
                "applicability": "uncertain",
                "bottom_line": "Fabricated completion documented.",
            },
            "limitations": [],
            "needs_more_evidence": False,
        }

    install(handler)
    result = review_patient_workup_with_note_search(
        notes,
        [{"name": "CT"}],
        config=primary,
        backup_config=backup,
        max_consecutive_failures=2,
    )
    item = result["assessments"][0]
    assert item["status"] == "completed"
    assert item["metadata"]["backup_events"][0]["stage"] == "followup_review"
    assert item["metadata"]["model"] == "backup"
    assert item["metadata"]["followup_review"]["status"] == "complete"
    assert any(
        e["note_date"] == workup._notes(notes)[1]["note_date"] for e in item["evidence"]
    )
    assert (
        Counter(m for m, _ in calls)["primary"] == 4
    )  # initial two + two failed terms
    assert discoveries == ["primary", "backup"]


def test_full_note_retry_repacks_for_backup_capacity_without_losing_text(
    harness, monkeypatch
):
    primary, backup, _, calls, install = harness
    primary.patient.update(chunk_size=500, chunk_overlap=0)

    class Tokenizer:
        is_fast = True

        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer()
    )
    original_fits = structured.StructuredClient.fits

    def fits(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        return original_fits(self, messages, **kwargs) and (
            self.config.model != "backup"
            or sum(len(n["text"]) for n in payload["raw_note_fragments"]) <= 80
        )

    monkeypatch.setattr(structured.StructuredClient, "fits", fits)
    text = "; ".join(f"Fabricated detail {i:02}" for i in range(16))

    def handler(model, payload):
        if model == "primary":
            return "invalid JSON"
        return {
            "assessments": [
                {
                    "name": "CT",
                    "status": "unclear",
                    "applicability": "uncertain",
                    "bottom_line": "Fabricated review.",
                }
            ]
        }

    install(handler)
    result = review_patient_workup(
        text,
        [{"name": "CT"}],
        config=primary,
        backup_config=backup,
        max_consecutive_failures=2,
    )
    reviewed = [p["raw_note_fragments"][0]["text"] for m, p in calls if m == "backup"]
    assert len(reviewed) > 1 and "".join(reviewed) == text
    assert result["metadata"]["validated_packets"] == len(reviewed)
    assert result["metadata"]["requests"] == len(calls)


def test_honest_unknown_does_not_trigger_backup(harness):
    primary, backup, discoveries, calls, install = harness

    def handler(model, payload):
        value = action(payload, code="search('not present')")
        if value["action"] == "final":
            value.update(status="unknown", limitations=["No documentation located."])
            value["answer"]["status"] = "not_documented"
        return value

    install(handler)
    result = review_patient_workup_with_note_search(
        NOTES,
        [{"name": "CT"}],
        config=primary,
        backup_config=backup,
        review=NO_FOLLOWUP,
    )
    assert result["assessments"][0]["status"] == "not_documented"
    assert not result["metadata"]["backup_used"]
    assert discoveries == ["primary"] and len(calls) == 2


def test_unavailable_backup_is_probed_once_for_parallel_failed_items(
    harness, monkeypatch
):
    primary, backup, discoveries, _, install = harness
    resolve = workup_search.resolve_structured_config

    def unavailable(runtime, **kwargs):
        result = resolve(runtime, **kwargs)
        if runtime["model_name"] == "backup":
            raise structured.EndpointError("PRIVATE PROVIDER STRING")
        return result

    monkeypatch.setattr(workup_search, "resolve_structured_config", unavailable)
    install(lambda *_: action({"search_only": True}, code="undefined_function()"))
    result = review_patient_workup_with_note_search(
        NOTES,
        [{"name": "CT"}, {"name": "ECG"}],
        config=primary,
        backup_config=backup,
        review=NO_FOLLOWUP,
    )
    assert discoveries == ["primary", "backup"]
    assert all(item["status"] == "error" for item in result["assessments"])
    assert all(
        item["metadata"]["backup_resolution_failed"] for item in result["assessments"]
    )
    assert "PRIVATE PROVIDER STRING" not in json.dumps(result)


def test_failed_followup_backup_keeps_last_validated_finding(harness):
    primary, backup, _, calls, install = harness

    def handler(model, payload):
        if "last_cell_result" not in payload:
            return "invalid JSON"
        value = action(payload, review=True)
        if value["action"] == "final":
            value["answer"]["status"] = "planned"
        return value

    install(handler)
    result = review_patient_workup_with_note_search(
        NOTES,
        [{"name": "CT"}],
        config=primary,
        backup_config=backup,
        max_consecutive_failures=2,
    )
    item = result["assessments"][0]
    assert item["status"] == "planned"
    assert item["metadata"]["model"] == "primary"  # last validated assessment
    assert item["metadata"]["backup_used"]
    assert item["metadata"]["followup_review"]["status"] == "incomplete"
    assert any("last validated" in s for s in item["limitations"])
    assert item["metadata"]["requests"] == len(calls)
    assert len(item["metadata"]["backup_events"]) == 1


@pytest.mark.parametrize("threshold", [0, 11, True, 1.5])
def test_invalid_threshold_rejected_before_endpoint(harness, threshold):
    primary, backup, discoveries, _, _ = harness
    with pytest.raises(ValueError, match="max_consecutive_failures"):
        review_patient_workup_with_note_search(
            NOTES,
            [{"name": "CT"}],
            config=primary,
            backup_config=backup,
            max_consecutive_failures=threshold,
        )
    assert not discoveries
