"""LLM grounding, backup behavior and failures in retrieved-chunk workup review."""

import json
from threading import Barrier, Event, Lock

import pytest
from test_colbert import encoder, sample  # noqa: F401

from matchminer_ai import load_default_preset
from matchminer_ai.cancellation import (
    CancellationToken,
    InferenceCancelled,
    cancellation_scope,
)
from matchminer_ai.llm.structured import StructuredConfig, EndpointError
from matchminer_ai.patients import review_patient_workup_with_colbert
from matchminer_ai.patients import review_patients_workup_with_colbert
from matchminer_ai.patients import workup_colbert as workup


@pytest.fixture
def harness(encoder, monkeypatch):  # noqa: F811
    calls, runtimes = [], []

    def resolve(runtime, **kwargs):
        runtimes.append(runtime)
        return StructuredConfig(
            model=runtime["model_name"],
            tokenizer_mode="bytes",
            attempts=runtime["max_retries"],
        ), {}

    monkeypatch.setattr(workup, "resolve_structured_config", resolve)
    failures = set()

    class Client:
        def __init__(self, config, cache):
            assert cache is None
            self.config = config

        def fits(self, messages):
            return True

        def complete(self, job, messages, schema, validate):
            payload = json.loads(messages[-1]["content"])
            calls.append((self.config.model, messages, payload, schema))
            self.requests += 1
            if self.config.model in failures:
                raise EndpointError("safe failure")
            result = {
                "assessments": [
                    dict(
                        name=payload["recommendations"][0]["name"],
                        status="not_documented",
                        applicability="uncertain",
                        bottom_line="Not located in the retrieved excerpts.",
                    )
                ]
            }
            validate(result)
            return result

    monkeypatch.setattr(workup, "StructuredClient", Client)
    config = load_default_preset()
    config.remote["enabled"] = True
    config.patient["remote"]["model_name"] = "primary"
    backup = load_default_preset()
    backup.remote["enabled"] = True
    backup.patient["remote"]["model_name"] = "backup"
    return sample()["a"], config, backup, calls, runtimes, failures, Client


def test_retrieves_once_and_prompts_only_selected_patient_fragments(harness):
    index, config, backup, calls, runtimes, failures, _ = harness
    items = [
        {
            "name": "scan",
            "conditions": "if indicated",
            "evidence": [{"source": "guideline"}],
        },
        {"name": "dental"},
    ]
    result = review_patient_workup_with_colbert(
        index, items, config=config, top_n=1, population_context="Fictional population"
    )
    assert len(calls) == 2
    assert result["metadata"]["method"] == "colbert"
    assert result["metadata"]["model"] == "primary"
    assert result["metadata"]["retrieval_model"] == "lightonai/GTE-ModernColBERT-v1"
    assert result["metadata"]["requests"] == 2
    assert (
        result["metadata"]["top_n"] == 1
        and not result["metadata"]["patient_summary_used"]
    )
    assert "subset" in result["notice"]
    for row, (_, messages, payload, schema) in zip(result["assessments"], calls):
        assert len(payload["retrieved_note_fragments"]) == 1
        assert "Other patient" not in json.dumps(messages)
        assert "No patient" not in json.dumps(messages)
        assert (
            row["evidence"][0]["quote"]
            == payload["retrieved_note_fragments"][0]["text"]
        )
        assert (
            row["evidence"][0]["note_date"]
            == payload["retrieved_note_fragments"][0]["note_date"]
        )
        assert (
            "evidence" not in schema["properties"]["assessments"]["items"]["properties"]
        )
        assert row["limitations"]
    assert result["assessments"][0]["recommendation"] == items[0]


def test_explicit_backup_switch_retains_retrieval_and_unknown_is_not_failure(harness):
    index, config, backup, calls, runtimes, failures, _ = harness
    failures.add("primary")
    result = review_patient_workup_with_colbert(
        index,
        [{"name": "scan"}, {"name": "dental"}],
        config=config,
        backup_config=backup,
        max_consecutive_failures=4,
        max_concurrent_requests=1,
        top_n=2,
    )
    assert [c[0] for c in calls] == ["primary", "backup", "backup"]
    assert calls[0][2] == calls[1][2]
    assert [r["max_retries"] for r in runtimes] == [4, 3]
    assert result["metadata"]["backup_used"]
    assert all(r["status"] == "not_documented" for r in result["assessments"])
    assert result["metadata"]["requests"] == 3
    assert result["metadata"]["backup_events"][0]["failure_threshold"] == 4


def test_no_implicit_backup_and_failed_items_are_not_missing_evidence(harness):
    index, config, _, calls, runtimes, failures, _ = harness
    failures.add("primary")
    result = review_patient_workup_with_colbert(
        index, [{"name": "scan"}], config=config
    )
    assert len(runtimes) == 1 and len(calls) == 1
    assert result["assessments"][0]["status"] == "error"
    assert not result["metadata"]["backup_used"]


def test_oversized_prompt_fails_without_silently_dropping_chunks(harness):
    index, config, _, calls, _, _, Client = harness
    Client.fits = lambda *args: False
    result = review_patient_workup_with_colbert(
        index, [{"name": "scan"}], config=config
    )
    assert not calls
    assert result["assessments"][0]["status"] == "error"
    assert not result["assessments"][0]["evidence"]
    assert "budget" in result["assessments"][0]["limitations"][0]


def test_backup_resolution_failure_visible_and_cancellation_propagates(
    harness, monkeypatch
):
    index, config, backup, calls, _, failures, _ = harness
    failures.add("primary")
    original = workup.resolve_structured_config

    def resolve(runtime, **kwargs):
        if runtime["model_name"] == "backup":
            raise EndpointError("unavailable")
        return original(runtime, **kwargs)

    monkeypatch.setattr(workup, "resolve_structured_config", resolve)
    result = review_patient_workup_with_colbert(
        index, [{"name": "scan"}], config=config, backup_config=backup
    )
    assert result["metadata"]["backup_resolution_failed"]
    assert result["assessments"][0]["status"] == "error"
    token = CancellationToken()
    token.cancel()
    with pytest.raises(InferenceCancelled), cancellation_scope(token):
        review_patient_workup_with_colbert(index, [{"name": "scan"}], config=config)


def test_real_structured_client_retries_invalid_quotes_before_backup(
    harness, monkeypatch
):
    from matchminer_ai.llm import structured

    index, config, backup, _, _, _, _ = harness
    monkeypatch.setattr(workup, "StructuredClient", structured.StructuredClient)
    monkeypatch.setattr(structured, "cancel_sleep", lambda seconds: None)
    calls = []

    def http(self, endpoint, body, **kwargs):
        calls.append((self.config.model, body))
        row = {
            "name": "scan",
            "status": "completed",
            "applicability": "uncertain",
            "bottom_line": "The retrieved text documents the scan.",
        }
        if self.config.model == "primary":
            row["evidence"] = ["Invented quotation must be rejected."]
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"assessments": [row]})},
                }
            ]
        }

    monkeypatch.setattr(structured.StructuredClient, "_http", http)
    result = review_patient_workup_with_colbert(
        index,
        [{"name": "scan"}],
        config=config,
        backup_config=backup,
        max_consecutive_failures=2,
        top_n=1,
    )
    assert [model for model, _ in calls] == ["primary", "primary", "backup"]
    assert result["metadata"]["requests"] == 3
    assert result["metadata"]["backup_events"][0]["failure_threshold"] == 2
    assert result["assessments"][0]["status"] == "completed"
    assert "Invented quotation" not in json.dumps(result)
    assert (
        len(calls[-1][1]["messages"]) == 2
    )  # backup does not inherit primary retry prompts


def test_batch_overlaps_patients_and_questions_with_one_global_bound(
    harness, monkeypatch
):
    _, config, _, calls, runtimes, _, Client = harness
    config.remote["max_concurrent_requests"] = 1
    original = Client.complete
    barrier, released, lock = Barrier(3), Event(), Lock()
    active = peak = started = 0
    finished = []

    def complete(self, job, messages, schema, validate):
        nonlocal active, peak, started
        payload = json.loads(messages[-1]["content"])
        text = json.dumps(payload["retrieved_note_fragments"])
        patient = "b" if "Other patient" in text else "a"
        question = payload["recommendations"][0]["name"]
        with lock:
            active += 1
            peak = max(peak, active)
            started += 1
            wave = started
        try:
            if wave <= 3:
                barrier.wait(timeout=5)
            if (patient, question) == ("a", "scan"):
                assert released.wait(5)
            value = original(self, job, messages, schema, validate)
            with lock:
                finished.append((patient, question))
            if (patient, question) == ("b", "scan"):
                released.set()
            return value
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(Client, "complete", complete)
    progress = []
    result = review_patients_workup_with_colbert(
        sample(),
        [{"name": "scan"}, {"name": "dental"}],
        config=config,
        max_concurrent_requests=3,
        top_n=1,
        progress_callback=progress.append,
    )
    assert peak == 3 and len(calls) == 4
    assert finished.index(("b", "scan")) < finished.index(("a", "scan"))
    assert list(result) == ["a", "b"]
    assert runtimes[0]["max_concurrent_requests"] == 3
    for pid, review in result.items():
        assert [r["recommendation_index"] for r in review["assessments"]] == [0, 1]
        for row in review["assessments"]:
            assert row["status"] == "not_documented"
            assert ("Other patient" in json.dumps(row["evidence"])) == (pid == "b")
    assert "4/4 patient–question checks complete" in progress[-1]
    assert "2/2 patients complete" in progress[-1]
    assert "0 in flight" in progress[-1]


def test_parallel_backup_resolves_once_and_keeps_prompts_isolated(harness, monkeypatch):
    index, config, backup, calls, runtimes, failures, Client = harness
    failures.add("primary")
    original, barrier = Client.complete, Barrier(2)

    def complete(self, *args):
        if self.config.model == "primary":
            barrier.wait(timeout=5)
        return original(self, *args)

    monkeypatch.setattr(Client, "complete", complete)
    result = review_patient_workup_with_colbert(
        index,
        [{"name": name} for name in ("one", "two", "three", "four")],
        config=config,
        backup_config=backup,
        max_concurrent_requests=2,
    )
    assert [r["model_name"] for r in runtimes] == ["primary", "backup"]
    assert all(r["max_concurrent_requests"] == 2 for r in runtimes)
    assert [model for model, *_ in calls].count("primary") == 2
    assert result["metadata"]["requests"] == 6
    assert len(result["metadata"]["backup_events"]) == 1
    assert all(r["metadata"]["backup_used"] for r in result["assessments"])
    for _, messages, payload, _ in calls:
        assert len(messages) == 2 and len(payload["recommendations"]) == 1


def test_batch_cancellation_reaches_workers_and_stops_dispatch(harness, monkeypatch):
    from matchminer_ai.cancellation import check_cancelled

    _, config, _, calls, _, _, Client = harness
    token, barrier, started = CancellationToken(), Barrier(2), []

    def complete(*args):
        started.append(1)
        barrier.wait(timeout=5)
        token.cancel()
        check_cancelled()
        pytest.fail("Worker lost its cancellation scope")

    monkeypatch.setattr(Client, "complete", complete)
    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        review_patients_workup_with_colbert(
            sample(),
            [{"name": "scan"}, {"name": "dental"}],
            config=config,
            max_concurrent_requests=2,
        )
    assert len(started) == 2 and not calls


def test_retrieval_failure_keeps_patient_in_matrix_without_exposing_text(
    harness, monkeypatch
):
    _, config, _, _, _, _, _ = harness
    original = workup.retrieve_patient_chunks_colbert

    def retrieve(index, *args, **kw):
        if index.patient_id == "b":
            raise RuntimeError("Never expose provider or patient text")
        return original(index, *args, **kw)

    monkeypatch.setattr(workup, "retrieve_patient_chunks_colbert", retrieve)
    result = review_patients_workup_with_colbert(
        sample(),
        [{"name": "scan"}, {"name": "dental"}],
        config=config,
    )
    assert all(r["status"] == "error" for r in result["b"]["assessments"])
    assert all(r["status"] != "error" for r in result["a"]["assessments"])
    assert "Never expose" not in json.dumps(result)


@pytest.mark.parametrize("concurrency", [0, 33, True, 1.5])
def test_invalid_concurrency_rejected_before_any_request(harness, concurrency):
    index, config, _, calls, runtimes, _, _ = harness
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        review_patient_workup_with_colbert(
            index,
            [{"name": "scan"}],
            config=config,
            max_concurrent_requests=concurrency,
        )
    assert not calls and not runtimes
