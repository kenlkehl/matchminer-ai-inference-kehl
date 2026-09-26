import copy
import json
from dataclasses import replace

import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.llm import structured
from matchminer_ai.patients import review_patient_workup
from matchminer_ai.patients import workup


class Tokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def finding(name="Imaging", status="not_documented", evidence=None):
    return dict(
        name=name,
        status=status,
        applicability="uncertain",
        bottom_line="Review documentation and indication.",
        evidence=evidence or [],
    )


@pytest.fixture
def harness(monkeypatch):
    config = load_default_preset()
    config.remote["enabled"] = True
    config.patient["chunk_size"] = 15
    config.patient["chunk_overlap"] = 2
    calls = []
    responder = [
        lambda payload: [finding(r["name"]) for r in payload["recommendations"]]
    ]
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer()
    )
    monkeypatch.setattr(
        workup,
        "resolve_structured_config",
        lambda *a, **k: (
            structured.StructuredConfig(model="test", tokenizer_mode="bytes"),
            {},
        ),
    )

    class Client:
        def __init__(self, config, cache_dir):
            assert cache_dir is None

        def fits(self, messages):
            return True

        def complete(self, job, messages, schema, validator):
            payload = json.loads(messages[-1]["content"])
            calls.append(payload)
            value = {"assessments": responder[0](payload)}
            validator(value)
            return value

    monkeypatch.setattr(workup, "StructuredClient", Client)
    return config, calls, responder, Client


def test_serial_review_retains_evidence_dates_and_conditions(harness):
    config, calls, responder, _ = harness
    responder[0] = lambda p: [
        finding(
            status="planned" if len(calls) == 1 else "completed",
            evidence=[
                dict(
                    note_number=p["raw_note_fragments"][0]["note_number"],
                    quote=p["raw_note_fragments"][0]["text"],
                )
            ],
        )
    ]
    notes = pd.DataFrame(
        [
            dict(
                patient_id="synthetic",
                note_text="Scan performed.",
                note_date="2026-02-01",
            ),
            dict(
                patient_id="synthetic",
                note_text="Scan ordered.",
                note_date="2026-01-01",
            ),
        ]
    )
    recommendations = [
        dict(name="Imaging", conditions="If indicated", evidence=[{"quote": "source"}])
    ]
    original = copy.deepcopy(recommendations)
    result = review_patient_workup(
        notes,
        recommendations,
        config=config,
        population_context="Fictional guideline population",
    )
    assert len(calls) == 2
    assert calls[1]["prior_assessments"][0]["status"] == "planned"
    row = result["assessments"][0]
    assert row["status"] == "completed"
    assert [e["quote"] for e in row["evidence"]] == ["Scan ordered.", "Scan performed."]
    assert row["evidence"][0]["note_date"].startswith("2026-01-01")
    assert row["recommendation"] == original[0]
    assert recommendations == original
    assert result["metadata"]["patient_summary_used"] is False
    assert calls[0]["guideline_population"] == "Fictional guideline population"
    assert result["metadata"]["reasoning_effort"] == "xhigh"
    assert len(result["metadata"]["prompt_sha256"]) == 64


def test_every_item_every_chunk_and_unknown_dates(harness):
    config, calls, _, _ = harness
    result = review_patient_workup(
        "A" * 40,
        [{"name": "First"}, {"name": "Second"}],
        config=config,
        recommendation_batch_size=1,
    )
    assert len(calls) == 6
    assert [r["name"] for r in result["assessments"]] == ["First", "Second"]
    assert all(n["note_date"] is None for p in calls for n in p["raw_note_fragments"])
    assert result["metadata"]["undated_notes"] == 1


@pytest.mark.parametrize(
    "row",
    [
        finding(status="completed"),
        finding(evidence=[dict(note_number=1, quote="Invented result")]),
        finding(evidence=[dict(note_number=2, quote="Scan ordered.")]),
        {**finding(), "name": "Wrong recommendation"},
        {**finding(), "applicability": "not_applicable"},
        {**finding(), "status": "eligible"},
    ],
)
def test_rejects_ungrounded_or_misaligned_results(harness, row):
    config, _, responder, _ = harness
    responder[0] = lambda _: [row]
    with pytest.raises(ValueError):
        review_patient_workup("Scan ordered.", [{"name": "Imaging"}], config=config)


def test_future_note_cannot_be_cited_before_review(harness):
    config, _, responder, _ = harness
    responder[0] = lambda _: [
        finding(
            status="completed", evidence=[dict(note_number=2, quote="Scan performed.")]
        )
    ]
    with pytest.raises(ValueError, match="verbatim"):
        review_patient_workup(
            pd.DataFrame(
                [
                    dict(note_text="Scan ordered.", note_date="2026-01-01"),
                    dict(note_text="Scan performed.", note_date="2026-02-01"),
                ]
            ),
            [{"name": "Imaging"}],
            config=config,
        )


def test_later_silence_cannot_erase_documented_findings(harness):
    config, calls, responder, _ = harness
    responder[0] = lambda _: [
        finding(status="planned", evidence=[dict(note_number=1, quote="Scan ordered.")])
        if len(calls) == 1
        else finding()
    ]
    with pytest.raises(ValueError, match="silence"):
        review_patient_workup(
            pd.DataFrame(
                [
                    dict(note_text="Scan ordered.", note_date="2026-01-01"),
                    dict(note_text="Unrelated note.", note_date="2026-02-01"),
                ]
            ),
            [{"name": "Imaging"}],
            config=config,
        )


def test_context_overflow_splits_without_dropping_text(harness):
    config, calls, _, client = harness
    client.fits = (
        lambda _, messages: sum(
            len(n["text"])
            for n in json.loads(messages[-1]["content"])["raw_note_fragments"]
        )
        <= 18
    )
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    review_patient_workup(
        text, [{"name": "Imaging"}], config=config, chunk_size=50, chunk_overlap=2
    )
    seen = "".join(n["text"] for p in calls for n in p["raw_note_fragments"])
    assert all(character in seen for character in text)
    assert len(calls) > 1


def test_rejects_multiple_patients_and_bad_dates(harness):
    config, calls, _, _ = harness
    for frame in [
        pd.DataFrame({"patient_id": ["a", "b"], "note_text": ["a", "b"]}),
        pd.DataFrame({"note_text": ["a"], "note_date": ["unknown date"]}),
    ]:
        with pytest.raises(ValueError):
            review_patient_workup(frame, [{"name": "Imaging"}], config=config)
    assert calls == []


def test_in_memory_client_retries_truncation_and_writes_nothing(monkeypatch):
    client = structured.StructuredClient(
        structured.StructuredConfig(
            model="test",
            tokenizer_mode="bytes",
            attempts=2,
        ),
        None,
    )
    requests = []

    def http(endpoint, body, **kwargs):
        requests.append(body)
        return {
            "choices": [
                {
                    "finish_reason": "length" if len(requests) == 1 else "stop",
                    "message": {"content": '{"ok": true}'},
                }
            ]
        }

    monkeypatch.setattr(client, "_http", http)
    monkeypatch.setattr(structured.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        structured, "atomic_json", lambda *a, **k: pytest.fail("Patient disk write")
    )
    result = client.complete(
        "test", [{"role": "user", "content": "Synthetic"}], {}, lambda _: None
    )
    assert result == {"ok": True}
    assert len(requests) == 2
    assert requests[0]["max_tokens"] == client.config.max_tokens
    assert requests[0]["chat_template_kwargs"]["enable_thinking"] is True


def test_in_memory_failure_does_not_echo_patient_content(monkeypatch):
    client = structured.StructuredClient(
        replace(structured.StructuredConfig(), tokenizer_mode="bytes", attempts=1), None
    )
    monkeypatch.setattr(
        client,
        "_http",
        lambda *a, **k: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "sensitive provider response"},
                }
            ]
        },
    )
    with pytest.raises(structured.EndpointError, match="failed validation") as exc:
        client.complete(
            "test", [{"role": "user", "content": "Synthetic"}], {}, lambda _: None
        )
    assert "sensitive" not in str(exc.value)
