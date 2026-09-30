"""Original-note pandas navigation, optional context and unchanged isolation."""

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from matchminer_ai.patients import (
    NoteSearchLimits,
    answer_patient_question_batch,
    answer_patient_questions,
    review_patient_workup_with_note_search,
)
from matchminer_ai.patients import note_search_qa as qa
from matchminer_ai.patients._note_record import prepare_record
from matchminer_ai.patients._note_repl import NoteREPL
from test_note_search_qa import cell, final, llm as llm_fixture, mock_model
from test_workup_search_review import harness as workup_harness_fixture

llm = llm_fixture
harness = workup_harness_fixture


@pytest.fixture
def notes():
    return pd.DataFrame(
        [
            {
                "note_text": "é: CT completed; report available.",
                "note_date": "2026-02-01",
                "note_type": "Radiology",
            },
            {
                "note_text": "CT ordered; pending.",
                "note_date": "2026-01-01",
                "note_type": "Oncology",
            },
            {
                "note_text": "Undated clinical note: biopsy performed.",
                "note_date": None,
                "note_type": None,
            },
        ],
        index=[9, 9, 77],
    )


def test_pandas_date_type_navigation_auto_retains_immutable_originals(notes):
    record = prepare_record(notes=notes)
    original = notes.copy(deep=True)
    with NoteREPL(record.history, NoteSearchLimits(), notes=record.spans) as worker:
        first = worker.execute(
            "selected = notes.loc[notes.note_text.str.contains('CT', case=False, na=False) & notes.note_type.str.contains('radiolog', case=False, na=False) & notes.note_date.between(pd.Timestamp('2026-02-01', tz='UTC'), pd.Timestamp('2026-02-28', tz='UTC'))]; selected[['note_date', 'note_type', 'note_text']]"
        )
        assert first["error"] is None
        shown = ast.literal_eval(first["output"])["notes"]
        assert len(shown) == 1 and shown[0]["note_number"] == 2
        assert shown[0]["quote"] == "é: CT completed; report available."
        assert shown[0]["note_type"] == "Radiology"
        assert shown[0]["note_date"].startswith("2026-02-01")
        assert first["source_spans"] == [[shown[0]["start"], shown[0]["end"]]]
        assert (
            worker.execute("selected.iloc[0]")["source_spans"] == first["source_spans"]
        )
        assert (
            worker.execute("print(selected.note_text)")["source_spans"]
            == first["source_spans"]
        )
        undated = worker.execute(
            "notes.loc[notes.note_date.isna() | notes.note_type.isna()]"
        )
        assert "biopsy performed" in undated["output"]
        assert "'note_date': None" in undated["output"]
        forged = worker.execute(
            "changed = selected.copy(); changed['note_text'] = 'INVENTED'; changed"
        )
        assert forged["error"] == "ValueError" and forged["source_spans"] == []
    pd.testing.assert_frame_equal(notes, original)


def test_long_note_pagination_and_focused_excerpt_preserve_unicode():
    text = "é " + "routine text " * 2000 + "CT completed."
    record = prepare_record(notes=pd.DataFrame({"note_text": [text, "Later note."]}))
    with NoteREPL(
        record.history, NoteSearchLimits(max_output_chars=1200), notes=record.spans
    ) as worker:
        found = worker.execute(
            "show_notes(notes, pattern='CT', chars=150, context=10, limit=1)"
        )
        assert found["error"] is None
        data = ast.literal_eval(found["output"])
        assert data["has_more"] and data["next_start"] == 1
        assert "CT completed" in data["notes"][0]["quote"]
        start, end = found["source_spans"][0]
        assert record.history[start:end] == data["notes"][0]["quote"]
        assert "Later note" in worker.execute("show_notes(notes, start=1)")["output"]


@pytest.mark.parametrize("form", ["frame", "text", "both", "positional_frame"])
def test_all_input_forms_summary_navigation_and_code_owned_dates(
    monkeypatch, llm, notes, form
):
    summary = "UNVERIFIED NAVIGATION SUMMARY"
    options = {
        "questions": ["Was CT performed?"],
        "llm": llm,
        "patient_summary": summary,
    }
    if form in {"frame", "both"}:
        options["notes"] = notes
    if form in {"text", "both"}:
        options["history"] = "CT completed. TEXT BUFFER ONLY"
    if form == "positional_frame":
        options["history"] = notes
    calls = []

    def handler(payload):
        calls.append(payload)
        assert payload["patient_summary"] == summary
        assert payload["notes_metadata"]["columns"][-3:] == [
            "note_date",
            "note_type",
            "note_text",
        ]
        if payload["last_cell_result"] is None:
            return cell(
                "notes.loc[notes.note_text.str.contains('CT completed', case=False, na=False)]"
            )
        excerpts = payload["last_cell_result"]["source_excerpts"]
        assert summary not in json.dumps(excerpts)
        return final("", "CT", "Completion documented in original notes.")

    mock_model(monkeypatch, handler)
    result = answer_patient_questions(**options)["answers"][0]
    assert result["status"] == "answered" and result["metadata"]["patient_summary_used"]
    assert len(calls) == 2
    if form == "text":
        assert result["evidence"][0]["note_date"] is None
    else:
        assert result["evidence"][0]["note_date"].startswith("2026-02-01")
        assert result["evidence"][0]["note_type"] == "Radiology"
        assert "TEXT BUFFER ONLY" not in json.dumps(result["evidence"])


def test_more_than_two_cells_keep_pandas_state_without_reasoning_replay(
    monkeypatch, llm, notes
):
    count = 0
    codes = [
        "candidates = notes.loc[notes.note_text.str.contains('CT', case=False, na=False)]; len(candidates)",
        "candidates[['note_date', 'note_type']]",
        "candidates.loc[candidates.note_date >= pd.Timestamp('2026-02-01', tz='UTC')]",
        "show_notes(candidates, pattern='CT', chars=100)",
    ]

    def handler(payload):
        nonlocal count
        count += 1
        return (
            cell(codes[count - 1], "Factual notebook")
            if count <= 4
            else final("", "CT")
        )

    seen = mock_model(monkeypatch, handler)
    answer = answer_patient_questions(notes=notes, questions=["CT status?"], llm=llm)[
        "answers"
    ][0]
    assert answer["status"] == "answered" and answer["metadata"]["cells"] == 4
    assert answer["metadata"]["max_calls"] == 16
    assert all(
        [message["role"] for message in messages] == ["system", "user"]
        for messages in seen
    )


def test_summary_does_not_create_original_evidence(monkeypatch, llm):
    mock_model(
        monkeypatch,
        lambda p: cell("notes[['note_date', 'note_type']]")
        if p["last_cell_result"] is None
        else final("", answer="Unknown: no original documentation."),
    )
    answer = answer_patient_questions(
        notes=pd.DataFrame({"note_text": ["Unrelated text."]}),
        patient_summary="CT definitely completed",
        questions=["CT status?"],
        llm=llm,
    )["answers"][0]
    assert answer["status"] == "unknown" and answer["evidence"] == []


def test_batch_mixes_dataframe_and_text_without_shared_patient_state(
    monkeypatch, llm, notes
):
    mock_model(
        monkeypatch,
        lambda p: cell("notes.loc[notes.note_text.str.contains('CT', na=False)]")
        if p["last_cell_result"] is None
        else final("", "CT"),
    )
    result = answer_patient_question_batch(
        [
            {
                "patient_id": "a",
                "notes": notes,
                "patient_summary": "Context A",
                "questions": ["CT?"],
            },
            {
                "patient_id": "b",
                "history": "CT canceled in fictional patient B.",
                "questions": ["CT?"],
            },
        ],
        llm=llm,
    )
    a, b = [p["answers"][0] for p in result["patients"]]
    assert a["evidence"][0]["note_date"] is not None
    assert b["evidence"][0]["note_date"] is None
    assert "patient B" not in json.dumps(a["evidence"])
    assert "Radiology" not in json.dumps(b["evidence"])


def test_pandas_io_and_process_access_remain_denied(tmp_path, notes):
    target = tmp_path / "secret.csv"
    target.write_text("PRIVATE SENTINEL")
    record = prepare_record(notes=notes)
    with NoteREPL(record.history, NoteSearchLimits(), notes=record.spans) as worker:
        for code in [
            f"pd.read_csv({str(target)!r})",
            f"notes.to_csv({str(target)!r})",
            "pd.read_json('http://127.0.0.1:1/private')",
            "import os; os.fork()",
        ]:
            result = worker.execute(code)
            assert result["error"] is not None
            assert "PRIVATE SENTINEL" not in result["output"]
        for family in [1, 2, 10]:
            result = worker.execute(
                f"import ctypes; libc = ctypes.CDLL(None, use_errno=True); (libc.socket({family}, 1, 0), ctypes.get_errno())"
            )
            assert result["output"].strip() == "(-1, 1)"


def test_every_dependency_thread_has_the_same_seccomp_filter():
    with NoteREPL("Fabricated note.", NoteSearchLimits()) as worker:
        policies = []
        for task in (Path("/proc") / str(worker.process.pid) / "task").iterdir():
            fields = {
                line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                for line in task.joinpath("status").read_text().splitlines()
                if line.startswith(("NoNewPrivs:", "Seccomp:", "Seccomp_filters:"))
            }
            assert fields["NoNewPrivs"] == "1" and fields["Seccomp"] == "2"
            policies.append(fields["Seccomp_filters"])
        assert policies and len(set(policies)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"notes": pd.DataFrame({"wrong": ["text"]})},
        {"notes": pd.DataFrame({"note_text": ["text"], "note_date": ["invalid"]})},
        {"notes": pd.DataFrame({"note_text": ["a", "b"], "patient_id": ["a", "b"]})},
        {"notes": pd.DataFrame({"note_text": ["text"], "note_type": [3]})},
        {"history": "text", "patient_summary": {"bad": "shape"}},
    ],
)
def test_invalid_sources_fail_before_worker_or_endpoint(monkeypatch, llm, kwargs):
    monkeypatch.setattr(qa, "NoteREPL", lambda *a, **kw: pytest.fail("No worker yet"))
    with pytest.raises((ValueError, TypeError)):
        answer_patient_questions(questions=["q"], llm=llm, **kwargs)


def test_workup_pandas_summary_and_followup_patient_free_vocabulary(harness, notes):
    config, install, calls = harness
    install(
        lambda p: {"terms": ["CT"]}
        if "unsuccessful_terms" in p
        else {
            "status": "answered",
            "answer": {
                "status": "completed",
                "applicability": "applies",
                "bottom_line": "Fabricated completion.",
            },
            "limitations": [],
            "needs_more_evidence": False,
        },
        code="notes.loc[notes.note_type.eq('Oncology').fillna(False)]",
    )
    result = review_patient_workup_with_note_search(
        notes,
        [{"name": "CT"}],
        history="IGNORED UNSTRUCTURED BUFFER",
        patient_summary="PRIVATE NAVIGATION CONTEXT",
        config=config,
    )
    row = result["assessments"][0]
    assert row["status"] == "completed" and result["metadata"]["patient_summary_used"]
    assert row["evidence"][-1]["note_type"] == "Radiology"
    vocabulary = [p for p in calls if "unsuccessful_terms" in p]
    assert vocabulary and "PRIVATE NAVIGATION CONTEXT" not in json.dumps(vocabulary)
    assert "IGNORED UNSTRUCTURED" not in json.dumps(row["evidence"])
