"""Fabricated population tests: isolation, denominators, cancellation and archives."""

import copy
import json
import zipfile
from dataclasses import replace

import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.cancellation import (
    CancellationToken,
    InferenceCancelled,
    cancellation_scope,
)
from matchminer_ai.population import (
    create_patient_cohort,
    subset_patient_cohort,
    compress_patient_cohort,
    review_population_workup,
    aggregate_population_workup,
    population_results_table,
    save_patient_cohort,
    load_patient_cohort,
)


@pytest.fixture
def cohort():
    summaries = pd.DataFrame(
        [
            {"patient_id": "b", "cancer_history_summary": "Fabricated second patient"},
            {"patient_id": "a", "cancer_history_summary": "Fabricated first patient"},
        ]
    )
    notes = pd.DataFrame(
        [
            {
                "patient_id": "a",
                "note_text": "Later test completed.",
                "note_date": "2026-02-01",
                "note_type": "Visit",
            },
            {
                "patient_id": "b",
                "note_text": "Separate patient planned test.",
                "note_date": None,
            },
            {
                "patient_id": "a",
                "note_text": "Earlier planned test.",
                "note_date": "2026-01-01",
                "note_type": "Visit",
            },
        ]
    )
    c = create_patient_cohort(summaries, notes, provenance={"source": "fabricated"})
    return replace(
        c,
        space={
            "space_trial_id": "s",
            "clinical_space_summary": "Fabricated population",
            "diagnostic_workup": [
                {
                    "name": "Test",
                    "conditions": "If indicated",
                    "evidence": [{"quote": "Fabricated guideline text"}],
                },
                {"name": "Second test", "conditions": ""},
            ],
        },
    )


def finding(rec, number, status="completed", applies="applies"):
    return dict(
        recommendation=copy.deepcopy(rec),
        recommendation_index=number,
        name=rec["name"],
        status=status,
        applicability=applies,
        bottom_line="Fabricated finding",
        evidence=[],
        limitations=[],
    )


def test_validation_and_stable_source_dates(cohort):
    assert cohort.summaries.patient_id.tolist() == ["b", "a"]
    assert cohort.notes.loc[cohort.notes.patient_id == "a", "note_text"].tolist() == [
        "Earlier planned test.",
        "Later test completed.",
    ]
    assert cohort.notes.loc[cohort.notes.patient_id == "b", "note_date"].iloc[0] is None
    with pytest.raises(ValueError, match="Every summary"):
        create_patient_cohort(
            cohort.summaries, cohort.notes.loc[cohort.notes.patient_id == "a"]
        )
    bad = cohort.notes.copy()
    bad.loc[0, "note_date"] = "wrong"
    with pytest.raises(ValueError, match="date"):
        create_patient_cohort(cohort.summaries, bad)
    with pytest.raises(ValueError, match="unique"):
        create_patient_cohort(pd.concat([cohort.summaries] * 2), cohort.notes)


def test_empty_selection_valid_but_cannot_review(cohort):
    selected = subset_patient_cohort(cohort, [])
    assert selected.summaries.empty and selected.notes.empty
    with pytest.raises(ValueError, match="nonempty"):
        review_population_workup(selected, config=load_default_preset())


def test_compression_keeps_originals_and_invalidates_the_report(cohort, monkeypatch):
    def compress(frame, **kwargs):
        assert kwargs["max_concurrent_requests"] == 8
        return frame.assign(
            compressed_note_text=frame.note_text.str.replace("planned", "plan")
        )

    monkeypatch.setattr("matchminer_ai.patients.compress_patient_notes", compress)
    c = compress_patient_cohort(replace(cohort, report={"stale": True}), config=load_default_preset())
    pd.testing.assert_frame_equal(c.notes, cohort.notes)
    assert c.report is None and not c.compression["losslessness_verified"]
    assert c.selected_notes("compressed").note_text.str.contains("plan").any()


def test_review_isolation_failure_counts_and_csv(cohort, monkeypatch):
    calls = []

    def review(notes, recommendations, **kwargs):
        pid = notes.patient_id.iloc[0] if "patient_id" in notes else None
        texts = notes.note_text.tolist()
        calls.append((texts, kwargs))
        if any("Separate" in text for text in texts):
            raise RuntimeError("fabricated endpoint failure")
        assert pid is None and len(texts) == 2
        return {
            "assessments": [finding(r, n, applies="applies" if n == 0 else "not_applicable")
                            for n, r in enumerate(recommendations)],
            "metadata": {"method": "full_notes"},
            "notice": "Fabricated",
        }

    monkeypatch.setattr("matchminer_ai.patients.review_patient_workup", review)
    result = review_population_workup(cohort, config=load_default_preset(), max_concurrent_patients=1)
    assert [len(texts) for texts, _ in calls] == [1, 2]
    assert all(kwargs["population_context"] == "Fabricated population" for _, kwargs in calls)
    table = population_results_table(result)
    assert len(table) == 4 and set(table.method) == {"full_notes"}
    row = result.report["aggregate"][0]
    assert row["statuses"]["error"] == 1
    assert row["completed_applicable"] == row["applicable_assessed"] == 1
    assert row["documented_completion_fraction"] == 1
    assert result.report["aggregate"][1]["documented_completion_fraction"] is None
    assert "RuntimeError" in json.dumps(result.report) and "fabricated endpoint failure" not in json.dumps(result.report)
    assert cohort.report is None


def test_denominators_distinguish_applicability_and_errors(cohort):
    rec = cohort.space["diagnostic_workup"][:1]
    cases = [
        ("completed", "applies"),
        ("not_documented", "applies"),
        ("completed", "uncertain"),
        ("not_done", "not_applicable"),
        ("error", "applies"),
    ]
    reviews = [
        {"patient_id": str(i), "review": {"assessments": [finding(rec[0], 0, s, a)]}}
        for i, (s, a) in enumerate(cases)
    ]
    row = aggregate_population_workup(reviews, rec)[0]
    assert sum(row["statuses"].values()) == 5
    assert row["applicable_assessed"] == 2 and row["completed_applicable"] == 1
    assert row["documented_completion_fraction"] == 0.5


def test_cancellation_does_not_publish_partial_report(cohort, monkeypatch):
    token = CancellationToken()

    def review(*args, **kwargs):
        token.cancel()
        token.check()

    monkeypatch.setattr("matchminer_ai.patients.review_patient_workup", review)
    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        review_population_workup(cohort, config=load_default_preset())
    assert cohort.report is None


def test_archive_roundtrip_full_cohort_with_compression(cohort, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "matchminer_ai.patients.compress_patient_notes",
        lambda frame, **kw: frame.assign(
            compressed_note_text="Compact fabricated note"
        ),
    )
    c = compress_patient_cohort(cohort, config=load_default_preset())
    path = save_patient_cohort(c, tmp_path / "cohort.zip")
    loaded = load_patient_cohort(path)
    pd.testing.assert_frame_equal(loaded.notes, c.notes)
    pd.testing.assert_frame_equal(loaded.summaries, c.summaries)
    pd.testing.assert_frame_equal(loaded.compressed_notes, c.compressed_notes)
    assert loaded.space == c.space
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="oversized"):
        load_patient_cohort(path, max_uncompressed_bytes=1)
    with zipfile.ZipFile(path) as z:
        contents = {name: z.read(name) for name in z.namelist()}
    contents["cohort.json"] += b" "
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as z:
        for name, content in contents.items():
            z.writestr(name, content)
    with pytest.raises(ValueError, match="checksum"):
        load_patient_cohort(tmp_path / "bad.zip")
