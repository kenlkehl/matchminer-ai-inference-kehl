"""Fabricated population tests: isolation, denominators, cancellation and archives."""

import copy
import json
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai.cancellation import (
    CancellationToken,
    InferenceCancelled,
    cancellation_scope,
)
from matchminer_ai.patients import (
    ColBERTConfig,
    ColBERTPatientIndex,
    colbert_notes_fingerprint,
)
from matchminer_ai.patients.workup import _notes
from matchminer_ai.population import (
    create_patient_cohort,
    subset_patient_cohort,
    compress_patient_cohort,
    encode_patient_cohort,
    review_population_workup_with_colbert,
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


def fake_encode(notes, *, config, **kw):
    result = {}
    for pid, frame in notes.groupby("patient_id", sort=False):
        source = _notes(frame)
        chunks = tuple({**n, "start": 0, "end": len(n["text"])} for n in source)
        result[pid] = ColBERTPatientIndex(
            pid,
            config,
            {"model": "fabricated"},
            tuple(source),
            chunks,
            tuple(np.array([[1.0, 0.0]], dtype=np.float32) for _ in chunks),
            colbert_notes_fingerprint(frame),
        )
    return result


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


@pytest.fixture
def encoded(cohort, monkeypatch):
    monkeypatch.setattr(
        "matchminer_ai.patients.encode_patient_notes_colbert", fake_encode
    )
    return encode_patient_cohort(cohort, config=ColBERTConfig(device="cpu"))


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


def test_empty_selection_valid_but_cannot_encode(cohort):
    selected = subset_patient_cohort(cohort, [])
    assert selected.summaries.empty and selected.notes.empty
    with pytest.raises(ValueError, match="nonempty"):
        encode_patient_cohort(selected, config=ColBERTConfig())


def test_compression_keeps_originals_and_invalidates_compressed_index(
    encoded, monkeypatch
):
    def compress(frame, **kwargs):
        assert kwargs["max_concurrent_requests"] == 8
        return frame.assign(
            compressed_note_text=frame.note_text.str.replace("planned", "plan")
        )

    monkeypatch.setattr("matchminer_ai.patients.compress_patient_notes", compress)
    c = compress_patient_cohort(encoded, config=load_default_preset())
    pd.testing.assert_frame_equal(c.notes, encoded.notes)
    assert c.indexes and c.index_source == "original"
    c = encode_patient_cohort(c, config=ColBERTConfig(), note_source="compressed")
    c = compress_patient_cohort(c, config=load_default_preset())
    assert (
        c.indexes == {}
        and c.index_source is None
        and not c.compression["losslessness_verified"]
    )


def test_review_isolation_failure_counts_and_csv(encoded, monkeypatch):
    calls = []

    def review(index, recommendations, **kwargs):
        calls.append((index.patient_id, kwargs))
        assert (
            all("Separate" not in n["text"] for n in index.notes)
            if index.patient_id == "a"
            else len(index.notes) == 1
        )
        if index.patient_id == "b":
            raise RuntimeError("Never expose provider or patient text")
        return {
            "assessments": [
                finding(r, n, applies="applies" if n == 0 else "not_applicable")
                for n, r in enumerate(recommendations)
            ],
            "metadata": {"method": "colbert"},
            "notice": "Fabricated",
        }

    monkeypatch.setattr(
        "matchminer_ai.patients.review_patient_workup_with_colbert", review
    )
    result = review_population_workup_with_colbert(
        encoded, config=load_default_preset(), top_n=7
    )
    assert [p for p, _ in calls] == ["b", "a"]
    assert all(
        kwargs["top_n"] == 7 and "patient_summary" not in kwargs for _, kwargs in calls
    )
    assert len(population_results_table(result)) == 4
    row = result.report["aggregate"][0]
    assert row["statuses"]["error"] == 1
    assert row["completed_applicable"] == row["applicable_assessed"] == 1
    assert row["documented_completion_fraction"] == 1
    assert result.report["aggregate"][1]["documented_completion_fraction"] is None
    assert "Never expose" not in json.dumps(result.report)
    assert encoded.report is None


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


def test_cancellation_does_not_publish_partial_report(encoded, monkeypatch):
    token = CancellationToken()

    def review(*args, **kwargs):
        token.cancel()
        token.check()

    monkeypatch.setattr(
        "matchminer_ai.patients.review_patient_workup_with_colbert", review
    )
    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        review_population_workup_with_colbert(encoded, config=load_default_preset())
    assert encoded.report is None


def test_wrong_index_source_or_chunking_rejected_before_review(encoded, monkeypatch):
    monkeypatch.setattr(
        "matchminer_ai.patients.review_patient_workup_with_colbert",
        lambda *a, **kw: pytest.fail("Must not review"),
    )
    with pytest.raises(ValueError, match="stale"):
        review_population_workup_with_colbert(
            encoded,
            config=load_default_preset(),
            retrieval_config=ColBERTConfig(chunk_size=128),
        )
    wrong = replace(
        encoded, indexes={"a": encoded.indexes["b"], "b": encoded.indexes["a"]}
    )
    with pytest.raises(ValueError, match="stale"):
        review_population_workup_with_colbert(wrong, config=load_default_preset())


def test_archive_roundtrip_full_cohort_with_compression_and_vectors(
    encoded, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "matchminer_ai.patients.compress_patient_notes",
        lambda frame, **kw: frame.assign(
            compressed_note_text="Compact fabricated note"
        ),
    )
    c = compress_patient_cohort(encoded, config=load_default_preset())
    c = encode_patient_cohort(c, config=ColBERTConfig(), note_source="compressed")
    path = save_patient_cohort(c, tmp_path / "cohort.zip")
    loaded = load_patient_cohort(path)
    pd.testing.assert_frame_equal(loaded.notes, c.notes)
    pd.testing.assert_frame_equal(loaded.summaries, c.summaries)
    pd.testing.assert_frame_equal(loaded.compressed_notes, c.compressed_notes)
    assert loaded.index_source == "compressed" and loaded.space == c.space
    assert path.stat().st_mode & 0o777 == 0o600
    for pid in c.indexes:
        assert loaded.indexes[pid].source_sha256 == c.indexes[pid].source_sha256
        np.testing.assert_array_equal(
            loaded.indexes[pid].embeddings[0], c.indexes[pid].embeddings[0]
        )
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
