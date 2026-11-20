"""Benchmark bindings reject stale data, and evidence checks retain date semantics."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from matchminer_ai.apps.guideline_fetcher.settings import Settings

# Load the example by path: an unrelated installed package named "examples" can shadow
# this repository's examples folder.
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_biliary_workup",
    Path(__file__).resolve().parents[3] / "examples" / "guideline_fetcher" / "benchmark_biliary_workup.py")
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def inputs(tmp_path, monkeypatch):
    csv = tmp_path / "biliary.csv"
    csv.write_text("patient_id,note_date,note_text\nsynthetic,2026-01-01,CT performed.\n")
    provenance = {
        "source": str(benchmark.synthetic_source()),
        "csv_sha256": benchmark.sha256(csv),
        "note_count": 1,
    }
    csv.with_suffix(".provenance.json").write_text(json.dumps(provenance))
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "paradigms.jsonl").write_text("{}")
    gold = {
        "notes_sha256": benchmark.sha256(csv),
        "catalog_sha256": benchmark.sha256(catalog / "paradigms.jsonl"),
        "space_trial_id": "fabricated",
        "items": [{"name": "CT", "accepted_statuses": ["completed"]}],
    }
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(gold))
    monkeypatch.setattr(
        benchmark,
        "load_guideline_catalog",
        lambda path: pd.DataFrame(
            [
                {
                    "space_trial_id": "fabricated",
                    "name": "Fabricated test space",
                    "diagnostic_workup": [{"name": "CT"}],
                }
            ]
        ),
    )
    return Settings(device="cpu", biliary_synthetic_notes=csv), catalog, annotations


def test_verified_synthetic_inputs_and_annotations(tmp_path, monkeypatch):
    settings, catalog, annotations = inputs(tmp_path, monkeypatch)
    source, space, gold = benchmark.load_inputs(settings, catalog, "fabricated", annotations)
    assert source["notes"][0]["note_text"] == "CT performed."
    assert space["name"] == "Fabricated test space" and len(gold["items"]) == 1


@pytest.mark.parametrize("field", ["notes_sha256", "catalog_sha256", "space_trial_id"])
def test_stale_annotations_rejected_before_inference(tmp_path, monkeypatch, field):
    settings, catalog, annotations = inputs(tmp_path, monkeypatch)
    gold = json.loads(annotations.read_text())
    gold[field] = "changed"
    annotations.write_text(json.dumps(gold))
    with pytest.raises(ValueError):
        benchmark.load_inputs(settings, catalog, "fabricated", annotations)


def test_modified_note_csv_rejected(tmp_path, monkeypatch):
    settings, catalog, annotations = inputs(tmp_path, monkeypatch)
    settings.biliary_synthetic_notes.write_text("changed source")
    with pytest.raises(ValueError, match="provenance hash"):
        benchmark.load_inputs(settings, catalog, "fabricated", annotations)


@pytest.mark.parametrize("thinking", [True, False])
def test_summary_reuse_requires_the_same_verified_case(tmp_path, thinking):
    gold = {"notes_sha256": "notes", "catalog_sha256": "catalog", "space_trial_id": "space"}
    saved = {
        **gold,
        "synthetic_only": True,
        "model": benchmark.GEMMA_PROFILE["model"],
        "thinking": thinking,
        "summary": "Fabricated patient summary.",
    }
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(saved))
    assert benchmark.reused_summary(path, gold) == saved["summary"]
    saved["notes_sha256"] = "different notes"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        benchmark.reused_summary(path, gold)


def test_status_score_excludes_ambiguity_and_checks_exact_source_evidence():
    notes = [{"note_text": "CT performed.", "note_date": "2026-01-01T00:00:00"}]
    gold = {
        "items": [
            {"name": "CT", "accepted_statuses": ["completed"], "rationale": "Performed."},
            {
                "name": "Initial consultation",
                "accepted_statuses": ["unclear"],
                "score": False,
                "rationale": "Timing ambiguous.",
            },
        ]
    }
    review = {
        "assessments": [
            {
                "status": "completed",
                "applicability": "applies",
                "evidence": [
                    {
                        "note_number": 1,
                        "note_date": "2026-01-01T00:00:00+00:00",
                        "quote": "CT performed.",
                    },
                    {
                        "note_number": 1,
                        "note_date": "2026-01-01T00:00:00+00:00",
                        "quote": "Invented result.",
                    },
                ],
            },
            {"status": "not_documented", "applicability": "uncertain", "evidence": []},
        ]
    }
    score = benchmark.score(review, gold, notes)
    assert score["status_matches"] == score["scored_items"] == 1
    assert score["excluded_ambiguous_items"] == 1
    assert score["evidence_items"] == 2 and score["invalid_source_quotes_or_dates"] == 1
