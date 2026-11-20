"""A miniature synthetic run in the real run's layout (fabricated text only), and fixtures with fakes.

No real model, encoder, endpoint or dataset file is touched: the decision model's package functions and the
query encoder are replaced by fakes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

# A fabricated site config: the trial table's name and columns, site names, the home-institution prefix, the
# teacher endpoint and the Vertex project (placeholders only).
SITE_CONFIG = """
endpoints:
  teacher:
    url: http://teacher.example:8000/v1
vertex:
  project: example-project
dashboard:
  trial_table:
    name: trial_listing
    columns:
      management_group: managing_group
      site_pi: local_pi
  site_names:
    Example Cancer Center: ECC Main
    Example General Hospital: EGH
  center_prefix: '^ECC/(?:MAIN|PEDS)\\s+'
  higher_threshold_groups: [ECC Phase I]
  branding:
    institution: Example Cancer Center
    pi_label: ECC PI
    institutions:
      - {label: ECC / Main, prefixes: [ECC/MAIN]}
    trial_links:
      - {label: Portal, url: 'https://portal.example.org/protocols/{protocol_no}'}
"""


@pytest.fixture(autouse=True)
def site_config(tmp_path, monkeypatch):
    """Every test runs with the fabricated site config and a scratch data root."""
    path = tmp_path / "site.yaml"
    path.write_text(SITE_CONFIG)
    monkeypatch.setenv("MMAI_SITE_CONFIG", str(path))
    monkeypatch.setenv("MMAI_DATA_ROOT", str(tmp_path / "data"))
    return path

REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DIM = 8

PATIENTS = [
    ("patient_a", "Age: 62\nSex: Male\nCancer type: Lung cancer\nHistology: Adenocarcinoma\nBiomarkers: EGFR L858R",
     "ECOG 1. Creatinine 0.9 mg/dL.", "positive", "NCT00000001", "NCT00000001-cohort-0"),
    ("patient_b", "Age: 55\nSex: Female\nCancer type: Lung cancer\nHistology: Squamous cell carcinoma",
     "ECOG 2.", "negative", "NCT00000001", "NCT00000001-cohort-0"),
    ("patient_c", "Age: 48\nSex: Female\nCancer type: Breast cancer\nHistology: Invasive ductal carcinoma\nBiomarkers: HER2 positive",
     "ECOG 0.", "positive", "NCT00000002", "NCT00000002-cohort-0"),
    ("patient_d", "Age: 70\nSex: Male\nCancer type: Breast cancer\nHistology: Invasive lobular carcinoma",
     "", "negative", "NCT00000002", "NCT00000002-cohort-1"),
    ("patient_e", "Age: 66\nSex: Female\nCancer type: Colorectal cancer\nHistology: Adenocarcinoma",
     "Prior myocardial infarction.", "positive", "NCT00000002", "NCT00000002-cohort-1"),
]


def items(lines):
    return [{"number": i, "heading": "Inclusion criteria:" if kind == "inclusion" else "Exclusion criteria:", "text": text,
             "criterion_type": kind, "boilerplate": bp} for i, (text, kind, bp) in enumerate(lines, 1)]


COHORTS = [
    ("NCT00000001-cohort-0", "NCT00000001", "protocol", "All participants",
     [("Metastatic lung adenocarcinoma with EGFR L858R.", "inclusion", False), ("ECOG 0-1.", "inclusion", True),
      ("Prior EGFR inhibitor.", "exclusion", False)]),
    ("NCT00000001-ctgov-cohort-0", "NCT00000001", "ctgov", "All participants",
     [("EGFR-mutant NSCLC.", "inclusion", False), ("Pregnancy.", "exclusion", True)]),
    ("NCT00000002-cohort-0", "NCT00000002", "protocol", "HER2-positive",
     [("HER2-positive breast cancer.", "inclusion", False)]),
    ("NCT00000002-cohort-1", "NCT00000002", "protocol", "HER2-negative",
     [("HER2-negative breast cancer.", "inclusion", False), ("Adequate organ function.", "inclusion", True)]),
    ("NCT00000002-ctgov-cohort-0", "NCT00000002", "ctgov", "Breast cancer",
     [("Breast cancer.", "inclusion", False)]),
]


def criteria_text(lines, boilerplate):
    chosen = [(t, k) for t, k, b in lines if b == boilerplate]
    out = []
    for label, kind in (("Inclusion criteria:", "inclusion"), ("Exclusion criteria:", "exclusion")):
        part = [t for t, k in chosen if k == kind]
        if part:
            out += [label, *part]
    return "\n".join(out)


def unit(seed: int, dim: int = DIM) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def encoder_identity_for(model: str = "google/embeddinggemma-2", revision: str | None = REVISION) -> dict:
    from matchminer_ai.embedding.encoder import EncoderSettings, encoder_identity
    from matchminer_ai.embedding.trialspace_contract import model_prompts

    return encoder_identity(EncoderSettings(model, revision=revision, devices=("cpu",), batch_size=32, max_seq_length=8192,
                                            dtype="float32", autocast="bfloat16", prompts=model_prompts("trialspace"),
                                            default_prompt_name="SentenceSimilarity", include_prompt_in_pooling=True,
                                            text_only=True))


def write_round(root: Path, number: int, corpus_receipt: str, digests: dict, encoder: dict, *, seed: int = 0,
                patients: int | None = None, cohorts: int | None = None) -> Path:
    mining = root / "trialspace" / f"trialspace_round{number}" / "mining"
    mining.mkdir(parents=True, exist_ok=True)
    P = np.stack([unit(seed + i) for i in range(patients if patients is not None else len(PATIENTS))])
    C = np.stack([unit(seed + 100 + i) for i in range(cohorts if cohorts is not None else len(COHORTS))])
    np.save(mining / "patients.npy", P)
    np.save(mining / "cohorts.npy", C)
    manifest = {"version": "trialspace-dense-mining-v3", "corpus": corpus_receipt, "contract": "c", "encoder": encoder,
                "patients": digests["patients"], "cohorts": digests["cohorts"], "dimension": DIM}
    write_json(mining / "manifest.json", manifest)
    write_json(mining / "stage_receipt.json", {
        "format": "mmai-stage-receipt-v1", "kind": "mine", "receipt_id": f"mine{number}",
        "identity": {"version": manifest["version"], "corpus": corpus_receipt, "encoder": encoder},
        "upstream": {"corpus": corpus_receipt},
        "outputs": {"patients.npy": (mining / "patients.npy").stat().st_size, "cohorts.npy": (mining / "cohorts.npy").stat().st_size},
        "provenance": {"finished_at": "2026-11-07T03:09:27+00:00"}})
    return mining


def build_dataset(root: Path) -> dict:
    """The miniature run; returns ids and paths the tests use."""
    from matchminer_ai.labeling.corpus import load_corpus
    from matchminer_ai.labeling.files import digest

    corpus = root / "trialspace" / "trialspace_corpus"
    write_jsonl(corpus / "patients.jsonl", [{"id": pid, "text": key, "boilerplate": bp, "split": "train",
                                             "source_trial_id": trial, "source_split": "train"}
                                            for pid, key, bp, _, trial, _ in PATIENTS])
    write_jsonl(corpus / "cohorts.jsonl", [{"id": cid, "trial_id": trial, "source": source, "cohort_name": name,
                                            "text": criteria_text(lines, False), "boilerplate": criteria_text(lines, True),
                                            "items": items(lines), "split": "train", "source_split": "train"}
                                           for cid, trial, source, name, lines in COHORTS])
    write_json(corpus / "corpus.json", {"version": "trialspace-corpus-v3", "patients": len(PATIENTS), "cohorts": len(COHORTS),
                                        "cohort_sources": {"protocol": 3, "ctgov": 2}})
    write_json(corpus / "stage_receipt.json", {"format": "mmai-stage-receipt-v1", "kind": "prepare", "receipt_id": "corpus-receipt-1"})
    patients, cohorts = load_corpus(corpus)
    digests = {"patients": digest(patients), "cohorts": digest(cohorts)}
    import pyarrow as pa
    import pyarrow.parquet as pq

    summaries = root / "patient_summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"patient_id": [p[0] for p in PATIENTS], "patient_class": [p[3] for p in PATIENTS],
                             "nct_id": [p[4] for p in PATIENTS], "cohort_trial_id": [p[5] for p in PATIENTS]}),
                   summaries / "synthetic_patient_metadata.parquet")
    notes = {"patient_id": ["patient_a", "patient_b", "patient_a", "patient_c"], "date": ["2024-05-02", "2024-01-01", "2024-03-01", "2024-02-02"],
             "event_type": ["clinic_note", "imaging_report", "pathology_report", "clinic_note"],
             "synthetic_note": ["Second visit note.", "CT chest.", "Biopsy: adenocarcinoma.", "Breast clinic."]}
    pq.write_table(pa.table(notes), summaries / "all_synthetic_notes.parquet", row_group_size=2)
    write_round(root, 1, "corpus-receipt-1", digests, encoder_identity_for())
    return {"root": root, "digests": digests, "corpus": corpus}


class FakeEncoder:
    """Stands in for the embedding worker: deterministic unit vectors per text."""

    def __init__(self, available=True, problem=None):
        self.available, self.problem, self.calls = available, problem, []

    def encode(self, texts, prompt_name):
        from matchminer_ai.apps.dashboard.core.encoder import EncoderUnavailable

        if not self.available:
            raise EncoderUnavailable(self.problem or "unavailable")
        self.calls.append((list(texts), prompt_name))
        return np.stack([unit(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)) for t in texts])

    def describe(self):
        return {"available": self.available, "problem": self.problem, "state": "fake", "python": "-", "cache": "-", "device": "cpu"}

    def close(self):
        pass


@pytest.fixture
def dataset(tmp_path):
    return build_dataset(tmp_path / "data")


@pytest.fixture
def settings(dataset, tmp_path):
    from matchminer_ai.apps.dashboard.core.settings import Settings

    return Settings(synthetic_run=dataset["root"], cache_dir=tmp_path / "cache", embedding_caches=(tmp_path / "hf",),
                    embedding_python=tmp_path / "missing-python")


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the decision model's package stages with fakes that record their calls."""
    import pandas as pd
    import matchminer_ai.matching as matching

    from matchminer_ai.apps.dashboard.core.decisions import DecisionService

    calls = {"match": 0, "boilerplate": 0, "criteria": 0, "attribution": 0}

    def score_match_quality(frame, config=None, filter_low_quality=True):
        calls["match"] += len(frame)
        out = frame[["patient_id", "cohort_trial_id"]].copy()
        p = [0.9 if "lung" in key.lower() and "EGFR" in cohort else 0.1
             for key, cohort in zip(frame["cancer_history_summary"], frame["cohort_key_criteria"])]
        out["match_quality_score"] = p
        out["match_quality_pass"] = [x >= 0.5 for x in p]  # stands in for P(reasonable) >= 0.5
        for label in ("reasonableness", "cancer_type", "burden", "treatment_history", "biomarker"):
            out[f"match_quality_{label}"] = p
        return out

    def exclusion_criteria_check(frame, config=None):
        calls["boilerplate"] += len(frame)
        out = frame[["patient_id", "cohort_trial_id"]].copy()
        out["exclusion_criteria_pass_probability"] = 0.7
        out["exclusion_score"] = 0.7
        out["exclusion_criteria_pass"] = True
        return out

    def check_eligibility_criteria(frame, config=None, return_metadata=False):
        calls["criteria"] += 1
        found = json.loads(frame.iloc[0]["cohort_criteria_items"])
        rows = [{"patient_id": frame.iloc[0]["patient_id"], "cohort_trial_id": frame.iloc[0]["cohort_trial_id"],
                 "criterion_number": i["number"], "criterion_type": i["criterion_type"], "boilerplate": i["boilerplate"],
                 "criterion_text": i["text"], "criterion_pass_probability": 0.3 if i["criterion_type"] == "exclusion" else 0.9,
                 "criterion_clearly_fails": i["criterion_type"] == "exclusion"} for i in found]
        meta = {"model_metadata": {"check_eligibility_criteria": {"unchecked_lines": 0, "input_tokens": 99, "over_max_length": 0}}}
        return (pd.DataFrame(rows), meta) if return_metadata else pd.DataFrame(rows)

    def interpret_match_quality(frame, config=None, top_k=30, head="reasonableness"):
        calls["attribution"] += 1
        key = frame.iloc[0]["cancer_history_summary"].strip()
        return pd.DataFrame([{"patient_id": frame.iloc[0]["patient_id"], "cohort_trial_id": frame.iloc[0]["cohort_trial_id"],
                              f"match_quality_{head}": 0.8, "attribution_target": f"{head}_yes",
                              "token_attributions": {"patient_key_summary": [{"text": key[:3], "start": 0, "end": 3,
                                                                              "raw_score": 1.0, "normalized_score": 1.0}],
                                                     "cohort_key_criteria": []},
                              "attribution_coverage": {}, "attribution_caveat": "Local gradient attribution caveat."}])

    monkeypatch.setattr(matching, "score_match_quality", score_match_quality)
    monkeypatch.setattr(matching, "exclusion_criteria_check", exclusion_criteria_check)
    monkeypatch.setattr(matching, "check_eligibility_criteria", check_eligibility_criteria)
    monkeypatch.setattr(matching, "interpret_match_quality", interpret_match_quality)

    def prepare(self):
        self.loaded_at, self.load_seconds = 1.0, 0.0

    monkeypatch.setattr(DecisionService, "_prepare", prepare)
    return calls


def wait_job(client, job_id, timeout=10.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")
