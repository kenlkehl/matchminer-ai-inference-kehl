"""Cohort-style key criteria for guideline spaces (2026-11-14): prompt, parsing, records, audit and loading."""

import copy
import json

import pandas as pd
import pytest

from matchminer_ai._storage import atomic_json, digest, read_json
from matchminer_ai.llm.structured import StructuredConfig
from matchminer_ai.trials import _guideline_cohort_criteria as step
from matchminer_ai.trials._guideline_audit import audit_catalog
from matchminer_ai.trials._guideline_canonical import TASK as CATALOG_TASK
from matchminer_ai.trials._guideline_cohort_backfill import audit_backfill, backfill_cohort_criteria
from matchminer_ai.trials._guideline_generation import Client
from matchminer_ai.trials._guideline_pipeline import run_guideline
from matchminer_ai.trials._guideline_schema import FIELDS
from matchminer_ai.trials._guideline_sources import load_guideline
from matchminer_ai.trials import _guideline_prompts as prompts
from matchminer_ai.trials.guideline_catalog import load_guideline_catalog
from test_guideline_extraction import catalog, extraction, make_library, quoted_state

SPACE = {**dict.fromkeys(FIELDS, "NA"), "age_range_allowed": "Age >=18 years",
         "cancer_type_allowed": "Fictional carcinoma", "biomarkers_excluded": "Marker Z"}
GOOD = {"criteria": [
    {"type": "inclusion", "text": "The participant is aged 18 years or older.", "fields": ["age_range_allowed"]},
    {"type": "exclusion", "text": "The participant has Marker Z-positive disease.", "fields": ["biomarkers_excluded"]},
    {"type": "inclusion", "text": "The participant has fictional carcinoma.", "fields": ["cancer_type_allowed"]},
]}


def test_validator_accepts_full_coverage_and_names_every_problem():
    assert step.validate_criteria(SPACE, copy.deepcopy(GOOD)) == GOOD
    bad = copy.deepcopy(GOOD)
    bad["criteria"][0]["fields"].append("sex_allowed")
    bad["criteria"][1]["text"] = "The participant has Marker Z (A) AND (B) and an ECOG performance status of 0-1."
    bad["criteria"] = bad["criteria"][:2]
    with pytest.raises(ValueError) as error:
        step.validate_criteria(SPACE, bad)
    message = str(error.value)
    for expected in ("cancer_type_allowed", "sex_allowed, which is NA", "Boolean syntax", "'ecog'"):
        assert expected in message
    with pytest.raises(ValueError, match="at least one inclusion"):
        step.validate_criteria(SPACE, {"criteria": [GOOD["criteria"][1]]})
    numbered = copy.deepcopy(GOOD)
    numbered["criteria"][0]["text"] = "1. The participant is aged 18 years or older."
    with pytest.raises(ValueError, match="number or bullet"):
        step.validate_criteria(SPACE, numbered)


def test_rendering_lists_inclusion_then_exclusion_like_a_cohort():
    assert step.render_criteria(GOOD["criteria"]) == (
        "Inclusion criteria:\nThe participant is aged 18 years or older.\nThe participant has fictional carcinoma.\n"
        "Exclusion criteria:\nThe participant has Marker Z-positive disease."
    )


def test_request_shows_task_examples_fields_and_cited_source_lines():
    record = {"name": "Fictional population", "space": SPACE, "evidence": [
        {"page_id": "p0002", "printed_label": "FIC-1", "source_lines": [
            {"line": 4, "text": "Fictional carcinoma, Marker Z negative"}, {"line": 3, "text": "Adults"}]}]}
    system, user = step.build_messages(record, "Fictional Guideline")
    assert system["content"] == step.SYSTEM
    text = user["content"]
    assert text.startswith(step.TASK) and step.EXAMPLES in text
    assert json.dumps(SPACE["cancer_type_allowed"]) in text and "POPULATION NAME: Fictional population" in text
    assert text.index("[FIC-1 line 3] Adults") < text.index("[FIC-1 line 4] Fictional carcinoma")


def run_fictional(tmp_path, monkeypatch):
    guideline = load_guideline(make_library(tmp_path), "fictional")
    output = tmp_path / "results"

    def respond(self, endpoint, body, **kwargs):
        prompt = body["messages"][1]["content"]
        value = extraction() if prompt.startswith(prompts.EXTRACT_TASK) else (
            catalog() if prompt.startswith(CATALOG_TASK) else quoted_state())
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(Client, "_http", respond)
    config = StructuredConfig(model="synthetic", tokenizer_mode="bytes")
    run_guideline(guideline, output, config, workers=1)
    return guideline, output, config


def rows_of(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def test_pipeline_stores_criteria_that_the_audit_ties_to_accepted_responses(tmp_path, monkeypatch):
    guideline, output, _ = run_fictional(tmp_path, monkeypatch)
    row = rows_of(output / "paradigms.jsonl")[0]
    assert row["key_criteria"] == step.render_criteria(row["key_criteria_items"])
    assert row["key_criteria"].startswith("Inclusion criteria:\nThe participant has Fictional disease.")
    assert "**Cohort key criteria:**" in (output / "report.md").read_text()
    result = audit_catalog(guideline, output)
    assert result["cohort_criteria"] == step.VERSION
    assert read_json(output / "run_config.json")["stage_versions"]["cohort_criteria"] == step.VERSION
    frame = load_guideline_catalog(output / "paradigms.jsonl")
    assert frame.iloc[0]["cohort_key_criteria"] == row["key_criteria"]
    row["key_criteria_items"][0]["text"] = "The participant has another disease."
    row["key_criteria"] = step.render_criteria(row["key_criteria_items"])
    write_rows(output / "paradigms.jsonl", [row])
    with pytest.raises(ValueError, match="not an accepted endpoint response"):
        audit_catalog(guideline, output)


def test_old_run_resumes_and_only_adds_the_cohort_criteria(tmp_path, monkeypatch):
    guideline, output, config = run_fictional(tmp_path, monkeypatch)
    old = read_json(output / "run_config.json")
    for name in step.PROMPT_FILES:
        old["prompt_resources_sha256"].pop(name)
    old["stage_versions"].pop("cohort_criteria")
    identity = {k: copy.deepcopy(v) for k, v in old.items() if k not in ("config_sha256", "runtime", "stage_versions")}
    for key in ("timeout", "attempts", "api_key_env", "stream", "max_concurrent_requests"):
        identity["llm"].pop(key)
    old["config_sha256"] = digest(identity)
    atomic_json(output / "run_config.json", old)
    for folder in (output / "checkpoints").iterdir():
        if (folder / "request.json").exists() and read_json(folder / "request.json")["job"].startswith(step.JOB_PREFIX):
            for path in folder.iterdir():
                path.unlink()
            folder.rmdir()
    rows = rows_of(output / "paradigms.jsonl")
    write_rows(output / "paradigms.jsonl", [{k: v for k, v in r.items() if not k.startswith("key_criteria")}
                                            for r in rows])

    def no_request(*args, **kwargs):
        raise AssertionError("Compatible saved work should not be regenerated")

    monkeypatch.setattr(Client, "_http", no_request)  # the cohort-criteria requests are answered by conftest
    run_guideline(guideline, output, config, workers=1)
    assert rows_of(output / "paradigms.jsonl")[0]["key_criteria"] == rows[0]["key_criteria"]
    assert read_json(output / "run_config.json")["stage_versions"]["cohort_criteria"] == step.VERSION


def test_loader_prefers_stored_criteria_and_keeps_the_derived_layout_for_old_catalogs(tmp_path, monkeypatch):
    _, output, _ = run_fictional(tmp_path, monkeypatch)
    row = rows_of(output / "paradigms.jsonl")[0]
    new = load_guideline_catalog(pd.DataFrame([row]), return_metadata=True)
    old_row = {k: v for k, v in row.items() if not k.startswith("key_criteria")}
    old = load_guideline_catalog(pd.DataFrame([old_row]), return_metadata=True)
    assert new[0].iloc[0]["cohort_key_criteria"] == row["key_criteria"]
    assert old[0].iloc[0]["cohort_key_criteria"].startswith("Inclusion criteria:\nCancer type allowed:")
    assert new[1]["catalog_sha256"] != old[1]["catalog_sha256"]
    other = {**row, "paradigm_id": row["paradigm_id"] + "x", "space_trial_id": row["space_trial_id"] + "x",
             "clinical_space_number": row["clinical_space_number"] + 1}
    mixed = load_guideline_catalog(pd.DataFrame([old_row, other]))
    assert mixed.iloc[0]["cohort_key_criteria"].startswith("Inclusion criteria:\nCancer type allowed:")
    assert mixed.iloc[1]["cohort_key_criteria"] == row["key_criteria"]
    with pytest.raises(ValueError, match="must be cohort criteria"):
        load_guideline_catalog(pd.DataFrame([{**row, "key_criteria": "Cancer type allowed: X."}]))


def test_backfill_writes_a_derived_edition_and_never_changes_the_base(tmp_path, monkeypatch):
    _, base, config = run_fictional(tmp_path, monkeypatch)
    rows = [{k: v for k, v in r.items() if not k.startswith("key_criteria")} for r in rows_of(base / "paradigms.jsonl")]
    write_rows(base / "paradigms.jsonl", rows)
    atomic_json(base / "validation.json", {"status": "passed", "paradigms": len(rows),
                                           "paradigms_sha256": digest((base / "paradigms.jsonl").read_bytes())})
    (base / "sources.json").write_text("{}")
    before = (base / "paradigms.jsonl").read_bytes()
    derived = tmp_path / "derived"
    records, result = backfill_cohort_criteria(base, derived, config, Client, workers=1, notify=lambda m: None)
    assert (base / "paradigms.jsonl").read_bytes() == before
    assert [{k: v for k, v in r.items() if not k.startswith("key_criteria")} for r in records] == rows
    assert result["status"] == "passed" and result["paradigms_sha256"] == digest((derived / "paradigms.jsonl").read_bytes())
    assert read_json(derived / "status.json")["status"] == "complete" and (derived / "sources.json").exists()
    frame = load_guideline_catalog(derived / "paradigms.jsonl")
    assert frame.iloc[0]["cohort_key_criteria"] == records[0]["key_criteria"]
    attempts = sorted(p.name for p in (derived / "checkpoints").glob("*/attempt-*.json"))
    backfill_cohort_criteria(base, derived, config, Client, workers=1, notify=lambda m: None)
    assert sorted(p.name for p in (derived / "checkpoints").glob("*/attempt-*.json")) == attempts
    with pytest.raises(ValueError, match="new directory"):
        backfill_cohort_criteria(base, base, config, Client, workers=1, notify=lambda m: None)
    changed = rows_of(derived / "paradigms.jsonl")
    changed[0]["name"] = "Changed"
    write_rows(derived / "paradigms.jsonl", changed)
    with pytest.raises(ValueError, match="differs from its base"):
        audit_backfill(derived)
