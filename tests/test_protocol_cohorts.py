"""Full-protocol cohort extraction (extract_protocol_cohorts) through the shared three-step workflow."""

import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import prompt_text
from matchminer_ai.config import load_default_preset
from matchminer_ai.trials import extract_protocol_cohorts
from matchminer_ai.trials.cohort_workflow import (
    COHORT_COLUMNS, JSON_SCHEMAS, SCHEMA_VERSION, build_messages, load_prompts,
)

YES, NO = {"cancer_treatment_trial": "yes"}, {"cancer_treatment_trial": "no"}
SOLE = "Inclusion criteria:\n1. Adults.\nExclusion criteria:\n1. Active infection."


def one_cohort(criteria=SOLE):
    return {"screen": [YES], "list": [{"cohorts": ["All participants"]}],
            "criteria": {"All participants": [{"eligibility_criteria": criteria}]}}


def two_parts():
    return {"screen": [YES], "list": [{"cohorts": ["Part A", "Part B"]}], "criteria": {
        "Part A": [{"eligibility_criteria": "Inclusion criteria:\n1. Any solid tumor."}],
        "Part B": [{"eligibility_criteria": "Inclusion criteria:\n1. Measurable melanoma."}]}}


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    return config


def protocol_files(tmp_path, texts):
    rows = []
    for trial_id, text in texts.items():
        path = tmp_path / f"{trial_id}.txt"
        path.write_text(text, encoding="utf-8")
        rows.append({"trial_id": trial_id, "protocol_path": str(path)})
    return pd.DataFrame(rows)


def test_protocols_use_the_ctgov_prompts_and_return_cohort_rows(tmp_path, config, cohort_backend):
    cohort_backend.script.update({"Protocol one": one_cohort(), "Protocol two": two_parts()})
    protocols = protocol_files(tmp_path, {"T1": "Protocol one {5.1} text", "T2": "Protocol two text"})
    result, metadata = extract_protocol_cohorts(protocols, config=config, return_metadata=True)
    calls = cohort_backend.calls
    assert [c[1]["remote"]["request_params"]["response_format"]["json_schema"]["schema"] for c in calls] == [
        JSON_SCHEMAS[s] for s in ("screen", "list", "criteria")]
    # The same prompt text as ctgov listings, with the whole protocol as the document.
    assert prompt_text(calls[0][0][0]) == build_messages("screen", "Protocol one {5.1} text", load_prompts())[0]["content"]
    assert list(result.columns) == COHORT_COLUMNS
    assert result.cohort_trial_id.tolist() == ["T1-cohort-0", "T2-cohort-0", "T2-cohort-1"]
    assert result.cohort_criteria_source.eq("protocol").all()
    assert result.cohort_eligibility_criteria.iloc[0] == SOLE
    assert result.attrs["schema_version"] == SCHEMA_VERSION and metadata["structured_outputs"]
    assert metadata["source_documents"]["T1"]["text_path"] == str(tmp_path / "T1.txt")


def test_only_failed_requests_retry_with_the_full_document(tmp_path, config, cohort_backend):
    script = two_parts()
    script["list"] = ['{"cohorts": ["Part A", "Part B"]}\n}', {"cohorts": ["Part A", "Part B"]}]
    cohort_backend.script.update({"First protocol": one_cohort(), "Second protocol body": script})
    protocols = protocol_files(tmp_path, {"T1": "First protocol", "T2": "Second protocol body"})
    result, metadata = extract_protocol_cohorts(protocols, config=config, return_metadata=True)
    retried = [p for prompts, _ in cohort_backend.calls for p in prompts if "failed validation" in prompt_text(p)]
    assert len(retried) == 1 and "Second protocol body" in prompt_text(retried[0])
    assert [a["attempt"] for a in metadata["trial_attempts"]["T2"]["list"]] == [1, 2]
    assert metadata["trial_errors"] == {} and set(result.trial_id) == {"T1", "T2"}


def test_rejected_and_exhausted_trials_yield_no_rows(tmp_path, config, cohort_backend):
    config.protocol_cohort_extraction["max_attempts"] = 1
    cohort_backend.script.update({"Supportive care": {"screen": [NO]}, "Broken answer": {"screen": ["not json"]}})
    protocols = protocol_files(tmp_path, {"T1": "Supportive care study", "T2": "Broken answer study"})
    result = extract_protocol_cohorts(protocols, config=config)
    assert result.empty and list(result.columns) == COHORT_COLUMNS
    assert set(result.attrs["trial_rejections"]) == {"T1"} and set(result.attrs["trial_errors"]) == {"T2"}


def test_pdfs_are_converted_locally_and_text_can_be_kept(tmp_path, config, cohort_backend, monkeypatch):
    pdf = tmp_path / "protocol.pdf"
    pdf.write_bytes(b"%PDF-1.4 fabricated")

    def fake_ocr(path, output_path, **kwargs):
        Path(output_path).write_text("Converted protocol text", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("matchminer_ai.trials.protocol_cohorts.ocr_pdf", fake_ocr)
    cohort_backend.script["Converted protocol"] = one_cohort()
    kept = tmp_path / "text"
    _, metadata = extract_protocol_cohorts(pd.DataFrame([{"trial_id": "T1", "protocol_path": str(pdf)}]),
                                           config=config, text_output_dir=kept, return_metadata=True)
    source = metadata["source_documents"]["T1"]
    assert Path(source["text_path"]).parent == kept and Path(source["text_path"]).exists()
    assert "Converted protocol text" in prompt_text(cohort_backend.calls[0][0][0])


@pytest.mark.parametrize("change,error", [
    (lambda df, p: df.drop(columns="protocol_path"), ValueError),
    (lambda df, p: pd.concat([df, df]), ValueError),
    (lambda df, p: df.assign(protocol_path=str(p.with_suffix(".docx"))), ValueError),
    (lambda df, p: df.assign(protocol_path=str(p.with_name("missing.txt"))), FileNotFoundError),
])
def test_invalid_inputs_fail_before_inference(tmp_path, config, cohort_backend, change, error):
    protocols = protocol_files(tmp_path, {"T1": "Protocol"})
    with pytest.raises(error):
        extract_protocol_cohorts(change(protocols, tmp_path / "T1.txt"), config=config)
    assert cohort_backend.calls == []


def test_oversized_documents_fail_rather_than_truncate(tmp_path, config, cohort_backend):
    config.protocol_cohort_extraction["max_document_characters"] = 10
    with pytest.raises(ValueError, match="not truncated"):
        extract_protocol_cohorts(protocol_files(tmp_path, {"T1": "x" * 11}), config=config)
    assert cohort_backend.calls == []


def test_local_generation_is_not_grammar_constrained(tmp_path, config, cohort_backend):
    config.remote["enabled"] = False
    cohort_backend.script["Protocol"] = one_cohort()
    result = extract_protocol_cohorts(protocol_files(tmp_path, {"T1": "Protocol"}), config=config)
    runtime = cohort_backend.calls[0][1]
    assert runtime["backend_mode"] == "local" and "response_format" not in runtime["sampling_params"]
    assert result.attrs["structured_outputs"] is False


def test_saved_decisions_are_reused_only_for_unchanged_documents(tmp_path, config, cohort_backend):
    config.protocol_cohort_extraction["max_attempts"] = 1
    protocols = pd.DataFrame([{"trial_id": t, "protocol_path": str(tmp_path / f"{t}.md")} for t in ("T1", "T2", "T3")])
    for trial_id in ("T1", "T2", "T3"):
        (tmp_path / f"{trial_id}.md").write_text(f"# Protocol {trial_id}", encoding="utf-8")
    out = tmp_path / "decisions"
    cohort_backend.script.update({"Protocol T1": one_cohort(), "Protocol T2": {"screen": [NO]},
                                  "Protocol T3": {"screen": ["not json"]}})
    first = extract_protocol_cohorts(protocols, config=config, output_dir=out)
    saved = json.loads((out / "T1" / "extraction.json").read_text())
    assert saved["decision"]["cohorts"] == [{"cohort_name": "All participants", "eligibility_criteria": SOLE}]
    assert saved["schema_version"] == SCHEMA_VERSION and saved["cohort_names"] == ["All participants"]
    assert saved["source_document"]["protocol_path"] == str(tmp_path / "T1.md")
    assert saved["attempts"]["screen"][0]["attempt"] == 1
    assert json.loads((out / "T1" / "reasoning.json").read_text())["criteria"] == {"0": "test reasoning"}
    assert not (out / "T3").exists()
    assert set(first.attrs["trial_rejections"]) == {"T2"} and set(first.attrs["trial_errors"]) == {"T3"}

    # Only the failed trial and the edited document are sent again.
    (tmp_path / "T2.md").write_text("# Protocol T2, amended", encoding="utf-8")
    del cohort_backend.script["Protocol T2"]
    cohort_backend.script.update({"Protocol T2, amended": two_parts(), "Protocol T3": one_cohort()})
    cohort_backend.calls.clear()
    again, metadata = extract_protocol_cohorts(protocols, config=config, output_dir=out, return_metadata=True)
    screened = [prompt_text(p) for p in cohort_backend.calls[0][0]]
    assert len(screened) == 2 and not any("# Protocol T1" in text for text in screened)
    assert metadata["skipped_completed"] == 1 and metadata["trial_errors"] == {}
    assert again.cohort_trial_id.tolist() == ["T1-cohort-0", "T2-cohort-0", "T2-cohort-1", "T3-cohort-0"]


def test_chunked_requests_retry_failures_after_later_chunks(tmp_path, config, cohort_backend):
    config.protocol_cohort_extraction.update(chunk_size=1, chunks_in_flight=1)
    script = one_cohort()
    script["screen"] = ["not json", YES]
    cohort_backend.script.update({"First protocol": script, "Second protocol": two_parts()})
    protocols = protocol_files(tmp_path, {"T1": "First protocol", "T2": "Second protocol"})
    result, metadata = extract_protocol_cohorts(protocols, config=config, return_metadata=True)
    assert all(len(prompts) == 1 for prompts, _ in cohort_backend.calls)
    assert [a["attempt"] for a in metadata["trial_attempts"]["T1"]["screen"]] == [1, 2]
    assert result.cohort_trial_id.tolist() == ["T1-cohort-0", "T2-cohort-0", "T2-cohort-1"]
