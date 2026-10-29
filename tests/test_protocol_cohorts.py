"""Full-protocol cohort extraction prompt and response contract."""

import json

import pytest

from matchminer_ai.trials.cohort_schema import (
    PROTOCOL_COHORT_JSON_SCHEMA, PROTOCOL_COHORT_SCHEMA_VERSION, build_protocol_cohort_messages,
    parse_cohort_response,
)

NO = {"cancer_treatment_trial": "no"}
SOLE = {"cancer_treatment_trial": "yes", "multiple_cohorts": "no",
        "eligibility_criteria": "Inclusion criteria:\n1. Adults.\nExclusion criteria:\n1. Active infection."}
MULTI = {"cancer_treatment_trial": "yes", "multiple_cohorts": "yes", "cohorts": [
    {"cohort_name": "Part A", "cohort_specific_eligibility_criteria": "Inclusion criteria:\n1. Any solid tumor."},
    {"cohort_name": "Part B", "cohort_specific_eligibility_criteria": "Inclusion criteria:\n1. Measurable melanoma."},
], "common_eligibility_criteria": "Exclusion criteria:\n1. Active infection."}


def parse(value):
    return parse_cohort_response(json.dumps(value), schema_version=PROTOCOL_COHORT_SCHEMA_VERSION)


@pytest.mark.parametrize("value", [NO, SOLE, MULTI])
def test_protocol_schema_accepts_its_three_terminal_states(value):
    assert parse(value) == value


@pytest.mark.parametrize("criteria", [None, "", "   ", 5])
def test_single_cohort_protocol_requires_its_criteria(criteria):
    value = {"cancer_treatment_trial": "yes", "multiple_cohorts": "no"}
    if criteria is not None:
        value["eligibility_criteria"] = criteria
    with pytest.raises(ValueError):
        parse(value)


def test_registry_schema_still_rejects_single_cohort_criteria_field():
    with pytest.raises(ValueError):
        parse_cohort_response(json.dumps(SOLE))


def test_trailing_characters_after_the_object_fail():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_cohort_response(json.dumps(MULTI) + "\n}", schema_version=PROTOCOL_COHORT_SCHEMA_VERSION)


def test_structured_output_schema_matches_parser_field_order():
    # Generation follows schema property order; the parser requires this order.
    shapes = [list(variant["properties"]) for variant in PROTOCOL_COHORT_JSON_SCHEMA["anyOf"]]
    assert shapes == [list(NO), list(SOLE), list(MULTI)]
    assert all(v["required"] == list(v["properties"]) and v["additionalProperties"] is False
               for v in PROTOCOL_COHORT_JSON_SCHEMA["anyOf"])


def test_protocol_prompt_keeps_source_literal_and_its_examples_parse():
    rendered = build_protocol_cohort_messages("Section {5.1} criteria")[0]["content"]
    assert "Section {5.1} criteria" in rendered and "{protocol_text}" not in rendered
    starts = [i for i in range(len(rendered)) if rendered.startswith('{\n  "cancer_treatment_trial"', i)]
    examples = [parse_cohort_response(rendered[s:rendered.index("\n}", s) + 2],
                                      schema_version=PROTOCOL_COHORT_SCHEMA_VERSION) for s in starts]
    assert [e["multiple_cohorts"] for e in examples] == ["no", "yes"]


# Public function: extract_protocol_cohorts


from copy import deepcopy
from pathlib import Path

import pandas as pd

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.trials import extract_protocol_cohorts
from matchminer_ai.trials.cohorts import COHORT_COLUMNS


@pytest.fixture
def backend(monkeypatch):
    config = load_default_preset()
    config.remote["enabled"] = True
    calls, answers = [], []

    class Backend:
        def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
            calls.append((prompt_list, deepcopy(llm_config)))
            batch = answers.pop(0)
            assert len(batch) == len(prompt_list)
            return LLMGenerationResult(
                final_outputs=[json.dumps(x) if isinstance(x, dict) else x for x in batch],
                finish_reasons=["stop"] * len(batch), reasoning_outputs=["r"] * len(batch),
                raw_outputs=[], model_metadata={"model_sha": "test"},
            )

    monkeypatch.setattr("matchminer_ai.trials.protocol_cohorts.get_llm_backend", lambda _: Backend())
    return config, calls, answers


def protocol_files(tmp_path, texts):
    rows = []
    for trial_id, text in texts.items():
        path = tmp_path / f"{trial_id}.txt"
        path.write_text(text, encoding="utf-8")
        rows.append({"trial_id": trial_id, "protocol_path": str(path)})
    return pd.DataFrame(rows)


def prompt_text(prompt):
    return getattr(prompt, "prompt_text", None) or json.dumps(prompt.messages)


def test_remote_requests_use_the_schema_and_return_cohort_rows(tmp_path, backend):
    config, calls, answers = backend
    protocols = protocol_files(tmp_path, {"T1": "Protocol one {5.1} text", "T2": "Protocol two text"})
    answers.append([SOLE, MULTI])
    result, metadata = extract_protocol_cohorts(protocols, config=config, return_metadata=True)
    request = calls[0][1]["remote"]["request_params"]["response_format"]
    assert request["type"] == "json_schema" and request["json_schema"]["schema"] == PROTOCOL_COHORT_JSON_SCHEMA
    assert list(result.columns) == COHORT_COLUMNS
    assert result.cohort_trial_id.tolist() == ["T1-cohort-0", "T2-cohort-0", "T2-cohort-1"]
    assert result.cohort_criteria_source.eq("protocol").all()
    assert result.iloc[0].cohort_eligibility_criteria == SOLE["eligibility_criteria"]
    assert pd.isna(result.iloc[0].cohort_specific_eligibility_criteria)
    assert result.iloc[2].cohort_eligibility_criteria.endswith(MULTI["common_eligibility_criteria"])
    assert result.attrs["schema_version"] == PROTOCOL_COHORT_SCHEMA_VERSION and metadata["structured_outputs"]
    assert metadata["source_documents"]["T1"]["text_path"] == str(tmp_path / "T1.txt")


def test_only_failed_trials_retry_with_the_full_document(tmp_path, backend):
    config, calls, answers = backend
    protocols = protocol_files(tmp_path, {"T1": "First protocol", "T2": "Second protocol body"})
    answers.extend([[SOLE, json.dumps(MULTI) + "\n}"], [MULTI]])
    result, metadata = extract_protocol_cohorts(protocols, config=config, return_metadata=True)
    assert len(calls[1][0]) == 1
    retried = calls[1][0][0]
    assert "Second protocol body" in prompt_text(retried) and "failed validation" in prompt_text(retried)
    assert [r["attempt"] for r in metadata["trial_attempts"]["T2"]] == [1, 2]
    assert metadata["trial_errors"] == {} and set(result.trial_id) == {"T1", "T2"}


def test_rejected_and_exhausted_trials_yield_no_rows(tmp_path, backend):
    config, calls, answers = backend
    config.protocol_cohort_extraction["max_attempts"] = 1
    protocols = protocol_files(tmp_path, {"T1": "Supportive care study", "T2": "Broken answer study"})
    answers.append([NO, "not json"])
    result = extract_protocol_cohorts(protocols, config=config)
    assert result.empty and list(result.columns) == COHORT_COLUMNS
    assert set(result.attrs["trial_rejections"]) == {"T1"} and set(result.attrs["trial_errors"]) == {"T2"}


def test_pdfs_are_converted_locally_and_text_can_be_kept(tmp_path, backend, monkeypatch):
    config, calls, answers = backend
    pdf = tmp_path / "protocol.pdf"
    pdf.write_bytes(b"%PDF-1.4 fabricated")

    def fake_ocr(path, output_path, **kwargs):
        Path(output_path).write_text("Converted protocol text", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr("matchminer_ai.trials.protocol_cohorts.ocr_pdf", fake_ocr)
    answers.append([SOLE])
    kept = tmp_path / "text"
    _, metadata = extract_protocol_cohorts(pd.DataFrame([{"trial_id": "T1", "protocol_path": str(pdf)}]),
                                           config=config, text_output_dir=kept, return_metadata=True)
    source = metadata["source_documents"]["T1"]
    assert Path(source["text_path"]).parent == kept and Path(source["text_path"]).exists()
    assert "Converted protocol text" in prompt_text(calls[0][0][0])


@pytest.mark.parametrize("change,error", [
    (lambda df, p: df.drop(columns="protocol_path"), ValueError),
    (lambda df, p: pd.concat([df, df]), ValueError),
    (lambda df, p: df.assign(protocol_path=str(p.with_suffix(".docx"))), ValueError),
    (lambda df, p: df.assign(protocol_path=str(p.with_name("missing.txt"))), FileNotFoundError),
])
def test_invalid_inputs_fail_before_inference(tmp_path, backend, change, error):
    config, calls, _ = backend
    protocols = protocol_files(tmp_path, {"T1": "Protocol"})
    with pytest.raises(error):
        extract_protocol_cohorts(change(protocols, tmp_path / "T1.txt"), config=config)
    assert calls == []


def test_oversized_documents_fail_rather_than_truncate(tmp_path, backend):
    config, calls, _ = backend
    config.protocol_cohort_extraction["max_document_characters"] = 10
    with pytest.raises(ValueError, match="not truncated"):
        extract_protocol_cohorts(protocol_files(tmp_path, {"T1": "x" * 11}), config=config)
    assert calls == []


def test_local_generation_is_not_grammar_constrained(tmp_path, backend):
    config, calls, answers = backend
    config.remote["enabled"] = False
    answers.append([SOLE])
    result = extract_protocol_cohorts(protocol_files(tmp_path, {"T1": "Protocol"}), config=config)
    runtime = calls[0][1]
    assert runtime["backend_mode"] == "local" and "response_format" not in runtime["sampling_params"]
    assert result.attrs["structured_outputs"] is False
