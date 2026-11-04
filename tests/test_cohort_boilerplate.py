"""Separating a cohort's boilerplate criteria from the rest of its criteria (remove_boilerplate_criteria)."""

import json

import pandas as pd
import pytest

from conftest import prompt_text
from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.trials import remove_boilerplate_criteria
from matchminer_ai.trials.cohort_boilerplate import (
    BOILERPLATE_COLUMN, DEBOILERPLATE_JSON_SCHEMA, DEBOILERPLATE_PROMPT, build_deboilerplate_messages, parse_deboilerplated,
)


def rows(trial_id, texts):
    return [dict(cohort_trial_id=f"{trial_id}-cohort-{i}", trial_id=trial_id, cohort_number=i, cohort_name=f"Arm {chr(65 + i)}",
                 cohort_eligibility_criteria=text, cohort_criteria_source="protocol", merged_from=f"[{i}]")
            for i, text in enumerate(texts)]


def answer(criteria, boilerplate):
    return {"eligibility_criteria": criteria, "boilerplate_criteria": boilerplate}


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    return config


def test_prompt_holds_one_cohort_literally_with_the_boilerplate_rules():
    prompt = load_prompt_text(DEBOILERPLATE_PROMPT)
    content = build_deboilerplate_messages("Arm {1}", "Inclusion criteria:\nAdults {x}.", prompt)[0]["content"]
    assert "<cohort_name>\nArm {1}\n</cohort_name>" in content and "Inclusion criteria:\nAdults {x}." in content
    for rule in ("informed consent", "measurable or evaluable disease or lesions",
                 "to have tumor tissue available or tumor amenable to biopsy, or to undergo a biopsy",
                 "Life expectancy or prognosis requirements", "Performance status requirements, such as ECOG",
                 "adequate organ or bone marrow function", "brain or other central nervous system metastases",
                 "toxicities of earlier treatment", "medical conditions other than cancer",
                 "an exclusion for having received or planning to receive a live or other vaccine",
                 "Every criterion about contraception, pregnancy, or lactation",
                 "central to what the trial studies or whom it targets", "a hemoglobin requirement in a trial for anemia",
                 "brain metastasis criteria in a trial for primary brain tumors",
                 "A biomarker or other test result that a criterion requires is not boilerplate",
                 "When you are unsure whether a criterion is boilerplate, treat it as not boilerplate",
                 "Every criterion appears in exactly one of the two lists, copied word for word"):
        assert rule in content, rule
    assert "<trial_document>" not in content  # the cohort's criteria only, not the trial document
    assert parse_deboilerplated(json.dumps(answer("Inclusion criteria:\nAdults.", "Inclusion criteria:\nECOG 0-1."))) == (
        "Inclusion criteria:\nAdults.", "Inclusion criteria:\nECOG 0-1.")
    assert parse_deboilerplated(json.dumps(answer("Adults.", ""))) == ("Adults.", "")  # no boilerplate is allowed
    assert DEBOILERPLATE_JSON_SCHEMA["required"] == ["eligibility_criteria", "boilerplate_criteria"]
    for bad in (answer("", "ECOG 0-1."), {"eligibility_criteria": "Adults."}, answer("Adults.", None)):
        with pytest.raises(ValueError):
            parse_deboilerplated(json.dumps(bad))


def test_each_cohort_gets_its_criteria_and_boilerplate_separately(config, cohort_backend):
    frame = pd.DataFrame(rows("T1", ["Trial one adults; consent.", "Trial one children; consent."]))
    cohort_backend.script["Trial one adults"] = {"deboilerplate": [answer("Trial one adults.", "Consent.")]}
    cohort_backend.script["Trial one children"] = {"deboilerplate": [answer("Trial one children.", "")]}
    result, metadata = remove_boilerplate_criteria(frame, config=config, section="protocol_cohort_extraction",
                                                   return_metadata=True)
    assert len(cohort_backend.calls) == 1 and len(cohort_backend.calls[0][0]) == 2
    request = cohort_backend.calls[0][1]["remote"]["request_params"]["response_format"]["json_schema"]["schema"]
    assert request == DEBOILERPLATE_JSON_SCHEMA
    assert result.cohort_eligibility_criteria.tolist() == ["Trial one adults.", "Trial one children."]
    assert result[BOILERPLATE_COLUMN].tolist() == ["Consent.", ""]
    unchanged = ["cohort_eligibility_criteria", BOILERPLATE_COLUMN]
    assert result.drop(columns=unchanged).equals(frame.drop(columns="cohort_eligibility_criteria"))
    assert metadata["cohort_attempts"]["T1-cohort-0"][0]["reasoning"] == "test reasoning"
    assert [(e["eligibility_criteria"], e["boilerplate_criteria"]) for e in metadata["trial_cohorts"]["T1"]] == [
        ("Trial one adults.", "Consent."), ("Trial one children.", "")]


def test_a_cohort_that_never_validates_keeps_its_criteria(config, cohort_backend):
    config.trial["cohorts"] = {**config.trial["cohorts"], "max_attempts": 2}
    frame = pd.DataFrame(rows("T1", ["Trial one adults; consent."]))
    cohort_backend.script["Trial one"] = {"deboilerplate": [json.dumps(answer("", "Consent."))]}
    result, metadata = remove_boilerplate_criteria(frame, config=config, return_metadata=True)
    assert len(cohort_backend.calls) == 2 and "failed validation" in prompt_text(cohort_backend.calls[1][0][0])
    assert result.cohort_eligibility_criteria.tolist() == ["Trial one adults; consent."]
    assert result[BOILERPLATE_COLUMN].tolist() == [None]
    assert "T1-cohort-0" in metadata["deboilerplate_errors"]


def test_saved_results_are_reused_only_for_unchanged_cohorts(tmp_path, config, cohort_backend):
    frame = pd.DataFrame(rows("T1", ["Trial one adults; consent.", "Trial one children; consent."]))
    cohort_backend.script["Trial one adults"] = {"deboilerplate": [answer("Trial one adults.", "Consent.")]}
    cohort_backend.script["Trial one children"] = {"deboilerplate": [answer("Trial one children.", "Consent.")]}
    remove_boilerplate_criteria(frame, config=config, output_dir=tmp_path)
    saved = json.loads((tmp_path / "T1/deboilerplate.json").read_text())
    assert [(c["eligibility_criteria"], c["boilerplate_criteria"]) for c in saved["cohorts"]] == [
        ("Trial one adults.", "Consent."), ("Trial one children.", "Consent.")]
    assert saved["cohorts"][0]["attempts"][0]["reasoning"] == "test reasoning"
    again, metadata = remove_boilerplate_criteria(frame, config=config, output_dir=tmp_path, return_metadata=True)
    assert len(cohort_backend.calls) == 1 and metadata["skipped_completed"] == 2
    assert again[BOILERPLATE_COLUMN].tolist() == ["Consent.", "Consent."]
    frame.loc[1, "cohort_eligibility_criteria"] = "Trial one children over 12; consent."
    cohort_backend.script["Trial one children"] = {"deboilerplate": [answer("Trial one children over 12.", "Consent.")]}
    changed, metadata = remove_boilerplate_criteria(frame, config=config, output_dir=tmp_path, return_metadata=True)
    assert len(cohort_backend.calls) == 2 and len(cohort_backend.calls[1][0]) == 1 and metadata["skipped_completed"] == 1
    assert changed.cohort_eligibility_criteria.tolist() == ["Trial one adults.", "Trial one children over 12."]
