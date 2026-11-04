"""Separating a cohort's boilerplate criteria from the rest of its criteria (remove_boilerplate_criteria)."""

import json

import pandas as pd
import pytest

from conftest import prompt_text
from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.trials import remove_boilerplate_criteria
from matchminer_ai.trials.cohort_boilerplate import (
    BOILERPLATE_COLUMN, DEBOILERPLATE_JSON_SCHEMA, DEBOILERPLATE_PROMPT, build_deboilerplate_messages, gather_criteria,
    parse_deboilerplated, split_criteria,
)

CRITERIA = ("Inclusion criteria:\nTrial one adults with {x} cancer.\n  a. Prior platinum therapy.\nECOG 0-1.\n\n"
            "Exclusion criteria:\nAny of the following:\n  - Active hepatitis B.\n  - Prior anti-PD-1 therapy.")


def rows(trial_id, texts):
    return [dict(cohort_trial_id=f"{trial_id}-cohort-{i}", trial_id=trial_id, cohort_number=i, cohort_name=f"Arm {chr(65 + i)}",
                 cohort_eligibility_criteria=text, cohort_criteria_source="protocol", merged_from=f"[{i}]")
            for i, text in enumerate(texts)]


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    return config


def test_items_are_lines_and_are_gathered_unchanged_in_order():
    items = split_criteria(CRITERIA.replace("Exclusion criteria:", "EXCLUSION CRITERIA"))
    assert items == [("Inclusion criteria:", "Trial one adults with {x} cancer."), ("Inclusion criteria:", "  a. Prior platinum therapy."),
                     ("Inclusion criteria:", "ECOG 0-1."), ("Exclusion criteria:", "Any of the following:"),
                     ("Exclusion criteria:", "  - Active hepatitis B."), ("Exclusion criteria:", "  - Prior anti-PD-1 therapy.")]
    assert gather_criteria(items, [False, False, True, False, True, False]) == (
        "Inclusion criteria:\nTrial one adults with {x} cancer.\n  a. Prior platinum therapy.\n"
        "Exclusion criteria:\nAny of the following:\n  - Prior anti-PD-1 therapy.",
        "Inclusion criteria:\nECOG 0-1.\nExclusion criteria:\n  - Active hepatitis B.")
    assert gather_criteria([("", "Adults."), ("", "Consent.")], [False, True]) == ("Adults.", "Consent.")
    assert gather_criteria(items[:2], [False, False])[1] == ""  # a list without items is empty


def test_prompt_shows_the_numbered_cohort_and_ends_with_the_item():
    prompt = load_prompt_text(DEBOILERPLATE_PROMPT)
    items = split_criteria(CRITERIA)
    first, fifth = (build_deboilerplate_messages("Arm {1}", items, n, prompt)[0]["content"] for n in (1, 5))
    assert "<cohort_name>\nArm {1}\n</cohort_name>" in first and "<trial_document>" not in first
    assert ("Inclusion criteria:\n[1] Trial one adults with {x} cancer.\n[2]   a. Prior platinum therapy.\n[3] ECOG 0-1.\n"
            "Exclusion criteria:\n[4] Any of the following:\n[5]   - Active hepatitis B.") in first
    assert fifth.endswith("ITEM TO CLASSIFY\n[5] - Active hepatitis B.")
    shared = first.rsplit("ITEM TO CLASSIFY", 1)[0]
    assert fifth.startswith(shared)  # the cohort's items share everything before the item (prefix caching)
    for rule in ("informed consent", "measurable or evaluable disease or lesions",
                 "to have tumor tissue available or tumor amenable to biopsy, or to undergo a biopsy",
                 "Life expectancy or prognosis requirements", "Performance status requirements, such as ECOG",
                 "adequate organ or bone marrow function", "brain or other central nervous system metastases",
                 "toxicities of earlier treatment", "medical conditions other than cancer",
                 "an exclusion for having received or planning to receive a live or other vaccine",
                 "a history of other cancers, unrelated to the cancer this cohort is for",
                 "participating in another clinical trial at the same time",
                 "Every criterion about contraception, pregnancy, or lactation",
                 "central to what the trial studies or whom it targets", "a hemoglobin requirement in a trial for anemia",
                 "brain metastasis criteria in a trial for primary brain tumors",
                 "A biomarker or other test result that a criterion requires is not boilerplate",
                 "An item that combines boilerplate with any requirement that is not boilerplate is not boilerplate",
                 "is boilerplate only when every item it introduces is boilerplate",
                 "When you are unsure whether the item is boilerplate, treat it as not boilerplate"):
        assert rule in first, rule
    assert parse_deboilerplated('{"boilerplate": true}') is True and parse_deboilerplated('{"boilerplate": false}') is False
    assert DEBOILERPLATE_JSON_SCHEMA["required"] == ["boilerplate"]
    for bad in ('{"boilerplate": "yes"}', '{"boilerplate": true, "why": ""}', '{}'):
        with pytest.raises(ValueError):
            parse_deboilerplated(bad)
    with pytest.raises(ValueError):
        parse_deboilerplated('{"boilerplate": true}', "length")


def test_each_item_is_classified_and_the_criteria_are_sorted_in_code(config, cohort_backend):
    frame = pd.DataFrame(rows("T1", [CRITERIA, "Inclusion criteria:\nTrial one children.\nConsent."]))
    cohort_backend.script["Trial one adults"] = {"deboilerplate": {"ECOG 0-1.": [{"boilerplate": True}],
                                                                   "- Active hepatitis B.": [{"boilerplate": True}]}}
    cohort_backend.script["Trial one children"] = {"deboilerplate": {"Consent.": [{"boilerplate": True}]}}
    result, metadata = remove_boilerplate_criteria(frame, config=config, section="protocol_cohort_extraction",
                                                   return_metadata=True)
    assert len(cohort_backend.calls) == 1 and len(cohort_backend.calls[0][0]) == 8  # one request per item
    request = cohort_backend.calls[0][1]["remote"]["request_params"]["response_format"]["json_schema"]["schema"]
    assert request == DEBOILERPLATE_JSON_SCHEMA
    assert result.cohort_eligibility_criteria.tolist() == [
        "Inclusion criteria:\nTrial one adults with {x} cancer.\n  a. Prior platinum therapy.\n"
        "Exclusion criteria:\nAny of the following:\n  - Prior anti-PD-1 therapy.", "Inclusion criteria:\nTrial one children."]
    assert result[BOILERPLATE_COLUMN].tolist() == ["Inclusion criteria:\nECOG 0-1.\nExclusion criteria:\n  - Active hepatitis B.",
                                                   "Inclusion criteria:\nConsent."]
    assert result.drop(columns=["cohort_eligibility_criteria", BOILERPLATE_COLUMN]).equals(
        frame.drop(columns="cohort_eligibility_criteria"))
    saved = metadata["trial_cohorts"]["T1"][0]
    assert [(i["number"], i["text"], i["boilerplate"]) for i in saved["items"]][2] == (3, "ECOG 0-1.", True)
    assert saved["items"][0]["attempts"][0]["reasoning"] == "test reasoning"
    assert metadata["items"] == 8 and not metadata["deboilerplate_errors"] and not metadata["unclassified_items"]


def test_unclassified_items_stay_and_an_all_boilerplate_cohort_keeps_its_criteria(config, cohort_backend):
    config.trial["cohorts"] = {**config.trial["cohorts"], "max_attempts": 2}
    frame = pd.DataFrame(rows("T1", ["Trial one adults.\nECOG 0-1.\nConsent.", "Trial one: ECOG 0-1.\nConsent."]))
    cohort_backend.script["Trial one adults"] = {"deboilerplate": {"ECOG 0-1.": ["not json"], "Consent.": [{"boilerplate": True}]}}
    cohort_backend.script["Trial one: ECOG"] = {"deboilerplate": {"Trial one: ECOG 0-1.": [{"boilerplate": True}],
                                                                  "Consent.": [{"boilerplate": True}]}}
    result, metadata = remove_boilerplate_criteria(frame, config=config, return_metadata=True)
    assert len(cohort_backend.calls) == 2 and "failed validation" in prompt_text(cohort_backend.calls[1][0][0])
    assert result.cohort_eligibility_criteria.tolist() == ["Trial one adults.\nECOG 0-1.", "Trial one: ECOG 0-1.\nConsent."]
    assert result[BOILERPLATE_COLUMN].tolist() == ["Consent.", None]
    assert list(metadata["unclassified_items"]["T1-cohort-0"]) == [2]
    assert list(metadata["deboilerplate_errors"]) == ["T1-cohort-1"]


def test_saved_classifications_are_reused_only_for_unchanged_cohorts(tmp_path, config, cohort_backend):
    frame = pd.DataFrame(rows("T1", ["Trial one adults.\nConsent.", "Trial one children.\nConsent."]))
    cohort_backend.script["Trial one"] = {"deboilerplate": {"Consent.": [{"boilerplate": True}]}}
    remove_boilerplate_criteria(frame, config=config, output_dir=tmp_path)
    saved = json.loads((tmp_path / "T1/deboilerplate.json").read_text())
    assert [(c["eligibility_criteria"], c["boilerplate_criteria"]) for c in saved["cohorts"]] == [
        ("Trial one adults.", "Consent."), ("Trial one children.", "Consent.")]
    assert saved["cohorts"][0]["items"][1]["attempts"][0]["reasoning"] == "test reasoning"
    again, metadata = remove_boilerplate_criteria(frame, config=config, output_dir=tmp_path, return_metadata=True)
    assert len(cohort_backend.calls) == 1 and metadata["skipped_completed"] == 2 and metadata["items_reused"] == 4
    assert again[BOILERPLATE_COLUMN].tolist() == ["Consent.", "Consent."]
    frame.loc[1, "cohort_eligibility_criteria"] = "Trial one children over 12.\nConsent."
    changed, metadata = remove_boilerplate_criteria(frame, config=config, output_dir=tmp_path, return_metadata=True)
    assert len(cohort_backend.calls) == 2 and len(cohort_backend.calls[1][0]) == 2 and metadata["skipped_completed"] == 1
    assert changed.cohort_eligibility_criteria.tolist() == ["Trial one adults.", "Trial one children over 12."]
