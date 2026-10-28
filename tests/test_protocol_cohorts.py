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
