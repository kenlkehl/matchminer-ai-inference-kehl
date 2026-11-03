"""Historical single-request cohort contract (cohort_schema.parse_cohort_response), kept for read-only audits.

Production extraction uses the three-step workflow (test_cohort_workflow.py, test_protocol_cohorts.py)."""

import json

import pytest

from matchminer_ai.trials.cohort_schema import parse_cohort_response

NO = {"cancer_treatment_trial": "no"}
SINGLE = {"cancer_treatment_trial": "yes", "multiple_cohorts": "no",
          "eligibility_criteria": "Inclusion criteria:\nAdults with cancer.\nExclusion criteria:\nActive infection."}
MULTI = {"cancer_treatment_trial": "yes", "multiple_cohorts": "yes", "cohorts": [
    {"cohort_name": "Cohort A", "cohort_specific_eligibility_criteria": "Inclusion: prior therapy; GFR >= 40 mL/min."},
    {"cohort_name": "Cohort B", "cohort_specific_eligibility_criteria": "Inclusion: untreated; GFR >= 60 mL/min."},
], "common_eligibility_criteria": "Inclusion: adult cancer. Exclusion: infection."}
SPECIFIC = [c["cohort_specific_eligibility_criteria"] for c in MULTI["cohorts"]]


@pytest.mark.parametrize("value", [NO, SINGLE, MULTI])
@pytest.mark.parametrize("language,newline", [
    (None, "\n"), ("json", "\n"), ("JSON", "\r\n"), ("", "\n"),
])
def test_three_terminal_json_states(value, language, newline):
    text = json.dumps(value)
    if language is not None:
        text = f" \n```{language}  {newline}{text}{newline}```\t\n"
    assert parse_cohort_response(text) == value


@pytest.mark.parametrize("value", [
    {}, [], {"cancer_treatment_trial": False},
    {**NO, "multiple_cohorts": "no"}, {**NO, "rationale": "not treatment"},
    {**NO, "common_eligibility_criteria": ""},
    {"cancer_treatment_trial": "yes"}, {**SINGLE, "cohorts": []},
    {**SINGLE, "common_eligibility_criteria": ""},
    {**SINGLE, "multiple_cohorts": False},
    {**MULTI, "cohorts": []}, {**MULTI, "cohorts": ["one"]},
    {**MULTI, "cohorts": ["one", " "]}, {**MULTI, "cohorts": ["one", 2]},
    {**MULTI, "cohorts": ["one", " ONE "]},
    {k: v for k, v in MULTI.items() if k != "common_eligibility_criteria"},
    *[{**MULTI, "common_eligibility_criteria": v} for v in (None, False, [], {}, " \n")],
    {"multiple_cohorts": "no", "cancer_treatment_trial": "yes"},
    {**SINGLE, "multiple_cohorts": "yes", "common_eligibility_criteria": "common",
     "cohorts": ["A", "B"]},
])
@pytest.mark.parametrize("fenced", [False, True])
def test_invalid_schema_never_becomes_a_rejection(value, fenced):
    text = json.dumps(value)
    if fenced:
        text = f"```json\n{text}\n```"
    with pytest.raises(ValueError):
        parse_cohort_response(text)


@pytest.mark.parametrize("text", [
    '{"cancer_treatment_trial":"no","cancer_treatment_trial":"no"}',
    '```json\n{"cancer_treatment_trial":"no","cancer_treatment_trial":"no"}\n```',
    '{"cancer_treatment_trial":"no"} extra',
])
def test_duplicate_fields_or_surrounding_text_fail(text):
    with pytest.raises(ValueError):
        parse_cohort_response(text)


@pytest.mark.parametrize("wrapper", [
    "Here is the answer:\n```json\n{payload}\n```",
    "```json\n{payload}\n```\nExplanation.",
    "```json\n{payload}",
    "```javascript\n{payload}\n```",
    "```json\n{payload}\n```\n```json\n{payload}\n```",
    "```json\n```json\n{payload}\n```\n```",
    "```json\n{payload} trailing junk\n```",
    "```json\n{payload},\n```",
])
def test_fences_do_not_salvage_ambiguous_or_malformed_output(wrapper):
    with pytest.raises(ValueError):
        parse_cohort_response(wrapper.format(payload=json.dumps(NO)))


def test_fenced_complete_json_with_truncated_finish_still_fails():
    with pytest.raises(ValueError, match="Incomplete"):
        parse_cohort_response(f"```json\n{json.dumps(NO)}\n```", "length")


@pytest.mark.parametrize("version", ["trial-treatment-cohorts-v1", "trial-treatment-cohorts-v2"])
def test_old_strings_require_explicit_historical_schema_selection(version):
    old = {**MULTI, "cohorts": SPECIFIC}
    if version.endswith("v1"):
        old.pop("common_eligibility_criteria")
    text = json.dumps(old)
    with pytest.raises(ValueError, match="cohort.*object"):
        parse_cohort_response(text)
    assert parse_cohort_response(text, schema_version=version) == old
    with pytest.raises(ValueError, match="Unsupported"):
        parse_cohort_response(text, schema_version="unknown")


@pytest.mark.parametrize("cohort", [
    {}, {"cohort_name": "A"},
    {"cohort_name": " ", "cohort_specific_eligibility_criteria": "Adults."},
    {"cohort_name": "A", "cohort_specific_eligibility_criteria": ""},
    {"cohort_name": 1, "cohort_specific_eligibility_criteria": "Adults."},
    {"cohort_name": "A", "cohort_specific_eligibility_criteria": ["Adults."]},
    {"cohort_name": "A", "cohort_specific_eligibility_criteria": "Adults.", "extra": "x"},
])
def test_named_cohort_fields_are_exact_nonempty_strings(cohort):
    with pytest.raises(ValueError):
        parse_cohort_response(json.dumps({**MULTI, "cohorts": [cohort, MULTI["cohorts"][1]]}))


def test_different_names_do_not_allow_duplicate_eligibility_populations():
    duplicate = {"cohort_name": "Different arm", "cohort_specific_eligibility_criteria": "  " + SPECIFIC[0].upper() + "  "}
    with pytest.raises(ValueError, match="Duplicate cohort eligibility"):
        parse_cohort_response(json.dumps({**MULTI, "cohorts": [MULTI["cohorts"][0], duplicate]}))
