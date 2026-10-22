import json
from pathlib import Path

import pytest

from matchminer_ai.trials.space_refinement import (
    FIELDS, build_refinement_messages, format_space, load_refinement_prompts,
    parse_refinement_response, parse_space_text, refinement_retry_messages,
)

PROMPT_DIR = Path(__file__).parents[1] / "src/matchminer_ai/prompts"
SPACE = dict(age="18 years or older", sex="no restriction", cancer_primary_site="lung",
             histology="adenocarcinoma", disease_state="advanced/metastatic",
             prior_treatment_requirements="prior platinum chemotherapy", biomarker_requirements="NA")


def test_full_verbatim_primer_precedes_source_and_space():
    prompts = load_refinement_prompts(PROMPT_DIR)
    a = build_refinement_messages("SOURCE A", SPACE, prompts)
    b = build_refinement_messages("SOURCE A", {**SPACE, "histology": "squamous cell carcinoma"}, prompts)
    text = a[1]["content"]
    assert text.count(prompts["primer"]) == 1
    assert text.index(prompts["primer"]) < text.index("REVIEW DIRECTIVES:") < text.index("TRIAL DOCUMENT:\nSOURCE A") < text.index("CURRENT COMPLETED SPACE:")
    assert text.endswith(json.dumps(SPACE, ensure_ascii=False))
    assert text.split("CURRENT COMPLETED SPACE:")[0] == b[1]["content"].split("CURRENT COMPLETED SPACE:")[0]
    assert "Reasoning: high" not in text


def test_text_round_trip_and_single_space_boundary():
    assert parse_space_text("17. " + format_space(SPACE)) == SPACE
    with pytest.raises(ValueError):
        parse_space_text(format_space(SPACE) + " " + format_space(SPACE))


def test_unchanged_must_retain_values():
    payload = dict(status="unchanged", spaces=[SPACE], explanation="All source criteria retained.")
    assert parse_refinement_response(json.dumps(payload), SPACE) == payload
    payload["spaces"] = [{**SPACE, "prior_treatment_requirements": "NA"}]
    with pytest.raises(ValueError, match="preserve"):
        parse_refinement_response(json.dumps(payload), SPACE)


def test_supported_split_and_markdown_wrapper():
    original = {**SPACE, "histology": "adenocarcinoma or squamous cell carcinoma"}
    payload = dict(status="refined", spaces=[SPACE, {**SPACE, "histology": "squamous cell carcinoma"}], explanation="Split combined histologies, preserving shared requirements.")
    parsed = parse_refinement_response("```json\n" + json.dumps(payload) + "\n```", original)
    assert parsed == payload


def test_unresolved_case_cannot_silently_replace_or_remove_population():
    payload = dict(status="needs_review", spaces=[], explanation="Conflicting source-defined treatment conditions.")
    assert parse_refinement_response(json.dumps(payload), SPACE) == payload
    payload["status"] = "refined"
    with pytest.raises(ValueError, match="silently remove"):
        parse_refinement_response(json.dumps(payload), SPACE)


@pytest.mark.parametrize("change", [
    {"spaces": [{**SPACE, "sex": "any"}]},
    {"spaces": [{**SPACE, "histology": "x\ny"}]},
    {"spaces": [{**SPACE, "boilerplate": "infection"}]},
    {"spaces": [{**SPACE, "age": ""}]},
    {"status": []},
    {"spaces": [SPACE, SPACE]},
])
def test_malformed_or_duplicated_fields_rejected(change):
    payload = dict(status="refined", spaces=[{**SPACE, "age": "21 years or older"}], explanation="Correction.")
    with pytest.raises(ValueError):
        parse_refinement_response(json.dumps({**payload, **change}), SPACE)


def test_truncation_and_duplicate_keys_rejected():
    payload = dict(status="unchanged", spaces=[SPACE], explanation="No correction.")
    with pytest.raises(ValueError, match="Incomplete"):
        parse_refinement_response(json.dumps(payload), SPACE, "length")
    text = json.dumps(payload).replace('"status": "unchanged"', '"status": "refined", "status": "unchanged"')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        parse_refinement_response(text, SPACE)


def test_retry_carries_specific_error_without_altering_prefix():
    messages = build_refinement_messages("SOURCE", SPACE, load_refinement_prompts(PROMPT_DIR))
    retry = refinement_retry_messages(messages, "bad json", "missing biomarker_requirements")
    assert retry[:2] == messages
    assert retry[-2]["content"] == "bad json"
    assert "missing biomarker_requirements" in retry[-1]["content"]
