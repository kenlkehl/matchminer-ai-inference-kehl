"""Three-step cohort extraction: prompts, step contracts, and the registry API (extract_cohorts)."""

import json
import re

import pandas as pd
import pytest

from conftest import prompt_text
from matchminer_ai.config import load_default_preset
from matchminer_ai.trials import extract_cohorts
from matchminer_ai.trials.cohort_workflow import (
    COHORT_COLUMNS, JSON_SCHEMAS, REPRESENTATION_VERSION, SCHEMA_VERSION, STEPS, assemble_decision,
    build_messages, load_prompts, parse_cohort_list, parse_criteria, parse_screen, registry_trial_text,
)

YES, NO = {"cancer_treatment_trial": "yes"}, {"cancer_treatment_trial": "no"}
CRITERIA = {"Part A": "Inclusion criteria:\nAny solid tumor; GFR >= 40 mL/min.\nExclusion criteria:\nInfection.",
            "Part B": "Inclusion criteria:\nMeasurable melanoma; GFR >= 60 mL/min.\nExclusion criteria:\nInfection."}
SOLE = "Inclusion criteria:\nAdults with cancer.\nExclusion criteria:\nActive infection."


def trials(ids):
    return pd.DataFrame([{"trial_id": t, "trial_title": f"Trial {t}", "brief_summary": "Fabricated {raw} summary.",
                          "eligibility_criteria": "Inclusion:\n  Cancer."} for t in ids],
                        columns=["trial_id", "trial_title", "brief_summary", "eligibility_criteria"])


def two_parts():
    return {"screen": [YES], "list": [{"cohorts": list(CRITERIA)}],
            "criteria": {name: [{"eligibility_criteria": text}] for name, text in CRITERIA.items()}}


def one_cohort():
    return {"screen": [YES], "list": [{"cohorts": ["All participants"]}],
            "criteria": {"All participants": [{"eligibility_criteria": SOLE}]}}


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    config.trial["cohorts"] = {**config.trial["cohorts"], "max_attempts": 2}
    return config


# Prompts


def test_prompts_share_one_document_preamble_and_put_the_target_last():
    prompts = load_prompts()
    preambles = {prompts[step].split("\nTASK\n")[0] for step in STEPS}
    assert len(preambles) == 1 and "{trial_text}" in preambles.pop()
    assert all(prompts[step].count("{trial_text}") == 1 for step in STEPS)
    criteria = prompts["criteria"]
    assert criteria.index("\nTARGET COHORT\n") > criteria.index("OUTPUT FORMAT")
    assert criteria.rstrip().endswith("{cohort_name}") and "{cohort_list}" in criteria.split("\nTARGET COHORT\n")[1]


def test_messages_keep_document_and_names_literal():
    prompts = load_prompts()
    document = "Section {5.1}: <trial_document> {cohort_name}"
    for step in STEPS:
        content = build_messages(step, document, prompts, cohort_names=["Arm {A}", "Arm B"], cohort_name="Arm {A}")[0]["content"]
        assert document in content and "{trial_text}" not in content
    content = build_messages("criteria", document, prompts, cohort_names=["Arm {A}", "Arm B"], cohort_name="Arm {A}")[0]["content"]
    assert content.endswith("this cohort only: Arm {A}") and "- Arm {A}\n- Arm B" in content


def test_prompt_examples_parse_with_their_step_contracts():
    prompts = load_prompts()
    rendered = {step: build_messages(step, "doc", prompts, cohort_names=["A"], cohort_name="A")[0]["content"] for step in STEPS}
    assert parse_screen('{"cancer_treatment_trial": "no"}') == "no" and '{"cancer_treatment_trial": "no"}' in rendered["screen"]
    example = re.search(r'\{"cohorts": \[.*?\]\}', rendered["list"]).group(0)
    assert len(parse_cohort_list(example)) == 2
    example = re.search(r'\{"eligibility_criteria": ".*?"\}', rendered["criteria"]).group(0)
    assert parse_criteria(example).startswith("Inclusion criteria:\n")


def test_criteria_prompt_extracts_every_criterion_verbatim_with_the_applicability_rules():
    criteria = load_prompts()["criteria"]
    # Boilerplate is kept here and removed by a separate, optional step (cohort_boilerplate).
    for rule in ("Never leave out a criterion because it is common or administrative",
                 "Copy each criterion verbatim from the document", "Do not paraphrase, summarize, shorten",
                 "only the version that applies in the United States",
                 "Never delete a qualifier", "even when its name also says so"):
        assert rule in criteria
    assert "boilerplate" not in criteria.lower()
    listing = load_prompts()["list"]
    # The list step names the trial's cohorts without comparing their criteria.
    assert "You do not need to compare the cohorts' eligibility criteria" in listing
    assert "stratification factors" in listing
    assert "crossover, extension, continuation, retreatment, and rollover phases" in listing
    assert not any(word in listing for word in ("identical", "combine"))


def test_registry_listing_is_one_document_of_title_summary_and_criteria():
    text = registry_trial_text("A trial", None, "Inclusion:\n  Adults.")
    assert text == "Title: A trial\n\nBrief summary:\n\n\nEligibility criteria:\nInclusion:\n  Adults."


# Step contracts


@pytest.mark.parametrize("text", ['{"cancer_treatment_trial": true}', '{"cancer_treatment_trial": "Yes"}',
                                  '{"cancer_treatment_trial": "yes", "reason": "x"}', "yes", "", '["yes"]'])
def test_screen_rejects_anything_but_the_exact_object(text):
    with pytest.raises(ValueError):
        parse_screen(text)


def test_malformed_or_truncated_answers_are_never_rejections():
    assert parse_screen('```json\n{"cancer_treatment_trial": "no"}\n```') == "no"
    with pytest.raises(ValueError, match="Incomplete"):
        parse_screen('{"cancer_treatment_trial": "no"}', "length")
    with pytest.raises(ValueError, match="valid JSON"):
        parse_screen('{"cancer_treatment_trial": "no"}\n}')
    with pytest.raises(ValueError, match="Duplicate"):
        parse_screen('{"cancer_treatment_trial": "no", "cancer_treatment_trial": "yes"}')


@pytest.mark.parametrize("names", [[], [""], ["A", " a "], ["A", 2], ["TBD"], "A"])
def test_cohort_list_needs_distinct_nonempty_names(names):
    with pytest.raises(ValueError):
        parse_cohort_list(json.dumps({"cohorts": names}))


@pytest.mark.parametrize("criteria", ["", "  ", "placeholder", "…", 5])
def test_criteria_must_be_real_text(criteria):
    with pytest.raises(ValueError):
        parse_criteria(json.dumps({"eligibility_criteria": criteria}))


def test_schemas_match_parsers_and_allow_escapes():
    assert [JSON_SCHEMAS[s]["required"] for s in STEPS] == [["cancer_treatment_trial"], ["cohorts"], ["eligibility_criteria"]]
    assert all(JSON_SCHEMAS[s]["additionalProperties"] is False for s in STEPS)
    # vLLM's grammar admits no JSON escape (line break, quote) in a minLength string.
    assert "minLength" not in json.dumps(JSON_SCHEMAS)


def test_cohorts_with_identical_criteria_are_combined():
    decision, merged = assemble_decision(["Arm A", "Arm B", "Arm C"], ["Adults.", " adults. ", "Children."])
    assert decision == {"cancer_treatment_trial": "yes", "multiple_cohorts": "yes", "cohorts": [
        {"cohort_name": "Arm A; Arm B", "eligibility_criteria": "Adults."},
        {"cohort_name": "Arm C", "eligibility_criteria": "Children."}]}
    assert merged == [[0, 1]]
    assert assemble_decision(["A", "B"], ["Same.", "Same."])[0]["multiple_cohorts"] == "no"


# Registry API


def test_summarize_trials_runs_three_steps_and_returns_complete_cohort_rows(config, cohort_backend):
    cohort_backend.script.update({"Trial T0": {"screen": [NO]}, "Trial T1": one_cohort(), "Trial T2": two_parts()})
    frame = trials(["T0", "T1", "T2"])
    original = frame.copy(deep=True)
    result, metadata, qc = extract_cohorts(frame, config=config, return_metadata=True, return_qc=True)
    calls = cohort_backend.calls
    assert [len(prompts) for prompts, _ in calls] == [3, 2, 3]
    assert [c[1]["remote"]["request_params"]["response_format"]["json_schema"]["name"] for c in calls] == [
        "trial_cohorts_screen", "trial_cohorts_list", "trial_cohorts_criteria"]
    assert [c[1]["remote"]["request_params"]["response_format"]["json_schema"]["schema"] for c in calls] == [
        JSON_SCHEMAS[s] for s in STEPS]
    sent = prompt_text(calls[2][0][1])
    assert "Title: Trial T2\n\nBrief summary:\nFabricated {raw} summary." in sent and "- Part A\n- Part B" in sent
    assert list(result) == COHORT_COLUMNS
    assert result.cohort_trial_id.tolist() == ["T1-cohort-0", "T2-cohort-0", "T2-cohort-1"]
    assert result.cohort_name.tolist() == ["All participants", "Part A", "Part B"]
    assert result.cohort_eligibility_criteria.tolist() == [SOLE, *CRITERIA.values()]
    assert result.multiple_cohorts.tolist() == ["no", "yes", "yes"]
    assert result.cohort_criteria_source.eq("registry").all()
    assert metadata["schema_version"] == SCHEMA_VERSION and metadata["representation_version"] == REPRESENTATION_VERSION
    assert metadata["trial_rejections"] == {"T0": NO} and metadata["trial_errors"] == {}
    assert metadata["trial_cohort_names"] == {"T1": ["All participants"], "T2": ["Part A", "Part B"]}
    assert set(metadata["prompt_sha256_by_step"]) == set(STEPS) and metadata["structured_outputs"] is True
    assert result.attrs["trial_decisions"]["T2"]["cohorts"][1] == {"cohort_name": "Part B", "eligibility_criteria": CRITERIA["Part B"]}
    assert qc.set_index("metric").loc["trials_failed_inference", "value"] == 0
    pd.testing.assert_frame_equal(frame, original)


def test_only_the_failed_request_is_retried_with_the_full_document(config, cohort_backend):
    script = two_parts()
    script["criteria"]["Part B"] = [("{\"eligibility_criteria\": \"cut", "length"), {"eligibility_criteria": CRITERIA["Part B"]}]
    cohort_backend.script["Trial T0"] = script
    result, metadata, qc = extract_cohorts(trials(["T0"]), config=config, return_metadata=True, return_qc=True)
    retry = cohort_backend.calls[-1][0]
    assert len(retry) == 1 and "this cohort only: Part B" in prompt_text(retry[0])
    assert "Trial T0" in prompt_text(retry[0]) and "failed validation: Incomplete response: length" in prompt_text(retry[0])
    assert [a["attempt"] for a in metadata["trial_attempts"]["T0"]["criteria"][1]] == [1, 2]
    assert len(result) == 2 and metadata["trial_errors"] == {}
    assert qc.set_index("metric").loc["trials_truncated_llm_response", "value"] == 1


def test_a_cohort_failing_every_attempt_fails_its_trial_only(config, cohort_backend):
    script = two_parts()
    script["criteria"]["Part A"] = ["not json"]
    cohort_backend.script.update({"Trial T0": script, "Trial T1": one_cohort()})
    result, metadata = extract_cohorts(trials(["T0", "T1"]), config=config, return_metadata=True)
    assert result.trial_id.tolist() == ["T1"]
    assert set(metadata["trial_errors"]) == {"T0"} and "criteria (cohort 0)" in metadata["trial_errors"]["T0"]
    assert metadata["trial_rejections"] == {}


def test_failed_screen_is_not_a_rejection(config, cohort_backend):
    cohort_backend.script["Trial T0"] = {"screen": ['{"cancer_treatment_trial": "maybe"}']}
    result = extract_cohorts(trials(["T0"]), config=config)
    assert result.empty and list(result) == COHORT_COLUMNS
    assert set(result.attrs["trial_errors"]) == {"T0"} and result.attrs["trial_rejections"] == {}
    assert len(cohort_backend.calls) == 2


def test_empty_input_and_duplicate_ids(config, cohort_backend):
    assert extract_cohorts(trials([]), config=config).empty and not cohort_backend.calls
    with pytest.raises(ValueError, match="unique"):
        extract_cohorts(trials(["T0", "T0"]), config=config)
    assert not cohort_backend.calls


def test_debug_mode_keeps_each_steps_answer(config, cohort_backend):
    config.debug_mode = True
    cohort_backend.script["Trial T0"] = two_parts()
    result, metadata = extract_cohorts(trials(["T0"]), config=config, return_metadata=True)
    responses = metadata["trial_responses"]["T0"]
    assert json.loads(responses["screen"]["answer"]) == YES and set(responses["criteria"]) == {0, 1}
    assert result.trial_input_text.str.startswith("Title: Trial T0").all()


def test_default_preset_uses_the_three_step_prompts():
    config = load_default_preset()
    assert config.trial["extraction_method"] == "cohorts" and "cohorts" not in config.trial["prompt_files"]
    assert "prompt_file" not in config.protocol_cohort_extraction
    assert config.trial["cohorts"]["structured_outputs"] is True


def test_process_trials_runs_every_trial_step_in_order(monkeypatch):
    import matchminer_ai.trials as trials_api
    calls = []
    listing = pd.DataFrame([dict(trial_id="NCT1", trial_title="Fabricated trial", brief_summary="s",
                                 eligibility_criteria="Adults.")])
    def extract(trials, *, config, return_metadata):
        calls.append(("extract", list(trials.trial_id)))
        rows = pd.DataFrame([dict(trial_id="NCT1", cohort_number=0, cohort_eligibility_criteria="Adults.")])
        rows.attrs["trial_decisions"] = {"NCT1": {"cancer_treatment_trial": "yes"}}
        return rows, {"trial_decisions": rows.attrs["trial_decisions"]}
    def merge(cohorts, *, config, section, output_dir, return_metadata):
        assert cohorts.attrs == {} and section == "trial" and output_dir == "saved"
        calls.append("merge")
        return cohorts.assign(merged_from="[0]"), {"step": "merge"}
    def clean(cohorts, *, config, section, output_dir, return_metadata):
        assert section == "trial" and output_dir == "saved"
        calls.append("boilerplate")
        return cohorts.assign(cohort_key_criteria="Adults.", cohort_boilerplate_criteria=""), {"step": "boilerplate"}
    monkeypatch.setattr(trials_api, "extract_cohorts", extract)
    monkeypatch.setattr(trials_api, "merge_equivalent_cohorts", merge)
    monkeypatch.setattr(trials_api, "remove_boilerplate_criteria", clean)
    result, metadata = trials_api.process_trials(listing, output_dir="saved", return_metadata=True)
    assert calls == [("extract", ["NCT1"]), "merge", "boilerplate"]
    assert result.cohort_key_criteria.tolist() == ["Adults."] and "merged_from" in result
    assert result.attrs["trial_decisions"] == {"NCT1": {"cancer_treatment_trial": "yes"}}
    assert set(metadata) == {"extraction", "merge", "boilerplate"}


def test_summarize_trials_is_a_deprecated_alias_of_extract_cohorts(monkeypatch):
    import matchminer_ai.trials as trials_api
    monkeypatch.setattr(trials_api, "extract_cohorts", lambda *args, **kwargs: ("rows", kwargs))
    with pytest.warns(DeprecationWarning, match="extract_cohorts"):
        assert trials_api.summarize_trials("listings", return_metadata=True) == ("rows", {"return_metadata": True})
