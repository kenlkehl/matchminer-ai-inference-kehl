"""Public treatment/cohort extraction contract and failure handling."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.trials import summarize_trials
from matchminer_ai.trials.cohort_schema import build_cohort_messages, parse_cohort_response
from matchminer_ai.trials.cohorts import COHORT_COLUMNS

NO = {"cancer_treatment_trial": "no"}
SINGLE = {"cancer_treatment_trial": "yes", "multiple_cohorts": "no"}
MULTI = {"cancer_treatment_trial": "yes", "multiple_cohorts": "yes", "cohorts": [
    "Cohort A. Inclusion: adult cancer, prior therapy. Exclusion: infection.",
    "Cohort B. Inclusion: adult cancer, untreated. Exclusion: infection.",
]}


def source(ids):
    return pd.DataFrame([{
        "trial_id": trial_id, "trial_title": f"Trial {trial_id}",
        "brief_summary": "Fabricated {raw} trial text.",
        "eligibility_criteria": "Inclusion:\n  Cancer.\n\nExclusion:\n  Infection.",
    } for trial_id in ids], columns=[
        "trial_id", "trial_title", "brief_summary", "eligibility_criteria",
    ])


@pytest.fixture
def harness(monkeypatch, default_config):
    config = deepcopy(default_config)
    config.trial["extraction_method"] = "cohorts"
    config.trial["cohorts"] = {"max_attempts": 2}
    config.remote["enabled"] = True
    calls, answers = [], []

    class Backend:
        def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
            calls.append((prompt_list, deepcopy(llm_config)))
            batch = answers.pop(0)
            assert len(batch) == len(prompt_list)
            return LLMGenerationResult(
                final_outputs=[json.dumps(x) if isinstance(x, dict) else x for x, _ in batch],
                finish_reasons=[finish for _, finish in batch],
                reasoning_outputs=["test reasoning"] * len(batch), raw_outputs=[],
                model_metadata={"model_sha": "test"},
            )

    monkeypatch.setattr("matchminer_ai.trials.cohorts.get_llm_backend", lambda _: Backend())
    return config, calls, answers


@pytest.mark.parametrize("value", [NO, SINGLE, MULTI])
def test_three_terminal_json_states(value):
    assert parse_cohort_response(json.dumps(value)) == value


@pytest.mark.parametrize("value", [
    {}, [], {"cancer_treatment_trial": False},
    {**NO, "multiple_cohorts": "no"}, {**NO, "rationale": "not treatment"},
    {"cancer_treatment_trial": "yes"}, {**SINGLE, "cohorts": []},
    {**SINGLE, "multiple_cohorts": False},
    {**MULTI, "cohorts": []}, {**MULTI, "cohorts": ["one"]},
    {**MULTI, "cohorts": ["one", " "]}, {**MULTI, "cohorts": ["one", 2]},
    {**MULTI, "cohorts": ["one", " ONE "]},
    {"multiple_cohorts": "no", "cancer_treatment_trial": "yes"},
])
def test_invalid_schema_never_becomes_a_rejection(value):
    with pytest.raises(ValueError):
        parse_cohort_response(json.dumps(value))


@pytest.mark.parametrize("text", [
    '{"cancer_treatment_trial":"no","cancer_treatment_trial":"no"}',
    '```json\n{"cancer_treatment_trial":"no"}\n```',
    '{"cancer_treatment_trial":"no"} extra',
])
def test_duplicate_fields_or_surrounding_text_fail(text):
    with pytest.raises(ValueError):
        parse_cohort_response(text)


def test_combined_prompt_has_only_literal_source_and_three_shapes():
    text = "Marker {cancer_treatment_trial}; literal <trial_text> text."
    messages = build_cohort_messages(text)
    assert text in messages[0]["content"]
    prompt = messages[0]["content"]
    assert '"cancer_treatment_trial": "no"' in prompt
    assert '"cancer_treatment_trial": "yes", "multiple_cohorts": "no"' in prompt
    assert "pharmacokinetic" in prompt and "background therapy" in prompt


def test_public_api_runs_one_combined_batch_and_keeps_sources(harness):
    config, calls, answers = harness
    answers.append([(NO, "stop"), (SINGLE, "stop"), (MULTI, "stop")])
    trials = source(["T0", "T1", "T2"])
    original = trials.copy(deep=True)
    result, metadata, qc = summarize_trials(trials, config=config, return_metadata=True, return_qc=True)
    assert len(calls) == 1
    assert list(result) == COHORT_COLUMNS
    assert result.trial_id.tolist() == ["T1", "T2", "T2"]
    assert result.cohort_trial_id.tolist() == ["T1-cohort-0", "T2-cohort-0", "T2-cohort-1"]
    assert result.iloc[0].cohort_eligibility_criteria == "\n".join(trials.iloc[1][
        ["trial_title", "brief_summary", "eligibility_criteria"]
    ])
    assert result.cohort_criteria_source.tolist() == ["trial_text", "llm", "llm"]
    assert result.cohort_eligibility_criteria.tolist()[1:] == MULTI["cohorts"]
    assert metadata["trial_decisions"] == {"T0": NO, "T1": SINGLE, "T2": MULTI}
    assert metadata["trial_rejections"] == {"T0": NO}
    assert metadata["trial_errors"] == {}
    assert result.attrs["trial_rejections"] == {"T0": NO}
    assert qc.set_index("metric").loc["trials_failed_inference", "value"] == 0
    pd.testing.assert_frame_equal(trials, original)


def test_retry_only_invalid_trial_preserves_sampling_and_full_source(harness):
    config, calls, answers = harness
    answers.extend([[(SINGLE, "stop"), (NO, "length")], [(MULTI, "stop")]])
    result, metadata = summarize_trials(source(["T0", "T1"]), config=config, return_metadata=True)
    assert len(result) == 3
    assert [len(prompts) for prompts, _ in calls] == [2, 1]
    assert calls[0][1] == calls[1][1]
    messages = calls[1][0][0].messages
    assert "Trial T1" in messages[0]["content"]
    assert "Incomplete trial cohort response: length" in messages[-1]["content"]
    assert metadata["trial_rejections"] == {} and metadata["trial_errors"] == {}


def test_exhausted_malformed_and_truncated_results_are_failures(harness):
    config, _, answers = harness
    answers.extend([[("invalid", "stop"), (NO, "length")]] * 2)
    result, metadata, qc = summarize_trials(source(["T0", "T1"]), config=config, return_metadata=True, return_qc=True)
    assert result.empty and list(result) == COHORT_COLUMNS
    assert set(metadata["trial_errors"]) == {"T0", "T1"}
    assert metadata["trial_rejections"] == {}
    assert qc.set_index("metric").loc["trials_failed_inference", "value"] == 2


def test_all_rejected_preserves_decisions_without_optional_metadata(harness):
    config, _, answers = harness
    answers.append([(NO, "stop")])
    result = summarize_trials(source(["T0"]), config=config)
    assert result.empty and result.attrs["trial_rejections"] == {"T0": NO}


def test_empty_input_does_not_call_backend(harness):
    config, calls, _ = harness
    result = summarize_trials(source([]), config=config)
    assert result.empty and not calls and list(result) == COHORT_COLUMNS


def test_debug_retains_rejected_and_accepted_responses(harness):
    config, _, answers = harness
    config.debug_mode = True
    answers.append([(NO, "stop"), (SINGLE, "stop")])
    result, meta = summarize_trials(source(["T0", "T1"]), config=config, return_metadata=True)
    assert "trial_answer_text" in result
    assert set(meta["trial_responses"]) == {"T0", "T1"}


def test_duplicate_ids_fail_before_inference(harness):
    config, calls, _ = harness
    with pytest.raises(ValueError, match="unique"):
        summarize_trials(source(["T0", "T0"]), config=config)
    assert not calls


def test_default_preset_uses_combined_prompt():
    config = load_default_preset()
    assert config.trial["extraction_method"] == "cohorts"
    assert config.trial["prompt_files"]["cohorts"] == "trial.cohorts.user.txt"
