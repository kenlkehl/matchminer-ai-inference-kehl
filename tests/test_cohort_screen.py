"""Screening trial documents for cancer treatment trials on their own (screen_trials)."""

import json

import pandas as pd
import pytest

from conftest import prompt_text
from matchminer_ai.config import load_default_preset
from matchminer_ai.trials import screen_trials
from matchminer_ai.trials.cohort_workflow import JSON_SCHEMAS


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    return config


def listings():
    return pd.DataFrame([dict(trial_id="T1", title="Drug A for {x} lung cancer", brief_summary="Treats cancer.",
                              eligibility_criteria="Adults."),
                         dict(trial_id="T2", title="Exercise for fatigue", brief_summary="Supportive care.",
                              eligibility_criteria="Survivors."),
                         dict(trial_id="T3", title="Unclear study", brief_summary="", eligibility_criteria="")])


def test_listings_are_screened_with_the_workflow_prompt_and_every_answer_is_saved(tmp_path, config, cohort_backend):
    cohort_backend.script.update({"Drug A": {"screen": [{"cancer_treatment_trial": "yes"}]},
                                  "Exercise": {"screen": ["not json", {"cancer_treatment_trial": "no"}]},
                                  "Unclear": {"screen": ["not json"]}})
    config.trial["cohorts"] = {**config.trial.get("cohorts", {}), "max_attempts": 2}
    result, metadata = screen_trials(listings(), config=config, output_dir=tmp_path, return_metadata=True)
    assert result.cancer_treatment_trial.tolist() == ["yes", "no", None]  # a failed answer is never a rejection
    assert result.screen_error.notna().tolist() == [False, False, True]
    first = prompt_text(cohort_backend.calls[0][0][0])
    assert "Title: Drug A for {x} lung cancer\n\nBrief summary:\nTreats cancer.\n\nEligibility criteria:\nAdults." in first
    schema = cohort_backend.calls[0][1]["remote"]["request_params"]["response_format"]["json_schema"]["schema"]
    assert schema == JSON_SCHEMAS["screen"]
    assert "failed validation" in prompt_text(cohort_backend.calls[1][0][0])
    saved = [json.loads(line) for line in (tmp_path / "screen.jsonl").read_text().splitlines()]
    assert sorted((s["trial_id"], s["decision"]) for s in saved) == [("T1", "yes"), ("T2", "no"), ("T3", None)]
    assert all(a["reasoning"] == "test reasoning" for s in saved for a in s["attempts"])
    assert metadata["accepted"] == 1 and metadata["rejected"] == 1 and list(metadata["screen_errors"]) == ["T3"]

    # A rerun reuses saved decisions for unchanged documents and asks only for the rest.
    calls = len(cohort_backend.calls)
    changed = listings()
    changed.loc[1, "eligibility_criteria"] = "Survivors over 18."
    again, metadata = screen_trials(changed, config=config, output_dir=tmp_path, return_metadata=True)
    assert metadata["reused"] == 1 and again.cancer_treatment_trial.tolist()[0] == "yes"
    asked = [prompt_text(p) for call in cohort_backend.calls[calls:] for p in call[0]]
    assert not any("Drug A" in text for text in asked) and any("Survivors over 18." in text for text in asked)
