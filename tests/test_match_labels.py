"""The five match labels asked in one prompt, their score, and the multi-head TrialChecker."""

import json

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.matching import score_match_quality
from matchminer_ai.matching.match_labels import (
    LABELS, MATCH_LABELS_JSON_SCHEMA, MAX_SCORE, build_match_label_messages, match_labels_prompt, match_score,
    parse_match_labels,
)


def answers(**yes):
    return {label: {"answer": "yes" if yes.get(label) else "no", "evidence_status": "supported", "reason": "Why."}
            for label in LABELS}


def test_one_prompt_asks_all_five_questions_about_key_text_only():
    prompt = match_labels_prompt()
    for label in LABELS:
        assert f'"{label}"' in prompt
    assert "key criteria" in prompt and "have been removed and are checked separately" in prompt
    assert "do not treat their absence from either text as a mismatch" in prompt
    messages = build_match_label_messages("Patient key section.", "Cohort key criteria.")
    assert messages[0]["content"] == prompt and messages[0]["role"] == "system"
    assert messages[1]["content"] == ("PATIENT HISTORY\nPatient key section.\n\n"
                                      "COHORT KEY ELIGIBILITY CRITERIA\nCohort key criteria.")
    assert MATCH_LABELS_JSON_SCHEMA["required"] == list(LABELS)


def test_answers_are_parsed_strictly_and_scored_zero_to_six():
    parsed = parse_match_labels(json.dumps(answers(reasonableness=True, burden=True, biomarker=True)))
    assert match_score(parsed) == 4 and MAX_SCORE == 6
    assert match_score(answers(cancer_type=True, burden=True, treatment_history=True, biomarker=True)) == 0
    assert match_score({label: True for label in LABELS}) == 6
    bad = answers()
    del bad["biomarker"]
    for text in (json.dumps(bad), "not json",
                 json.dumps({**answers(), "burden": {"answer": "maybe", "evidence_status": "supported", "reason": "x"}}),
                 json.dumps({**answers(), "burden": {"answer": "no", "evidence_status": "unknown", "reason": "x"}}),
                 json.dumps({**answers(), "burden": {"answer": "no", "evidence_status": "supported", "reason": " "}})):
        with pytest.raises(ValueError):
            parse_match_labels(text)
    with pytest.raises(ValueError):
        parse_match_labels(json.dumps(answers()), "length")


def test_multi_head_trial_checker_returns_each_head_and_the_expected_score(monkeypatch):
    heads = dict(reasonableness=0.9, cancer_type=0.8, burden=0.5, treatment_history=0.2, biomarker=0.1)
    seen = {}

    def run_checker(prompts, *, checker_config, model_metadata_cache_dir=None, return_all_scores=False):
        seen.update(prompts=prompts, all_scores=return_all_scores)
        return [[{"label": label, "score": value} for label, value in heads.items()]], {"model_name": "multi-head"}

    monkeypatch.setattr("matchminer_ai.matching.rerank.run_checker", run_checker)
    pairs = pd.DataFrame([dict(patient_id="P1", cohort_trial_id="T1-cohort-1",
                               patient_summary="Lung cancer.\n\nOther relevant history:\nDialysis.",
                               cohort_eligibility_criteria="Lung cancer.\nNo dialysis.",
                               cohort_key_criteria="Lung cancer.", cohort_boilerplate_criteria="No dialysis.")])
    result = score_match_quality(pairs, config=load_default_preset(), filter_low_quality=False)
    expected = 0.9 * (1 + 0.8 + 0.5 + 0.2 + 2 * 0.1) / 6
    assert result.match_quality_score.tolist() == pytest.approx([expected])
    assert [result[f"match_quality_{label}"].iloc[0] for label in LABELS] == pytest.approx(list(heads.values()))
    assert list(result.columns[:2]) == ["patient_id", "cohort_trial_id"] and seen["all_scores"]
    assert seen["prompts"] == ["Lung cancer.\nNow here is the patient summary:Lung cancer."]  # key text only
