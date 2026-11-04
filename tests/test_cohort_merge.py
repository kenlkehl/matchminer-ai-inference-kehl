"""Combining a trial's cohorts with essentially the same criteria (merge_equivalent_cohorts) for training."""

import json

import pandas as pd
import pytest

from conftest import prompt_text
from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.trials import merge_equivalent_cohorts
from matchminer_ai.trials.cohort_merge import MERGE_JSON_SCHEMA, MERGE_PROMPT, build_merge_messages, parse_merge_groups
from matchminer_ai.trials.cohort_workflow import COHORT_COLUMNS


def rows(trial_id, texts, names=None):
    names = names or [f"Arm {chr(65 + i)}" for i in range(len(texts))]
    return [dict(cohort_trial_id=f"{trial_id}-cohort-{i}", trial_id=trial_id, cohort_number=i, cohort_name=name,
                 cohort_eligibility_criteria=text, cohort_criteria_source="protocol", cancer_treatment_trial="yes",
                 multiple_cohorts="yes" if len(texts) > 1 else "no", protocol_origin="synthetic")
            for i, (name, text) in enumerate(zip(names, texts))]


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    return config


def test_prompt_lists_numbered_cohorts_literally_and_asks_only_for_groups():
    prompt = load_prompt_text(MERGE_PROMPT)
    content = build_merge_messages(["Arm {1}", "Arm 2"], ["Adults {x}.", "Children."], prompt)[0]["content"]
    assert "### Cohort 1: Arm {1}\nAdults {x}.\n\n### Cohort 2: Arm 2\nChildren." in content
    assert "do not rewrite, summarize, or repeat any criteria" in content
    assert "When you are unsure whether cohorts define different populations, keep them apart" in content
    assert "Ignore differences of extraction" in content and "unless it concerns what defines a cohort's population" in content
    assert parse_merge_groups('{"groups": [[1, 3], [2], [4, 5]]}', "stop", 5) == [[0, 2], [1], [3, 4]]
    assert list(MERGE_JSON_SCHEMA["properties"]) == ["groups"] and "minLength" not in json.dumps(MERGE_JSON_SCHEMA)


@pytest.mark.parametrize("text", ['{"groups": [[1], [1, 2]]}', '{"groups": [[1]]}', '{"groups": [[1, 2, 3]]}',
                                  '{"groups": [[1, "2"]]}', '{"groups": [[], [1, 2]]}', '{"groups": []}',
                                  '{"groups": [[1, 2]], "reason": "x"}', "[[1, 2]]"])
def test_groups_must_partition_the_cohort_numbers(text):
    with pytest.raises(ValueError):
        parse_merge_groups(text, "stop", 2)


def test_truncated_answers_fail():
    with pytest.raises(ValueError, match="Incomplete"):
        parse_merge_groups('{"groups": [[1, 2]]}', "length", 2)


def test_groups_become_one_row_with_joined_names_and_the_first_cohorts_criteria(config, cohort_backend):
    frame = pd.DataFrame(rows("T1", ["Trial one adults.", "Trial one children.", "Trial one adults, reworded."])
                         + rows("T2", ["Trial two adults."]))
    cohort_backend.script["Trial one"] = {"merge": [{"groups": [[1, 3], [2]]}]}
    merged, metadata = merge_equivalent_cohorts(frame, config=config, section="protocol_cohort_extraction",
                                                return_metadata=True)
    assert len(cohort_backend.calls) == 1 and len(cohort_backend.calls[0][0]) == 1  # T2 has one cohort: no request
    request = cohort_backend.calls[0][1]["remote"]["request_params"]["response_format"]["json_schema"]["schema"]
    assert request == MERGE_JSON_SCHEMA
    assert merged.cohort_trial_id.tolist() == ["T1-cohort-0", "T1-cohort-1", "T2-cohort-0"]
    assert merged.cohort_name.tolist() == ["Arm A; Arm C", "Arm B", "Arm A"]
    assert merged.cohort_eligibility_criteria.tolist() == ["Trial one adults.", "Trial one children.", "Trial two adults."]
    assert merged.merged_from.tolist() == ["[0, 2]", "[1]", "[0]"]
    assert merged.multiple_cohorts.tolist() == ["yes", "yes", "no"]
    assert merged.protocol_origin.eq("synthetic").all() and set(COHORT_COLUMNS) <= set(merged)
    assert metadata["trial_groups"] == {"T1": [[0, 2], [1]], "T2": [[0]]}
    assert (metadata["cohorts_in"], metadata["cohorts_out"]) == (4, 3)


def test_a_trial_that_never_validates_keeps_its_cohorts(config, cohort_backend):
    config.trial["cohorts"] = {**config.trial["cohorts"], "max_attempts": 2}
    frame = pd.DataFrame(rows("T1", ["Trial one adults.", "Trial one adults again."]))
    cohort_backend.script["Trial one"] = {"merge": ['{"groups": [[1]]}']}
    merged, metadata = merge_equivalent_cohorts(frame, config=config, return_metadata=True)
    assert len(cohort_backend.calls) == 2 and "failed validation" in prompt_text(cohort_backend.calls[1][0][0])
    assert merged.cohort_name.tolist() == ["Arm A", "Arm B"] and "T1" in metadata["merge_errors"]


def test_saved_groupings_are_reused_only_for_unchanged_cohorts(tmp_path, config, cohort_backend):
    frame = pd.DataFrame(rows("T1", ["Trial one adults.", "Trial one adults, reworded."]))
    cohort_backend.script["Trial one"] = {"merge": [{"groups": [[1, 2]]}]}
    first = merge_equivalent_cohorts(frame, config=config, output_dir=tmp_path)
    saved = json.loads((tmp_path / "T1/merge.json").read_text())
    assert saved["groups"] == [[0, 1]] and len(first) == 1
    assert saved["attempts"][0]["reasoning"] == "test reasoning" and json.loads(saved["attempts"][0]["answer"]) == {"groups": [[1, 2]]}
    again, metadata = merge_equivalent_cohorts(frame, config=config, output_dir=tmp_path, return_metadata=True)
    assert len(cohort_backend.calls) == 1 and metadata["skipped_completed"] == 1 and len(again) == 1
    frame.loc[1, "cohort_eligibility_criteria"] = "Trial one children."
    cohort_backend.script["Trial one"] = {"merge": [{"groups": [[1], [2]]}]}
    changed = merge_equivalent_cohorts(frame, config=config, output_dir=tmp_path)
    assert len(cohort_backend.calls) == 2 and len(changed) == 2
