import json
from pathlib import Path

import pytest

from matchminer_ai.trials.space_residual_exclusions import (
    build_residual_exclusion_messages, load_residual_exclusion_prompts,
    parse_residual_exclusions, residual_exclusion_retry_messages,
)

PROMPT_DIR = Path(__file__).parents[1] / "src/matchminer_ai/prompts"


def test_only_task_source_and_exact_space_reach_model(tmp_path):
    # It must work without a primer or any hierarchy/refinement prompt present.
    (tmp_path / "trial.space_residual_exclusions.txt").write_bytes(
        (PROMPT_DIR / "trial.space_residual_exclusions.txt").read_bytes())
    prompts = load_residual_exclusion_prompts(tmp_path)
    source, space = "SOURCE DATA: no active infection.", "FULL SPACE DATA: male."
    messages = build_residual_exclusion_messages(source, space, prompts)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert messages[0]["content"] == (prompts["request"].rstrip()
        + "\n\nTRIAL DOCUMENT:\n" + source + "\nEND TRIAL DOCUMENT\n\nCOMPLETED TRIAL SPACE:\n" + space)
    lowered = messages[0]["content"].lower()
    for forbidden in ["residual", "boilerplate", "primer", "reasoning: high", "-re histology"]:
        assert forbidden not in lowered
    assert "pregnancy requirements for a male-restricted space" in lowered


def test_empty_list_and_complete_outer_fence_are_valid():
    assert parse_residual_exclusions('{"exclusions": []}') == {"exclusions": []}
    obj = {"exclusions": ["ECOG performance status greater than 1."]}
    assert parse_residual_exclusions("```json\n" + json.dumps(obj) + "\n```") == obj


@pytest.mark.parametrize("payload", [[], {}, {"exclusions": "none"},
    {"exclusions": [], "reasoning": "extra"}, {"exclusions": [None]},
    {"exclusions": [""]}, {"exclusions": ["a\nb"]}, {"exclusions": ["same", "same"]}])
def test_invalid_output_is_not_silently_accepted(payload):
    with pytest.raises(ValueError):
        parse_residual_exclusions(json.dumps(payload))


def test_truncation_and_duplicate_keys_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        parse_residual_exclusions('{"exclusions": []}', "length")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        parse_residual_exclusions('{"exclusions": [], "exclusions": []}')


def test_retry_includes_error_but_no_old_field_rules():
    messages = build_residual_exclusion_messages("SOURCE", "SPACE", load_residual_exclusion_prompts(PROMPT_DIR))
    retry = residual_exclusion_retry_messages(messages, "bad output", "not a JSON object")
    assert retry[:1] == messages
    assert "not a JSON object" in retry[-1]["content"]
    assert "boilerplate" not in json.dumps(retry).lower()
