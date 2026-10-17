import json
from pathlib import Path

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.trials import summarize_trials
from matchminer_ai.trials._hierarchical_backend import summarize_hierarchically
from matchminer_ai.trials.hierarchy import (
    FIELDS,
    Completion,
    Hierarchy,
    ResponseCache,
    load_prompts,
    parse_response,
)
from matchminer_ai.trials.postprocess import _strip_numerical_prefix

PROMPTS = load_prompts(Path(__file__).parents[1] / "src/matchminer_ai/prompts")


def branches(*values):
    return json.dumps(
        {
            "branches": [{"value": value, "scope": scope} for value, scope in values],
            "prune_reason": "",
        }
    )


def reply(request):
    field = request.field_name
    if field == "age":
        return branches(
            ("18 to 64 years", "younger pathway"),
            ("65 years or older", "older pathway"),
        )
    if field == "sex":
        return branches(("Female", request.scope))
    if field == "cancer_type":
        return branches(("breast cancer", request.scope))
    if field == "histology":
        return branches(("invasive ductal carcinoma", request.scope))
    if field == "disease_state":
        return branches(("localized", request.scope))
    if field == "prior_treatment_required":
        return branches(("lumpectomy with negative margins", request.scope))
    if field == "biomarkers_required":
        value = (
            "Oncotype DX recurrence score >=26"
            if "younger" in request.scope
            else "estrogen receptor positive"
        )
        return branches((value, request.scope))
    if field == "boilerplate":
        return json.dumps(
            {"exclusions": ["For the younger pathway only: active autoimmune disease."]}
        )
    return branches(("NA", request.scope))


def test_parent_branches_preserve_dependencies_and_assemble_without_llm_numbering():
    state = Hierarchy(0, "SYNTHETIC TRIAL", PROMPTS)
    calls = []
    for stage in range(10):
        requests = state.requests()
        calls.extend(requests)
        for request in requests:
            assert (
                list(request.parent) == [key for key, _ in FIELDS[:stage]]
                if stage < 9
                else request.parent == {}
            )
            assert "SYNTHETIC TRIAL" in request.messages[-1]["content"]
        state.accept(requests, [Completion(reply(r)) for r in requests])
    output = state.output()
    lines = output.split("\nBoilerplate")[0].splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1. Age range allowed: 18 to 64 years.")
    assert "Oncotype DX recurrence score >=26" in lines[0]
    assert "estrogen receptor positive" not in lines[0]
    assert "estrogen receptor positive" in lines[1]
    assert "Oncotype" not in lines[1]
    assert all(
        "Prior treatment required: lumpectomy with negative margins." in line
        for line in lines
    )
    assert [r.field_name for r in calls].count("boilerplate") == 1
    assert len(calls) == 18
    assert not state.requests()


def test_equal_field_values_with_different_scopes_are_not_merged_early():
    state = Hierarchy(0, "trial", PROMPTS)
    state.accept(
        state.requests(), [Completion(branches(("18+", "path A"), ("18+", "path B")))]
    )
    assert len(state.requests()) == 2
    assert state.requests()[0].scope != state.requests()[1].scope


@pytest.mark.parametrize(
    "text,field,finish",
    [
        ("bad JSON", "age", "stop"),
        ('{"branches": [], "prune_reason": "uncertain"}', "age", "stop"),
        ('{"branches": [], "prune_reason": ""}', "sex", "stop"),
        (
            '{"branches": [{"value": "18+", "scope": "all", "sex": "Female"}], "prune_reason": ""}',
            "age",
            "stop",
        ),
        (
            '{"branches": [{"value": "18+\\nSex: Female", "scope": "all"}], "prune_reason": ""}',
            "age",
            "stop",
        ),
        ('{"exclusions": "None"}', "boilerplate", "stop"),
        ('{"exclusions": [], "exclusions": []}', "boilerplate", "stop"),
        (branches(("18+", "all")), "age", "length"),
    ],
)
def test_invalid_output_never_means_a_valid_empty_space(text, field, finish):
    with pytest.raises(ValueError):
        parse_response(text, field, finish)


def test_limits_and_all_pruned_fail_without_partial_output():
    state = Hierarchy(0, "trial", PROMPTS, max_nodes=1)
    state.accept(state.requests(), [Completion(branches(("18+", "A"), ("40+", "B")))])
    with pytest.raises(ValueError, match="max_nodes"):
        state.output()
    state = Hierarchy(0, "trial", PROMPTS, max_calls=1)
    state.accept(state.requests(), [Completion(branches(("18+", "A")))])
    assert state.requests() == []
    assert "max_calls" in state.error
    state = Hierarchy(0, "trial", PROMPTS)
    state.accept(state.requests(), [Completion(branches(("18+", "A")))])
    state.accept(
        state.requests(),
        [Completion('{"branches": [], "prune_reason": "source excludes this group"}')],
    )
    with pytest.raises(ValueError, match="All branches"):
        state.output()


def test_checkpoint_identity_covers_source_parent_prompt_and_runtime(tmp_path):
    request = Hierarchy(0, "trial", PROMPTS).requests()[0]
    cache = ResponseCache(tmp_path, {"model": "A"})
    value = Completion(branches(("18+", "A")))
    cache.put(request, value)
    assert cache.get(request) == value
    assert ResponseCache(tmp_path, {"model": "B"}).get(request) is None
    assert cache.get(Hierarchy(0, "changed trial", PROMPTS).requests()[0]) is None
    changed = {**PROMPTS, "age": PROMPTS["age"] + " changed"}
    assert cache.get(Hierarchy(0, "trial", changed).requests()[0]) is None


class FakeBackend:
    def __init__(self, invalid=0, fail_trial=False):
        self.calls = []
        self.invalid = invalid
        self.fail_trial = fail_trial

    def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
        answers, finishes = [], []
        for prompt in prompt_list:
            messages = prompt.messages
            text = messages[1]["content"]
            self.calls.append((messages, llm_config))
            if self.invalid:
                self.invalid -= 1
                answers.append("not JSON")
                finishes.append("stop")
                continue
            if self.fail_trial and text.endswith("BAD"):
                answers.append("transport failed")
                finishes.append("error")
                continue
            if "CURRENT FIELD:" not in text:
                answers.append('{"exclusions": ["Active infection."]}')
            else:
                name = text.split("CURRENT FIELD: ")[1].splitlines()[0]
                value = {
                    "age": "18+",
                    "sex": "Both",
                    "cancer_type": "lung cancer",
                    "histology": "adenocarcinoma",
                    "disease_state": "metastatic",
                }.get(name, "NA")
                answers.append(branches((value, "All eligible patients")))
            finishes.append("stop")
        return LLMGenerationResult(
            answers, {"model": "fake"}, finishes, ["reasoning"] * len(answers), []
        )


def runtime():
    return {
        "backend_mode": "remote",
        "model_name": "fake",
        "sampling_params": {"max_tokens": 100},
        "remote": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
    }


def test_adapter_retries_with_error_then_switches_thinking_and_reuses_cache(tmp_path):
    backend = FakeBackend(invalid=2)
    result = summarize_hierarchically(
        ["SOURCE"],
        backend,
        trial_config=runtime(),
        settings={"checkpoint_dir": str(tmp_path)},
    )
    assert result.finish_reasons == ["stop"]
    assert len(backend.calls) == 12
    assert "failed validation" in backend.calls[1][0][-1]["content"]
    assert backend.calls[1][0][-2]["content"] == "not JSON"
    assert (
        backend.calls[2][1]["remote"]["extra_body"]["chat_template_kwargs"][
            "enable_thinking"
        ]
        is False
    )
    second = summarize_hierarchically(
        ["SOURCE"],
        backend,
        trial_config=runtime(),
        settings={"checkpoint_dir": str(tmp_path)},
    )
    assert second.final_outputs == result.final_outputs
    assert len(backend.calls) == 12
    assert "validation_error" in result.reasoning_outputs[0]


def test_failed_trial_does_not_export_partial_spaces_or_break_other_trials():
    backend = FakeBackend(fail_trial=True)
    result = summarize_hierarchically(["GOOD", "BAD"], backend, trial_config=runtime())
    assert result.finish_reasons == ["stop", "error"]
    assert result.final_outputs[1] == ""
    assert len(backend.calls) == 11


def test_public_default_uses_hierarchy_and_preserves_output_schema(monkeypatch):
    backend = FakeBackend()
    config = load_default_preset()
    config.remote["enabled"] = True
    config.trial["remote"]["model_name"] = "fake"
    # This test isolates orchestration; token accounting has its own test below.
    config.trial["local"]["engine"].pop("max_model_len")
    monkeypatch.setattr(
        "matchminer_ai.trials.summarize.get_llm_backend", lambda config: backend
    )
    monkeypatch.setattr(
        "matchminer_ai._qc.trials.count_embedding_tokens",
        lambda texts, embedding_config: [1] * len(texts),
    )
    trials = pd.DataFrame(
        [
            {
                "trial_id": "T1",
                "trial_title": "Title",
                "brief_summary": "Summary",
                "eligibility_criteria": "SOURCE",
            }
        ]
    )
    result = summarize_trials(trials, config=config)
    assert len(backend.calls) == 10
    assert result.space_trial_id.tolist() == ["T1-0"]
    assert result.general_exclusion_criteria.tolist() == ["Active infection."]
    assert "Disease state allowed: metastatic." in result.clinical_space_summary.iloc[0]


def test_programmatic_numbering_supports_more_than_99_spaces():
    assert (
        _strip_numerical_prefix("123. Age range allowed: 18+.")
        == "Age range allowed: 18+."
    )


def test_context_budget_is_recounted_without_truncating(monkeypatch):
    from types import SimpleNamespace

    from matchminer_ai.trials._hierarchical_backend import _bounded_prompts

    tokenizer = SimpleNamespace(
        encode=lambda text, **kw: text.split(),
        apply_chat_template=lambda conversation, **kw: conversation[-1]["content"],
    )
    monkeypatch.setattr(
        "matchminer_ai.trials._hierarchical_backend._get_chat_template_tokenizer",
        lambda name: tokenizer,
    )
    config = {**runtime(), "max_model_len": 300}
    messages = [[{"role": "user", "content": "source text"}]]
    prompts, _ = _bounded_prompts(messages, config)
    assert prompts[0].max_tokens == 42
    assert prompts[0].messages == messages[0]
    with pytest.raises(ValueError, match="not truncated"):
        _bounded_prompts(messages, {**config, "max_model_len": 256})
