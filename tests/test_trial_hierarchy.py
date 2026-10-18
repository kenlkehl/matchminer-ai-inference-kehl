import json
from pathlib import Path

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.trials import summarize_trials
from matchminer_ai.trials._hierarchical_backend import summarize_hierarchically
from matchminer_ai.trials.hierarchy import (
    FIELD_QUESTIONS,
    FIELDS,
    Completion,
    Hierarchy,
    ResponseCache,
    build_request,
    load_prompts,
    parse_response,
)
from matchminer_ai.trials.postprocess import _strip_numerical_prefix

PROMPTS = load_prompts(Path(__file__).parents[1] / "src/matchminer_ai/prompts")


@pytest.mark.parametrize("field", [name for name, _ in FIELDS])
def test_each_component_gets_only_its_field_rules_and_one_parent(field):
    source = "SYNTHETIC FULL SOURCE\nAdults; no known hepatitis; ECOG 0 or 1."
    index = [name for name, _ in FIELDS].index(field)
    parent = {name: f"locked {name}" for name, _ in FIELDS[:index]}
    scope = "Synthetic source-defined pathway with unresolved dependencies."
    request = build_request(0, 0, source, field, parent, scope, PROMPTS)
    text = request.messages[1]["content"]
    expected_parts = [
        PROMPTS["common"].strip(),
        f"CURRENT FIELD: {field}\n",
        f"TRIAL DOCUMENT:\n{source}\nEND TRIAL DOCUMENT",
        "For a patient with this set of characteristics:\n" + json.dumps(parent),
        "SOURCE-DEFINED CONTEXT FOR THIS BRANCH:\n" + scope,
        f"What sets of {FIELD_QUESTIONS[field]} are relevant for determining eligibility for this trial?",
        "ANTI-BOILERPLATE INSTRUCTIONS:",
        PROMPTS["anti_boilerplate"],
        "OMISSIONS (verbatim):\n" + PROMPTS["omissions"],
        "INSTRUCTIONS FOR THIS STEP:\n" + PROMPTS[field].strip(),
        f"RETURN CHILD SPACES FOR CURRENT FIELD: {field}\n"
        + PROMPTS["response"].strip(),
    ]
    positions = [text.index(part) for part in expected_parts]
    assert positions == sorted(positions)
    assert text.startswith(expected_parts[0])
    assert text.endswith(expected_parts[-1])
    assert text.count(source) == 1
    assert request.parent == parent
    assert request.scope == scope
    field_sections = {
        "sex criteria": {"sex"},
        "cancer primary site": {"cancer_primary_site"},
        "histology": {"histology"},
        "disease state": {"disease_state"},
        "treatment history": {"prior_treatment_requirements"},
        "biomarkers": {"biomarker_requirements"},
    }
    for label, relevant_fields in field_sections.items():
        assert (f"-Re {label}:" in text) == (field in relevant_fields)
    assert text.count("For a patient with this set of characteristics:") == 1
    assert "uncontrolled brain metastases, HIV or hepatitis" in text
    assert "Make sure your boilerplate exclusions" not in text
    assert "Do NOT include this type of time-specific treatment washout" in text
    assert "DO NOT BELONG IN YOUR TRIAL SPACE OUTPUT TEXT AT ALL" in text
    assert ("exhaustive list of spaces" in text) == (
        field in {"cancer_primary_site"}
    )
    assert ("ALK-rearranged" in text) == (
        field in {"biomarker_requirements"}
    )
    assert "FULL FIELD DEFINITIONS" not in text
    assert "Structure your output like this" not in text
    assert "After you output the trial spaces" not in text
    assert "1. Age range allowed:" not in text


def test_boilerplate_gets_full_source_and_only_boilerplate_rules():
    request = build_request(
        0, 0, "FULL SOURCE", "boilerplate", {}, "All populations", PROMPTS
    )
    text = request.messages[1]["content"]
    parts = [
        PROMPTS["common"].strip(),
        "TRIAL DOCUMENT:\nFULL SOURCE\nEND TRIAL DOCUMENT",
        "EXTRACTION SCOPE:\nAll populations",
        "OMISSIONS (verbatim):\n" + PROMPTS["omissions"],
        "INSTRUCTIONS FOR THIS STEP:\n" + PROMPTS["boilerplate"].strip(),
        "RETURN GENERAL EXCLUSIONS:\n" + PROMPTS["boilerplate_response"].strip(),
    ]
    assert [text.index(part) for part in parts] == sorted(
        text.index(part) for part in parts
    )
    assert text.endswith(parts[-1])
    assert "CURRENT FIELD:" not in text
    assert "CURRENT CANDIDATE SPACE" not in text
    assert "RETURN CHILD SPACES" not in text
    assert "inclusion and exclusion sections" in text
    assert "-Re " not in text
    assert "For a patient with this set of characteristics:" not in text
    assert '"ECOG > 1"' in text
    assert '"Hct <= 30 and Hgb <= 10"' in text


@pytest.mark.parametrize(
    "field,start,end,example_label",
    [
        ("sex", "-Re sex criteria:", "-Re cancer primary site:", None),
        (
            "cancer_primary_site",
            "-Re cancer primary site:",
            "-Re histology:",
            "cancer primary site",
        ),
        ("histology", "-Re histology:", "-Re disease state:", "histology"),
        (
            "disease_state",
            "-Re disease state:",
            "-Re treatment history:",
            "disease state",
        ),
        (
            "prior_treatment_requirements",
            "-Re treatment history:",
            "-Re biomarkers:",
            "treatment history",
        ),
        (
            "biomarker_requirements",
            "-Re biomarkers:",
            "Some trials have only one space",
            "biomarker",
        ),
        (
            "boilerplate",
            "Then, list exclusion criteria described in the trial text",
            "ALWAYS output plain text only.",
            None,
        ),
    ],
)
def test_field_instruction_includes_exact_authored_section_and_examples_once(
    field, start, end, example_label
):
    primer = (
        Path(__file__).parents[1] / "src/matchminer_ai/prompts/trial.user.primer.txt"
    ).read_text()
    expected = primer[primer.index(start) : primer.index(end)].strip()
    request = build_request(0, 0, "source", field, {}, "all", PROMPTS)
    text = request.messages[1]["content"]
    reminder = text.split("FIELD-SPECIFIC INSTRUCTIONS (verbatim):\n")
    assert len(reminder) == 2
    assert reminder[1].startswith(expected)
    assert text.count(expected) == 1
    assert "{{field_rules}}" not in text
    if example_label:
        for kind in ("valid", "invalid"):
            assert f"Examples of {kind} {example_label} components:" in reminder[1]


def test_age_reminder_uses_existing_rules_without_invented_examples():
    request = build_request(0, 0, "source", "age", {}, "all", PROMPTS)
    text = request.messages[1]["content"]
    reminder = text.split("FIELD-SPECIFIC INSTRUCTIONS:\n")[1]
    assert reminder.startswith("Identify every age group targeted by the trial")
    assert "Do not invent age cutoffs when unstated; use NA." in reminder
    assert "Examples of" not in reminder


@pytest.mark.parametrize("field", [name for name, _ in FIELDS])
def test_known_status_omission_is_scoped_to_biomarker_fields(field):
    source = "BRAF status must be known; either result is allowed."
    text = build_request(0, 0, source, field, {}, "all", PROMPTS).messages[1]["content"]
    rules = (
        "A requirement for biomarker status to be known, tested, assessed, or available is not part of a trial space.",
        "Include only requirements for positive or negative biomarker results, including specific alterations, expression levels, or thresholds.",
        "output NA when no other biomarker result restriction applies.",
        "If the trial requires a particular biomarker result and specifies whether it will be assessed during screening, note that.",
    )
    for rule in rules:
        assert (rule in text) == (
            field in {"biomarker_requirements"}
        )


def test_authored_edits_reach_only_relevant_fields_and_invalidate_their_cache(
    tmp_path,
):
    source_dir = Path(__file__).parents[1] / "src/matchminer_ai/prompts"
    for source in source_dir.glob("trial.hierarchy.*.txt"):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    primer_path = tmp_path / "trial.user.primer.txt"
    primer = (source_dir / primer_path.name).read_text()
    primer_path.write_text(primer)
    assert PROMPTS["anti_boilerplate"] in primer
    assert all(p in primer for p in PROMPTS["omissions"].split("\n\n"))
    original = build_request(
        0, 0, "trial", "biomarker_requirements", {}, "all", load_prompts(tmp_path)
    )
    original_age = Hierarchy(0, "trial", load_prompts(tmp_path)).requests()[0]
    cache = ResponseCache(tmp_path / "cache", {"model": "synthetic"})
    cache.put(original, Completion(branches(("18+", "all"))))
    primer_path.write_text(
        primer.replace("-Re biomarkers:", "-Re biomarkers: NEW AUTHORED RULE.")
    )
    changed = build_request(
        0, 0, "trial", "biomarker_requirements", {}, "all", load_prompts(tmp_path)
    )
    assert "NEW AUTHORED RULE." in changed.messages[1]["content"]
    assert cache.get(changed) is None
    assert (
        Hierarchy(0, "trial", load_prompts(tmp_path)).requests()[0].key
        == original_age.key
    )
    for field in ("biomarker_requirements",):
        text = build_request(
            0, 0, "trial", field, {}, "all", load_prompts(tmp_path)
        ).messages[1]["content"]
        assert (
            "NEW AUTHORED RULE."
            in text.split("FIELD-SPECIFIC INSTRUCTIONS (verbatim):\n")[1]
        )
    primer_path.write_text(
        primer.replace("-Re disease state:", "Changed field heading:")
    )
    with pytest.raises(ValueError, match="field-reminder sections"):
        load_prompts(tmp_path)
    primer_path.write_text(
        primer.replace("Details about measurable disease", "Unexpected section layout")
    )
    with pytest.raises(ValueError, match="shared-rule sections"):
        load_prompts(tmp_path)


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
    if field == "cancer_primary_site":
        return branches(("breast cancer", request.scope))
    if field == "histology":
        return branches(("invasive ductal carcinoma", request.scope))
    if field == "disease_state":
        return branches(("localized", request.scope))
    if field == "prior_treatment_requirements":
        return branches(("lumpectomy with negative margins", request.scope))
    if field == "biomarker_requirements":
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
    for stage in range(len(FIELDS) + 1):
        requests = state.requests()
        calls.extend(requests)
        for request in requests:
            assert (
                list(request.parent) == [key for key, _ in FIELDS[:stage]]
                if stage < len(FIELDS)
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
        "Prior treatment requirements: lumpectomy with negative margins." in line
        for line in lines
    )
    assert [r.field_name for r in calls].count("boilerplate") == 1
    assert len(calls) == 2 * len(FIELDS)
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


def test_previous_hierarchy_version_cannot_reuse_node_cache(tmp_path, monkeypatch):
    from matchminer_ai.trials import hierarchy

    cache = ResponseCache(tmp_path, {"model": "synthetic"})
    with monkeypatch.context() as old:
        old.setattr(hierarchy, "VERSION", "trial-hierarchy-v4")
        request = Hierarchy(0, "trial", PROMPTS).requests()[0]
        cache.put(request, Completion(branches(("18+", "all"))))
    assert hierarchy.VERSION == "trial-hierarchy-v5"
    assert cache.get(Hierarchy(0, "trial", PROMPTS).requests()[0]) is None


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
            if self.invalid and "CURRENT FIELD: age\n" in text:
                self.invalid -= 1
                answers.append("not JSON")
                finishes.append("stop")
                continue
            if self.fail_trial and "TRIAL DOCUMENT:\nBAD\nEND TRIAL DOCUMENT" in text:
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
                    "cancer_primary_site": "lung cancer",
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
    assert len(backend.calls) == 10
    age_calls = [
        call
        for call in backend.calls
        if "CURRENT FIELD: age\n" in call[0][1]["content"]
    ]
    assert "failed validation" in age_calls[1][0][-1]["content"]
    assert age_calls[1][0][-2]["content"] == "not JSON"
    assert (
        age_calls[2][1]["remote"]["extra_body"]["chat_template_kwargs"][
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
    assert len(backend.calls) == 10
    assert "validation_error" in result.reasoning_outputs[0]


def test_failed_trial_does_not_export_partial_spaces_or_break_other_trials():
    backend = FakeBackend(fail_trial=True)
    result = summarize_hierarchically(["GOOD", "BAD"], backend, trial_config=runtime())
    assert result.finish_reasons == ["stop", "error"]
    assert result.final_outputs[1] == ""
    assert len(backend.calls) == 10  # Failed trial's age and independent boilerplate.


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
    assert len(backend.calls) == 8
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
    config["sampling_params"]["max_tokens"] = 50000
    messages = [[{"role": "user", "content": "source text"}]]
    prompts, _ = _bounded_prompts(messages, config)
    assert prompts[0].max_tokens == 42
    assert prompts[0].messages == messages[0]
    large_context, _ = _bounded_prompts(messages, {**config, "max_model_len": 100000})
    assert large_context[0].max_tokens == 50000
    assert large_context[0].messages == messages[0]
    with pytest.raises(ValueError, match="not truncated"):
        _bounded_prompts(messages, {**config, "max_model_len": 256})
