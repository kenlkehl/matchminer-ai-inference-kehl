import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from matchminer_ai.config import MMAIConfig
from matchminer_ai.matching import (
    exclusion_criteria_check,
    exclusion_criteria_check_with_llm,
    generate_candidate_matches,
    interpret_match_quality,
    score_match_quality,
    score_match_quality_with_llm,
)


def test_generate_candidate_matches_returns_top_k_per_query():
    """Return top-k matches per query sorted by similarity."""
    query_df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    corpus_df = pd.DataFrame(
        {
            "space_trial_id": ["T1-1", "T2-1", "T3-1"],
            "trial_id": ["T1", "T2", "T3"],
            "clinical_space_number": [1, 1, 1],
            "embedding": [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        }
    )

    result = generate_candidate_matches(query_df, corpus_df, k=2)

    assert list(result.columns[:4]) == [
        "patient_id",
        "space_trial_id",
        "similarity_score",
        "rank",
    ]
    assert (result["patient_id"].value_counts() == 2).all()
    p1 = result[result["patient_id"] == "P1"]["space_trial_id"].tolist()
    p2 = result[result["patient_id"] == "P2"]["space_trial_id"].tolist()
    assert p1[0] == "T1-1"
    assert p2[0] == "T3-1"


def test_generate_candidate_matches_returns_all_when_k_none():
    """Return all corpus rows per query when k is None, sorted by similarity."""
    query_df = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "embedding": [[1.0, 0.0]],
        }
    )
    corpus_df = pd.DataFrame(
        {
            "space_trial_id": ["T1-1", "T2-1"],
            "trial_id": ["T1", "T2"],
            "clinical_space_number": [1, 1],
            "embedding": [[0.0, 1.0], [1.0, 0.0]],
        }
    )

    result = generate_candidate_matches(query_df, corpus_df, k=None)

    assert result["space_trial_id"].tolist() == ["T2-1", "T1-1"]
    assert result["rank"].tolist() == [1, 2]


class _FakeDecisions:
    """Stands in for the decision model: records each request and answers P(yes) by question name."""

    def __init__(self, answer):
        self.answer, self.requests = answer, []

    def __call__(self, requests, *, config=None, progress=None):
        self.requests.extend(requests)
        return ([{name: self.answer(state, name) for name in questions} for state, questions in requests],
                {"model_name": "LiquidAI/d1-2B", "model_sha": "pinned"})


def _config(**sections):
    return MMAIConfig(preset_name="default", debug_mode=False, trial={}, patient={}, local={}, remote={}, embedding={},
                      model_metadata_cache_dir=None, raw=sections)


def test_score_match_quality_asks_the_five_match_questions_and_scores_them(monkeypatch):
    """One state per pair (patient key section, cohort key criteria), five noul questions, expected score."""
    probabilities = {"reasonableness": 0.9, "cancer_type": 0.5, "burden": 0.0, "treatment_history": 0.0, "biomarker": 1.0}
    fake = _FakeDecisions(lambda state, name: probabilities[name] if "Lung" in state else 0.01)
    monkeypatch.setattr("matchminer_ai.matching.rerank.run_decisions", fake)
    pairs = pd.DataFrame({"patient_id": ["P1", "P2"], "space_trial_id": ["T1-1", "T2-1"],
                          "cancer_history_summary": ["Lung adenocarcinoma.", "Breast cancer."],
                          "cohort_key_criteria": ["Metastatic NSCLC.", "Melanoma."]})
    result, metadata = score_match_quality(pairs, config=_config(match_quality={"score_cutoff": 0.2}),
                                           return_metadata=True)
    state, questions = fake.requests[0]
    assert state.index("Lung adenocarcinoma.") < state.index("Metastatic NSCLC.")
    assert list(questions) == ["reasonableness", "cancer_type", "burden", "treatment_history", "biomarker"]
    assert questions["reasonableness"].startswith("Is this patient potentially eligible for this cohort")
    assert "if the patient is clearly ineligible, answer no" in questions["reasonableness"]
    assert "evidence_status" not in "".join(questions.values())
    assert result["patient_id"].tolist() == ["P1"]  # P2 scores below the cutoff
    assert result["match_quality_score"].iloc[0] == pytest.approx(0.9 * (1 + 0.5 + 2) / 6)
    assert result["match_quality_biomarker"].iloc[0] == 1.0
    assert metadata["model_metadata"]["score_match_quality"]["model_name"] == "LiquidAI/d1-2B"


def test_exclusion_criteria_check_yes_means_potentially_eligible(monkeypatch):
    fake = _FakeDecisions(lambda state, name: 0.9 if "ECOG 1" in state else 0.2)
    monkeypatch.setattr("matchminer_ai.matching.exclusion_check.run_decisions", fake)
    pairs = pd.DataFrame({"patient_id": ["P1", "P2"], "cohort_trial_id": ["T1-1", "T1-1"],
                          "general_exclusion_criteria_evidence": ["ECOG 1.", "Active hepatitis B."],
                          "cohort_boilerplate_criteria": ["ECOG 0-1. No active hepatitis B."] * 2})
    result = exclusion_criteria_check(pairs, config=_config(exclusion_criteria={"pass_cutoff": 0.5}))
    assert list(fake.requests[0][1]) == ["boilerplate"]
    assert fake.requests[0][1]["boilerplate"].startswith("Is this patient potentially eligible")
    assert result["exclusion_criteria_pass"].tolist() == [True, False]
    assert result["exclusion_criteria_pass_probability"].tolist() == [0.9, 0.2]
    assert result["exclusion_score"].tolist() == pytest.approx([0.9, 0.8])  # the predicted outcome's probability
    kept = exclusion_criteria_check(pairs, config=_config(), filter_excluded=True)
    assert kept["patient_id"].tolist() == ["P1"]


def test_check_eligibility_criteria_asks_each_tagged_line_once_per_state(monkeypatch):
    """Yes = the patient does not clearly fail the line; a low P(yes) flags a clearly failed line."""
    from matchminer_ai.matching import check_eligibility_criteria
    items = json.dumps([
        {"number": 1, "heading": "Inclusion criteria:", "text": "Age >= 18.", "criterion_type": "inclusion", "boilerplate": False},
        {"number": 2, "heading": "Exclusion criteria:", "text": "Active hepatitis B.", "criterion_type": "exclusion",
         "boilerplate": True},
        {"number": 3, "heading": "Exclusion criteria:", "text": "Unclassified.", "criterion_type": None, "boilerplate": False}])
    fake = _FakeDecisions(lambda state, name: {"criterion_1": 0.95, "criterion_2": 0.1}[name])
    monkeypatch.setattr("matchminer_ai.matching.criteria_check.run_decisions", fake)
    pairs = pd.DataFrame({"patient_id": ["P1", "P1"], "cohort_trial_id": ["T1-1", "T1-2"],
                          "cancer_history_summary": ["Lung cancer."] * 2,
                          "general_exclusion_criteria_evidence": ["Chronic hepatitis B, active."] * 2,
                          "cohort_criteria_items": [items, items]})
    result, metadata = check_eligibility_criteria(pairs, config=_config(), return_metadata=True)
    assert len(fake.requests) == 1  # the same summary and criteria are asked once
    state, questions = fake.requests[0]
    assert "Other relevant history:\nChronic hepatitis B" in state and "[2] Active hepatitis B." in state
    assert list(questions) == ["criterion_1", "criterion_2"]
    assert questions["criterion_2"].endswith("CRITERION TO ASSESS\n[2] Exclusion criterion: Active hepatitis B.")
    assert result[["cohort_trial_id", "criterion_number", "criterion_clearly_fails"]].values.tolist() == [
        ["T1-1", 1, False], ["T1-1", 2, True], ["T1-2", 1, False], ["T1-2", 2, True]]
    assert metadata["model_metadata"]["check_eligibility_criteria"]["unchecked_lines"] == 2
    with pytest.raises(ValueError, match="remove_boilerplate_criteria"):
        check_eligibility_criteria(pairs.drop(columns=["cohort_criteria_items"]), config=_config())


def test_interpret_match_quality_maps_state_ranges_and_orients_to_the_decision(monkeypatch):
    fake = _FakeDecisions(lambda state, name: 0.2)
    monkeypatch.setattr("matchminer_ai.matching.rerank.run_decisions", fake)
    seen = {}

    def attribute(state, questions, target, ranges, *, config=None, top_k=30):
        seen.update({name: state[a:b] for name, (a, b) in ranges.items()}, target=target)
        return {"method": "gradient_x_input", "target": target, "coverage": {},
                "token_attributions": {"patient_key_summary": [
                    {"text": "Lung", "start": 0, "end": 4, "raw_score": 2.0, "normalized_score": 1.0,
                     "importance": 1.0, "direction": "supports_prediction"}]}}

    monkeypatch.setattr("matchminer_ai.matching.rerank.attribute_decision", attribute)
    pairs = pd.DataFrame({"patient_id": ["P1"], "space_trial_id": ["T1-1"],
                          "cancer_history_summary": ["Lung cancer."], "cohort_key_criteria": ["Melanoma."]})
    result = interpret_match_quality(pairs, config=_config())
    assert seen == {"cohort_key_criteria": "Melanoma.", "patient_key_summary": "Lung cancer.", "target": "reasonableness"}
    assert result["attribution_target"].tolist() == ["reasonableness_no"]
    token = result["token_attributions"].iloc[0]["patient_key_summary"][0]
    assert token["normalized_score"] == -1.0 and token["direction"] == "opposes_prediction"  # supports yes -> opposes no


def test_run_checker_reuses_cached_pipeline(monkeypatch):
    """Repeated checker calls with the same config should reuse the pipeline."""
    from matchminer_ai.matching import inference

    inference.clear_checker_pipeline_cache()
    tokenizer_calls = []
    pipeline_calls = []

    class FakeTokenizer:
        """Fake tokenizer factory used by the checker pipeline."""

        @staticmethod
        def from_pretrained(model_name, **kwargs):
            """Record tokenizer construction and return a sentinel tokenizer."""
            tokenizer_calls.append((model_name, kwargs))
            return f"tokenizer:{model_name}"

    class FakePipeline:
        """Fake text-classification pipeline that records prompt batches."""

        def __init__(self):
            """Initialize prompt batch capture."""
            self.prompt_batches = []

        def __call__(self, prompts, **_kwargs):
            """Return deterministic positive-label outputs for each prompt."""
            self.prompt_batches.append(list(prompts))
            return [
                {"label": "POSITIVE", "score": float(index)}
                for index, _prompt in enumerate(prompts)
            ]

    def fake_pipeline(*args, **kwargs):
        """Build and record a fake checker pipeline."""
        checker_pipeline = FakePipeline()
        pipeline_calls.append((args, kwargs, checker_pipeline))
        return checker_pipeline

    fake_transformers = SimpleNamespace(
        AutoTokenizer=FakeTokenizer,
        pipeline=fake_pipeline,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "matchminer_ai.matching.inference.get_model_metadata",
        lambda model_name, cache_dir=None: {"model_name": model_name},
    )

    checker_config = {
        "model_name": "checker/model",
        "device": "cpu",
        "max_length": 3192,
    }
    first_outputs, first_metadata = inference.run_checker(
        ["prompt 1"],
        checker_config=checker_config,
        model_metadata_cache_dir=".cache/models",
    )
    second_outputs, second_metadata = inference.run_checker(
        ["prompt 2", "prompt 3"],
        checker_config=checker_config,
        model_metadata_cache_dir=".cache/models",
    )

    assert first_outputs == [{"label": "POSITIVE", "score": 0.0}]
    assert second_outputs == [
        {"label": "POSITIVE", "score": 0.0},
        {"label": "POSITIVE", "score": 1.0},
    ]
    assert first_metadata == {"model_name": "checker/model"}
    assert second_metadata == {"model_name": "checker/model"}
    assert len(tokenizer_calls) == 1
    assert len(pipeline_calls) == 1
    assert pipeline_calls[0][0] == ("text-classification", "checker/model")
    assert pipeline_calls[0][1]["tokenizer"] == "tokenizer:checker/model"
    assert pipeline_calls[0][1]["device"] == "cpu"
    assert pipeline_calls[0][1]["max_length"] == 3192
    assert pipeline_calls[0][2].prompt_batches == [
        ["prompt 1"],
        ["prompt 2", "prompt 3"],
    ]

    inference.clear_checker_pipeline_cache()


def test_clear_checker_pipeline_cache_forces_reload(monkeypatch):
    """Clearing the checker cache should make the next run rebuild the pipeline."""
    from matchminer_ai.matching import inference

    inference.clear_checker_pipeline_cache()
    pipeline_calls = []

    class FakeTokenizer:
        """Fake tokenizer factory for checker cache reload tests."""

        @staticmethod
        def from_pretrained(model_name, **kwargs):
            """Return a sentinel tokenizer value."""
            return f"tokenizer:{model_name}"

    def fake_pipeline(*args, **kwargs):
        """Record checker pipeline construction and return fixed outputs."""
        pipeline_calls.append((args, kwargs))
        return lambda prompts, **_kwargs: [{"label": "NEGATIVE", "score": 0.5} for _ in prompts]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeTokenizer, pipeline=fake_pipeline),
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.inference.get_model_metadata",
        lambda model_name, cache_dir=None: {"model_name": model_name},
    )

    checker_config = {"model_name": "checker/model", "device": "cpu"}
    inference.run_checker(["prompt 1"], checker_config=checker_config)
    inference.clear_checker_pipeline_cache()
    inference.run_checker(["prompt 2"], checker_config=checker_config)

    assert len(pipeline_calls) == 2

    inference.clear_checker_pipeline_cache()


class _MockLLMBackend:
    """Minimal LLM backend that records calls and returns fixed outputs."""

    def __init__(self, outputs):
        """Store fixed outputs and initialize call capture fields."""
        self.outputs = outputs
        self.last_prompt_list = None
        self.last_llm_config = None
        self.last_model_metadata_cache_dir = None

    def generate_llm_outputs(
        self,
        *,
        prompt_list,
        llm_config,
        model_metadata_cache_dir=None,
    ):
        """Record generation inputs and return deterministic LLM outputs."""
        self.last_prompt_list = prompt_list
        self.last_llm_config = llm_config
        self.last_model_metadata_cache_dir = model_metadata_cache_dir
        return SimpleNamespace(
            final_outputs=self.outputs,
            reasoning_outputs=[
                f"reasoning {idx}" for idx, _ in enumerate(self.outputs)
            ],
            model_metadata={"model_name": llm_config["model_name"]},
        )


def _llm_config(*, debug_mode: bool = False) -> MMAIConfig:
    """Build a minimal config for LLM checker tests."""
    return MMAIConfig(
        preset_name="default",
        debug_mode=debug_mode,
        trial={},
        patient={},
        local={},
        remote={},
        embedding={},
        model_metadata_cache_dir=".mmai_cache/model_metadata",
        llm_match_quality={
            "reasoning_parser": "auto",
            "local": {
                "model_name": "llm/model",
                "engine": {
                    "max_model_len": 50000,
                    "tensor_parallel_size": 1,
                },
                "generation": {"max_tokens": 15000},
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "remote": {
                "model_name": "llm/model",
                "request_params": {"max_tokens": 15000},
                "extra_body": {},
            },
            "prompt_file": "match_quality.labels.system.txt",
        },
        llm_exclusion_criteria={
            "reasoning_parser": "auto",
            "local": {
                "model_name": "llm/model",
                "engine": {
                    "max_model_len": 50000,
                    "tensor_parallel_size": 1,
                },
                "generation": {"max_tokens": 20000},
                "chat_template_kwargs": {"enable_thinking": True},
            },
            "remote": {
                "model_name": "llm/model",
                "request_params": {"max_tokens": 20000},
                "extra_body": {},
            },
            "prompt_file": "llm_exclusion_criteria.user.txt",
        },
        raw={},
    )


_LABELS_ANSWER = json.dumps({label: {"answer": "yes" if label in ("reasonableness", "cancer_type", "biomarker") else "no",
                                     "evidence_status": "supported", "reason": "Fabricated reason."}
                             for label in ("reasonableness", "cancer_type", "burden", "treatment_history", "biomarker")})


@pytest.mark.parametrize("model_name,expects_header", [("llm/model", False), ("google/gemma-4-31B-it", False), ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", False), ("openai/gpt-oss-120b", True)])
def test_score_match_quality_with_llm_builds_training_prompt_and_parses(monkeypatch, model_name, expects_header):
    """Run LLM match scoring through package backend config and score parser."""
    config = _llm_config()
    config.llm_match_quality["local"]["model_name"] = model_name
    captured_messages = []

    def fake_build_prompt_list(messages_list, *, llm_config):
        """Capture match-quality messages and return one rendered prompt."""
        captured_messages.extend(messages_list)
        return [SimpleNamespace(prompt_text="rendered", max_tokens=15000)]

    backend = _MockLLMBackend([_LABELS_ANSWER])
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.build_prompt_list",
        fake_build_prompt_list,
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.get_llm_backend",
        lambda config: backend,
    )
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cohort_trial_id": "T1-cohort-1",
                "patient_summary": "Patient has metastatic lung cancer.\n\nOther relevant history:\nDialysis.",
                "cohort_key_criteria": "Trial for metastatic lung cancer.",
                "cohort_boilerplate_criteria": "No dialysis.",
            }
        ]
    )

    result, metadata = score_match_quality_with_llm(
        pairs,
        config=config,
        return_metadata=True,
    )

    # Reasonable (1) + cancer type (1) + biomarker (2); each label is also returned.
    assert result["llm_match_quality_score"].tolist() == [4]
    assert result["llm_reasonableness"].tolist() == [True] and result["llm_burden"].tolist() == [False]
    assert "llm_match_quality_verdict" not in result.columns
    assert "llm_match_quality_response" not in result.columns
    assert "llm_match_quality_reasoning" not in result.columns
    assert "llm_match_quality_answer_text" not in result.columns
    assert "llm_match_quality_reasoning_text" not in result.columns
    assert "llm_match_quality_parse_status" not in result.columns
    system = captured_messages[0][0]["content"]
    assert system.startswith("Reasoning: high\n\n") == expects_header and "reasonableness" in system
    user = captured_messages[0][1]["content"]
    assert "metastatic lung cancer" in user and "COHORT KEY ELIGIBILITY CRITERIA" in user
    assert "Dialysis" not in user and "No dialysis" not in user  # key text only
    assert backend.last_llm_config["max_model_len"] == 50000
    assert backend.last_llm_config["sampling_params"]["max_tokens"] == 15000
    assert metadata["model_metadata"]["score_match_quality_with_llm"]["model_name"] == (
        model_name
    )


def test_score_match_quality_with_llm_includes_debug_columns(monkeypatch):
    """Include raw LLM response details only when debug mode is enabled."""

    def fake_build_prompt_list(messages_list, *, llm_config):
        """Return one rendered prompt for match-quality debug output tests."""
        return [SimpleNamespace(prompt_text="rendered", max_tokens=15000)]

    backend = _MockLLMBackend([_LABELS_ANSWER])
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.build_prompt_list",
        fake_build_prompt_list,
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.get_llm_backend",
        lambda config: backend,
    )
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "space_trial_id": "T1-1",
                "cancer_history_summary": "Patient has metastatic lung cancer.",
                "clinical_space_summary": "Trial for metastatic lung cancer.",
            }
        ]
    )

    result = score_match_quality_with_llm(pairs, config=_llm_config(debug_mode=True))

    assert result["llm_match_quality_answer_text"].tolist() == [_LABELS_ANSWER]
    assert result["llm_match_quality_reasoning_text"].tolist() == ["reasoning 0"]
    assert result["llm_match_quality_parse_status"].tolist() == ["parsed"]


@pytest.mark.parametrize("model_name,expects_header", [("llm/model", False), ("google/gemma-4-31B-it", False), ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", False), ("openai/gpt-oss-120b", True)])
def test_exclusion_criteria_check_with_llm_builds_training_prompt_and_parses(monkeypatch, model_name, expects_header):
    """Run LLM exclusion scoring through package backend config and yes/no parser."""
    config = _llm_config()
    config.llm_exclusion_criteria["local"]["model_name"] = model_name
    captured_messages = []

    def fake_build_prompt_list(messages_list, *, llm_config):
        """Capture exclusion-check messages and return one rendered prompt."""
        captured_messages.extend(messages_list)
        return [SimpleNamespace(prompt_text="rendered", max_tokens=20000)]

    backend = _MockLLMBackend(["No!"])
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.build_prompt_list",
        fake_build_prompt_list,
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.get_llm_backend",
        lambda config: backend,
    )
    matches = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "T1",
                "general_exclusion_criteria": "Excludes uncontrolled brain mets.",
                "general_exclusion_criteria_evidence": "No active brain mets.",
            }
        ]
    )

    result, metadata = exclusion_criteria_check_with_llm(
        matches,
        config=config,
        return_metadata=True,
    )

    assert result["llm_exclusion_criteria_pass"].tolist() == [True]
    assert "llm_exclusion_criteria_verdict" not in result.columns
    assert "llm_exclusion_criteria_response" not in result.columns
    assert "llm_exclusion_criteria_reasoning" not in result.columns
    assert "llm_exclusion_criteria_answer_text" not in result.columns
    assert "llm_exclusion_criteria_reasoning_text" not in result.columns
    assert "llm_exclusion_criteria_parse_status" not in result.columns
    assert captured_messages[0][0] == {"role": "system", "content": "Reasoning: high" if expects_header else ""}
    assert "uncontrolled brain mets" in captured_messages[0][1]["content"]
    assert backend.last_llm_config["max_model_len"] == 50000
    assert backend.last_llm_config["sampling_params"]["max_tokens"] == 20000
    assert metadata["model_metadata"]["exclusion_criteria_check_with_llm"] == {
        "model_name": model_name
    }


def test_llm_checks_report_remote_model_metadata(monkeypatch):
    """HTTP-backed checks keep the backend's endpoint provenance."""

    def fake_build_prompt_list(messages_list, *, llm_config):
        """Return one rendered prompt per checker input."""
        return [
            SimpleNamespace(prompt_text="rendered", max_tokens=10)
            for _ in messages_list
        ]

    backend = _MockLLMBackend(["Final score: 4"])
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.build_prompt_list",
        fake_build_prompt_list,
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.get_llm_backend",
        lambda config: backend,
    )
    config = _llm_config()
    config.remote = {
        "enabled": True,
        "server_urls": ["http://endpoint/v1"],
    }
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "space_trial_id": "T1-1",
                "cancer_history_summary": "Patient summary",
                "clinical_space_summary": "Trial summary",
            }
        ]
    )

    _result, metadata = score_match_quality_with_llm(
        pairs,
        config=config,
        return_metadata=True,
    )

    assert metadata["model_metadata"] == {
        "score_match_quality_with_llm": {"model_name": "llm/model"}
    }


def test_exclusion_criteria_check_with_llm_includes_debug_columns(monkeypatch):
    """Include raw LLM exclusion details only when debug mode is enabled."""

    def fake_build_prompt_list(messages_list, *, llm_config):
        """Return one rendered prompt for exclusion debug output tests."""
        return [SimpleNamespace(prompt_text="rendered", max_tokens=20000)]

    backend = _MockLLMBackend(["No!"])
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.build_prompt_list",
        fake_build_prompt_list,
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.llm_checks.get_llm_backend",
        lambda config: backend,
    )
    matches = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "T1",
                "general_exclusion_criteria": "Excludes uncontrolled brain mets.",
                "general_exclusion_criteria_evidence": "No active brain mets.",
            }
        ]
    )

    result = exclusion_criteria_check_with_llm(
        matches,
        config=_llm_config(debug_mode=True),
    )

    assert result["llm_exclusion_criteria_answer_text"].tolist() == ["No!"]
    assert result["llm_exclusion_criteria_reasoning_text"].tolist() == ["reasoning 0"]
    assert result["llm_exclusion_criteria_parse_status"].tolist() == ["parsed"]
