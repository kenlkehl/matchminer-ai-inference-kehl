from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
import pytest

import matchminer_ai.trials.drug_catalog as catalog_module
from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.model_profiles import apply_model_profile
from matchminer_ai.matching import (
    RUBRIC_CRITERIA,
    build_good_option_checker_text,
    build_good_option_messages,
    check_good_options,
    evaluate_good_options,
    good_option_evidence_budget,
    pack_good_option_evidence,
    parse_good_option_response,
    score_good_options,
    score_good_options_with_llm,
)
from matchminer_ai.trials import (
    DrugIdentity,
    DrugSummary,
    EvidencePassage,
    GoodOptionCatalog,
    ResearchSettings,
    TrialDrugAssignment,
    build_good_option_catalog,
    load_good_option_catalog,
    research_drug,
)
from matchminer_ai.trials.drug_research import (
    GeneralWebEvidenceSource,
    HostRateLimiter,
    build_facet_query,
)


def no_classes(_drug, _evidence):
    """Class assignment double: the class axis is exercised in its own tests."""

    return {"classes": []}


def _summary(drug_id: str, name: str, *, status: str = "complete") -> DrugSummary:
    return DrugSummary(
        drug_id=drug_id,
        preferred_name=name,
        ncit_code="",
        research_status=status,
        synthesis_status="ok" if status == "complete" else "blocked",
        structured_facts={},
        good_option_summary=(
            f"Drug: {name}\nMechanism and targets:\n- TARGET_MARKER inhibition.\n"
            "Human efficacy by tumor type:\n- Responses in synthetic cancer.\n"
            "Biomarker prevalence:\n- TARGET_MARKER occurs in 30% of the full population.\n"
            "Biomarker-directed human efficacy:\n- Human responses were observed."
            if status == "complete"
            else ""
        ),
        help_me_choose_summary=(
            f"Drug: {name}\nMechanism, efficacy, and safety synthesis."
            if status == "complete"
            else ""
        ),
        evidence_count=4,
    )


def _catalog(*, blocked_second: bool = False) -> GoodOptionCatalog:
    summaries = [
        _summary("D1", "Novel Agent"),
        _summary(
            "D2", "Second Agent", status="blocked" if blocked_second else "complete"
        ),
        _summary("D3", "Control Agent"),
    ]
    assignments = [
        TrialDrugAssignment(
            trial_id="NCT12345678",
            drug_id="D1",
            preferred_name="Novel Agent",
            registry_name="Novel Agent",
            intervention_type="DRUG",
            role="investigational",
            role_confidence="high",
            scoreable=True,
        ),
        TrialDrugAssignment(
            trial_id="NCT12345678",
            drug_id="D2",
            preferred_name="Second Agent",
            registry_name="Second Agent",
            intervention_type="BIOLOGICAL",
            role="uncertain",
            role_confidence="low",
            scoreable=True,
        ),
        TrialDrugAssignment(
            trial_id="NCT12345678",
            drug_id="D3",
            preferred_name="Control Agent",
            registry_name="Control Agent",
            intervention_type="DRUG",
            role="control",
            role_confidence="high",
            scoreable=False,
        ),
    ]
    return GoodOptionCatalog(
        path=Path("/tmp/test-catalog"),
        manifest={"compatibility_id": "compat-v2"},
        trial_registry=pd.DataFrame(
            [
                {
                    "trial_id": "NCT12345678",
                    "title": "Trial",
                    "registry_status": "ok",
                    "phases_json": "[]",
                    "brief_summary": "",
                }
            ]
        ),
        trial_drug_index=pd.DataFrame([item.to_record() for item in assignments]),
        drug_summaries=pd.DataFrame([item.to_record() for item in summaries]),
        drug_evidence=pd.DataFrame(
            [
                {
                    "drug_id": "D1",
                    "title": "Evidence",
                    "url": "https://example.test/evidence",
                }
            ]
        ),
        drug_research_attempts=pd.DataFrame(),
    )


def _response() -> str:
    assessments = []
    for index, drug in enumerate(("Novel Agent", "Second Agent")):
        points = (1, 1, 1, 1) if index == 0 else (1, 0, 1, 0)
        assessment: dict[str, Any] = {
            "drug_name": drug,
            "targeted_biomarkers": ["TARGET_MARKER"],
        }
        for criterion, point in zip(RUBRIC_CRITERIA, points, strict=True):
            assessment[criterion] = {
                "point": point,
                "rationale": f"Synthetic rationale for {criterion}.",
            }
        assessments.append(assessment)
    return json.dumps(
        {
            "patient_disease_type": "Synthetic cancer",
            "drug_assessments": assessments,
            "key_uncertainties": ["Small studies"],
        }
    )


def test_prompt_is_patient_first_metadata_free_and_omits_control() -> None:
    catalog = _catalog()
    messages = build_good_option_messages(
        patient_summary="PRIVATE_PATIENT TARGET_MARKER synthetic cancer",
        drug_summaries=catalog.scoreable_summaries_for_trial("NCT12345678"),
    )
    prompt = messages[1]["content"]

    assert (
        prompt.index("PATIENT CANCER HISTORY")
        < prompt.index("EXPERIMENTAL DRUGS IN THIS TRIAL")
        < prompt.index("HOW TO SCORE")
    )
    assert "PRIVATE_PATIENT" in prompt
    assert "Novel Agent" in prompt and "Second Agent" in prompt
    assert "Control Agent" not in prompt
    for forbidden in (
        "https://",
        "NCT12345678",
        "drug_only_query",
        "source_id",
        "research_notices",
        "overall_status",
    ):
        assert forbidden not in prompt
    assert "evidence_labels" not in prompt
    assert "control arms" in messages[0]["content"]


def test_response_parser_keeps_four_binary_criteria_without_evidence_ids() -> None:
    summaries = _catalog().scoreable_summaries_for_trial("NCT12345678")
    parsed = parse_good_option_response(_response(), drug_summaries=summaries)

    assert parsed.status == "ok"
    assert parsed.points == 6
    assert parsed.max_points == 8
    assert parsed.score == pytest.approx(0.75)
    assert all(
        "evidence_labels" not in assessment[criterion]
        for assessment in parsed.drug_assessments
        for criterion in RUBRIC_CRITERIA
    )


def test_response_parser_normalizes_nested_criteria_wrapper() -> None:
    response = json.loads(_response())
    for assessment in response["drug_assessments"]:
        assessment["criteria"] = {
            criterion: assessment.pop(criterion) for criterion in RUBRIC_CRITERIA
        }
    summaries = _catalog().scoreable_summaries_for_trial("NCT12345678")

    parsed = parse_good_option_response(
        json.dumps(response),
        drug_summaries=summaries,
    )

    assert parsed.status == "ok"
    assert parsed.points == 6
    assert parsed.max_points == 8
    assert all(
        criterion in assessment
        for assessment in parsed.drug_assessments
        for criterion in RUBRIC_CRITERIA
    )
    assert all("criteria" not in assessment for assessment in parsed.drug_assessments)


def test_response_parser_keeps_nested_criteria_validation_strict() -> None:
    response = json.loads(_response())
    for assessment in response["drug_assessments"]:
        assessment["criteria"] = {
            criterion: assessment.pop(criterion) for criterion in RUBRIC_CRITERIA
        }
    response["drug_assessments"][0]["criteria"]["disease_type_benefit"]["point"] = 2

    parsed = parse_good_option_response(
        json.dumps(response),
        drug_summaries=_catalog().scoreable_summaries_for_trial("NCT12345678"),
    )

    assert parsed.status == "parse_failed"
    assert parsed.parse_error == "Novel Agent: invalid disease_type_benefit result."


def test_llm_scoring_uses_catalog_and_marks_blocked_trial_unscored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_run(messages_list, *, config):
        del config
        captured.extend(messages_list)
        return SimpleNamespace(
            final_outputs=[_response()],
            reasoning_outputs=[""],
            finish_reasons=["stop"],
            model_metadata={"model_name": "teacher"},
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.good_options._run_good_option_llm", fake_run
    )
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "NCT12345678",
                "cancer_history_summary": "Synthetic TARGET_MARKER cancer",
                "clinical_space_summary": "MUST_NOT_APPEAR",
            }
        ]
    )
    output = score_good_options_with_llm(
        pairs, catalog=_catalog(), config=load_default_preset()
    )

    assert output.loc[0, "good_option_score"] == pytest.approx(0.75)
    assert "MUST_NOT_APPEAR" not in captured[0][1]["content"]

    blocked = score_good_options_with_llm(
        pairs, catalog=_catalog(blocked_second=True), config=load_default_preset()
    )
    assert blocked.loc[0, "good_option_status"] == "drug_research_blocked"
    assert pd.isna(blocked.loc[0, "good_option_score"])


def test_llm_scoring_rejects_token_limited_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(messages_list, *, config):
        del messages_list, config
        return SimpleNamespace(
            final_outputs=[_response()],
            reasoning_outputs=["unfinished"],
            finish_reasons=["length"],
            model_metadata={"model_name": "teacher"},
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.good_options._run_good_option_llm", fake_run
    )
    config = load_default_preset()
    config.debug_mode = True
    output = score_good_options_with_llm(
        pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "trial_id": "NCT12345678",
                    "cancer_history_summary": "Synthetic TARGET_MARKER cancer",
                }
            ]
        ),
        catalog=_catalog(),
        config=config,
    )

    assert output.loc[0, "good_option_status"] == "parse_failed"
    assert "output token limit" in output.loc[0, "good_option_parse_error"]
    assert output.loc[0, "good_option_finish_reason"] == "length"


def test_llm_scoring_retries_parse_failures_with_feedback_then_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    generations = [
        ([_response(), "{}"], ["stop", "stop"]),
        ([json.dumps({"drug_assessments": []})], ["stop"]),
        ([_response()], ["length"]),
        ([_response()], ["stop"]),
    ]

    def fake_run(messages_list, *, config):
        outputs, finish_reasons = generations[len(calls)]
        calls.append(
            {
                "messages": [list(messages) for messages in messages_list],
                "local_thinking": config.llm_good_option["local"][
                    "chat_template_kwargs"
                ]["enable_thinking"],
                "remote_thinking": config.llm_good_option["remote"]["extra_body"][
                    "chat_template_kwargs"
                ]["enable_thinking"],
            }
        )
        return SimpleNamespace(
            final_outputs=outputs,
            reasoning_outputs=["reasoning"] * len(outputs),
            finish_reasons=finish_reasons,
            model_metadata={"model_name": "teacher"},
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.good_options._run_good_option_llm", fake_run
    )
    config = load_default_preset()
    config.debug_mode = True
    output = score_good_options_with_llm(
        pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "trial_id": "NCT12345678",
                    "cancer_history_summary": "Synthetic TARGET_MARKER cancer",
                },
                {
                    "patient_id": "P2",
                    "trial_id": "NCT12345678",
                    "cancer_history_summary": "Synthetic TARGET_MARKER cancer",
                },
            ]
        ),
        catalog=_catalog(),
        config=config,
        max_parse_attempts=3,
        reasoning_off_fallback=True,
    )

    assert output["good_option_status"].tolist() == ["ok", "ok"]
    assert [len(call["messages"]) for call in calls] == [2, 1, 1, 1]
    assert [len(calls[index]["messages"][0]) for index in (1, 2, 3)] == [4, 4, 4]
    assert calls[1]["messages"][0][-2] == {"role": "assistant", "content": "{}"}
    assert (
        "No JSON object containing drug_assessments"
        in calls[1]["messages"][0][-1]["content"]
    )
    assert (
        "patient_disease_type must be non-empty"
        in calls[2]["messages"][0][-1]["content"]
    )
    assert "reached its output token limit" in calls[3]["messages"][0][-1]["content"]
    assert [call["local_thinking"] for call in calls] == [True, True, True, False]
    assert [call["remote_thinking"] for call in calls] == [True, True, True, False]
    assert config.llm_good_option["local"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def _check_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "NCT12345678",
                "cancer_history_summary": "Synthetic TARGET_MARKER cancer",
            }
        ]
    )


def _recording_run(calls: list[Any], outputs: list[str]):
    def fake_run(messages_list, *, config):
        calls.append(config)
        output = outputs[min(len(calls) - 1, len(outputs) - 1)]
        return SimpleNamespace(
            final_outputs=[output] * len(messages_list),
            reasoning_outputs=[""] * len(messages_list),
            finish_reasons=["stop"] * len(messages_list),
            model_metadata={"model_name": "teacher"},
        )

    return fake_run


def test_check_good_options_caps_output_and_retries_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        "matchminer_ai.matching.good_options._run_good_option_llm",
        _recording_run(calls, ["{}", "{}", _response()]),
    )
    config = load_default_preset()

    output, metadata = check_good_options(
        _check_pairs(), catalog=_catalog(), config=config, return_metadata=True
    )

    assert output["good_option_status"].tolist() == ["ok"]
    assert len(calls) == 3
    for call in calls:
        assert call.llm_good_option["local"]["generation"]["max_tokens"] == 50000
        assert call.llm_good_option["remote"]["request_params"]["max_tokens"] == 50000
    assert metadata["max_parse_attempts"] == 3
    assert metadata["reasoning_off_fallback"] is True
    assert metadata["good_option_check"]["max_output_tokens"] == 50000
    # The caller's config, and so catalog-build fingerprints, are untouched.
    assert config.llm_good_option["local"]["generation"]["max_tokens"] == 100000
    assert config.llm_good_option["remote"]["request_params"]["max_tokens"] == 100000


def test_check_good_options_sizes_evidence_to_the_served_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        "matchminer_ai.matching.good_options._run_good_option_llm",
        _recording_run(calls, [_response()]),
    )
    served = {"http://a:8000/v1": 131072, "http://b:8000/v1": 98304}
    discovered: list[str] = []

    def fake_discover(url, *, api_key=None, timeout=10.0):
        discovered.append(url)
        return served[url]

    monkeypatch.setattr(
        "matchminer_ai.matching.good_options.discover_served_context_tokens",
        fake_discover,
    )
    config = load_default_preset()
    config.remote["enabled"] = True
    config.remote["server_urls"] = list(served)

    _output, metadata = check_good_options(
        _check_pairs(), catalog=_catalog(), config=config, return_metadata=True
    )

    assert discovered == list(served)
    assert metadata["good_option_check"]["context_source"] == "endpoint"
    assert metadata["good_option_check"]["context_tokens"] == 98304
    assert calls[0].raw["good_option_prompt"]["context_tokens"] == 98304
    assert config.raw["good_option_prompt"]["context_tokens"] is None
    expected = good_option_evidence_budget(calls[0], fixed_prompt_chars=0)
    assert expected["context_tokens"] == 98304
    assert expected["output_tokens"] == 50000

    # An explicit context wins without asking the servers.
    discovered.clear()
    config.raw["good_option_prompt"]["context_tokens"] = 65536
    _output, metadata = check_good_options(
        _check_pairs(), catalog=_catalog(), config=config, return_metadata=True
    )
    assert discovered == []
    assert metadata["good_option_check"]["context_source"] == "configured"

    # A failed lookup is reported and falls back to the configured context.
    config.raw["good_option_prompt"]["context_tokens"] = None

    def refuse(url, *, api_key=None, timeout=10.0):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(
        "matchminer_ai.matching.good_options.discover_served_context_tokens", refuse
    )
    output, metadata = check_good_options(
        _check_pairs(), catalog=_catalog(), config=config, return_metadata=True
    )
    assert output["good_option_status"].tolist() == ["ok"]
    assert metadata["good_option_check"]["context_source"] == "preset"
    assert "refused" in metadata["good_option_check"]["context_warning"]


def test_catalog_cache_reloads_only_when_bundle_files_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    reads: list[Path] = []

    def fake_read(root, *, validate):
        reads.append(root)
        return _catalog()

    monkeypatch.setattr(catalog_module, "_read_good_option_catalog", fake_read)
    monkeypatch.setattr(catalog_module, "_CATALOG_CACHE", {})

    first = load_good_option_catalog(tmp_path, cache=True)
    assert load_good_option_catalog(tmp_path, cache=True) is first
    assert len(reads) == 1
    load_good_option_catalog(tmp_path)
    assert len(reads) == 2

    (tmp_path / "manifest.json").write_text('{"changed": true}', encoding="utf-8")
    assert load_good_option_catalog(tmp_path, cache=True) is not first
    assert len(reads) == 3

    calls: list[Any] = []
    monkeypatch.setattr(
        "matchminer_ai.matching.good_options._run_good_option_llm",
        _recording_run(calls, [_response()]),
    )
    check_good_options(_check_pairs(), catalog=str(tmp_path))
    assert len(reads) == 3


def test_classifier_aggregates_every_drug_by_criterion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_checker(
        prompts,
        *,
        checker_config,
        model_metadata_cache_dir=None,
        return_all_scores=False,
    ):
        del checker_config, model_metadata_cache_dir
        assert return_all_scores
        captured.extend(prompts)
        first = [{"label": criterion, "score": 1.0} for criterion in RUBRIC_CRITERIA]
        second = [
            {"label": criterion, "score": value}
            for criterion, value in zip(
                RUBRIC_CRITERIA, (1.0, 0.0, 1.0, 0.0), strict=True
            )
        ]
        return [first, second], {"model_name": "checker"}

    config = load_default_preset()
    config.raw["good_option_checker"]["model_name"] = "local/checker"
    monkeypatch.setattr("matchminer_ai.matching.good_options.run_checker", fake_checker)
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "NCT12345678",
                "cancer_history_summary": "Synthetic patient",
            }
        ]
    )
    with pytest.deprecated_call(match="GoodOptionChecker"):
        output, metadata = score_good_options(
            pairs, catalog=_catalog(), config=config, return_metadata=True
        )

    assert output.loc[0, "good_option_score"] == pytest.approx(0.75)
    assert len(captured) == 2
    assert all("Control Agent" not in prompt for prompt in captured)
    assert metadata["checker_input_version"].endswith("four-logit")
    with pytest.deprecated_call(match="score_good_options is deprecated"):
        evaluate_good_options(
            pairs, catalog=_catalog(), config=config, method="classifier"
        )


def test_legacy_research_argument_is_rejected() -> None:
    with pytest.raises(ValueError, match="Legacy snippet research"):
        evaluate_good_options(
            pd.DataFrame(), catalog=None, research=[object()], method="llm"
        )


class _FlakySource:
    name = "flaky"
    source_type = "authoritative"

    def __init__(self, *, always_fail: bool = False, empty: bool = False) -> None:
        self.calls = 0
        self.always_fail = always_fail
        self.empty = empty

    async def fetch(self, drug, *, facet, query, client, settings):
        del client, settings
        self.calls += 1
        if self.always_fail or self.calls == 1:
            request = httpx.Request("GET", "https://example.test")
            raise httpx.ReadTimeout("temporary", request=request)
        if self.empty:
            return []
        passage = f"{drug.preferred_name} {facet} human evidence"
        return [
            EvidencePassage(
                evidence_id=f"fake:{facet}",
                drug_id=drug.drug_id,
                facet=facet,
                source=self.name,
                source_type=self.source_type,
                title="Evidence",
                passage=passage,
                url="https://example.test/evidence",
                source_locator=facet,
                query=query,
                content_sha256=facet,
            )
        ]


def test_research_retries_and_distinguishes_empty_from_technical_failure() -> None:
    drug = DrugIdentity(drug_id="D1", preferred_name="Novel Agent")
    flaky = _FlakySource()

    async def no_sleep(_seconds: float) -> None:
        return None

    evidence, attempts, status, failures = asyncio.run(
        research_drug(
            drug,
            sources=[flaky],
            settings=ResearchSettings(max_attempts=2),
            sleep=no_sleep,
        )
    )
    assert status == "complete"
    assert evidence
    assert {item.status for item in attempts} >= {"failed", "ok"}
    assert not any("Unresolved technical facet" in item for item in failures)

    empty = _FlakySource(empty=True)
    _, empty_attempts, empty_status, _ = asyncio.run(
        research_drug(
            drug,
            sources=[empty],
            settings=ResearchSettings(max_attempts=2),
            sleep=no_sleep,
        )
    )
    assert empty_status == "complete"
    assert any(item.status == "empty" for item in empty_attempts)

    failed = _FlakySource(always_fail=True)
    _, _, failed_status, failed_messages = asyncio.run(
        research_drug(
            drug,
            sources=[failed],
            settings=ResearchSettings(max_attempts=2),
            sleep=no_sleep,
        )
    )
    assert failed_status == "blocked"
    assert any("Unresolved technical facet" in item for item in failed_messages)


class _FlakyWebProvider:
    name = "flaky-web"

    def __init__(self) -> None:
        self.max_results: list[int] = []
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int):
        self.queries.append(query)
        self.max_results.append(max_results)
        if len(self.max_results) == 1:
            raise RuntimeError("temporary search outage")
        return []


def test_general_web_search_failures_retry_with_a_bounded_result_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _FlakyWebProvider()

    async def no_sleep(_seconds: float) -> None:
        return None

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(
        "matchminer_ai.trials.drug_research.asyncio.to_thread", run_inline
    )

    _, attempts, status, _ = asyncio.run(
        research_drug(
            DrugIdentity(drug_id="D1", preferred_name="Novel Agent"),
            sources=[GeneralWebEvidenceSource(provider)],
            settings=ResearchSettings(
                max_attempts=2,
                web_results_per_query=10,
                max_web_results_per_drug=60,
            ),
            sleep=no_sleep,
        )
    )

    assert status == "complete"
    assert any(item.status == "failed" for item in attempts)
    assert any(item.status == "empty" for item in attempts)
    assert provider.max_results
    assert max(provider.max_results) == 4  # 60 results across at most 15 queries.
    assert provider.queries
    assert all("cancer treatment" in query for query in provider.queries)


def test_drug_web_queries_focus_every_adaptive_round_on_cancer_treatment() -> None:
    drug = DrugIdentity(
        drug_id="D1",
        preferred_name="CAR",
        aliases=("Synthetic CAR-T Agent",),
    )

    for round_index in range(3):
        query = build_facet_query(drug, "mechanism_targets", round_index=round_index)
        assert '"CAR"' in query
        assert '"Synthetic CAR-T Agent"' in query
        assert "cancer treatment" in query


def test_synthesis_retries_invalid_json_and_preserves_ledger_support_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(_messages, *, config, stage, reasoning_off=False):
        nonlocal calls
        del config
        assert stage == "synthesis"
        calls += 1
        if calls == 1:
            return ["not valid synthesis JSON"]
        return [
            json.dumps(
                {
                    "mechanism_and_targets": [
                        {"text": "Targets Marker A.", "support_ids": ["P1"]}
                    ],
                    "efficacy_by_tumor": [],
                    "biomarker_prevalence": [],
                    "biomarker_directed_efficacy": [],
                    "safety": [],
                    "limitations": [],
                }
            )
        ]

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)
    config = load_default_preset()
    config.good_option_catalog["synthesis_max_attempts"] = 2
    evidence = EvidencePassage(
        evidence_id="ledger:E1",
        drug_id="D1",
        facet="mechanism_targets",
        source="test",
        source_type="authoritative",
        title="Evidence",
        passage="Novel Agent targets Marker A.",
        url="https://example.test/evidence",
        source_locator="E1",
    )

    result = asyncio.run(
        catalog_module.synthesize_serially(
            [(DrugIdentity(drug_id="D1", preferred_name="Novel Agent"), [evidence])],
            config=config,
        )
    )

    assert calls == 2
    # Serial synthesis returns validated facts, so support_ids are already
    # resolved to ledger IDs rather than left as prompt positions.
    assert result["D1"]["mechanism_and_targets"][0]["support_ids"] == ["ledger:E1"]


def test_synthesis_retries_token_limited_or_all_empty_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(messages_list, *, config, stage, reasoning_off=False):
        nonlocal calls
        del config
        calls += 1
        assert stage == "synthesis"
        assert len(messages_list) == 1
        if calls == 1:
            return [
                catalog_module._CatalogLLMOutput(
                    text="", finish_reason="length", reasoning="unfinished"
                )
            ]
        if calls == 2:
            assert "reached its output token limit" in messages_list[0][-1]["content"]
            return [
                json.dumps(
                    {category: [] for category in catalog_module._SYNTHESIS_CATEGORIES}
                )
            ]
        assert "all arrays were empty" in messages_list[0][-1]["content"]
        # The last error was not a token limit, so thinking stays on.
        assert not reasoning_off
        return [
            json.dumps(
                {
                    "mechanism_and_targets": [
                        {"text": "Targets Marker A.", "support_ids": ["P1"]}
                    ],
                    "efficacy_by_tumor": [],
                    "biomarker_prevalence": [],
                    "biomarker_directed_efficacy": [],
                    "safety": [],
                    "limitations": [],
                }
            )
        ]

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)
    evidence = EvidencePassage(
        evidence_id="ledger:E1",
        drug_id="D1",
        facet="mechanism_targets",
        source="test",
        source_type="authoritative",
        title="Evidence",
        passage="Novel Agent targets Marker A.",
        url="https://example.test/evidence",
        source_locator="E1",
    )

    result = asyncio.run(
        catalog_module.synthesize_serially(
            [(DrugIdentity(drug_id="D1", preferred_name="Novel Agent"), [evidence])],
            config=load_default_preset(),
        )
    )

    assert calls == 3
    assert result["D1"]["mechanism_and_targets"][0]["support_ids"] == ["ledger:E1"]


def test_ncit_definition_is_citable_ledger_evidence() -> None:
    drug = DrugIdentity(
        drug_id="NCIT:C123",
        preferred_name="Phase One Agent",
        ncit_code="C123",
        definition="An investigational inhibitor of Marker A.",
    )
    evidence = catalog_module._ncit_definition_evidence(
        drug, ncit_version="test-version"
    )

    assert len(evidence) == 1
    assert evidence[0].evidence_id == "ncit_definition:C123"
    prompt = catalog_module.build_synthesis_messages(drug, evidence)
    assert "ontology_definition" in prompt[1]["content"]
    assert "no supplied passage supports a relevant fact" in prompt[1]["content"]
    assert "minimum evidence-grade threshold" in prompt[1]["content"]
    assert "qualifying evidence" not in prompt[1]["content"]

    facts = catalog_module._validate_structured_facts(
        {
            "mechanism_and_targets": [
                {
                    "text": "Phase One Agent inhibits Marker A.",
                    "support_ids": ["ncit_definition"],
                }
            ]
        },
        evidence=evidence,
    )
    assert facts["mechanism_and_targets"][0]["support_ids"] == ["ncit_definition:C123"]


def test_validator_stamps_scope_and_projection_renders_it() -> None:
    drug = DrugIdentity(drug_id="NCIT:C123", preferred_name="Phase One Agent")
    evidence = [
        EvidencePassage(
            evidence_id="pubmed:1",
            drug_id=drug.drug_id,
            facet="efficacy_by_tumor",
            source="pubmed",
            source_type="literature_abstract",
            title="Abstract",
            passage="Responses were seen.",
            url="",
            source_locator="PMID 1",
            content_sha256="1",
        )
    ]
    raw = {
        "efficacy_by_tumor": [
            {
                "text": "Prolonged PFS (15.1 vs 10.6 months).",
                "tumor_type": "colon adenocarcinoma",
                "support_ids": ["P1"],
            }
        ]
    }

    agent_facts = catalog_module._validate_structured_facts(raw, evidence=evidence)
    class_facts = catalog_module._validate_structured_facts(
        raw, evidence=evidence, scope="class"
    )

    assert agent_facts["efficacy_by_tumor"][0]["scope"] == "agent"
    assert class_facts["efficacy_by_tumor"][0]["scope"] == "class"
    rendered = catalog_module._render_summary(
        drug, class_facts, include_safety=False, max_chars=4000
    )
    assert "[scope: class | tumor type: colon adenocarcinoma]" in rendered
    with pytest.raises(ValueError, match="Unknown evidence scope"):
        catalog_module._validate_structured_facts(raw, evidence=evidence, scope="other")


def test_stored_facts_get_a_scope_without_being_revalidated() -> None:
    stored = {
        "efficacy_by_tumor": [
            {"text": "Responses were seen.", "support_ids": ["civic:EID11238"]}
        ],
        "limitations": [{"text": "Single arm.", "support_ids": [], "scope": "class"}],
    }

    stamped = catalog_module.stamp_fact_scope(stored)

    # Resolved evidence IDs survive: re-running the P#-based validator here would
    # match nothing and drop the fact entirely.
    assert stamped["efficacy_by_tumor"][0]["support_ids"] == ["civic:EID11238"]
    assert stamped["efficacy_by_tumor"][0]["scope"] == "agent"
    assert stamped["limitations"][0]["scope"] == "class"
    assert stamped["safety"] == []


def test_indication_retrieval_is_bounded_across_diseases_and_sources() -> None:
    from matchminer_ai.trials import drug_research

    class _Source:
        source_type = "literature_abstract"

        def __init__(self, name: str) -> None:
            self.name = name

        async def fetch(self, drug, *, facet, query, client, settings):
            del client, settings
            return [
                EvidencePassage(
                    evidence_id=f"{self.name}:{facet}:{query[:24]}:{index}",
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=self.name,
                    source_type=self.source_type,
                    title="Abstract",
                    passage="Evidence.",
                    url="",
                    source_locator=str(index),
                    query=query,
                    content_sha256=f"{self.name}:{facet}:{query[:24]}:{index}",
                )
                for index in range(20)
            ]

    settings = ResearchSettings(
        max_attempts=1,
        indication_sources=("pubmed", "europe_pmc"),
        max_indication_passages_per_drug=12,
    )
    selected, attempts = asyncio.run(
        drug_research.research_indications(
            DrugIdentity(drug_id="D1", preferred_name="Novel Agent"),
            ["Colon Adenocarcinoma", "Lynch Syndrome", "Colon Adenocarcinoma"],
            sources=[_Source("pubmed"), _Source("europe_pmc"), _Source("civic")],
            settings=settings,
            client=None,
        )
    )

    assert len(selected) == 12
    assert all(item.query_scope == "drug_indication" for item in selected)
    # Both diseases and both configured sources are represented; the third
    # source searches by exact agent name and is not on this axis.
    assert {item.source for item in selected} == {"pubmed", "europe_pmc"}
    diseases = {
        "Colon Adenocarcinoma" if "Colon" in item.query else "Lynch Syndrome"
        for item in selected
    }
    assert diseases == {"Colon Adenocarcinoma", "Lynch Syndrome"}
    assert {item.source for item in attempts} == {"pubmed", "europe_pmc"}


def test_projection_gives_a_full_section_the_budget_thin_ones_do_not_use() -> None:
    drug = DrugIdentity(drug_id="D1", preferred_name="Novel Agent")
    facts = {
        "mechanism_and_targets": [
            {"text": "Targets Marker A.", "scope": "agent", "support_ids": []}
        ],
        "efficacy_by_tumor": [
            {
                "text": f"Result {index}: " + "detail " * 30,
                "tumor_type": f"tumor {index}",
                "scope": "agent",
                "support_ids": [],
            }
            for index in range(12)
        ],
        "biomarker_prevalence": [],
        "biomarker_directed_efficacy": [],
        "limitations": [{"text": "Single arm.", "scope": "agent", "support_ids": []}],
    }

    rendered = catalog_module._render_summary(
        drug, facts, include_safety=False, max_chars=4000
    )

    # An equal split would give efficacy a quarter of the budget and strand the
    # rest in sections holding one short line each.
    kept = sum(1 for index in range(12) if f"Result {index}:" in rendered)
    assert kept >= 10
    assert "Targets Marker A." in rendered
    assert "Single arm." in rendered
    assert len(rendered) <= 4000


def _efficacy(text: str, level: str | None, tumor: str = "prostate cancer") -> dict:
    return {
        "text": text + " " + "detail " * 25,
        "tumor_type": tumor,
        "evidence_level": level,
        "scope": "class",
        "support_ids": [],
    }


def test_projection_keeps_the_strongest_evidence_when_it_must_truncate() -> None:
    drug = DrugIdentity(drug_id="C1", preferred_name="Radioligand")
    facts = {
        "efficacy_by_tumor": [
            *(_efficacy(f"Registry {index}.", "retrospective cohort") for index in range(8)),
            _efficacy("Case.", "case report"),
            _efficacy("VISION result.", "phase III randomized trial"),
            _efficacy("NETTER-1 result.", "phase 3", tumor="neuroendocrine tumor"),
            _efficacy("Phase 2 result.", "single-arm phase II"),
        ],
        "limitations": [
            {"text": f"Limitation {index}. " + "detail " * 25, "support_ids": []}
            for index in range(12)
        ],
    }

    rendered = catalog_module._render_summary(
        drug, facts, include_safety=False, max_chars=3000
    )
    efficacy = rendered.split("Human efficacy by tumor type:")[1].split("\n", 1)[1]
    efficacy = efficacy.split("Biomarker prevalence:")[0]

    # Emission order put these after every registry line, where truncation
    # dropped them; both tumor types' phase III results lead now.
    assert efficacy.index("VISION result.") < efficacy.index("Phase 2 result.")
    assert efficacy.index("NETTER-1 result.") < efficacy.index("Phase 2 result.")
    assert "Case." not in efficacy
    assert "further finding(s) omitted for length." in efficacy
    # Efficacy outweighs limitations when both overflow.
    assert efficacy.count("\n- [") > rendered.count("Limitation ")
    assert len(rendered) <= 3000


def test_projection_files_preclinical_findings_apart_from_human_evidence() -> None:
    drug = DrugIdentity(drug_id="D1", preferred_name="Novel Agent")
    facts = {
        "efficacy_by_tumor": [
            _efficacy("Xenograft shrinkage.", "preclinical mouse model"),
            _efficacy("PDX response.", None) | {
                "text": "In patient-derived xenografts, tumors regressed."
            },
            _efficacy("Murine and patient data.", "preclinical murine study and case report"),
        ],
        "biomarker_directed_efficacy": [
            _efficacy("Cell-line sensitivity.", "preclinical in vitro"),
        ],
    }

    rendered = catalog_module._render_summary(
        drug, facts, include_safety=False, max_chars=6000
    )
    human, preclinical = rendered.split("Preclinical efficacy (not human evidence):")

    assert "Xenograft shrinkage." in preclinical
    assert "patient-derived xenografts" in preclinical
    assert "Cell-line sensitivity." in preclinical
    # A label naming a patient keeps the finding in the human section.
    assert "Murine and patient data." in human
    assert "No human evidence for this category" in human


def test_projection_strips_passage_handles_and_repeats() -> None:
    drug = DrugIdentity(drug_id="D1", preferred_name="Novel Agent")
    facts = {
        "efficacy_by_tumor": [
            {"text": "Responses in 7 of 12 patients (P3,P10).", "support_ids": []},
        ],
        "biomarker_directed_efficacy": [
            {"text": "Responses in 7 of 12 patients [P3].", "support_ids": []},
        ],
        "limitations": [{"text": "Small cohort. [P2, P5]", "support_ids": []}],
    }

    rendered = catalog_module._render_summary(
        drug, facts, include_safety=False, max_chars=4000
    )

    assert "(P3" not in rendered and "[P" not in rendered
    assert rendered.count("Responses in 7 of 12 patients.") == 1
    assert "- Small cohort." in rendered


def test_structural_background_read_survives_the_screen_calling_it_investigational() -> None:
    """Lymphodepletion sits in the experimental arm and is not the tested agent.

    The screen reliably reads "administered as part of the experimental regimen"
    as investigational, which put CAR-T conditioning chemotherapy into the scored
    drug set and diluted the cell therapy's own score threefold.
    """

    interventions = catalog_module._extract_registry_interventions(
        "NCT12345678",
        {
            "protocolSection": {
                "armsInterventionsModule": {
                    "armGroups": [{"label": "Experimental", "type": "EXPERIMENTAL"}],
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "Fludarabine",
                            "description": "Lymphodepleting chemotherapy before cells.",
                            "armGroupLabels": ["Experimental"],
                        },
                        {
                            "type": "BIOLOGICAL",
                            "name": "Novel Agent",
                            "description": "Autologous CAR T cells.",
                            "armGroupLabels": ["Experimental"],
                        },
                    ],
                }
            }
        },
    )
    assert [item.initial_role for item in interventions] == ["background", "uncertain"]
    assert interventions[0].role_confidence == "high"

    screen = {
        str(index): {
            "research_disposition": "include",
            "exclusion_category": "none",
            "role": "investigational",
            "confidence": "high",
            "rationale": "Administered as part of the experimental regimen.",
            "active_entity_names": [item.registry_name],
        }
        for index, item in enumerate(interventions)
    }
    _identities, assignments, _rows = catalog_module.derive_trial_assignments(
        {"NCT12345678": interventions}, {"NCT12345678": screen}, ncit_index=_NoNCIt()
    )
    roles = {item.preferred_name: (item.role, item.scoreable) for item in assignments}

    assert roles["Fludarabine"] == ("background", False)
    assert roles["Novel Agent"] == ("investigational", True)


def test_serial_synthesis_carries_facts_across_chunks_with_resolved_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The running summary must survive renumbering between chunks.

    Each chunk numbers its own passages from P1, so a fact carried forward cites
    an evidence ID while a new fact cites a position. Resolving the carried one
    against the new chunk's map would match nothing and delete it.
    """

    def passage(index: int) -> EvidencePassage:
        return EvidencePassage(
            evidence_id=f"ledger:E{index}",
            drug_id="D1",
            facet="efficacy_by_tumor",
            source="pubmed",
            source_type="literature_abstract",
            title="Abstract",
            passage="x" * 3000,
            url="",
            source_locator=str(index),
            content_sha256=f"E{index}",
        )

    seen: list[str] = []

    def fake_run(messages_list, *, config, stage, reasoning_off=False):
        del config, stage
        body = messages_list[0][-1]["content"]
        seen.append(body)
        if len(seen) == 1:
            facts = {"efficacy_by_tumor": [{"text": "First.", "support_ids": ["P1"]}]}
        else:
            # Carry the earlier fact by its resolved ID; add one from this chunk.
            facts = {
                "efficacy_by_tumor": [
                    {"text": "First.", "support_ids": ["ledger:E1"]},
                    {"text": "Second.", "support_ids": ["P1"]},
                ]
            }
        return [
            json.dumps(
                {
                    **{category: [] for category in catalog_module._SYNTHESIS_CATEGORIES},
                    **facts,
                }
            )
        ]

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)

    result = asyncio.run(
        catalog_module.synthesize_serially(
            [
                (
                    DrugIdentity(drug_id="D1", preferred_name="Novel Agent"),
                    [passage(1), passage(2)],
                )
            ],
            config=load_default_preset(),
            chunk_token_limit=1000,  # 4,000 characters: one passage per chunk.
        )
    )

    assert len(seen) == 2
    assert "THE SYNTHESIS SO FAR" in seen[1] and "ledger:E1" in seen[1]
    items = result["D1"]["efficacy_by_tumor"]
    assert [item["text"] for item in items] == ["First.", "Second."]
    assert [item["support_ids"] for item in items] == [["ledger:E1"], ["ledger:E2"]]
    assert all(item["scope"] == "agent" for item in items)


def _synthesis_passage(drug_id: str, index: int) -> EvidencePassage:
    return EvidencePassage(
        evidence_id=f"ledger:{drug_id}-{index}",
        drug_id=drug_id,
        facet="efficacy_by_tumor",
        source="pubmed",
        source_type="literature_abstract",
        title="Abstract",
        passage="x" * 3000,
        url="",
        source_locator=f"{drug_id}-{index}",
        content_sha256=f"{drug_id}-{index}",
    )


def _one_fact_synthesis() -> str:
    return json.dumps(
        {
            **{category: [] for category in catalog_module._SYNTHESIS_CATEGORIES},
            "efficacy_by_tumor": [{"text": "Responds.", "support_ids": ["P1"]}],
        }
    )


def test_synthesis_chains_finish_each_subject_without_waiting_for_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-chunk subject is checkpointed while a three-chunk one is mid-chain.

    Lock-step rounds held every subject until the longest finished, so a single
    long ledger delayed every checkpoint in its batch.
    """

    short_done = threading.Event()
    lock = threading.Lock()
    in_flight = 0
    peak = 0
    long_calls: list[str] = []
    short_done_before_long_last_round: list[bool] = []

    def fake_run(messages_list, *, config, stage, reasoning_off=False):
        nonlocal in_flight, peak
        del config, stage, reasoning_off
        body = messages_list[0][-1]["content"]
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
            if "Long Agent" in body:
                long_calls.append(body)
            is_long_last_round = "Long Agent" in body and len(long_calls) == 3
        try:
            if is_long_last_round:
                short_done_before_long_last_round.append(short_done.wait(timeout=5))
            else:
                time.sleep(0.05)
            return [_one_fact_synthesis()]
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)
    config = load_default_preset()
    config.remote["enabled"] = True
    config.remote["max_concurrent_requests"] = 2
    completed: list[str] = []
    progress: list[tuple[int, int]] = []

    def on_complete(drug, facts):
        assert facts["efficacy_by_tumor"]
        completed.append(drug.drug_id)
        if drug.drug_id == "SHORT":
            short_done.set()

    result = asyncio.run(
        catalog_module.synthesize_serially(
            [
                (
                    DrugIdentity(drug_id="LONG", preferred_name="Long Agent"),
                    [_synthesis_passage("LONG", index) for index in (1, 2, 3)],
                ),
                (
                    DrugIdentity(drug_id="SHORT", preferred_name="Short Agent"),
                    [_synthesis_passage("SHORT", 1)],
                ),
            ],
            config=config,
            chunk_token_limit=1000,  # 4,000 characters: one passage per chunk.
            on_subject_complete=on_complete,
            progress_callback=lambda done, total: progress.append((done, total)),
        )
    )

    assert len(long_calls) == 3
    assert short_done_before_long_last_round == [True]
    assert completed == ["SHORT", "LONG"]
    assert progress == [(1, 2), (2, 2)]
    assert peak == 2
    assert set(result) == {"LONG", "SHORT"}


def test_synthesis_disables_thinking_only_after_repeated_token_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[bool] = []

    def fake_run(messages_list, *, config, stage, reasoning_off=False):
        del messages_list, config, stage
        seen.append(reasoning_off)
        if not reasoning_off:
            return [
                catalog_module._CatalogLLMOutput(
                    text="", finish_reason="length", reasoning="unfinished"
                )
            ]
        return [_one_fact_synthesis()]

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)

    result = asyncio.run(
        catalog_module.synthesize_serially(
            [
                (
                    DrugIdentity(drug_id="D1", preferred_name="Novel Agent"),
                    [_synthesis_passage("D1", 1)],
                )
            ],
            config=load_default_preset(),
        )
    )

    assert seen == [False, False, True]
    assert result["D1"]["efficacy_by_tumor"][0]["support_ids"] == ["ledger:D1-1"]


def test_reasoning_off_overrides_thinking_enabled_by_a_stage_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    class FakeBackend:
        def generate_llm_outputs(self, *, prompt_list, llm_config, **_kwargs):
            captured.append(llm_config)
            return SimpleNamespace(
                final_outputs=["{}"] * len(prompt_list),
                finish_reasons=["stop"] * len(prompt_list),
                reasoning_outputs=[""] * len(prompt_list),
            )

    monkeypatch.setattr(catalog_module, "get_llm_backend", lambda _config: FakeBackend())
    config = load_default_preset()
    config.remote["enabled"] = True
    # Served-model profiles set enable_thinking on the stage override as well.
    apply_model_profile(
        config,
        "Inferact/Qwen3.8-Flash-Next-NVFP4",
        sections=("llm_good_option", "good_option_catalog.synthesis_llm"),
    )
    messages = [[{"role": "user", "content": "Synthesize."}]]

    catalog_module._run_llm_messages(messages, config=config, stage="synthesis")
    catalog_module._run_llm_messages(
        messages, config=config, stage="synthesis", reasoning_off=True
    )

    thinking = [
        runtime["remote"]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        for runtime in captured
    ]
    assert thinking == [True, False]
    assert "reasoning_effort" not in captured[1]["remote"]["request_params"]
    assert captured[1]["remote"]["request_params"]["max_tokens"] == 100000
    # The fallback is per call; the shared config still thinks.
    assert config.good_option_catalog["synthesis_llm"]["remote"]["extra_body"][
        "chat_template_kwargs"
    ]["enable_thinking"] is True


def test_class_ids_are_stable_across_wording_of_the_same_class() -> None:
    same = {
        catalog_module.class_id_for(name)
        for name in ("PD-L1 inhibitor", "PD-L1 Inhibitors", "anti-PD-L1 inhibitor")
    }
    assert len(same) == 1
    assert catalog_module.class_id_for("ADC") == catalog_module.class_id_for(
        "Antibody-Drug Conjugates (ADCs)"
    )
    assert catalog_module.class_id_for("PD-L1 inhibitor") != (
        catalog_module.class_id_for("immune checkpoint inhibitor")
    )
    assert catalog_module.class_id_for("   ") == ""


def test_class_assignment_rejects_unusable_items_and_caps_the_list() -> None:
    evidence = [
        EvidencePassage(
            evidence_id="pubmed:1",
            drug_id="D1",
            facet="mechanism_targets",
            source="pubmed",
            source_type="literature_abstract",
            title="Abstract",
            passage="An anti-PD-L1 antibody.",
            url="",
            source_locator="PMID 1",
            content_sha256="1",
        )
    ]
    passage_map = {"P1": "pubmed:1"}

    unsupported = {
        "classes": [{"name": "PD-L1 inhibitor", "basis": "target", "confidence": "high"}],
        "__passage_id_map__": passage_map,
    }
    assert (
        catalog_module._class_output_validation_error(unsupported, evidence=evidence)
        == "PD-L1 inhibitor: no supplied P# identifier supports it"
    )
    assert catalog_module._class_output_validation_error(
        {"classes": [], "__passage_id_map__": passage_map}, evidence=evidence
    ) is None

    valid = {
        "classes": [
            {
                "name": "PD-L1 inhibitor",
                "basis": "target",
                "target": "CD274",
                "aliases": ["anti-PD-L1 antibody"],
                "confidence": "high",
                "support_ids": ["P1"],
            },
            {
                "name": "PD-L1 Inhibitors",
                "basis": "target",
                "confidence": "low",
                "support_ids": ["P1"],
            },
            {
                "name": "immune checkpoint inhibitor",
                "basis": "mechanism",
                "confidence": "medium",
                "support_ids": ["P1"],
            },
        ],
        "__passage_id_map__": passage_map,
    }
    assert catalog_module._class_output_validation_error(valid, evidence=evidence) is None
    records = catalog_module._validate_drug_classes("D1", valid, max_classes=3)

    # The duplicate spelling collapses into the first record rather than
    # consuming one of the three slots.
    assert [item.class_name for item in records] == [
        "PD-L1 inhibitor",
        "immune checkpoint inhibitor",
    ]
    assert records[0].support_ids == ("pubmed:1",)
    assert records[0].aliases == ("anti-PD-L1 antibody",)


def test_evidence_bounding_reserves_room_for_every_query_scope() -> None:
    def passage(index: int, scope: str) -> EvidencePassage:
        return EvidencePassage(
            evidence_id=f"{scope}:{index}",
            drug_id="D1",
            facet="efficacy_by_tumor",
            source="pubmed",
            source_type="literature_abstract",
            title="Abstract",
            passage="x" * 100,
            url="",
            source_locator=str(index),
            content_sha256=f"{scope}:{index}",
            query_scope=scope,
        )

    items = [passage(index, "drug") for index in range(20)]
    items += [passage(index, "drug_indication") for index in range(3)]
    selected = catalog_module.bound_evidence(items, character_limit=500)

    assert len(selected) == 5
    assert sum(1 for item in selected if item.query_scope == "drug_indication") == 2


def test_class_blocks_are_deduplicated_and_name_the_drugs_they_cover() -> None:
    from matchminer_ai.trials.drug_evidence import TrialClassEvidence

    messages = build_good_option_messages(
        patient_summary="PRIVATE_PATIENT",
        drug_summaries=_catalog().scoreable_summaries_for_trial("NCT12345678"),
        class_evidence=(
            TrialClassEvidence(
                class_id="abc",
                class_name="PD-L1 inhibitor",
                drug_names=("Novel Agent", "Second Agent"),
                class_option_summary="Drug class: PD-L1 inhibitor\nEvidence.",
            ),
        ),
    )
    prompt = messages[1]["content"]

    assert prompt.count("DRUG CLASS EVIDENCE") == 2  # One block, plus the legend.
    assert "PD-L1 inhibitor (covers: Novel Agent, Second Agent)" in prompt
    assert prompt.index("Drug: Novel Agent") < prompt.index(
        "DRUG CLASS EVIDENCE — PD-L1 inhibitor"
    )


def _facts(prefix: str, count: int) -> dict:
    return {
        "mechanism_and_targets": [
            {"text": f"{prefix} targets Marker A.", "scope": "agent", "support_ids": []}
        ],
        "efficacy_by_tumor": [
            _efficacy(f"{prefix} result {index}.", "phase II", tumor=f"tumor {index}")
            | {"text": f"{prefix} result {index}. " + "detail " * 60}
            for index in range(count)
        ],
    }


def _packable(name: str, count: int) -> DrugSummary:
    return DrugSummary(
        drug_id=name,
        preferred_name=name,
        ncit_code="",
        research_status="complete",
        synthesis_status="ok",
        structured_facts=_facts(name, count),
        good_option_summary=f"Drug: {name}\nSTALE STORED PROJECTION",
        help_me_choose_summary="",
        evidence_count=count,
    )


def _class_block(name: str, count: int, covers: tuple[str, ...]):
    from matchminer_ai.trials.drug_evidence import TrialClassEvidence

    return TrialClassEvidence(
        class_id=name,
        class_name=name,
        drug_names=covers,
        class_option_summary=f"Drug class: {name}\nSTALE STORED PROJECTION",
        structured_facts=_facts(name, count),
    )


def test_packing_renders_everything_from_facts_when_the_budget_allows() -> None:
    text, report = pack_good_option_evidence(
        [_packable("Agent A", 30)],
        [_class_block("Class K", 40, ("Agent A",))],
        max_chars=500_000,
    )

    # Re-rendered from the facts rather than the catalog's fixed-size projection.
    assert "STALE STORED PROJECTION" not in text
    assert all(f"Agent A result {index}." in text for index in range(30))
    assert all(f"Class K result {index}." in text for index in range(40))
    assert "omitted for length" not in text
    assert text.index("Drug: Agent A") < text.index(
        "DRUG CLASS EVIDENCE — Class K (covers: Agent A)\nDrug class: Class K"
    )
    assert [item["granularity"] for item in report] == ["full", "full"]


def test_packing_coarsens_before_it_drops_and_weights_drugs_over_classes() -> None:
    drug, block = _packable("Agent A", 30), _class_block("Class K", 30, ("Agent A",))
    _full, report = pack_good_option_evidence([drug], [block], max_chars=10**9)
    full_total = sum(item["full_chars"] for item in report)

    text, report = pack_good_option_evidence(
        [drug], [block], max_chars=int(full_total * 0.8)
    )

    # The drug's weighted share covers it in full; the class block coarsens,
    # and shorter attribution and statements keep every result visible.
    assert "omitted for length" not in text
    assert all(f"Class K result {index}." in text for index in range(30))
    assert report[0]["granularity"] == "full"
    assert report[1]["granularity"] in {"condensed", "brief"}
    assert len(text) <= int(full_total * 0.8)

    text, report = pack_good_option_evidence(
        [drug], [block], max_chars=int(full_total * 0.3)
    )
    drug_report, class_report = report
    assert class_report["truncated"] and drug_report["truncated"]
    assert drug_report["budget_chars"] >= 2 * class_report["budget_chars"] - 1
    assert text.count("omitted for length") == 2
    assert len(text) <= int(full_total * 0.3)


def test_packing_releases_unused_share_and_caps_each_subject() -> None:
    small, large = _packable("Small", 1), _packable("Large", 60)
    _text, report = pack_good_option_evidence([small, large], max_chars=10**9)
    small_full = report[0]["full_chars"]

    _text, report = pack_good_option_evidence(
        [small, large], max_chars=small_full + 6000
    )
    assert report[0]["granularity"] == "full"
    assert report[1]["budget_chars"] >= 6000 - 2

    _text, report = pack_good_option_evidence(
        [small, large], max_chars=10**9, max_drug_chars=3000
    )
    assert report[1]["rendered_chars"] <= 3000
    assert report[1]["truncated"]


def test_packing_uses_stored_summaries_without_structured_facts() -> None:
    summaries = _catalog().scoreable_summaries_for_trial("NCT12345678")
    text, report = pack_good_option_evidence(summaries, max_chars=10**9)

    assert text == "\n\n".join(item.good_option_summary for item in summaries)
    assert not any(item["from_structured_facts"] for item in report)


def test_evidence_budget_follows_the_teacher_context_and_completion() -> None:
    config = load_default_preset()
    budget = good_option_evidence_budget(config, fixed_prompt_chars=20_000)

    assert budget["context_tokens"] == 262144
    assert budget["output_tokens"] == 100000
    assert budget["evidence_max_chars"] == int((262144 - 100000 - 4096) * 3.5) - (
        20_000 + 14_000
    )
    assert budget["max_class_chars"] == 70_000

    config.raw["good_option_prompt"]["context_tokens"] = 131072
    config.remote["enabled"] = True
    config.llm_good_option["remote"]["request_params"]["max_tokens"] = 32000
    budget = good_option_evidence_budget(config, fixed_prompt_chars=0)
    assert budget["output_tokens"] == 32000
    assert budget["evidence_max_chars"] == int((131072 - 32000 - 4096) * 3.5) - 14_000

    config.raw["good_option_prompt"]["context_tokens"] = 40000
    with pytest.raises(ValueError, match="characters for evidence"):
        build_good_option_messages(
            patient_summary="Synthetic patient",
            drug_summaries=[_packable("Agent A", 2)],
            config=config,
        )


def test_llm_scoring_packs_class_facts_and_records_the_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[dict[str, str]]] = []

    def fake_run(messages_list, *, config):
        del config
        captured.extend(messages_list)
        return SimpleNamespace(
            final_outputs=[_response()],
            reasoning_outputs=[""],
            finish_reasons=["stop"],
            model_metadata={},
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.good_options._run_good_option_llm", fake_run
    )
    catalog = _catalog()
    block = _class_block("Class K", 3, ("Novel Agent",))
    monkeypatch.setattr(catalog, "class_evidence_for_trial", lambda _trial: (block,))
    config = load_default_preset()
    config.debug_mode = True
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "NCT12345678",
                "cancer_history_summary": "Synthetic TARGET_MARKER cancer",
            }
        ]
    )

    output, metadata = score_good_options_with_llm(
        pairs, catalog=catalog, config=config, return_metadata=True
    )

    prompt = captured[0][1]["content"]
    assert "Class K result 2." in prompt
    assert "STALE STORED PROJECTION" not in prompt
    assert metadata["evidence_packing_version"] == "good-option-evidence-packing-v1"
    assert metadata["evidence_packing"]["subjects"] == 3
    assert metadata["evidence_packing"]["truncated_subjects"] == 0
    packing = output.loc[0, "good_option_evidence_packing"]
    assert [item["kind"] for item in packing["subjects"]] == ["drug", "drug", "class"]


STUDY = {
    "protocolSection": {
        "identificationModule": {"briefTitle": "Novel agent study"},
        "statusModule": {
            "overallStatus": "RECRUITING",
            "lastUpdatePostDateStruct": {"date": "2026-08-01"},
        },
        "designModule": {"phases": ["PHASE1"]},
        "descriptionModule": {"briefSummary": "A synthetic public trial."},
        "armsInterventionsModule": {
            "armGroups": [
                {"label": "Experimental", "type": "EXPERIMENTAL"},
                {"label": "Control", "type": "ACTIVE_COMPARATOR"},
            ],
            "interventions": [
                {
                    "type": "DRUG",
                    "name": "Novel Agent",
                    "armGroupLabels": ["Experimental"],
                },
                {
                    "type": "DRUG",
                    "name": "Control Agent",
                    "armGroupLabels": ["Control"],
                },
            ],
        },
    }
}


class _NoNCIt:
    def search(self, _query: str, *, limit: int):
        del limit
        return []


def _screening_result() -> dict[str, dict[str, Any]]:
    return {
        "0": {
            "research_disposition": "include",
            "exclusion_category": "none",
            "role": "investigational",
            "confidence": "high",
            "rationale": "The named anticancer agent's contribution is tested.",
            "active_entity_names": ["Novel Agent"],
        },
        "1": {
            "research_disposition": "include",
            "exclusion_category": "none",
            "role": "control",
            "confidence": "high",
            "rationale": "This is a named anticancer comparator agent.",
            "active_entity_names": ["Control Agent"],
        },
    }


class _EvidenceSource:
    name = "fake"
    source_type = "authoritative"

    async def fetch(self, drug, *, facet, query, client, settings):
        del client, settings
        passage = f"{drug.preferred_name} has human {facet} evidence."
        return [
            EvidencePassage(
                evidence_id=f"{drug.drug_id}:{facet}",
                drug_id=drug.drug_id,
                facet=facet,
                source=self.name,
                source_type=self.source_type,
                title="Synthetic evidence",
                passage=passage,
                url="https://example.test/source",
                source_locator=facet,
                query=query,
                content_sha256=f"{drug.drug_id}:{facet}",
            )
        ]


class _NamedEvidenceSource(_EvidenceSource):
    """A source the class and indication axes are configured to call."""

    name = "pubmed"


def test_catalog_retrieves_and_merges_class_and_indication_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = json.loads(json.dumps(STUDY))
    study["protocolSection"]["conditionsModule"] = {
        "conditions": ["Colon Adenocarcinoma", "Lynch Syndrome"]
    }

    async def fake_fetch(_nct_id: str, *, client):
        del client
        return json.loads(json.dumps(study))

    def role_resolver(_trial_id, _interventions):
        return _screening_result()

    def classify(drug, _evidence):
        return {
            "classes": [
                {
                    "name": "TARGET_MARKER inhibitor",
                    "basis": "target",
                    "target": "TARGET_MARKER",
                    "aliases": ["anti-TARGET_MARKER antibody"],
                    "confidence": "high",
                    "support_ids": ["P1"],
                }
            ]
            if drug.preferred_name == "Novel Agent"
            else []
        }

    def synthesize(_drug, evidence):
        return {
            "mechanism_and_targets": [
                {"text": "Targets TARGET_MARKER.", "support_ids": ["P1"]}
            ]
            if evidence
            else [],
            "efficacy_by_tumor": [],
            "biomarker_prevalence": [],
            "biomarker_directed_efficacy": [],
            "safety": [],
            "limitations": [],
        }

    monkeypatch.setattr(catalog_module, "fetch_trial_study", fake_fetch)
    monkeypatch.setattr(catalog_module, "load_ncit_drug_index", lambda _path: _NoNCIt())
    output = tmp_path / "catalog"
    catalog = asyncio.run(
        build_good_option_catalog(
            ["NCT12345678"],
            output,
            config=load_default_preset(),
            sources=[_NamedEvidenceSource()],
            settings=ResearchSettings(max_attempts=1),
            role_resolver=role_resolver,
            synthesizer=synthesize,
            class_resolver=classify,
        )
    )

    class_id = catalog_module.class_id_for("TARGET_MARKER inhibitor")
    assert list(catalog.drug_classes["class_id"]) == [class_id]
    assert list(catalog.class_summaries["class_id"]) == [class_id]

    # The class corpus is retrieved for the class, not for any one drug, and is
    # conditioned on the diseases the class's member trials name.
    assert set(catalog.class_evidence["class_id"]) == {class_id}
    assert set(catalog.class_evidence["query_scope"]) == {"class", "class_indication"}
    assert (catalog.class_evidence["drug_id"] == "").all()
    class_disease_queries = catalog.class_evidence.loc[
        catalog.class_evidence["query_scope"].eq("class_indication"), "query"
    ]
    assert any("Colon Adenocarcinoma" in value for value in class_disease_queries)

    # Both diseases named by the trial reached retrieval, and the passages they
    # produced are marked with the axis that found them.
    indication_queries = catalog.drug_evidence.loc[
        catalog.drug_evidence["query_scope"].eq("drug_indication"), "query"
    ]
    assert any("Colon Adenocarcinoma" in value for value in indication_queries)
    assert any("Lynch Syndrome" in value for value in indication_queries)

    facts = json.loads(
        catalog.class_summaries.iloc[0]["structured_facts_json"]
    )
    assert facts["mechanism_and_targets"][0]["scope"] == "class"
    novel = next(
        item
        for item in catalog.scoreable_summaries_for_trial("NCT12345678")
        if item.preferred_name == "Novel Agent"
    )
    assert novel.structured_facts["mechanism_and_targets"][0]["scope"] == "agent"

    blocks = catalog.class_evidence_for_trial("NCT12345678")
    assert [item.drug_names for item in blocks] == [("Novel Agent",)]
    prompt = build_good_option_messages(
        patient_summary="Synthetic patient",
        drug_summaries=[novel],
        class_evidence=blocks,
    )[1]["content"]
    assert "DRUG CLASS EVIDENCE — TARGET_MARKER inhibitor (covers: Novel Agent)" in prompt
    assert "[scope: class]" in prompt

    # Reloading re-runs every contract check over the new tables.
    load_good_option_catalog(output)


def test_catalog_build_deduplicates_drugs_and_indexes_control_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_fetch(_nct_id: str, *, client):
        del client
        return json.loads(json.dumps(STUDY))

    def role_resolver(_trial_id, interventions):
        assert len(interventions) == 2
        return _screening_result()

    def synthesize(_drug, evidence):
        support = ["P1"] if evidence else []
        return {
            "mechanism_and_targets": [
                {"text": "Targets TARGET_MARKER.", "support_ids": support}
            ],
            "efficacy_by_tumor": [],
            "biomarker_prevalence": [],
            "biomarker_directed_efficacy": [],
            "safety": [],
            "limitations": [],
        }

    monkeypatch.setattr(
        "matchminer_ai.trials.drug_catalog.fetch_trial_study", fake_fetch
    )
    monkeypatch.setattr(
        "matchminer_ai.trials.drug_catalog.load_ncit_drug_index",
        lambda _resource: _NoNCIt(),
    )
    output = tmp_path / "catalog"
    catalog = asyncio.run(
        build_good_option_catalog(
            ["NCT12345678", "NCT87654321"],
            output,
            config=load_default_preset(),
            sources=[_EvidenceSource()],
            settings=ResearchSettings(max_attempts=1),
            role_resolver=role_resolver,
            synthesizer=synthesize,
            class_resolver=no_classes,
        )
    )

    assert len(catalog.drug_summaries) == 2  # Novel + control, deduplicated by name.
    scoreable = catalog.assignments_for_trial("NCT12345678", scoreable_only=True)
    assert [item.preferred_name for item in scoreable] == ["Novel Agent"]
    all_roles = {
        item.preferred_name: item.role
        for item in catalog.assignments_for_trial("NCT12345678")
    }
    assert all_roles["Control Agent"] == "control"
    novel_summary = catalog.summary_for_drug(scoreable[0].drug_id)
    assert novel_summary is not None
    support_ids = novel_summary.structured_facts["mechanism_and_targets"][0][
        "support_ids"
    ]
    assert support_ids == [f"{scoreable[0].drug_id}:mechanism_targets"]
    assert support_ids != ["P1"]
    assert load_good_option_catalog(output).compatibility_id


def test_catalog_screens_non_treatment_registry_entries_before_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [
        "Selumetinib Sulfate",
        "Sotorasib",
        "Tranexamic Acid",
        "Lidocaine",
        "Control (1% lidocaine with 1:100,000 epinephrine)",
        "Zirconium Zr 89 Girentuximab",
        "Extended Dosing Interval - A",
        "Extended Dosing Interval - B",
        "Extended Dosing Interval - C",
        "Standard of Care - A",
        "Standard of Care - B",
        "Standard of Care - C",
    ]
    study = json.loads(json.dumps(STUDY))
    module = study["protocolSection"]["armsInterventionsModule"]
    module["armGroups"] = [
        {
            "label": "Experimental",
            "type": "EXPERIMENTAL",
            "description": "Cancer treatment and procedural substudies.",
        }
    ]
    module["interventions"] = [
        {"type": "DRUG", "name": name, "armGroupLabels": ["Experimental"]}
        for name in names
    ]

    async def fake_fetch(_nct_id: str, *, client):
        del client
        return study

    def screen(_trial_id, interventions):
        assert [item.registry_name for item in interventions] == names
        output: dict[str, dict[str, Any]] = {}
        for index, name in enumerate(names):
            included = index < 2
            if index in {2, 3, 4}:
                category = "supportive_or_procedural"
            elif index == 5:
                category = "diagnostic_or_imaging"
            elif index >= 9:
                category = "unspecified_standard_of_care"
            else:
                category = "not_a_concrete_agent"
            output[str(index)] = {
                "research_disposition": "include" if included else "exclude",
                "exclusion_category": "none" if included else category,
                "role": "investigational" if included else "supportive",
                "confidence": "high",
                "rationale": (
                    "A named anticancer treatment agent."
                    if included
                    else "Not a concrete agent with direct anticancer treatment intent."
                ),
                "active_entity_names": [name] if included else [],
            }
        return output

    researched: list[str] = []

    async def fake_research(drug, **_kwargs):
        researched.append(drug.preferred_name)
        return [], [], "complete", []

    def synthesize(_drug, _evidence):
        return {category: [] for category in catalog_module._SYNTHESIS_CATEGORIES}

    monkeypatch.setattr(catalog_module, "fetch_trial_study", fake_fetch)
    monkeypatch.setattr(catalog_module, "load_ncit_drug_index", lambda _path: _NoNCIt())
    monkeypatch.setattr(catalog_module, "research_drug", fake_research)
    output = tmp_path / "catalog"
    catalog = asyncio.run(
        build_good_option_catalog(
            ["NCT12345678"],
            output,
            config=load_default_preset(),
            sources=[_EvidenceSource()],
            settings=ResearchSettings(max_attempts=1),
            role_resolver=screen,
            synthesizer=synthesize,
            class_resolver=no_classes,
        )
    )

    assert researched == ["Selumetinib Sulfate", "Sotorasib"]
    assert list(catalog.drug_summaries["preferred_name"]) == researched
    assert set(catalog.trial_drug_index["registry_name"]) == set(researched)
    screening = catalog.trial_intervention_screening.set_index("registry_name")
    assert bool(screening.loc["Selumetinib Sulfate", "included"])
    assert not bool(screening.loc["Tranexamic Acid", "included"])
    assert (
        screening.loc["Zirconium Zr 89 Girentuximab", "exclusion_category"]
        == "diagnostic_or_imaging"
    )
    assert (
        screening.loc["Extended Dosing Interval - A", "exclusion_category"]
        == "not_a_concrete_agent"
    )
    assert catalog.manifest["counts"]["screened_interventions"] == len(names)
    assert catalog.manifest["counts"]["excluded_interventions"] == len(names) - 2

    prompt = "\n".join(
        message["content"]
        for message in catalog_module.build_intervention_screening_messages(
            "NCT12345678",
            catalog_module._extract_registry_interventions("NCT12345678", study),
        )
    )
    assert "diagnostic or imaging tracers" in prompt
    assert "dosing intervals" in prompt
    assert "unspecified standard of care" in prompt


def test_default_intervention_screen_retries_invalid_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(messages_list, *, config, stage):
        nonlocal calls
        del config
        calls += 1
        assert stage == "screening"
        assert len(messages_list) == 1
        if calls == 1:
            assert (
                "previous response failed validation"
                not in messages_list[0][-1]["content"]
            )
            return ["not valid screening JSON"]
        assert "previous response failed validation" in messages_list[0][-1]["content"]
        assert "expected intervention indexes" in messages_list[0][-1]["content"]
        records = [
            {"index": int(index), **value}
            for index, value in _screening_result().items()
        ]
        return [json.dumps({"interventions": records})]

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)
    monkeypatch.setattr(catalog_module.asyncio, "to_thread", run_inline)
    config = load_default_preset()
    config.good_option_catalog["screening_max_attempts"] = 2
    result = asyncio.run(
        catalog_module._resolve_roles_with_default_llm(
            {
                "NCT12345678": catalog_module._extract_registry_interventions(
                    "NCT12345678", STUDY
                )
            },
            config=config,
        )
    )

    assert calls == 2
    assert result["NCT12345678"]["0"]["research_disposition"] == "include"


def test_default_intervention_screen_accepts_code_fenced_bare_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(messages_list, *, config, stage):
        nonlocal calls
        del config
        calls += 1
        assert stage == "screening"
        assert len(messages_list) == 1
        records = [
            {"index": int(index), **value}
            for index, value in _screening_result().items()
        ]
        return [f"```json\n{json.dumps(records)}\n```"]

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)
    monkeypatch.setattr(catalog_module.asyncio, "to_thread", run_inline)
    config = load_default_preset()
    config.good_option_catalog["screening_max_attempts"] = 1
    result = asyncio.run(
        catalog_module._resolve_roles_with_default_llm(
            {
                "NCT12345678": catalog_module._extract_registry_interventions(
                    "NCT12345678", STUDY
                )
            },
            config=config,
        )
    )

    assert calls == 1
    assert result["NCT12345678"]["0"]["research_disposition"] == "include"


def test_intervention_screen_accepts_active_entities_explicit_in_description() -> None:
    study = json.loads(json.dumps(STUDY))
    study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
        {
            "type": "DRUG",
            "name": "PCV chemotherapy",
            "description": "Lomustine, vincristine, and procarbazine are administered.",
            "armGroupLabels": ["Experimental"],
        }
    ]
    interventions = catalog_module._extract_registry_interventions("NCT12345678", study)
    output = {
        "0": {
            "research_disposition": "include",
            "exclusion_category": "none",
            "role": "investigational",
            "confidence": "high",
            "rationale": "The regimen explicitly names three anticancer agents.",
            "active_entity_names": ["Lomustine", "Vincristine", "Procarbazine"],
        }
    }

    assert catalog_module._role_output_is_complete(interventions, output)

    output["0"]["active_entity_names"] = ["Invented Agent"]
    error = catalog_module._role_output_validation_error(interventions, output)
    assert error is not None
    assert "registry name, aliases, or descriptions" in error

    study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
        {
            "type": "DRUG",
            "name": "S1",
            "description": "S1 is given orally twice daily.",
            "armGroupLabels": ["Experimental"],
        }
    ]
    short_name_intervention = catalog_module._extract_registry_interventions(
        "NCT12345678", study
    )
    output["0"]["active_entity_names"] = ["S1"]
    assert catalog_module._role_output_is_complete(short_name_intervention, output)


def test_catalog_fails_closed_invalid_screen_item_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_fetch(_nct_id: str, *, client):
        del client
        return json.loads(json.dumps(STUDY))

    async def invalid_default_screen(by_trial, *, config):
        del config
        trial_id = next(iter(by_trial))
        output = _screening_result()
        output["0"]["active_entity_names"] = ["Invented Agent"]
        return {trial_id: output}

    researched: list[str] = []

    async def fake_research(drug, **_kwargs):
        researched.append(drug.preferred_name)
        return [], [], "complete", []

    def synthesize(_drug, _evidence):
        return {category: [] for category in catalog_module._SYNTHESIS_CATEGORIES}

    monkeypatch.setattr(catalog_module, "fetch_trial_study", fake_fetch)
    monkeypatch.setattr(catalog_module, "load_ncit_drug_index", lambda _path: _NoNCIt())
    monkeypatch.setattr(
        catalog_module, "_resolve_roles_with_default_llm", invalid_default_screen
    )
    monkeypatch.setattr(catalog_module, "research_drug", fake_research)
    progress: list[tuple[str, str]] = []
    catalog = asyncio.run(
        build_good_option_catalog(
            ["NCT12345678"],
            tmp_path / "catalog",
            config=load_default_preset(),
            sources=[_EvidenceSource()],
            settings=ResearchSettings(max_attempts=1),
            synthesizer=synthesize,
            class_resolver=no_classes,
            progress_callback=lambda stage, _done, _total, label: progress.append(
                (stage, label)
            ),
        )
    )

    screening = catalog.trial_intervention_screening.set_index("registry_name")
    assert screening.loc["Novel Agent", "research_disposition"] == "uncertain"
    assert not bool(screening.loc["Novel Agent", "included"])
    assert "excluded fail-closed" in screening.loc["Novel Agent", "rationale"]
    assert bool(screening.loc["Control Agent", "included"])
    assert researched == ["Control Agent"]
    assert any(
        stage == "screening" and "NCT12345678 (fail-closed:" in label
        for stage, label in progress
    )


def test_catalog_resumes_registry_roles_and_completed_drug_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SimulatedDisconnect(BaseException):
        pass

    fetch_calls: list[str] = []
    role_calls: list[str] = []
    research_calls: list[str] = []

    async def fake_fetch(nct_id: str, *, client):
        del client
        fetch_calls.append(nct_id)
        return json.loads(json.dumps(STUDY))

    def role_resolver(trial_id, _interventions):
        role_calls.append(trial_id)
        return _screening_result()

    def synthesize(_drug, evidence):
        return {
            "mechanism_and_targets": [
                {"text": "Targets TARGET_MARKER.", "support_ids": ["P1"]}
            ]
            if evidence
            else [],
            "efficacy_by_tumor": [],
            "biomarker_prevalence": [],
            "biomarker_directed_efficacy": [],
            "safety": [],
            "limitations": [],
        }

    def evidence_for(drug: DrugIdentity) -> list[EvidencePassage]:
        return [
            EvidencePassage(
                evidence_id=f"{drug.drug_id}:mechanism_targets",
                drug_id=drug.drug_id,
                facet="mechanism_targets",
                source="fake",
                source_type="authoritative",
                title="Synthetic evidence",
                passage=f"{drug.preferred_name} targets TARGET_MARKER.",
                url="https://example.test/source",
                source_locator="mechanism_targets",
                content_sha256=f"{drug.drug_id}:mechanism_targets",
            )
        ]

    async def interrupting_research(drug, **_kwargs):
        research_calls.append(drug.preferred_name)
        if drug.preferred_name == "Control Agent":
            raise SimulatedDisconnect()
        return evidence_for(drug), [], "complete", []

    async def resumed_research(drug, **_kwargs):
        research_calls.append(drug.preferred_name)
        return evidence_for(drug), [], "complete", []

    monkeypatch.setattr(catalog_module, "fetch_trial_study", fake_fetch)
    monkeypatch.setattr(catalog_module, "load_ncit_drug_index", lambda _path: _NoNCIt())
    monkeypatch.setattr(catalog_module, "research_drug", interrupting_research)
    output = tmp_path / "catalog"
    checkpoints = tmp_path / "catalog-checkpoints"
    settings = ResearchSettings(max_attempts=1)
    with pytest.raises(SimulatedDisconnect):
        asyncio.run(
            build_good_option_catalog(
                ["NCT12345678"],
                output,
                checkpoint_path=checkpoints,
                config=load_default_preset(),
                sources=[_EvidenceSource()],
                settings=settings,
                role_resolver=role_resolver,
                synthesizer=synthesize,
                class_resolver=no_classes,
            )
        )

    assert not output.exists()
    assert fetch_calls == ["NCT12345678"]
    assert role_calls == ["NCT12345678"]
    assert research_calls == ["Novel Agent", "Control Agent"]
    assert len(list((checkpoints / "research").glob("*.json"))) == 1

    with pytest.raises(ValueError, match="checkpoints are incompatible"):
        asyncio.run(
            build_good_option_catalog(
                ["NCT12345678"],
                output,
                checkpoint_path=checkpoints,
                config=load_default_preset(),
                sources=[_EvidenceSource()],
                settings=ResearchSettings(max_attempts=2),
                role_resolver=role_resolver,
                synthesizer=synthesize,
                class_resolver=no_classes,
            )
        )

    research_calls.clear()
    monkeypatch.setattr(catalog_module, "research_drug", resumed_research)
    progress: list[tuple[str, str]] = []
    catalog = asyncio.run(
        build_good_option_catalog(
            ["NCT12345678"],
            output,
            checkpoint_path=checkpoints,
            config=load_default_preset(),
            sources=[_EvidenceSource()],
            settings=settings,
            role_resolver=role_resolver,
            synthesizer=synthesize,
            class_resolver=no_classes,
            progress_callback=lambda stage, _done, _total, label: progress.append(
                (stage, label)
            ),
        )
    )

    assert len(catalog.drug_summaries) == 2
    assert fetch_calls == ["NCT12345678"]
    assert role_calls == ["NCT12345678"]
    assert research_calls == ["Control Agent"]
    assert ("registry", "NCT12345678 (checkpoint)") in progress
    assert ("screening", "NCT12345678 (checkpoint)") in progress
    assert ("research", "Novel Agent (checkpoint)") in progress
    checkpoint_manifest = json.loads(
        (checkpoints / "manifest.json").read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["completed_catalog"] == str(output)


def test_catalog_researches_drugs_concurrently_and_resumes_at_any_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research_calls: list[str] = []
    in_flight = 0
    peak = 0

    async def fake_fetch(_nct_id: str, *, client):
        del client
        return json.loads(json.dumps(STUDY))

    async def slow_research(drug, **_kwargs):
        nonlocal in_flight, peak
        research_calls.append(drug.preferred_name)
        in_flight += 1
        peak = max(peak, in_flight)
        # The first drug finishes last, so completion order differs from input.
        await asyncio.sleep(0.05 if drug.preferred_name == "Novel Agent" else 0.01)
        in_flight -= 1
        return [], [], "complete", []

    def synthesize(_drug, _evidence):
        return {category: [] for category in catalog_module._SYNTHESIS_CATEGORIES}

    monkeypatch.setattr(catalog_module, "fetch_trial_study", fake_fetch)
    monkeypatch.setattr(catalog_module, "load_ncit_drug_index", lambda _path: _NoNCIt())
    monkeypatch.setattr(catalog_module, "research_drug", slow_research)

    def build(output: Path, checkpoints: Path, concurrency: int, progress: list):
        return asyncio.run(
            build_good_option_catalog(
                ["NCT12345678"],
                output,
                checkpoint_path=checkpoints,
                config=load_default_preset(),
                sources=[_EvidenceSource()],
                settings=ResearchSettings(max_attempts=1),
                role_resolver=lambda _trial_id, _items: _screening_result(),
                synthesizer=synthesize,
                class_resolver=no_classes,
                overwrite=True,
                subject_concurrency=concurrency,
                progress_callback=lambda stage, done, _total, label: progress.append(
                    (done, label)
                )
                if stage == "research"
                else None,
            )
        )

    parallel_progress: list[tuple[int, str]] = []
    parallel = build(tmp_path / "a", tmp_path / "a-checkpoints", 2, parallel_progress)
    assert peak == 2
    assert parallel_progress == [(1, "Control Agent"), (2, "Novel Agent")]

    serial = build(tmp_path / "b", tmp_path / "b-checkpoints", 1, [])
    assert list(parallel.drug_summaries["drug_id"]) == list(
        serial.drug_summaries["drug_id"]
    )

    # Concurrency only schedules work, so a resume at another level reuses
    # every research checkpoint instead of rejecting the directory.
    research_calls.clear()
    resumed_progress: list[tuple[int, str]] = []
    build(tmp_path / "a", tmp_path / "a-checkpoints", 1, resumed_progress)
    assert research_calls == []
    assert all(label.endswith("(checkpoint)") for _done, label in resumed_progress)


def test_host_rate_limiter_paces_only_the_configured_host() -> None:
    started: dict[str, list[float]] = {"eutils.ncbi.nlm.nih.gov": [], "example.test": []}

    def handler(request: httpx.Request) -> httpx.Response:
        started[request.url.host].append(asyncio.get_event_loop().time())
        return httpx.Response(200)

    async def run() -> None:
        limiter = HostRateLimiter({"eutils.ncbi.nlm.nih.gov": 20.0})
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            event_hooks={"request": [limiter]},
        ) as client:
            await asyncio.gather(
                *(client.get("https://eutils.ncbi.nlm.nih.gov/esearch") for _ in range(5)),
                *(client.get("https://example.test/page") for _ in range(5)),
            )

    asyncio.run(run())
    paced = sorted(started["eutils.ncbi.nlm.nih.gov"])
    assert all(later - earlier >= 0.045 for earlier, later in zip(paced, paced[1:]))
    unpaced = started["example.test"]
    assert max(unpaced) - min(unpaced) < 0.045


def test_catalog_resumes_completed_drug_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class SimulatedDisconnect(BaseException):
        pass

    async def fake_fetch(_nct_id: str, *, client):
        del client
        return json.loads(json.dumps(STUDY))

    def role_resolver(_trial_id, _interventions):
        return _screening_result()

    async def fake_research(drug, **_kwargs):
        evidence = EvidencePassage(
            evidence_id=f"{drug.drug_id}:mechanism_targets",
            drug_id=drug.drug_id,
            facet="mechanism_targets",
            source="fake",
            source_type="authoritative",
            title="Synthetic evidence",
            passage=f"{drug.preferred_name} targets TARGET_MARKER.",
            url="https://example.test/source",
            source_locator="mechanism_targets",
            content_sha256=f"{drug.drug_id}:mechanism_targets",
        )
        return [evidence], [], "complete", []

    class InterruptingSynthesizer:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.interrupt = True

        def __call__(self, drug, _evidence):
            self.calls.append(drug.preferred_name)
            if drug.preferred_name == "Control Agent" and self.interrupt:
                self.interrupt = False
                raise SimulatedDisconnect()
            return {
                "mechanism_and_targets": [
                    {"text": "Targets TARGET_MARKER.", "support_ids": ["P1"]}
                ],
                "efficacy_by_tumor": [],
                "biomarker_prevalence": [],
                "biomarker_directed_efficacy": [],
                "safety": [],
                "limitations": [],
            }

    monkeypatch.setattr(catalog_module, "fetch_trial_study", fake_fetch)
    monkeypatch.setattr(catalog_module, "load_ncit_drug_index", lambda _path: _NoNCIt())
    monkeypatch.setattr(catalog_module, "research_drug", fake_research)
    synthesizer = InterruptingSynthesizer()
    arguments = {
        "nct_ids": ["NCT12345678"],
        "output_path": tmp_path / "catalog",
        "checkpoint_path": tmp_path / "catalog-checkpoints",
        "config": load_default_preset(),
        "sources": [_EvidenceSource()],
        "settings": ResearchSettings(max_attempts=1),
        "role_resolver": role_resolver,
        "synthesizer": synthesizer,
        "class_resolver": no_classes,
    }

    with pytest.raises(SimulatedDisconnect):
        asyncio.run(build_good_option_catalog(**arguments))
    assert synthesizer.calls == ["Novel Agent", "Control Agent"]
    assert (
        len(list((tmp_path / "catalog-checkpoints" / "synthesis").glob("*.json"))) == 1
    )

    asyncio.run(build_good_option_catalog(**arguments))
    assert synthesizer.calls == [
        "Novel Agent",
        "Control Agent",
        "Control Agent",
    ]


def test_checker_text_contains_only_patient_and_clean_drug_summary() -> None:
    with pytest.deprecated_call():
        text = build_good_option_checker_text(
            "Synthetic patient", _summary("D1", "Novel")
        )
    assert text.index("Patient cancer history") < text.index(
        "Investigational drug evidence summary"
    )
    assert "https://" not in text
