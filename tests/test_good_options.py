from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
import pytest

import matchminer_ai.trials.drug_catalog as catalog_module
from matchminer_ai.config import load_default_preset
from matchminer_ai.matching import (
    RUBRIC_CRITERIA,
    build_good_option_checker_text,
    build_good_option_messages,
    evaluate_good_options,
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
    build_facet_query,
)


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
        < prompt.index("SCOREABLE DRUG SUMMARIES")
        < prompt.index("RUBRIC")
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
    assert "control, background" in messages[0]["content"]


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
    output, metadata = score_good_options(
        pairs, catalog=_catalog(), config=config, return_metadata=True
    )

    assert output.loc[0, "good_option_score"] == pytest.approx(0.75)
    assert len(captured) == 2
    assert all("Control Agent" not in prompt for prompt in captured)
    assert metadata["checker_input_version"].endswith("four-logit")


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

    def fake_run(_messages, *, config, stage):
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
                        {"claim": "Targets Marker A.", "support_ids": ["P1"]}
                    ],
                    "efficacy_by_tumor": [],
                    "biomarker_prevalence": [],
                    "biomarker_directed_efficacy": [],
                    "safety": [],
                    "limitations": [],
                }
            )
        ]

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)
    monkeypatch.setattr(catalog_module.asyncio, "to_thread", run_inline)
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
        catalog_module._default_synthesize_many(
            [(DrugIdentity(drug_id="D1", preferred_name="Novel Agent"), [evidence])],
            config=config,
        )
    )

    assert calls == 2
    assert result["D1"]["__passage_id_map__"] == {"P1": "ledger:E1"}


def test_synthesis_retries_token_limited_or_all_empty_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(messages_list, *, config, stage):
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
        return [
            json.dumps(
                {
                    "mechanism_and_targets": [
                        {"claim": "Targets Marker A.", "support_ids": ["P1"]}
                    ],
                    "efficacy_by_tumor": [],
                    "biomarker_prevalence": [],
                    "biomarker_directed_efficacy": [],
                    "safety": [],
                    "limitations": [],
                }
            )
        ]

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(catalog_module, "_run_llm_messages", fake_run)
    monkeypatch.setattr(catalog_module.asyncio, "to_thread", run_inline)
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
        catalog_module._default_synthesize_many(
            [(DrugIdentity(drug_id="D1", preferred_name="Novel Agent"), [evidence])],
            config=load_default_preset(),
        )
    )

    assert calls == 3
    assert result["D1"]["mechanism_and_targets"][0]["support_ids"] == ["P1"]


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
                    "claim": "Phase One Agent inhibits Marker A.",
                    "support_ids": ["ncit_definition"],
                }
            ]
        },
        evidence=evidence,
    )
    assert facts["mechanism_and_targets"][0]["support_ids"] == ["ncit_definition:C123"]


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
                {"claim": "Targets TARGET_MARKER.", "support_ids": support}
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
                {"claim": "Targets TARGET_MARKER.", "support_ids": ["P1"]}
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
                    {"claim": "Targets TARGET_MARKER.", "support_ids": ["P1"]}
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
    text = build_good_option_checker_text("Synthetic patient", _summary("D1", "Novel"))
    assert text.index("Patient cancer history") < text.index(
        "Investigational drug evidence summary"
    )
    assert "https://" not in text
