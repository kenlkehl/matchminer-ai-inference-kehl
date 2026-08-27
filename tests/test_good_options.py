from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
import pytest

import matchminer_ai.good_options.catalog as catalog_module
from matchminer_ai.config import load_default_preset
from matchminer_ai.good_options import (
    RUBRIC_CRITERIA,
    DrugIdentity,
    DrugSummary,
    EvidencePassage,
    GoodOptionCatalog,
    ResearchSettings,
    TrialDrugAssignment,
    build_good_option_catalog,
    build_good_option_checker_text,
    build_good_option_messages,
    evaluate_good_options,
    load_good_option_catalog,
    parse_good_option_response,
    research_drug,
    score_good_options,
    score_good_options_with_llm,
)
from matchminer_ai.good_options.research import GeneralWebEvidenceSource


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
        _summary("D2", "Second Agent", status="blocked" if blocked_second else "complete"),
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

    assert prompt.index("PATIENT CANCER HISTORY") < prompt.index(
        "SCOREABLE DRUG SUMMARIES"
    ) < prompt.index("RUBRIC")
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
        "matchminer_ai.good_options.scoring._run_good_option_llm", fake_run
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
        first = [
            {"label": criterion, "score": 1.0}
            for criterion in RUBRIC_CRITERIA
        ]
        second = [
            {"label": criterion, "score": value}
            for criterion, value in zip(
                RUBRIC_CRITERIA, (1.0, 0.0, 1.0, 0.0), strict=True
            )
        ]
        return [first, second], {"model_name": "checker"}

    config = load_default_preset()
    config.raw["good_option_checker"]["model_name"] = "local/checker"
    monkeypatch.setattr("matchminer_ai.good_options.scoring.run_checker", fake_checker)
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

    def search(self, _query: str, *, max_results: int):
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
        "matchminer_ai.good_options.research.asyncio.to_thread", run_inline
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
        assert len(interventions) == 1
        return {
            "0": {
                "role": "investigational",
                "confidence": "high",
                "rationale": "The contribution is tested.",
                "active_entity_names": ["Novel Agent"],
            }
        }

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
        "matchminer_ai.good_options.catalog.fetch_trial_study", fake_fetch
    )
    monkeypatch.setattr(
        "matchminer_ai.good_options.catalog.load_ncit_drug_index",
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
    scoreable = catalog.assignments_for_trial(
        "NCT12345678", scoreable_only=True
    )
    assert [item.preferred_name for item in scoreable] == ["Novel Agent"]
    all_roles = {item.preferred_name: item.role for item in catalog.assignments_for_trial("NCT12345678")}
    assert all_roles["Control Agent"] == "control"
    novel_summary = catalog.summary_for_drug(scoreable[0].drug_id)
    assert novel_summary is not None
    support_ids = novel_summary.structured_facts["mechanism_and_targets"][0][
        "support_ids"
    ]
    assert support_ids == [f"{scoreable[0].drug_id}:mechanism_targets"]
    assert support_ids != ["P1"]
    assert load_good_option_catalog(output).compatibility_id


def test_checker_text_contains_only_patient_and_clean_drug_summary() -> None:
    text = build_good_option_checker_text("Synthetic patient", _summary("D1", "Novel"))
    assert text.index("Patient cancer history") < text.index(
        "Investigational drug evidence summary"
    )
    assert "https://" not in text
