from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.good_options import (
    BIOMARKER_EXPRESSION_QUERY_SUFFIX,
    build_biomarker_expression_search_queries,
    build_good_option_checker_text,
    evaluate_good_options,
    extract_registry_drug_interventions,
    parse_experimental_drug_selection,
    research_good_options,
    score_good_options,
    score_good_options_with_llm,
)
from matchminer_ai.help_me_choose import (
    DrugIntervention,
    DrugSearchResult,
    TrialDrugResearch,
    build_comparison_messages,
    generate_trial_comparison,
)

STUDY = {
    "protocolSection": {
        "identificationModule": {"briefTitle": "Investigational agent study"},
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {"phases": ["PHASE1"]},
        "descriptionModule": {"briefSummary": "A first-in-human study."},
        "armsInterventionsModule": {
            "armGroups": [
                {
                    "label": "Experimental arm",
                    "type": "EXPERIMENTAL",
                    "description": "Novel Agent with standard carboplatin.",
                },
                {
                    "label": "Control arm",
                    "type": "ACTIVE_COMPARATOR",
                    "description": "Standard Drug alone.",
                },
            ],
            "interventions": [
                {
                    "type": "DRUG",
                    "name": "Novel Agent 10 mg IV",
                    "otherNames": ["Novel Agent"],
                    "armGroupLabels": ["Experimental arm"],
                },
                {
                    "type": "DRUG",
                    "name": "Carboplatin",
                    "armGroupLabels": ["Experimental arm"],
                },
                {
                    "type": "DRUG",
                    "name": "Standard Drug",
                    "armGroupLabels": ["Control arm"],
                },
            ],
        },
    }
}


def _research() -> TrialDrugResearch:
    return TrialDrugResearch(
        nct_id="NCT12345678",
        title="Novel Agent trial",
        phases=("PHASE1",),
        brief_summary="Novel Agent is being evaluated.",
        interventions=(
            DrugIntervention(
                name="Novel Agent",
                intervention_type="DRUG",
                description="Investigational targeted agent.",
            ),
            DrugIntervention(
                name="Second Agent",
                intervention_type="BIOLOGICAL",
                description="Investigational antibody.",
            ),
        ),
        search_results=(
            DrugSearchResult(
                query='"Novel Agent" oncology mechanism efficacy safety clinical trial',
                title="Clinical evidence",
                snippet="RESEARCH_EXTRACT reported human responses.",
                url="https://example.test/efficacy",
            ),
            DrugSearchResult(
                query=(f'"Novel Agent" {BIOMARKER_EXPRESSION_QUERY_SUFFIX}'),
                title="Biomarker prevalence",
                snippet="The target occurs in 30% of the disease population.",
                url="https://example.test/prevalence",
            ),
        ),
    )


def _six_of_eight_response() -> str:
    assessments = []
    for index, drug in enumerate(("Novel Agent", "Second Agent")):
        points = (1, 1, 1, 1) if index == 0 else (1, 0, 1, 0)
        assessments.append(
            {
                "drug_name": drug,
                "targeted_biomarkers": ["Marker A"],
                "disease_type_benefit": {
                    "point": points[0],
                    "rationale": "Human benefit evidence is supplied.",
                    "evidence_labels": ["S1"] if points[0] else [],
                },
                "common_biomarker_in_disease": {
                    "point": points[1],
                    "rationale": "Population prevalence evidence is supplied.",
                    "evidence_labels": ["S2"] if points[1] else [],
                },
                "patient_biomarker_targeted": {
                    "point": points[2],
                    "rationale": "The patient marker and target are supplied.",
                    "evidence_labels": ["PATIENT", "S1"] if points[2] else [],
                },
                "biomarker_targeted_benefit": {
                    "point": points[3],
                    "rationale": "Human target-benefit evidence is supplied.",
                    "evidence_labels": ["PATIENT", "S1"] if points[3] else [],
                },
            }
        )
    return json.dumps(
        {
            "patient_disease_type": "Synthetic cancer",
            "drug_assessments": assessments,
            "key_uncertainties": ["Small studies"],
        }
    )


def test_selection_uses_llm_roles_but_code_excludes_control_only_drug() -> None:
    registry_interventions = extract_registry_drug_interventions(STUDY)
    registry = TrialDrugResearch(
        nct_id="NCT12345678",
        interventions=registry_interventions,
    )
    response = json.dumps(
        {
            "interventions": [
                {
                    "source_index": 0,
                    "experimental_role": "investigational",
                    "canonical_drug_names": ["Novel Agent"],
                    "rationale": "The agent is evaluated in the experimental arm.",
                },
                {
                    "source_index": 1,
                    "experimental_role": "not_investigational",
                    "canonical_drug_names": [],
                    "rationale": "Carboplatin is the standard backbone.",
                },
                {
                    "source_index": 2,
                    "experimental_role": "investigational",
                    "canonical_drug_names": ["Standard Drug"],
                    "rationale": "Incorrectly selected by the model.",
                },
            ]
        }
    )

    selected, status, notices = parse_experimental_drug_selection(
        response,
        research=registry,
    )

    assert [item.name for item in selected] == ["Novel Agent"]
    assert status == "ok"
    assert any("control arm" in notice for notice in notices)


def test_research_queries_are_drug_only_and_include_biomarker_prevalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_queries: list[str] = []

    async def fake_fetch(_nct_id, *, client):
        del client
        study = json.loads(json.dumps(STUDY))
        study["protocolSection"]["armsInterventionsModule"]["interventions"] = [
            study["protocolSection"]["armsInterventionsModule"]["interventions"][0]
        ]
        return study

    def fake_search(queries):
        captured_queries.extend(queries)
        return (), ()

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("matchminer_ai.good_options.fetch_trial_study", fake_fetch)
    monkeypatch.setattr("matchminer_ai.good_options.asyncio.to_thread", run_inline)
    results = asyncio.run(
        research_good_options(
            ["NCT12345678"],
            use_llm_drug_selection=False,
            search_function=fake_search,
        )
    )

    assert [item.name for item in results[0].interventions] == ["Novel Agent 10 mg IV"]
    assert captured_queries
    assert any(BIOMARKER_EXPRESSION_QUERY_SUFFIX in query for query in captured_queries)
    assert all("PRIVATE_PATIENT" not in query for query in captured_queries)
    assert (
        "patient"
        not in inspect.signature(build_biomarker_expression_search_queries).parameters
    )
    assert "patient" not in inspect.signature(research_good_options).parameters


def test_llm_scoring_batches_independent_single_patient_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_messages: list[list[dict[str, str]]] = []

    def fake_run(messages_list, *, config):
        del config
        captured_messages.extend(messages_list)
        return SimpleNamespace(
            final_outputs=[_six_of_eight_response(), _six_of_eight_response()],
            reasoning_outputs=["reasoning-a", "reasoning-b"],
            finish_reasons=["stop", "stop"],
            model_metadata={"model_name": "teacher"},
        )

    monkeypatch.setattr("matchminer_ai.good_options._run_good_option_llm", fake_run)
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "NCT12345678",
                "cancer_history_summary": "PATIENT_ONE_MARKER synthetic cancer.",
                "clinical_space_summary": "Must not be used.",
            },
            {
                "patient_id": "P2",
                "trial_id": "NCT12345678",
                "cancer_history_summary": "PATIENT_TWO_MARKER synthetic cancer.",
                "clinical_space_summary": "Must not be used.",
            },
        ]
    )

    output = score_good_options_with_llm(
        pairs,
        research=[_research()],
        config=load_default_preset(),
    )

    assert output["good_option_score"].tolist() == pytest.approx([0.75, 0.75])
    assert output["good_option_points"].tolist() == [6, 6]
    assert output["good_option_max_points"].tolist() == [8, 8]
    assert len(captured_messages) == 2
    assert "PATIENT_ONE_MARKER" in captured_messages[0][1]["content"]
    assert "PATIENT_TWO_MARKER" not in captured_messages[0][1]["content"]
    assert "PATIENT_TWO_MARKER" in captured_messages[1][1]["content"]
    assert "PATIENT_ONE_MARKER" not in captured_messages[1][1]["content"]
    assert all(
        "Must not be used" not in messages[1]["content"]
        for messages in captured_messages
    )


def test_classifier_receives_patient_research_extract_and_registry_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_checker(prompts, *, checker_config, model_metadata_cache_dir=None):
        captured["prompts"] = prompts
        captured["config"] = checker_config
        captured["cache"] = model_metadata_cache_dir
        return [{"score": 0.0}], {"model_name": checker_config["model_name"]}

    config = load_default_preset()
    config.raw["good_option_checker"]["model_name"] = "local/good-option-checker"
    monkeypatch.setattr("matchminer_ai.good_options.run_checker", fake_checker)
    pairs = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "trial_id": "NCT12345678",
                "cancer_history_summary": "SYNTHETIC_PATIENT_MARKER",
                "clinical_space_summary": "SPACE_TEXT_MUST_NOT_APPEAR",
            }
        ]
    )

    output, metadata = score_good_options(
        pairs,
        research=[_research()],
        config=config,
        return_metadata=True,
    )

    prompt = captured["prompts"][0]
    assert "SYNTHETIC_PATIENT_MARKER" in prompt
    assert "RESEARCH_EXTRACT" in prompt
    assert "Registry investigational-drug context:" in prompt
    assert "SPACE_TEXT_MUST_NOT_APPEAR" not in prompt
    assert output["good_option_score"].tolist() == pytest.approx([0.5])
    assert output["good_option_method"].tolist() == ["classifier"]
    assert metadata["checker_input_version"] == (
        "patient-drug-research-plus-registry-v1"
    )


def test_dispatch_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        evaluate_good_options(
            pd.DataFrame(),
            research=[],
            method="guess",
        )


def test_help_me_choose_prompt_receives_auditable_good_option_result() -> None:
    messages, _sources = build_comparison_messages(
        patient_summary="Synthetic patient",
        patient_exclusion_evidence="No evidence",
        match_contexts=[{"nct_id": "NCT12345678"}],
        research=[_research()],
        good_option_results=[
            {
                "trial_id": "NCT12345678",
                "good_option_method": "llm",
                "good_option_score": 0.75,
                "good_option_points": 6,
                "good_option_max_points": 8,
                "good_option_drug_count": 2,
                "good_option_status": "ok",
                "good_option_drug_assessments": [
                    {
                        "drug_name": "Novel Agent",
                        "disease_type_benefit": {
                            "point": 1,
                            "evidence_labels": ["S1", "CT"],
                        },
                    }
                ],
                "good_option_uncertainties": ["Small studies"],
            }
        ],
    )

    prompt = messages[1]["content"]
    assert '"score_0_to_1": 0.75' in prompt
    assert '"points": 6' in prompt
    assert '"T1-S1"' in prompt
    assert '"T1-CT"' in prompt
    assert "without converting it into predicted benefit" in prompt


def test_generate_trial_comparison_runs_selected_good_option_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate(frame, *, research, method, config, return_metadata):
        del research, config
        captured["method"] = method
        assert return_metadata
        return (
            pd.DataFrame(
                [
                    {
                        "patient_id": frame.iloc[0]["patient_id"],
                        "trial_id": "NCT12345678",
                        "good_option_method": method,
                        "good_option_score": 0.5,
                        "good_option_status": "ok",
                    }
                ]
            ),
            {"method": method},
        )

    def fake_prompts(messages_list, *, llm_config):
        del llm_config
        captured["comparison_messages"] = messages_list
        return [SimpleNamespace(prompt_text="rendered", max_tokens=100)]

    class Backend:
        def generate_llm_outputs(self, **_kwargs):
            return SimpleNamespace(
                final_outputs=["## Trial ranking\n\n1. NCT12345678"],
                reasoning_outputs=[""],
                finish_reasons=["stop"],
                model_metadata={"model_name": "comparison-model"},
            )

    async def run_inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(
        "matchminer_ai.good_options.evaluate_good_options",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "matchminer_ai.llm.prompt_rendering.build_prompt_list", fake_prompts
    )
    monkeypatch.setattr(
        "matchminer_ai.llm.backends.get_llm_backend", lambda _config: Backend()
    )
    monkeypatch.setattr("matchminer_ai.help_me_choose.asyncio.to_thread", run_inline)

    report, metadata = asyncio.run(
        generate_trial_comparison(
            patient_summary="Synthetic patient",
            patient_exclusion_evidence="",
            match_contexts=[{"nct_id": "NCT12345678"}],
            research=[_research()],
            good_option_method="classifier",
            config=load_default_preset(),
            return_metadata=True,
        )
    )

    assert captured["method"] == "classifier"
    assert '"score_0_to_1": 0.5' in captured["comparison_messages"][0][1]["content"]
    assert metadata["good_option_method"] == "classifier"
    assert "NCT12345678" in report


def test_classifier_text_orders_patient_then_research_then_registry() -> None:
    text = build_good_option_checker_text("Synthetic patient", _research())

    assert (
        text.index("Patient cancer history:")
        < text.index("Investigational-drug public research evidence:")
        < text.index("Registry investigational-drug context:")
    )
