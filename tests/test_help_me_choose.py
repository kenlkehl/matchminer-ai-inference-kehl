from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pandas as pd
import pytest

from matchminer_ai.good_options import (
    DrugSummary,
    GoodOptionCatalog,
    TrialDrugAssignment,
)
from matchminer_ai.help_me_choose import (
    DrugIntervention,
    TrialDrugResearch,
    build_comparison_messages,
    build_drug_search_queries,
    extract_drug_interventions,
    extract_trial_eligibility_criteria,
    fetch_trial_eligibility_criteria,
    fetch_trial_registry_document,
    normalize_nct_reference,
    request_vllm_comparison,
    research_trials,
)


STUDY = {
    "protocolSection": {
        "identificationModule": {
            "briefTitle": "Drug A plus Drug B study",
            "officialTitle": "Official Drug A plus Drug B Study",
        },
        "designModule": {"phases": ["PHASE2"]},
        "descriptionModule": {
            "briefSummary": "A combination study.",
            "detailedDescription": "A detailed registry description.",
        },
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\r\n\r\n* EGFR &amp; ALK testing required.\r\n"
                "\r\nExclusion Criteria:\r\n\r\n* Active brain metastases."
            )
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
            "lastUpdatePostDateStruct": {
                "date": "2026-07-15",
                "type": "ACTUAL",
            },
        },
        "armsInterventionsModule": {
            "interventions": [
                {
                    "type": "DRUG",
                    "name": "Drug A",
                    "description": "A targeted agent.",
                    "otherNames": ["Agent A"],
                },
                {
                    "type": "BIOLOGICAL",
                    "name": "Drug B",
                    "description": "An antibody.",
                },
                {"type": "DRUG", "name": "Placebo"},
                {"type": "PROCEDURE", "name": "Tumor biopsy"},
            ]
        },
    }
}


def test_extracts_only_non_placebo_drug_interventions():
    interventions = extract_drug_interventions(STUDY)

    assert [item.name for item in interventions] == ["Drug A", "Drug B"]
    assert [item.intervention_type for item in interventions] == [
        "DRUG",
        "BIOLOGICAL",
    ]


def test_extracts_and_fetches_complete_eligibility_criteria():
    criteria = extract_trial_eligibility_criteria(STUDY)
    assert "EGFR & ALK testing required" in criteria
    assert "\n\n" in criteria

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/v2/studies/NCT12345678")
        return httpx.Response(200, json=STUDY)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = fetch_trial_eligibility_criteria(
            "nct12345678",
            client=client,
        )

    assert result.nct_id == "NCT12345678"
    assert result.eligibility_criteria == criteria
    assert result.source_url.endswith("/study/NCT12345678")
    assert result.last_update_post_date == "2026-07-15"
    assert result.fetched_at_utc


def test_missing_complete_eligibility_criteria_is_rejected():
    with pytest.raises(ValueError, match="does not provide complete"):
        extract_trial_eligibility_criteria({"protocolSection": {}})


def test_nct_url_fetch_is_wrangled_for_trial_space_extraction():
    assert normalize_nct_reference(
        "https://clinicaltrials.gov/study/NCT12345678?format=json"
    ) == "NCT12345678"
    assert normalize_nct_reference(
        "clinicaltrials.gov/ct2/show/nct12345678"
    ) == "NCT12345678"
    with pytest.raises(ValueError, match="ClinicalTrials.gov"):
        normalize_nct_reference("https://example.test/study/NCT12345678")

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/v2/studies/NCT12345678")
        return httpx.Response(200, json=STUDY)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        document = fetch_trial_registry_document(
            "https://clinicaltrials.gov/study/NCT12345678",
            client=client,
        )

    assert document.nct_id == "NCT12345678"
    assert document.trial_title == "Official Drug A plus Drug B Study"
    assert document.brief_summary == "A combination study."
    assert document.detailed_description == "A detailed registry description."
    assert "Active brain metastases" in document.eligibility_criteria


def test_search_query_api_has_no_patient_parameter():
    query_parameters = inspect.signature(build_drug_search_queries).parameters
    research_parameters = inspect.signature(research_trials).parameters

    assert list(query_parameters) == ["interventions"]
    assert "patient_summary" not in research_parameters
    assert "patient_history" not in research_parameters


def test_queries_contain_only_drug_names_and_generic_search_terms():
    marker = "PRIVATE_PATIENT_MARKER"
    queries = build_drug_search_queries(
        (
            DrugIntervention(name="Drug A", intervention_type="DRUG"),
            DrugIntervention(name="Drug B", intervention_type="BIOLOGICAL"),
        )
    )

    assert any("Drug A" in query for query in queries)
    assert any("Drug B" in query for query in queries)
    assert all(marker not in query for query in queries)


def test_research_reports_progress_as_each_trial_finishes():
    updates: list[tuple[int, int, str]] = []

    async def fake_research(nct_id: str, *, client: httpx.AsyncClient):
        del client
        return TrialDrugResearch(nct_id=nct_id)

    async def run():
        with patch(
            "matchminer_ai.help_me_choose.research_trial_drugs",
            new=fake_research,
        ):
            return await research_trials(
                ["NCT12345678", "NCT87654321"],
                progress_callback=lambda completed, total, nct_id: updates.append(
                    (completed, total, nct_id)
                ),
            )

    results = asyncio.run(run())

    assert [item.nct_id for item in results] == ["NCT12345678", "NCT87654321"]
    assert [item[0] for item in updates] == [1, 2]
    assert all(item[1] == 2 for item in updates)


def test_patient_marker_reaches_comparison_only_after_catalog_load():
    assignment = TrialDrugAssignment(
        trial_id="NCT12345678",
        drug_id="D1",
        preferred_name="Drug A",
        registry_name="Drug A",
        intervention_type="DRUG",
        role="investigational",
        role_confidence="high",
        scoreable=True,
    )
    summary = DrugSummary(
        drug_id="D1",
        preferred_name="Drug A",
        ncit_code="",
        research_status="complete",
        synthesis_status="ok",
        structured_facts={},
        good_option_summary="Drug A evidence summary.",
        help_me_choose_summary="Drug A mechanism, efficacy, and safety summary.",
        evidence_count=1,
    )
    catalog = GoodOptionCatalog(
        path=Path("/tmp/catalog"),
        manifest={"compatibility_id": "v2"},
        trial_registry=pd.DataFrame(
            [{"trial_id": "NCT12345678", "title": "Trial", "registry_status": "ok", "phases_json": "[]", "brief_summary": ""}]
        ),
        trial_drug_index=pd.DataFrame([assignment.to_record()]),
        drug_summaries=pd.DataFrame([summary.to_record()]),
        drug_evidence=pd.DataFrame(
            [{"drug_id": "D1", "title": "Drug A results", "url": "https://example.org/drug-a"}]
        ),
        drug_research_attempts=pd.DataFrame(),
    )
    marker = "PRIVATE_PATIENT_MARKER"
    messages, sources = build_comparison_messages(
        patient_summary=f"Patient summary {marker}",
        patient_exclusion_evidence="No additional evidence.",
        match_contexts=[
            {
                "nct_id": "NCT12345678",
                "clinical_space_summary": "A matched disease space.",
                "general_exclusion_criteria": "Standard exclusions.",
                "match_quality_score": 0.9,
                "similarity_score": 0.8,
            }
        ],
        catalog=catalog,
    )

    assert marker in messages[1]["content"]
    assert [source.label for source in sources] == ["T1-CT", "T1-S1"]
    prompt = messages[1]["content"]
    assert "https://example.org" not in prompt
    assert "drug_only_query" not in prompt
    assert (
        "The drug mechanism, efficacy, and safety subsection must always be first"
        in prompt
    )
    assert prompt.index("#### Drug mechanism, efficacy, and safety") < prompt.index(
        "#### Potential advantages for this patient"
    )


def test_google_comparison_preserves_openapi_url_and_omits_vllm_extensions():
    captured: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Comparison report")
                    )
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            return None

    async def token_provider():
        return "google-token"

    base_url = (
        "https://aiplatform.googleapis.com/v1/projects/profile-notes/"
        "locations/global/endpoints/openapi"
    )
    with patch("matchminer_ai.help_me_choose.AsyncOpenAI", FakeClient):
        report = asyncio.run(
            request_vllm_comparison(
                messages=[
                    {"role": "system", "content": "Follow the evidence."},
                    {"role": "user", "content": "Compare the trials."},
                ],
                base_url=base_url,
                model="google/gemma-4-26b-a4b-it-maas",
                api_key=token_provider,
                send_vllm_extra_body=True,
                remote_config={"provider": "google_agent_platform"},
            )
        )

    assert report == "Comparison report"
    assert captured["client"]["base_url"] == base_url
    request = captured["request"]
    assert request["extra_body"] is None
    assert request["messages"] == [
        {
            "role": "user",
            "content": (
                "Instructions:\nFollow the evidence.\n\n"
                "Request:\nCompare the trials."
            ),
        }
    ]
