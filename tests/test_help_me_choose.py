from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch

import httpx
import pytest

from matchminer_ai.help_me_choose import (
    DrugIntervention,
    DrugSearchResult,
    TrialDrugResearch,
    build_comparison_messages,
    build_drug_search_queries,
    extract_drug_interventions,
    extract_trial_eligibility_criteria,
    fetch_trial_eligibility_criteria,
    fetch_trial_registry_document,
    normalize_nct_reference,
    research_trial_drugs,
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


def test_patient_marker_reaches_llm_prompt_but_not_web_query():
    captured_queries: list[str] = []

    async def fake_fetch(_nct_id: str, *, client: httpx.AsyncClient):
        del client
        return STUDY

    def fake_search(queries):
        captured_queries.extend(queries)
        return (
            (
                DrugSearchResult(
                    query=queries[0],
                    title="Drug A results",
                    snippet="Reported findings.",
                    url="https://example.org/drug-a",
                ),
            ),
            (),
        )

    async def run_inline(function, *args):
        return function(*args)

    async def run():
        async with httpx.AsyncClient() as client:
            with (
                patch(
                    "matchminer_ai.help_me_choose.fetch_trial_study",
                    new=fake_fetch,
                ),
                patch(
                    "matchminer_ai.help_me_choose.asyncio.to_thread",
                    new=run_inline,
                ),
            ):
                return await research_trial_drugs(
                    "NCT12345678",
                    client=client,
                    search_function=fake_search,
                )

    research = asyncio.run(run())

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
        research=[research],
    )

    assert captured_queries
    assert all(marker not in query for query in captured_queries)
    assert marker in messages[1]["content"]
    assert [source.label for source in sources] == ["T1-CT", "T1-S1"]
    prompt = messages[1]["content"]
    assert (
        "The drug mechanism, efficacy, and safety subsection must always be first"
        in prompt
    )
    assert prompt.index("#### Drug mechanism, efficacy, and safety") < prompt.index(
        "#### Potential advantages for this patient"
    )
