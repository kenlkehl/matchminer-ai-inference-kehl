from __future__ import annotations

import asyncio
import json

import httpx

from matchminer_ai.contextualization.models import TrialSpaceQuery
from matchminer_ai.contextualization.sources import (
    EUROPE_PMC_API,
    FDA_COMPANION_DIAGNOSTICS_URL,
    CIViCSource,
    EuropePMCOpenGuidelinesSource,
    FDAClinicalSource,
    NCIPDQSource,
    PubMedSource,
)


QUERY = TrialSpaceQuery(
    space_trial_id="NCT12345678-1",
    trial_id="NCT12345678",
    clinical_space_summary="Synthetic trial space",
    disease="non-small cell lung cancer",
    histology="adenocarcinoma",
    biomarkers_required="EGFR L858R mutation",
)


def test_nci_pdq_adapter_selects_disease_relevant_diagnostic_sections():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "webapis.cancer.gov":
            assert "PDQ" in request.url.path
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Non-Small Cell Lung Cancer Treatment (PDQ)",
                            "url": (
                                "https://www.cancer.gov/types/lung/hp/"
                                "lung-treatment-pdq"
                            ),
                            "contentType": "pdqCancerInfoSummary",
                            "description": "Evidence summary description.",
                        },
                        {
                            "title": "Salivary Gland Cancer Treatment (PDQ)",
                            "url": (
                                "https://www.cancer.gov/types/head-and-neck/hp/"
                                "salivary-gland-treatment-pdq"
                            ),
                            "contentType": "pdqCancerInfoSummary",
                            "description": "Unrelated disease summary.",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            text=(
                "<html><head>"
                "<meta name='dcterms.issued' content='2025-01-01'>"
                "<meta name='dcterms.modified' content='2026-01-01'>"
                "</head><body><main id='main-content'><article>"
                "<h1>Non-Small Cell Lung Cancer</h1>"
                "<h2>Diagnostic Evaluation</h2>"
                "<p>Biopsy and imaging establish pathology and stage.</p>"
                "<h2>Treatment</h2><p>Evidence summary text.</p>"
                "</article></main></body></html>"
            ),
            headers={"content-type": "text/html"},
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await NCIPDQSource().fetch(
                QUERY,
                client=client,
                max_items=3,
                settings={},
            )

    items, notices = asyncio.run(run())

    assert items[0].evidence_id.startswith("nci-pdq:lung-treatment-pdq:section-")
    assert "Biopsy and imaging" in items[0].excerpt
    assert items[0].evidence_type == "diagnostic_evidence_summary"
    assert "Diagnostic Evaluation" in items[0].source_locator
    assert items[0].published_at == "2025-01-01"
    assert items[0].updated_at == "2026-01-01"
    assert items[0].attributes["is_clinical_practice_guideline"] is False
    assert notices[0].status == "ok"


def test_fda_adapter_parses_companion_diagnostic_and_label():
    table = """
    <table>
      <tr><th>Diagnostic Name</th><th>Indication - Sample Type</th>
          <th>Drug Trade Name (Generic)</th><th>Biomarker</th>
          <th>Details</th><th>PMA</th></tr>
      <tr><td>EGFR Test</td><td>Non-Small Cell Lung Cancer - Tissue</td>
          <td>TAGRISSO (osimertinib) NDA 208065</td><td>EGFR</td>
          <td>L858R mutation</td><td>P123 (01/01/2026)</td></tr>
    </table>
    """
    spl = """
    <document xmlns="urn:hl7-org:v3">
      <component><structuredBody><component><section>
        <title>INDICATIONS AND USAGE</title>
        <text>Indicated for synthetic test use.</text>
      </section></component></structuredBody></component>
    </document>
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == FDA_COMPANION_DIAGNOSTICS_URL:
            return httpx.Response(200, text=table)
        if request.url.path.endswith("/spls.json"):
            assert request.url.params["drug_name"] == "osimertinib"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "setid": "synthetic-set",
                            "title": "TAGRISSO label",
                            "published_date": "Jan 01, 2026",
                            "spl_version": "2",
                        }
                    ]
                },
            )
        if request.url.path.endswith("/synthetic-set.xml"):
            return httpx.Response(200, text=spl)
        return httpx.Response(404)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await FDAClinicalSource().fetch(
                QUERY,
                client=client,
                max_items=3,
                settings={},
            )

    items, notices = asyncio.run(run())

    assert {item.source for item in items} == {
        "fda_companion_diagnostics",
        "dailymed",
    }
    assert any("L858R" in item.excerpt for item in items)
    assert any("INDICATIONS" in item.excerpt for item in items)
    assert notices[0].status == "ok"


def test_civic_adapter_requests_only_accepted_disease_evidence():
    captured_payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        captured_payloads.append(payload)
        if "diseaseTypeahead" in payload["query"]:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "diseaseTypeahead": [
                            {
                                "id": 8,
                                "name": "Lung Non-small Cell Carcinoma",
                                "displayName": "Lung Non-small Cell Carcinoma",
                                "doid": "3908",
                            }
                        ]
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "evidenceItems": {
                        "nodes": [
                            {
                                "id": 7,
                                "status": "ACCEPTED",
                                "molecularProfile": {
                                    "id": 1,
                                    "name": "EGFR L858R",
                                    "link": "/molecular-profiles/1",
                                },
                                "evidenceType": "PREDICTIVE",
                                "evidenceLevel": "A",
                                "evidenceRating": 5,
                                "evidenceDirection": "SUPPORTS",
                                "significance": "SENSITIVITYRESPONSE",
                                "description": "Curated synthetic statement.",
                                "disease": {
                                    "id": 2,
                                    "doid": "3908",
                                    "name": "Lung Cancer",
                                    "displayName": "Lung Cancer",
                                },
                                "therapies": [{"id": 3, "name": "Osimertinib"}],
                                "source": {
                                    "id": 4,
                                    "citationId": "123",
                                    "citation": "Example et al.",
                                    "journal": "Example",
                                    "sourceType": "PUBMED",
                                },
                            }
                        ]
                    }
                }
            },
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await CIViCSource().fetch(
                QUERY,
                client=client,
                max_items=3,
                settings={},
            )

    items, notices = asyncio.run(run())

    evidence_payload = captured_payloads[-1]
    assert (
        evidence_payload["variables"]["diseaseName"]
        == "Lung Non-small Cell Carcinoma"
    )
    assert "status: ACCEPTED" in evidence_payload["query"]
    assert items[0].attributes["molecular_profile"] == "EGFR L858R"
    assert items[0].license == "CC0-1.0"
    assert notices[0].status == "ok"


def test_pubmed_adapter_splits_facets_then_ranks_guideline_abstracts():
    requests: list[httpx.Request] = []
    xml = """
    <PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>12345</PMID><Article>
        <ArticleTitle>Synthetic guideline article</ArticleTitle>
        <Abstract><AbstractText>Abstract evidence only.</AbstractText></Abstract>
        <Journal><Title>Example Journal</Title><JournalIssue>
          <PubDate><Year>2026</Year></PubDate>
        </JournalIssue></Journal>
        <PublicationTypeList>
          <PublicationType>Practice Guideline</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/esearch.fcgi"):
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": ["12345"]}},
            )
        return httpx.Response(200, text=xml)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await PubMedSource().fetch(
                QUERY,
                client=client,
                max_items=3,
                settings={"ncbi_email": "developer@example.org"},
            )

    items, notices = asyncio.run(run())

    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "esearch.fcgi",
        "esearch.fcgi",
        "esearch.fcgi",
        "efetch.fcgi",
    ]
    assert requests[0].url.params["tool"] == "matchminer-ai"
    assert requests[0].url.params["sort"] == "relevance"
    assert "diagnosis" in requests[0].url.params["term"]
    assert "molecular" in requests[1].url.params["term"]
    assert "treatment" in requests[2].url.params["term"]
    assert "PRIVATE_PATIENT_MARKER" not in str(requests[0].url)
    assert items[0].source_locator == "PMID 12345"
    assert items[0].excerpt == "Abstract evidence only."
    assert items[0].evidence_type == "diagnostic_guideline_abstract"
    assert items[0].attributes["search_facets"] == [
        "diagnostic_workup",
        "molecular_testing",
        "treatment_guidance",
    ]
    assert notices[0].status == "ok"


def test_europe_pmc_adapter_requires_permissive_license_and_extracts_sections():
    requests: list[httpx.Request] = []
    search_payload = {
        "resultList": {
            "result": [
                {
                    "title": (
                        "Consensus recommendations for tissue acquisition in "
                        "non-small cell lung cancer"
                    ),
                    "pmcid": "PMC111",
                    "pmid": "111",
                    "doi": "10.1/example",
                    "inEPMC": "Y",
                    "license": "cc by",
                    "firstPublicationDate": "2026-01-01",
                    "abstractText": "Diagnostic biopsy and staging guidance.",
                    "pubTypeList": {"pubType": ["Consensus Statement"]},
                },
                {
                    "title": "Guideline for non-small cell lung cancer staging",
                    "pmcid": "PMC222",
                    "inEPMC": "Y",
                    "license": "cc by-nc-nd",
                    "abstractText": "This record must be rejected.",
                    "pubTypeList": {"pubType": ["Practice Guideline"]},
                },
            ]
        }
    }
    full_text = """
    <article><body><sec><title>Materials and methods</title>
      <p>Diagnostic terms in a survey methodology section.</p>
    </sec><sec><title>Diagnostic evaluation and staging</title>
      <p>Obtain tissue by biopsy for histologic and molecular testing.</p>
      <p>Use imaging to establish disease extent.</p>
    </sec><sec><title>References</title><p>Not selected.</p></sec>
    </body></article>
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if str(request.url).startswith(f"{EUROPE_PMC_API}/search"):
            return httpx.Response(200, json=search_payload)
        if request.url.path.endswith("/PMC111/fullTextXML"):
            return httpx.Response(200, text=full_text)
        return httpx.Response(500, text="A rejected-license record was fetched")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            return await EuropePMCOpenGuidelinesSource().fetch(
                QUERY,
                client=client,
                max_items=3,
                settings={},
            )

    items, notices = asyncio.run(run())

    assert len(items) == 1
    assert items[0].source == "europe_pmc_open_guidelines"
    assert items[0].evidence_type == "diagnostic_guideline_full_text"
    assert items[0].license == "CC BY"
    assert "histologic and molecular testing" in items[0].excerpt
    assert all("methods" not in item.source_locator.casefold() for item in items)
    assert all("PMC222" not in str(request.url) for request in requests)
    assert "Rejected 1 result" in notices[0].message
