from __future__ import annotations

import asyncio
import json

import httpx

from matchminer_ai.contextualization.models import TrialSpaceQuery
from matchminer_ai.contextualization.sources import (
    FDA_COMPANION_DIAGNOSTICS_URL,
    CIViCSource,
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


def test_nci_pdq_adapter_parses_syndicated_content():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "webapis.cancer.gov":
            assert "PDQ" in request.url.path
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Lung Cancer Treatment (PDQ)",
                            "url": (
                                "https://www.cancer.gov/types/lung/hp/"
                                "lung-treatment-pdq"
                            ),
                            "contentType": "pdqCancerInfoSummary",
                            "description": "Evidence summary description.",
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
                "<h1>Treatment</h1><p>Evidence summary text.</p>"
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

    assert items[0].evidence_id == "nci-pdq:lung-treatment-pdq"
    assert "Evidence summary text." in items[0].excerpt
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


def test_pubmed_adapter_uses_esearch_then_abstract_only_efetch():
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
        "efetch.fcgi",
    ]
    assert requests[0].url.params["tool"] == "matchminer-ai"
    assert "PRIVATE_PATIENT_MARKER" not in str(requests[0].url)
    assert items[0].source_locator == "PMID 12345"
    assert items[0].excerpt == "Abstract evidence only."
    assert notices[0].status == "ok"
