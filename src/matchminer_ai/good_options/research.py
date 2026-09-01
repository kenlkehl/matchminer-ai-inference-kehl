"""Patient-free, retrying evidence retrieval for canonical oncology drugs."""

from __future__ import annotations

import asyncio
import hashlib
import html
import io
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any, Protocol
from urllib.parse import quote

import httpx
from ddgs import DDGS

from .models import DrugIdentity, EvidencePassage, ResearchAttempt


FACETS = (
    "mechanism_targets",
    "efficacy_by_tumor",
    "biomarker_prevalence",
    "biomarker_directed_efficacy",
    "safety",
)
FACET_QUERY_TERMS = {
    "mechanism_targets": "oncology mechanism molecular target biomarker",
    "efficacy_by_tumor": (
        "oncology clinical efficacy response PFS OS tumor histology trial"
    ),
    "biomarker_prevalence": (
        "target biomarker prevalence frequency expression across cancer types"
    ),
    "biomarker_directed_efficacy": (
        "biomarker selected targeted therapy clinical response efficacy"
    ),
    "safety": "safety adverse events toxicity prescribing information",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any, *, max_chars: int = 5000) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return f"{text[: max_chars - 1].rstrip()}…"
    return text


def evidence_id(source: str, locator: str, passage: str) -> str:
    digest = hashlib.sha256(
        f"{source}\0{locator}\0{passage}".encode("utf-8", errors="replace")
    ).hexdigest()[:24]
    return f"{source}:{digest}"


@dataclass(frozen=True)
class ResearchSettings:
    """Bounded source and retry policy for one catalog build."""

    request_timeout: float = 30.0
    max_attempts: int = 5
    registry_max_attempts: int = 10
    initial_backoff: float = 1.0
    maximum_backoff: float = 60.0
    max_concurrency: int = 6
    web_results_per_query: int = 10
    max_web_results_per_drug: int = 60
    max_web_documents_per_drug: int = 24
    max_pubmed_records: int = 40
    max_registry_studies: int = 25
    max_civic_records: int = 40
    max_europe_pmc_records: int = 12
    max_regulatory_records: int = 8
    max_passage_chars: int = 5000


class GeneralWebProvider(Protocol):
    """Pluggable patient-free general-web search interface."""

    name: str

    def search(self, query: str, *, max_results: int) -> Sequence[Mapping[str, Any]]:
        """Return title/body/href mappings for one drug-only query."""


class DDGSWebProvider:
    """Built-in anonymous general-web provider."""

    name = "ddgs"

    def search(self, query: str, *, max_results: int) -> Sequence[Mapping[str, Any]]:
        with DDGS() as client:
            return list(client.text(query, max_results=max_results) or [])


class RetryableResearchError(RuntimeError):
    """A technical source failure that should be retried with backoff."""


class DrugEvidenceSource(Protocol):
    """Common async interface for one patient-free drug evidence source."""

    name: str
    source_type: str

    async def fetch(
        self,
        drug: DrugIdentity,
        *,
        facet: str,
        query: str,
        client: httpx.AsyncClient,
        settings: ResearchSettings,
    ) -> list[EvidencePassage]:
        """Return zero or more bounded passages; raise on technical failure."""


class _MainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag in {"script", "style", "nav", "footer", "header", "noscript"}:
            self.suppressed += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "noscript"}:
            self.suppressed = max(0, self.suppressed - 1)

    def handle_data(self, data: str) -> None:
        if not self.suppressed:
            self.parts.append(data)


def _html_text(document: str, *, max_chars: int) -> str:
    try:
        import trafilatura

        extracted = trafilatura.extract(
            document,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except (ImportError, RuntimeError, ValueError):
        extracted = None
    if not extracted:
        parser = _MainTextParser()
        parser.feed(document)
        parser.close()
        extracted = " ".join(parser.parts)
    return clean_text(extracted, max_chars=max_chars)


def _pdf_text(content: bytes, *, max_chars: int) -> str:
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(io.BytesIO(content))
        pages: list[str] = []
        for page_index in range(len(document)):
            page = document[page_index]
            text_page = page.get_textpage()
            pages.append(text_page.get_text_range())
            text_page.close()
            page.close()
            if sum(len(value) for value in pages) >= max_chars:
                break
        document.close()
        return clean_text("\n".join(pages), max_chars=max_chars)
    except Exception:  # noqa: BLE001 - extraction fallback is handled by caller.
        return ""


async def _fetch_document_text(
    url: str, *, client: httpx.AsyncClient, settings: ResearchSettings
) -> str:
    response = await client.get(url)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").casefold()
    if "pdf" in content_type or url.casefold().split("?", 1)[0].endswith(".pdf"):
        return _pdf_text(response.content, max_chars=settings.max_passage_chars)
    return _html_text(response.text, max_chars=settings.max_passage_chars)


class GeneralWebEvidenceSource:
    """Search and fetch substantive passages from a pluggable web provider."""

    source_type = "general_web"

    def __init__(self, provider: GeneralWebProvider | None = None) -> None:
        self.provider = provider or DDGSWebProvider()
        self.name = f"web:{self.provider.name}"

    async def fetch(
        self,
        drug: DrugIdentity,
        *,
        facet: str,
        query: str,
        client: httpx.AsyncClient,
        settings: ResearchSettings,
    ) -> list[EvidencePassage]:
        try:
            hits = await asyncio.to_thread(
                self.provider.search,
                query,
                max_results=settings.web_results_per_query,
            )
        except Exception as error:  # noqa: BLE001 - normalized retry boundary.
            raise RetryableResearchError(
                f"General-web search failed via {self.provider.name}: "
                f"{type(error).__name__}: {clean_text(error, max_chars=500)}"
            ) from error
        if not hits:
            return []
        passages: list[EvidencePassage] = []
        fetch_failures = 0
        seen_urls: set[str] = set()
        for raw in hits[: settings.web_results_per_query]:
            url = str(raw.get("href") or raw.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                text = await _fetch_document_text(url, client=client, settings=settings)
            except Exception:  # noqa: BLE001 - source operation decides terminality.
                fetch_failures += 1
                continue
            if not text:
                fetch_failures += 1
                continue
            title = clean_text(raw.get("title"), max_chars=500)
            passages.append(
                EvidencePassage(
                    evidence_id=evidence_id(self.name, url, text),
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=self.name,
                    source_type=self.source_type,
                    title=title,
                    passage=text,
                    url=url,
                    source_locator=url,
                    retrieved_at=utc_now(),
                    query=query,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
            if len(passages) >= settings.max_web_documents_per_drug:
                break
        if not passages and fetch_failures:
            raise RetryableResearchError(
                f"Search returned {len(hits)} hits but no document could be fetched."
            )
        return passages


def _xml_text(node: ET.Element | None, *, max_chars: int = 6000) -> str:
    if node is None:
        return ""
    return clean_text(" ".join(node.itertext()), max_chars=max_chars)


class PubMedDrugSource:
    name = "pubmed"
    source_type = "literature_abstract"

    async def fetch(
        self,
        drug: DrugIdentity,
        *,
        facet: str,
        query: str,
        client: httpx.AsyncClient,
        settings: ResearchSettings,
    ) -> list[EvidencePassage]:
        term = f'"{drug.preferred_name}"[Title/Abstract] AND oncology AND ({FACET_QUERY_TERMS[facet]})'
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": settings.max_pubmed_records,
            "sort": "relevance",
            "tool": "matchminer-ai",
        }
        search = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params=params,
        )
        search.raise_for_status()
        ids = search.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetched = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(str(value) for value in ids),
                "retmode": "xml",
                "tool": "matchminer-ai",
            },
        )
        fetched.raise_for_status()
        root = ET.fromstring(fetched.text)
        passages: list[EvidencePassage] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = _xml_text(article.find(".//PMID"), max_chars=40)
            title = _xml_text(article.find(".//ArticleTitle"), max_chars=500)
            abstract = clean_text(
                " ".join(
                    _xml_text(node)
                    for node in article.findall(".//Abstract/AbstractText")
                ),
                max_chars=settings.max_passage_chars,
            )
            if not pmid or not abstract:
                continue
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            passages.append(
                EvidencePassage(
                    evidence_id=f"pubmed:{pmid}",
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=self.name,
                    source_type=self.source_type,
                    title=title or f"PubMed {pmid}",
                    passage=abstract,
                    url=url,
                    source_locator=f"PMID {pmid}",
                    retrieved_at=utc_now(),
                    license="Abstract/citation metadata; NLM and publisher terms apply",
                    query=term,
                    content_sha256=hashlib.sha256(abstract.encode()).hexdigest(),
                )
            )
        return passages[: settings.max_pubmed_records]


class ClinicalTrialsDrugSource:
    name = "clinicaltrials_gov"
    source_type = "trial_registry"

    async def fetch(
        self,
        drug: DrugIdentity,
        *,
        facet: str,
        query: str,
        client: httpx.AsyncClient,
        settings: ResearchSettings,
    ) -> list[EvidencePassage]:
        del query
        response = await client.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={
                "query.intr": drug.preferred_name,
                "format": "json",
                "pageSize": min(settings.max_registry_studies, 100),
                "countTotal": "true",
            },
        )
        response.raise_for_status()
        studies = response.json().get("studies", [])
        passages: list[EvidencePassage] = []
        for study in studies[: settings.max_registry_studies]:
            protocol = study.get("protocolSection", {})
            ident = protocol.get("identificationModule", {})
            description = protocol.get("descriptionModule", {})
            outcomes = protocol.get("outcomesModule", {})
            results = study.get("resultsSection", {})
            nct_id = str(ident.get("nctId") or "").strip()
            text = clean_text(
                " ".join(
                    str(value or "")
                    for value in (
                        description.get("briefSummary"),
                        description.get("detailedDescription"),
                        json.dumps(outcomes, ensure_ascii=False),
                        json.dumps(results, ensure_ascii=False),
                    )
                ),
                max_chars=settings.max_passage_chars,
            )
            if not nct_id or not text:
                continue
            url = f"https://clinicaltrials.gov/study/{nct_id}"
            passages.append(
                EvidencePassage(
                    evidence_id=f"clinicaltrials_gov:{nct_id}:{facet}",
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=self.name,
                    source_type=self.source_type,
                    title=clean_text(
                        ident.get("briefTitle") or ident.get("officialTitle"),
                        max_chars=500,
                    ),
                    passage=text,
                    url=url,
                    source_locator=nct_id,
                    retrieved_at=utc_now(),
                    license="ClinicalTrials.gov terms apply",
                    query=drug.preferred_name,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        return passages


class EuropePMCDrugSource:
    name = "europe_pmc"
    source_type = "open_literature"

    async def fetch(
        self,
        drug: DrugIdentity,
        *,
        facet: str,
        query: str,
        client: httpx.AsyncClient,
        settings: ResearchSettings,
    ) -> list[EvidencePassage]:
        term = f'TITLE_ABS:"{drug.preferred_name}" AND (oncology OR cancer) AND ({FACET_QUERY_TERMS[facet]})'
        response = await client.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={
                "query": term,
                "format": "json",
                "pageSize": settings.max_europe_pmc_records,
                "resultType": "core",
            },
        )
        response.raise_for_status()
        results = response.json().get("resultList", {}).get("result", [])
        passages: list[EvidencePassage] = []
        for record in results[: settings.max_europe_pmc_records]:
            source = str(record.get("source") or "MED")
            record_id = str(record.get("id") or record.get("pmid") or "").strip()
            abstract = clean_text(
                record.get("abstractText"), max_chars=settings.max_passage_chars
            )
            license_name = clean_text(record.get("license"), max_chars=80).casefold()
            pmcid = str(record.get("pmcid") or "").strip()
            if pmcid and license_name in {"cc by", "cc0", "cc zero"}:
                full_text = await client.get(
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                )
                if full_text.status_code == 200:
                    with_text = clean_text(
                        " ".join(ET.fromstring(full_text.text).itertext()),
                        max_chars=settings.max_passage_chars,
                    )
                    abstract = with_text or abstract
            if not record_id or not abstract:
                continue
            url = f"https://europepmc.org/article/{source}/{record_id}"
            passages.append(
                EvidencePassage(
                    evidence_id=f"europe_pmc:{source}:{record_id}",
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=self.name,
                    source_type=self.source_type,
                    title=clean_text(record.get("title"), max_chars=500),
                    passage=abstract,
                    url=url,
                    source_locator=f"{source}:{record_id}",
                    published_at=clean_text(
                        record.get("firstPublicationDate"), max_chars=40
                    ),
                    retrieved_at=utc_now(),
                    license=license_name,
                    query=term,
                    content_sha256=hashlib.sha256(abstract.encode()).hexdigest(),
                )
            )
        return passages


def _daily_med_sections(xml_text: str, *, max_chars: int) -> str:
    root = ET.fromstring(xml_text)
    requested = (
        "INDICATIONS AND USAGE",
        "WARNINGS AND PRECAUTIONS",
        "ADVERSE REACTIONS",
        "CLINICAL STUDIES",
        "CLINICAL PHARMACOLOGY",
    )
    parts: list[str] = []
    for section in root.findall(".//{*}section"):
        title_node = section.find("./{*}title")
        title = _xml_text(title_node, max_chars=200).upper()
        if title and any(value in title for value in requested):
            parts.append(_xml_text(section, max_chars=max_chars))
    return clean_text(" ".join(parts), max_chars=max_chars)


class DailyMedDrugSource:
    name = "dailymed"
    source_type = "regulatory_label"

    async def fetch(
        self,
        drug: DrugIdentity,
        *,
        facet: str,
        query: str,
        client: httpx.AsyncClient,
        settings: ResearchSettings,
    ) -> list[EvidencePassage]:
        del query
        response = await client.get(
            "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
            params={
                "drug_name": drug.preferred_name,
                "name_type": "both",
                "pagesize": settings.max_regulatory_records,
                "page": 1,
            },
        )
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data", payload) if isinstance(payload, Mapping) else payload
        if isinstance(records, Mapping):
            records = records.get("results", [])
        passages: list[EvidencePassage] = []
        for record in list(records or [])[: settings.max_regulatory_records]:
            set_id = str(record.get("setid") or "").strip()
            if not set_id:
                continue
            label = await client.get(
                f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
            )
            label.raise_for_status()
            text = _daily_med_sections(
                label.text, max_chars=settings.max_passage_chars
            )
            if not text:
                continue
            url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={quote(set_id)}"
            passages.append(
                EvidencePassage(
                    evidence_id=f"dailymed:{set_id}",
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=self.name,
                    source_type=self.source_type,
                    title=clean_text(record.get("title"), max_chars=500),
                    passage=text,
                    url=url,
                    source_locator=f"setid={set_id}",
                    published_at=clean_text(record.get("published_date"), max_chars=40),
                    retrieved_at=utc_now(),
                    license="FDA structured product label; label-specific terms apply",
                    query=drug.preferred_name,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        return passages


class CIViCDrugSource:
    name = "civic"
    source_type = "curated_clinical_evidence"

    _QUERY = """
    query DrugEvidence($therapyName: String!, $first: Int!) {
      evidenceItems(status: ACCEPTED, therapyName: $therapyName, first: $first) {
        nodes {
          id evidenceType evidenceLevel evidenceRating evidenceDirection significance
          description molecularProfile { name } disease { name displayName }
          therapies { name } source { citation citationId }
        }
      }
    }
    """.strip()

    async def fetch(
        self,
        drug: DrugIdentity,
        *,
        facet: str,
        query: str,
        client: httpx.AsyncClient,
        settings: ResearchSettings,
    ) -> list[EvidencePassage]:
        del query
        response = await client.post(
            "https://civicdb.org/api/graphql",
            json={
                "query": self._QUERY,
                "variables": {
                    "therapyName": drug.preferred_name,
                    "first": settings.max_civic_records,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise ValueError(f"CIViC GraphQL errors: {payload['errors']}")
        nodes = payload.get("data", {}).get("evidenceItems", {}).get("nodes", [])
        passages: list[EvidencePassage] = []
        for node in nodes[: settings.max_civic_records]:
            item_id = str(node.get("id") or "").strip()
            text = clean_text(node.get("description"), max_chars=settings.max_passage_chars)
            if not item_id or not text:
                continue
            url = f"https://civicdb.org/evidence/{item_id}/summary"
            passages.append(
                EvidencePassage(
                    evidence_id=f"civic:EID{item_id}",
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=self.name,
                    source_type=self.source_type,
                    title=f"CIViC EID {item_id}",
                    passage=text,
                    url=url,
                    source_locator=f"CIViC EID {item_id}",
                    retrieved_at=utc_now(),
                    license="CC0-1.0",
                    query=drug.preferred_name,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    attributes={
                        "evidence_type": node.get("evidenceType"),
                        "evidence_level": node.get("evidenceLevel"),
                        "molecular_profile": (node.get("molecularProfile") or {}).get("name"),
                        "disease": (node.get("disease") or {}).get("displayName")
                        or (node.get("disease") or {}).get("name"),
                    },
                )
            )
        return passages


class NCIDrugSource:
    name = "nci"
    source_type = "government_drug_information"

    async def fetch(
        self,
        drug: DrugIdentity,
        *,
        facet: str,
        query: str,
        client: httpx.AsyncClient,
        settings: ResearchSettings,
    ) -> list[EvidencePassage]:
        search_query = f"{drug.preferred_name} cancer drug {FACET_QUERY_TERMS[facet]}"
        response = await client.get(
            f"https://webapis.cancer.gov/sitewidesearch/v1/Search/cgov/en/{quote(search_query, safe='')}",
            params={"size": settings.max_regulatory_records, "from": 0, "site": "all"},
        )
        response.raise_for_status()
        payload = response.json()
        records = payload.get("results") or payload.get("Results") or []
        passages: list[EvidencePassage] = []
        for record in records[: settings.max_regulatory_records]:
            url = str(record.get("url") or "").strip()
            if not url.startswith("https://www.cancer.gov/"):
                continue
            text = await _fetch_document_text(url, client=client, settings=settings)
            if not text:
                continue
            passages.append(
                EvidencePassage(
                    evidence_id=evidence_id(self.name, url, text),
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=self.name,
                    source_type=self.source_type,
                    title=clean_text(
                        record.get("name") or record.get("title"), max_chars=500
                    ),
                    passage=text,
                    url=url,
                    source_locator=url,
                    retrieved_at=utc_now(),
                    license="US Government text; page-specific terms apply",
                    query=search_query,
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        return passages


def default_sources(
    web_provider: GeneralWebProvider | None = None,
) -> tuple[DrugEvidenceSource, ...]:
    """Return authoritative adapters plus the configured general-web provider."""

    return (
        ClinicalTrialsDrugSource(),
        PubMedDrugSource(),
        EuropePMCDrugSource(),
        CIViCDrugSource(),
        DailyMedDrugSource(),
        NCIDrugSource(),
        GeneralWebEvidenceSource(web_provider),
    )


def build_facet_query(drug: DrugIdentity, facet: str, *, round_index: int) -> str:
    """Build a drug-only adaptive query; no patient input is accepted."""

    if facet not in FACETS:
        raise ValueError(f"Unknown drug research facet: {facet!r}")
    names = [drug.preferred_name, *drug.aliases[:3]]
    quoted = " OR ".join(f'"{name.replace(chr(34), " ")}"' for name in names if name)
    suffix = FACET_QUERY_TERMS[facet]
    if round_index == 1:
        suffix += " review phase 1 phase 2 phase 3"
    elif round_index >= 2:
        suffix += " tumor subtype mutation amplification overexpression antigen"
    return f"({quoted}) cancer treatment {suffix}".strip()


def _retry_after_seconds(error: Exception) -> float | None:
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    raw = error.response.headers.get("Retry-After", "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            target = parsedate_to_datetime(raw)
            now = datetime.now(target.tzinfo or timezone.utc)
            return max(0.0, (target - now).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _retryable(error: Exception) -> bool:
    if isinstance(
        error,
        (RetryableResearchError, httpx.TimeoutException, httpx.TransportError),
    ):
        return True
    return isinstance(error, httpx.HTTPStatusError) and (
        error.response.status_code in {408, 425, 429}
        or error.response.status_code >= 500
    )


async def _run_source_with_retries(
    source: DrugEvidenceSource,
    drug: DrugIdentity,
    *,
    facet: str,
    query: str,
    client: httpx.AsyncClient,
    settings: ResearchSettings,
    sleep: Callable[[float], Awaitable[None]],
) -> tuple[list[EvidencePassage], list[ResearchAttempt], str]:
    max_attempts = (
        settings.registry_max_attempts
        if source.name == "clinicaltrials_gov"
        else settings.max_attempts
    )
    attempts: list[ResearchAttempt] = []
    for attempt in range(1, max_attempts + 1):
        started = utc_now()
        try:
            items = await source.fetch(
                drug,
                facet=facet,
                query=query,
                client=client,
                settings=settings,
            )
        except Exception as error:  # noqa: BLE001 - normalized retry/audit boundary.
            finished = utc_now()
            retry_after = _retry_after_seconds(error)
            attempts.append(
                ResearchAttempt(
                    drug_id=drug.drug_id,
                    facet=facet,
                    source=source.name,
                    query=query,
                    attempt=attempt,
                    status="failed",
                    started_at=started,
                    finished_at=finished,
                    error_type=type(error).__name__,
                    error_message=clean_text(error, max_chars=1000),
                    retry_after_seconds=retry_after,
                )
            )
            if attempt >= max_attempts or not _retryable(error):
                return [], attempts, "failed"
            backoff = min(
                settings.maximum_backoff,
                settings.initial_backoff * (2 ** (attempt - 1)),
            )
            delay = max(retry_after or 0.0, backoff * random.uniform(0.75, 1.25))
            await sleep(delay)
            continue
        attempts.append(
            ResearchAttempt(
                drug_id=drug.drug_id,
                facet=facet,
                source=source.name,
                query=query,
                attempt=attempt,
                status="ok" if items else "empty",
                started_at=started,
                finished_at=utc_now(),
                result_count=len(items),
            )
        )
        return items, attempts, "ok" if items else "empty"
    raise AssertionError("retry loop terminated unexpectedly")


async def research_drug(
    drug: DrugIdentity,
    *,
    sources: Sequence[DrugEvidenceSource] | None = None,
    settings: ResearchSettings | None = None,
    client: httpx.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[list[EvidencePassage], list[ResearchAttempt], str, list[str]]:
    """Research every required facet, using alternate sources before blocking."""

    resolved_settings = settings or ResearchSettings()
    resolved_sources = tuple(sources or default_sources())
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(resolved_settings.request_timeout),
            follow_redirects=True,
            headers={"Accept": "application/json, text/html, application/xml"},
        )
    evidence: list[EvidencePassage] = []
    attempts: list[ResearchAttempt] = []
    failures: list[str] = []
    semaphore = asyncio.Semaphore(max(1, resolved_settings.max_concurrency))
    maximum_web_queries = len(FACETS) * 3
    web_results_per_operation = max(
        1,
        min(
            resolved_settings.web_results_per_query,
            math.ceil(
                resolved_settings.max_web_results_per_drug / maximum_web_queries
            ),
        ),
    )
    try:
        for round_index in range(3):
            facets_to_run = []
            for facet in FACETS:
                existing = [item for item in evidence if item.facet == facet]
                if round_index == 0 or len(existing) < 2:
                    facets_to_run.append(facet)
            if not facets_to_run:
                break

            async def run_one(
                source: DrugEvidenceSource, facet: str
            ) -> tuple[str, str, list[EvidencePassage], list[ResearchAttempt], str]:
                query = build_facet_query(drug, facet, round_index=round_index)
                operation_settings = resolved_settings
                if source.source_type == "general_web":
                    operation_settings = replace(
                        resolved_settings,
                        web_results_per_query=web_results_per_operation,
                    )
                async with semaphore:
                    items, source_attempts, status = await _run_source_with_retries(
                        source,
                        drug,
                        facet=facet,
                        query=query,
                        client=client,
                        settings=operation_settings,
                        sleep=sleep,
                    )
                return facet, source.name, items, source_attempts, status

            round_results = await asyncio.gather(
                *(run_one(source, facet) for facet in facets_to_run for source in resolved_sources)
            )
            statuses_by_facet: dict[str, list[tuple[str, str]]] = {
                facet: [] for facet in facets_to_run
            }
            for facet, source_name, items, source_attempts, status in round_results:
                attempts.extend(source_attempts)
                statuses_by_facet[facet].append((source_name, status))
                evidence.extend(items)
            for facet, statuses in statuses_by_facet.items():
                if statuses and all(status == "failed" for _, status in statuses):
                    failures.append(
                        f"{facet}: all sources failed in adaptive round {round_index + 1}"
                    )

        deduplicated: list[EvidencePassage] = []
        seen: set[tuple[str, str]] = set()
        for item in evidence:
            key = (item.source, item.content_sha256 or item.evidence_id)
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)

        unresolved: list[str] = []
        for facet in FACETS:
            facet_attempts = [item for item in attempts if item.facet == facet]
            terminal_by_source: dict[str, str] = {}
            for item in facet_attempts:
                terminal_by_source[item.source] = item.status
            if not terminal_by_source or all(
                status == "failed" for status in terminal_by_source.values()
            ):
                unresolved.append(facet)
        status = "blocked" if unresolved else "complete"
        failures.extend(f"Unresolved technical facet: {facet}" for facet in unresolved)
        return deduplicated, attempts, status, list(dict.fromkeys(failures))
    finally:
        if owns_client:
            await client.aclose()


__all__ = [
    "CIViCDrugSource",
    "ClinicalTrialsDrugSource",
    "DDGSWebProvider",
    "DailyMedDrugSource",
    "DrugEvidenceSource",
    "EuropePMCDrugSource",
    "FACETS",
    "GeneralWebEvidenceSource",
    "GeneralWebProvider",
    "NCIDrugSource",
    "PubMedDrugSource",
    "ResearchSettings",
    "build_facet_query",
    "clean_text",
    "default_sources",
    "research_drug",
]
