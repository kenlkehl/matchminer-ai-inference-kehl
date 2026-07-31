"""Public-source adapters for trial-space clinical context."""

from __future__ import annotations

import asyncio
import html
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import quote, quote_plus

import httpx

from .models import EvidenceItem, SourceNotice, TrialSpaceQuery


NCI_SEARCH_API = "https://webapis.cancer.gov/sitewidesearch/v1"
FDA_COMPANION_DIAGNOSTICS_URL = (
    "https://www.fda.gov/medical-devices/in-vitro-diagnostics/"
    "list-cleared-or-approved-companion-diagnostic-devices-in-vitro-and-imaging-tools"
)
DAILYMED_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
CIVIC_GRAPHQL_URL = "https://civicdb.org/api/graphql"
NCBI_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class SourceAdapter(Protocol):
    """Common async source-adapter interface."""

    name: str

    async def fetch(
        self,
        query: TrialSpaceQuery,
        *,
        client: httpx.AsyncClient,
        max_items: int,
        settings: Mapping[str, Any],
    ) -> tuple[list[EvidenceItem], list[SourceNotice]]:
        """Return normalized evidence and source notices."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any, *, max_chars: int = 5000) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return f"{text[: max_chars - 1].rstrip()}…"
    return text


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.suppressed = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag in {"script", "style", "noscript"}:
            self.suppressed += 1
        elif tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.suppressed:
            self.suppressed -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.suppressed:
            self.parts.append(data)


def _strip_html(value: Any, *, max_chars: int = 5000) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(str(value or ""))
        parser.close()
    except Exception:
        return _clean_text(value, max_chars=max_chars)
    return _clean_text(" ".join(parser.parts), max_chars=max_chars)


class _TableExtractor(HTMLParser):
    """Small dependency-free HTML table parser for the FDA listing."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._table_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del attrs
        if tag == "table":
            self._table_depth += 1
        elif tag == "tr" and self._table_depth:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            if self._row is not None:
                self._row.append(_clean_text(" ".join(self._cell), max_chars=3000))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(self._row):
                self.rows.append(self._row)
            self._row = None
        elif tag == "table" and self._table_depth:
            self._table_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


class _MainContentExtractor(HTMLParser):
    """Extract text from the cancer.gov main content region."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.metadata: dict[str, str] = {}
        self._main_depth = 0
        self._suppressed = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "meta":
            key = str(
                attributes.get("name") or attributes.get("property") or ""
            ).casefold()
            content = str(attributes.get("content") or "").strip()
            if key and content:
                self.metadata[key] = content
        if tag == "main":
            self._main_depth += 1
            return
        if not self._main_depth:
            return
        if tag in {"script", "style", "noscript"}:
            self._suppressed += 1
        elif tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self._main_depth:
            return
        if tag == "main":
            self._main_depth -= 1
            return
        if tag in {"script", "style", "noscript"} and self._suppressed:
            self._suppressed -= 1
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._main_depth and not self._suppressed:
            self.parts.append(data)


def _extract_nci_page(
    value: str,
    *,
    max_chars: int = 7000,
) -> tuple[str, dict[str, str]]:
    parser = _MainContentExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        return _strip_html(value, max_chars=max_chars), {}
    extracted = _clean_text(" ".join(parser.parts), max_chars=max_chars)
    return (
        extracted or _strip_html(value, max_chars=max_chars),
        parser.metadata,
    )


def _response_results(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    results = payload.get("results", payload.get("data", []))
    if isinstance(results, Mapping):
        results = [results]
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, Mapping)]


class NCIPDQSource:
    """NCI PDQ health-professional summaries via NCI sitewide search."""

    name = "nci_pdq"

    async def fetch(
        self,
        query: TrialSpaceQuery,
        *,
        client: httpx.AsyncClient,
        max_items: int,
        settings: Mapping[str, Any],
    ) -> tuple[list[EvidenceItem], list[SourceNotice]]:
        del settings
        search_query = (
            f"{query.disease_query} PDQ treatment health professional"
        ).strip()
        response = await client.get(
            (
                f"{NCI_SEARCH_API}/Search/cgov/en/"
                f"{quote(search_query, safe='')}"
            ),
            params={
                "size": max(5, min(max_items * 3, 20)),
                "from": 0,
                "site": "all",
            },
        )
        response.raise_for_status()
        records = _response_results(response.json())
        items: list[EvidenceItem] = []
        content_failures: list[str] = []
        pdq_records = [
            record
            for record in records
            if str(record.get("contentType") or "") == "pdqCancerInfoSummary"
            and "/hp/" in str(record.get("url") or "")
            and str(record.get("url") or "").startswith("https://www.cancer.gov/")
        ]
        for record in pdq_records[:max_items]:
            page_url = str(record.get("url") or "").strip()
            if not page_url:
                continue
            try:
                content_response = await client.get(page_url)
                content_response.raise_for_status()
                content, page_metadata = _extract_nci_page(
                    content_response.text,
                    max_chars=7000,
                )
            except Exception as exc:
                content_failures.append(
                    f"{page_url}: {_clean_text(exc, max_chars=240)}"
                )
                content = ""
                page_metadata = {}
            excerpt = _clean_text(content, max_chars=7000)
            if not excerpt:
                excerpt = _clean_text(record.get("description"), max_chars=3000)
            slug = page_url.rstrip("/").rsplit("/", 1)[-1]
            items.append(
                EvidenceItem(
                    evidence_id=f"nci-pdq:{slug}",
                    space_trial_id=query.space_trial_id,
                    trial_id=query.trial_id,
                    source=self.name,
                    evidence_type="evidence_summary",
                    title=_clean_text(record.get("name"), max_chars=500)
                    or _clean_text(record.get("title"), max_chars=500)
                    or f"NCI PDQ: {query.disease}",
                    excerpt=excerpt,
                    url=page_url,
                    source_locator=f"NCI PDQ page {slug}",
                    published_at=_clean_text(
                        page_metadata.get("dcterms.issued")
                    ),
                    updated_at=_clean_text(
                        page_metadata.get("dcterms.modified")
                        or page_metadata.get("dcterms.date")
                    ),
                    retrieved_at=_now(),
                    jurisdiction="United States",
                    license=(
                        "US Government text; page-specific third-party material "
                        "and reuse notices may apply"
                    ),
                    query=search_query,
                    attributes={
                        "content_kind": "NCI PDQ health professional summary",
                        "is_clinical_practice_guideline": False,
                        "search_content_type": str(
                            record.get("contentType") or ""
                        ),
                    },
                )
            )
        notice = SourceNotice(
            source=self.name,
            status=(
                "partial"
                if content_failures and items
                else "ok"
                if items
                else "empty"
            ),
            message=(
                f"Retrieved {len(items)} NCI PDQ evidence summaries."
                + (
                    " Some PDQ page fetches failed: "
                    + "; ".join(content_failures)
                    if content_failures
                    else ""
                )
                if items
                else "No NCI PDQ health-professional page matched the disease query."
            ),
        )
        return items, [notice]


def _significant_tokens(text: str) -> set[str]:
    ignored = {
        "and",
        "the",
        "with",
        "cancer",
        "carcinoma",
        "tumor",
        "tumors",
        "allowed",
        "any",
        "other",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2 and token not in ignored
    }


def _row_matches(query: TrialSpaceQuery, row_text: str) -> bool:
    lowered = row_text.casefold()
    disease_phrase = query.disease.casefold()
    if disease_phrase and disease_phrase in lowered:
        return True
    disease_tokens = _significant_tokens(query.disease_query)
    overlap = disease_tokens.intersection(_significant_tokens(row_text))
    return bool(overlap) and len(overlap) >= min(2, len(disease_tokens))


def _extract_drug_names(cell: str) -> list[str]:
    names: list[str] = []
    for value in re.findall(r"\(([^()]{2,80})\)", cell):
        if re.search(r"\b(?:NDA|BLA|PMA|HDE)\b", value, re.IGNORECASE):
            continue
        names.extend(re.split(r",|/|\band\b|\bplus\b", value))
    if not names:
        names.append(re.split(r"\b(?:NDA|BLA)\b", cell, maxsplit=1)[0])
    cleaned: list[str] = []
    for name in names:
        value = re.sub(r"[^A-Za-z0-9 +\-]", " ", name)
        value = _clean_text(value, max_chars=100)
        if value and len(value.split()) <= 6 and value.casefold() not in {
            item.casefold() for item in cleaned
        }:
            cleaned.append(value)
    return cleaned[:4]


def _extract_spl_sections(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    requested = (
        "INDICATIONS AND USAGE",
        "DOSAGE AND ADMINISTRATION",
        "CONTRAINDICATIONS",
        "WARNINGS AND PRECAUTIONS",
        "ADVERSE REACTIONS",
        "CLINICAL STUDIES",
    )
    sections: list[str] = []
    for section in root.findall(".//{*}section"):
        title_node = section.find("./{*}title")
        title = _clean_text(
            " ".join(title_node.itertext()) if title_node is not None else "",
            max_chars=200,
        )
        if not title or not any(name in title.upper() for name in requested):
            continue
        text = _clean_text(" ".join(section.itertext()), max_chars=2500)
        if text:
            sections.append(text)
        if len(sections) >= 4:
            break
    return _clean_text(" ".join(sections), max_chars=7000)


class FDAClinicalSource:
    """FDA companion-diagnostic rows plus corresponding DailyMed labels."""

    name = "fda"

    async def fetch(
        self,
        query: TrialSpaceQuery,
        *,
        client: httpx.AsyncClient,
        max_items: int,
        settings: Mapping[str, Any],
    ) -> tuple[list[EvidenceItem], list[SourceNotice]]:
        del settings
        response = await client.get(FDA_COMPANION_DIAGNOSTICS_URL)
        response.raise_for_status()
        parser = _TableExtractor()
        parser.feed(response.text)
        parser.close()
        matched_rows = [
            row
            for row in parser.rows
            if len(row) >= 5
            and "diagnostic name" not in row[0].casefold()
            and _row_matches(query, " | ".join(row))
        ][:max_items]
        items: list[EvidenceItem] = []
        drug_names: list[str] = []
        for index, row in enumerate(matched_rows, start=1):
            row_text = " | ".join(row)
            locator = row[-1] or str(index)
            items.append(
                EvidenceItem(
                    evidence_id=f"fda-cdx:{query.space_trial_id}:{locator}",
                    space_trial_id=query.space_trial_id,
                    trial_id=query.trial_id,
                    source="fda_companion_diagnostics",
                    evidence_type="diagnostic_regulatory",
                    title=f"FDA companion diagnostic: {row[0]}",
                    excerpt=_clean_text(row_text, max_chars=5000),
                    url=FDA_COMPANION_DIAGNOSTICS_URL,
                    source_locator=locator,
                    retrieved_at=_now(),
                    jurisdiction="United States",
                    license=(
                        "US Government work; page-specific third-party material "
                        "and reuse notices may apply"
                    ),
                    query=" ".join(
                        part
                        for part in (query.disease_query, query.biomarker_query)
                        if part
                    ),
                    attributes={
                        "diagnostic_name": row[0],
                        "indication_and_sample_type": row[1] if len(row) > 1 else "",
                        "therapeutic_product": row[2] if len(row) > 2 else "",
                        "biomarker": row[3] if len(row) > 3 else "",
                        "biomarker_details": row[4] if len(row) > 4 else "",
                        "approval_or_clearance_date": (
                            re.findall(r"\d{2}/\d{2}/\d{4}", row[-1])
                            or [""]
                        )[-1],
                    },
                )
            )
            if len(row) > 2:
                for drug_name in _extract_drug_names(row[2]):
                    if drug_name.casefold() not in {
                        item.casefold() for item in drug_names
                    }:
                        drug_names.append(drug_name)

        label_failures: list[str] = []
        for drug_name in drug_names[: min(3, max_items)]:
            try:
                label_response = await client.get(
                    f"{DAILYMED_API}/spls.json",
                    params={
                        "drug_name": drug_name,
                        "name_type": "both",
                        "pagesize": 1,
                        "page": 1,
                    },
                )
                label_response.raise_for_status()
                labels = _response_results(label_response.json())
                if not labels:
                    continue
                label = labels[0]
                set_id = str(label.get("setid") or "").strip()
                if not set_id:
                    continue
                xml_response = await client.get(f"{DAILYMED_API}/spls/{set_id}.xml")
                xml_response.raise_for_status()
                excerpt = _extract_spl_sections(xml_response.text)
                if not excerpt:
                    continue
                items.append(
                    EvidenceItem(
                        evidence_id=f"dailymed:{set_id}",
                        space_trial_id=query.space_trial_id,
                        trial_id=query.trial_id,
                        source="dailymed",
                        evidence_type="drug_label",
                        title=_clean_text(label.get("title"), max_chars=500)
                        or f"DailyMed label for {drug_name}",
                        excerpt=excerpt,
                        url=(
                            "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?"
                            f"setid={quote_plus(set_id)}"
                        ),
                        source_locator=f"setid={set_id}",
                        published_at=_clean_text(label.get("published_date")),
                        retrieved_at=_now(),
                        jurisdiction="United States",
                        license=(
                            "FDA structured product label; included material may "
                            "have product-specific reuse terms"
                        ),
                        query=drug_name,
                        attributes={
                            "set_id": set_id,
                            "spl_version": str(label.get("spl_version") or ""),
                        },
                    )
                )
            except Exception as exc:
                label_failures.append(
                    f"{drug_name}: {_clean_text(exc, max_chars=240)}"
                )
        notices = [
            SourceNotice(
                source=self.name,
                status="ok" if items else "empty",
                message=(
                    f"Retrieved {len(matched_rows)} FDA companion-diagnostic "
                    f"rows and {sum(item.source == 'dailymed' for item in items)} "
                    "DailyMed labels."
                    if items
                    else "No FDA companion-diagnostic row matched the disease context."
                ),
            )
        ]
        if label_failures:
            notices.append(
                SourceNotice(
                    source="dailymed",
                    status="partial",
                    message="Some DailyMed label lookups failed: "
                    + "; ".join(label_failures),
                )
            )
        return items[: max_items * 2], notices


_CIVIC_QUERY = """
query TrialSpaceEvidence($diseaseName: String!, $first: Int!) {
  evidenceItems(status: ACCEPTED, diseaseName: $diseaseName, first: $first) {
    nodes {
      id
      status
      molecularProfile { id name link }
      evidenceType
      evidenceLevel
      evidenceRating
      evidenceDirection
      significance
      description
      disease { id doid name displayName }
      therapies { id name }
      source { id citationId citation journal sourceType }
    }
  }
}
""".strip()
_CIVIC_DISEASE_QUERY = """
query ResolveTrialSpaceDisease($queryTerm: String!) {
  diseaseTypeahead(queryTerm: $queryTerm) {
    id
    name
    displayName
    doid
  }
}
""".strip()


class CIViCSource:
    """Accepted CIViC evidence items queried by trial-space disease."""

    name = "civic"

    async def fetch(
        self,
        query: TrialSpaceQuery,
        *,
        client: httpx.AsyncClient,
        max_items: int,
        settings: Mapping[str, Any],
    ) -> tuple[list[EvidenceItem], list[SourceNotice]]:
        del settings
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = os.environ.get("CIVIC_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        disease_response = await client.post(
            CIVIC_GRAPHQL_URL,
            headers=headers,
            json={
                "query": _CIVIC_DISEASE_QUERY,
                "variables": {"queryTerm": query.disease},
            },
        )
        disease_response.raise_for_status()
        disease_payload = disease_response.json()
        disease_data = (
            disease_payload.get("data", {})
            if isinstance(disease_payload, Mapping)
            else {}
        )
        candidates = (
            disease_data.get("diseaseTypeahead", [])
            if isinstance(disease_data, Mapping)
            else []
        )
        canonical_disease = query.disease
        if isinstance(candidates, list):
            candidate_names = [
                str(candidate.get("name") or "").strip()
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and str(candidate.get("name") or "").strip()
            ]
            if candidate_names:
                query_tokens = _significant_tokens(
                    f"{query.disease} {query.histology}"
                )
                canonical_disease = max(
                    candidate_names,
                    key=lambda name: (
                        len(query_tokens.intersection(_significant_tokens(name))),
                        -len(_significant_tokens(name)),
                    ),
                )
        await asyncio.sleep(0.34)
        response = await client.post(
            CIVIC_GRAPHQL_URL,
            headers=headers,
            json={
                "query": _CIVIC_QUERY,
                "variables": {
                    "diseaseName": canonical_disease,
                    "first": max(1, min(max_items * 3, 50)),
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError("CIViC returned a non-object GraphQL response.")
        errors = payload.get("errors")
        if errors:
            raise ValueError(f"CIViC GraphQL errors: {_clean_text(errors)}")
        data = payload.get("data") or {}
        connection = data.get("evidenceItems") if isinstance(data, Mapping) else {}
        nodes = connection.get("nodes") if isinstance(connection, Mapping) else []
        if not isinstance(nodes, list):
            nodes = []

        biomarker_tokens = _significant_tokens(query.biomarker_query)
        selected: list[Mapping[str, Any]] = []
        fallback: list[Mapping[str, Any]] = []
        for node in nodes:
            if not isinstance(node, Mapping) or node.get("status") != "ACCEPTED":
                continue
            fallback.append(node)
            profile = node.get("molecularProfile") or {}
            profile_name = (
                str(profile.get("name") or "")
                if isinstance(profile, Mapping)
                else ""
            )
            if not biomarker_tokens or biomarker_tokens.intersection(
                _significant_tokens(profile_name)
            ):
                selected.append(node)
        if not selected:
            selected = fallback

        items: list[EvidenceItem] = []
        for node in selected[:max_items]:
            evidence_id = str(node.get("id") or "").strip()
            profile = node.get("molecularProfile") or {}
            disease = node.get("disease") or {}
            source = node.get("source") or {}
            therapies = node.get("therapies") or []
            profile_name = (
                _clean_text(profile.get("name"), max_chars=300)
                if isinstance(profile, Mapping)
                else ""
            )
            therapy_names = [
                _clean_text(therapy.get("name"), max_chars=120)
                for therapy in therapies
                if isinstance(therapy, Mapping)
            ]
            items.append(
                EvidenceItem(
                    evidence_id=f"civic:EID{evidence_id}",
                    space_trial_id=query.space_trial_id,
                    trial_id=query.trial_id,
                    source=self.name,
                    evidence_type=str(node.get("evidenceType") or "").casefold(),
                    title=" — ".join(
                        part
                        for part in (
                            f"CIViC EID {evidence_id}",
                            profile_name,
                            ", ".join(therapy_names),
                        )
                        if part
                    ),
                    excerpt=_clean_text(node.get("description"), max_chars=5000),
                    url=f"https://civicdb.org/evidence/{evidence_id}/summary",
                    source_locator=f"CIViC EID {evidence_id}",
                    retrieved_at=_now(),
                    jurisdiction="Global evidence; disease context as curated",
                    license="CC0-1.0",
                    query=" ".join(
                        part
                        for part in (query.disease_query, query.biomarker_query)
                        if part
                    ),
                    attributes={
                        "molecular_profile": profile_name,
                        "disease": (
                            _clean_text(
                                disease.get("displayName") or disease.get("name"),
                                max_chars=200,
                            )
                            if isinstance(disease, Mapping)
                            else ""
                        ),
                        "therapies": therapy_names,
                        "evidence_level": str(node.get("evidenceLevel") or ""),
                        "evidence_rating": node.get("evidenceRating"),
                        "evidence_direction": str(
                            node.get("evidenceDirection") or ""
                        ),
                        "significance": str(node.get("significance") or ""),
                        "citation": (
                            _clean_text(source.get("citation"), max_chars=300)
                            if isinstance(source, Mapping)
                            else ""
                        ),
                        "citation_id": (
                            str(source.get("citationId") or "")
                            if isinstance(source, Mapping)
                            else ""
                        ),
                        "civic_disease_query": canonical_disease,
                    },
                )
            )
        notice = SourceNotice(
            source=self.name,
            status="ok" if items else "empty",
            message=(
                f"Retrieved {len(items)} accepted CIViC evidence items."
                if items
                else "No accepted CIViC evidence item matched the disease query."
            ),
        )
        return items, [notice]


def _pubmed_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _clean_text(" ".join(element.itertext()), max_chars=6000)


def _pubmed_date(article: ET.Element) -> str:
    date = article.find(".//PubDate")
    if date is None:
        return ""
    year = _pubmed_text(date.find("Year"))
    month = _pubmed_text(date.find("Month"))
    day = _pubmed_text(date.find("Day"))
    medline = _pubmed_text(date.find("MedlineDate"))
    return "-".join(part for part in (year, month, day) if part) or medline


class PubMedSource:
    """PubMed citations and abstracts through NCBI E-utilities."""

    name = "pubmed"

    async def fetch(
        self,
        query: TrialSpaceQuery,
        *,
        client: httpx.AsyncClient,
        max_items: int,
        settings: Mapping[str, Any],
    ) -> tuple[list[EvidenceItem], list[SourceNotice]]:
        configured_email = str(settings.get("ncbi_email") or "").strip()
        email = os.environ.get("NCBI_EMAIL", configured_email).strip()
        search_query = (
            f'("{query.disease_query}"[Title/Abstract]) AND '
            '(guideline[Publication Type] OR practice guideline[Title/Abstract] '
            'OR standard of care[Title/Abstract] OR diagnosis[Title/Abstract] '
            'OR treatment[Title/Abstract])'
        )
        common_params: dict[str, Any] = {
            "db": "pubmed",
            "tool": "matchminer-ai",
        }
        if email:
            common_params["email"] = email
        api_key = os.environ.get("NCBI_API_KEY", "").strip()
        if api_key:
            common_params["api_key"] = api_key
        search_response = await client.get(
            f"{NCBI_EUTILS_URL}/esearch.fcgi",
            params={
                **common_params,
                "term": search_query,
                "retmode": "json",
                "retmax": max(1, max_items),
                "sort": "pub date",
            },
        )
        search_response.raise_for_status()
        payload = search_response.json()
        search_result = (
            payload.get("esearchresult", {}) if isinstance(payload, Mapping) else {}
        )
        ids = (
            search_result.get("idlist", [])
            if isinstance(search_result, Mapping)
            else []
        )
        ids = [str(item) for item in ids if str(item).strip()][:max_items]
        if not ids:
            if not api_key:
                await asyncio.sleep(0.34)
            return [], [
                SourceNotice(
                    source=self.name,
                    status="empty",
                    message="PubMed returned no matching citations.",
                )
            ]
        # NCBI asks unkeyed clients to remain at or below three requests/second.
        if not api_key:
            await asyncio.sleep(0.34)
        fetch_response = await client.get(
            f"{NCBI_EUTILS_URL}/efetch.fcgi",
            params={
                **common_params,
                "id": ",".join(ids),
                "retmode": "xml",
            },
        )
        fetch_response.raise_for_status()
        root = ET.fromstring(fetch_response.text)
        items: list[EvidenceItem] = []
        for article in root.findall(".//PubmedArticle"):
            pmid = _pubmed_text(article.find(".//PMID"))
            if not pmid:
                continue
            title = _pubmed_text(article.find(".//ArticleTitle"))
            abstract_parts = [
                _pubmed_text(node)
                for node in article.findall(".//Abstract/AbstractText")
            ]
            excerpt = _clean_text(" ".join(abstract_parts), max_chars=7000)
            if not excerpt:
                excerpt = "No abstract was supplied by PubMed."
            publication_types = [
                _pubmed_text(node)
                for node in article.findall(".//PublicationType")
                if _pubmed_text(node)
            ]
            items.append(
                EvidenceItem(
                    evidence_id=f"pubmed:{pmid}",
                    space_trial_id=query.space_trial_id,
                    trial_id=query.trial_id,
                    source=self.name,
                    evidence_type="literature_abstract",
                    title=title or f"PubMed {pmid}",
                    excerpt=excerpt,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    source_locator=f"PMID {pmid}",
                    published_at=_pubmed_date(article),
                    retrieved_at=_now(),
                    jurisdiction="Publication-specific",
                    license=(
                        "Citation metadata and abstract only; publisher and NLM "
                        "reuse terms apply"
                    ),
                    query=search_query,
                    attributes={
                        "publication_types": publication_types,
                        "journal": _pubmed_text(article.find(".//Journal/Title")),
                    },
                )
            )
        if not api_key:
            await asyncio.sleep(0.34)
        return items, [
            SourceNotice(
                source=self.name,
                status="ok" if items else "empty",
                message=(
                    f"Retrieved {len(items)} PubMed citation abstracts."
                    if items
                    else "PubMed records contained no parseable articles."
                ),
            )
        ]


SOURCE_ADAPTERS: dict[str, SourceAdapter] = {
    "nci_pdq": NCIPDQSource(),
    "fda": FDAClinicalSource(),
    "civic": CIViCSource(),
    "pubmed": PubMedSource(),
}


def resolve_sources(names: Sequence[str]) -> list[SourceAdapter]:
    """Validate source names and return adapters in requested order."""

    unknown = sorted(set(names).difference(SOURCE_ADAPTERS))
    if unknown:
        raise ValueError(
            "Unsupported contextualization source(s): "
            + ", ".join(unknown)
            + ". Supported sources: "
            + ", ".join(sorted(SOURCE_ADAPTERS))
            + "."
        )
    return [SOURCE_ADAPTERS[name] for name in dict.fromkeys(names)]


__all__ = [
    "CIVIC_GRAPHQL_URL",
    "DAILYMED_API",
    "FDA_COMPANION_DIAGNOSTICS_URL",
    "NCI_SEARCH_API",
    "NCBI_EUTILS_URL",
    "CIViCSource",
    "FDAClinicalSource",
    "NCIPDQSource",
    "PubMedSource",
    "SOURCE_ADAPTERS",
    "SourceAdapter",
    "resolve_sources",
]
