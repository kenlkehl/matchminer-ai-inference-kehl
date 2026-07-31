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
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"

_DIAGNOSTIC_TERMS = {
    "assay",
    "assessment",
    "baseline",
    "biomarker",
    "biopsy",
    "cytology",
    "diagnosis",
    "diagnostic",
    "evaluation",
    "genomic",
    "histology",
    "imaging",
    "molecular",
    "mri",
    "pathology",
    "pet",
    "pretreatment",
    "specimen",
    "stage",
    "staging",
    "testing",
    "workup",
}
_THERAPEUTIC_TERMS = {
    "management",
    "radiation",
    "radiotherapy",
    "standard of care",
    "surgery",
    "systemic therapy",
    "therapeutic",
    "therapy",
    "treatment",
}


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
    """Extract structured blocks from the cancer.gov main content region."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, str]] = []
        self.metadata: dict[str, str] = {}
        self._main_depth = 0
        self._suppressed = 0
        self._capture_tag = ""
        self._capture_depth = 0
        self._capture_parts: list[str] = []

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
        elif not self._suppressed and tag in {
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
            "tr",
        }:
            if not self._capture_tag:
                self._capture_tag = tag
                self._capture_depth = 1
                self._capture_parts = []
            else:
                self._capture_depth += 1
        elif self._capture_tag and tag == "br":
            self._capture_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self._main_depth:
            return
        if tag == "main":
            self._main_depth -= 1
            return
        if tag in {"script", "style", "noscript"} and self._suppressed:
            self._suppressed -= 1
        elif self._capture_tag and tag in {
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
            "tr",
        }:
            self._capture_depth -= 1
            if self._capture_depth == 0:
                text = _clean_text(" ".join(self._capture_parts), max_chars=60000)
                if text:
                    self.blocks.append((self._capture_tag, text))
                self._capture_tag = ""
                self._capture_parts = []

    def handle_data(self, data: str) -> None:
        if self._main_depth and not self._suppressed and self._capture_tag:
            self._capture_parts.append(data)


def _section_category(heading: str, text: str) -> str:
    heading_value = heading.casefold()
    text_value = text.casefold()
    heading_diagnostic_score = sum(
        term in heading_value for term in _DIAGNOSTIC_TERMS
    )
    heading_therapeutic_score = sum(
        term in heading_value for term in _THERAPEUTIC_TERMS
    )
    diagnostic_score = sum(term in text_value for term in _DIAGNOSTIC_TERMS)
    therapeutic_score = sum(term in text_value for term in _THERAPEUTIC_TERMS)
    if heading_diagnostic_score > heading_therapeutic_score:
        return "diagnostic"
    if (
        heading_therapeutic_score
        and heading_therapeutic_score >= heading_diagnostic_score
    ):
        return "therapeutic"
    if diagnostic_score >= max(2, therapeutic_score):
        return "diagnostic"
    if therapeutic_score:
        return "therapeutic"
    return "general"


def _extract_nci_page(
    value: str,
    *,
    max_chars: int = 60000,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    parser = _MainContentExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        fallback = _strip_html(value, max_chars=max_chars)
        return ([{"heading": "Page", "text": fallback, "category": "general"}], {})

    heading_levels: dict[int, str] = {}
    grouped: list[dict[str, str]] = []
    for tag, text in parser.blocks:
        if tag.startswith("h"):
            level = int(tag[1])
            heading_levels[level] = text
            for deeper in range(level + 1, 5):
                heading_levels.pop(deeper, None)
            continue
        heading = " > ".join(
            heading_levels[level] for level in sorted(heading_levels)
        ) or "Main content"
        if grouped and grouped[-1]["heading"] == heading:
            if text not in grouped[-1]["text"]:
                grouped[-1]["text"] += f"\n{text}"
        else:
            grouped.append({"heading": heading, "text": text})

    if not grouped:
        fallback = _strip_html(value, max_chars=max_chars)
        grouped = [{"heading": "Page", "text": fallback}]
    for section in grouped:
        section["text"] = _clean_text(section["text"], max_chars=max_chars)
        section["category"] = _section_category(
            section["heading"], section["text"]
        )
    return grouped, parser.metadata


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
        search_queries = [
            (
                f"{query.disease_query} PDQ diagnosis staging "
                "health professional"
            ).strip(),
            (
                f"{query.disease_query} PDQ treatment health professional"
            ).strip(),
        ]
        records_by_url: dict[str, Mapping[str, Any]] = {}
        for search_query in search_queries:
            response = await client.get(
                (
                    f"{NCI_SEARCH_API}/Search/cgov/en/"
                    f"{quote(search_query, safe='')}"
                ),
                params={
                    "size": max(8, min(max_items * 4, 30)),
                    "from": 0,
                    "site": "all",
                },
            )
            response.raise_for_status()
            for record in _response_results(response.json()):
                page_url = str(record.get("url") or "").strip()
                if page_url:
                    records_by_url.setdefault(page_url, record)
        items: list[EvidenceItem] = []
        content_failures: list[str] = []
        pdq_records: list[tuple[int, Mapping[str, Any]]] = []
        disease_tokens = _significant_tokens(query.disease)
        required_overlap = min(2, len(disease_tokens))
        for record in records_by_url.values():
            page_url = str(record.get("url") or "")
            if (
                str(record.get("contentType") or "")
                != "pdqCancerInfoSummary"
                or "/hp/" not in page_url
                or not page_url.startswith("https://www.cancer.gov/")
            ):
                continue
            searchable = " ".join(
                str(record.get(field) or "")
                for field in ("name", "title", "description", "url")
            )
            overlap = len(
                disease_tokens.intersection(_significant_tokens(searchable))
            )
            exact = query.disease.casefold() in searchable.casefold()
            if disease_tokens and not exact and overlap < required_overlap:
                continue
            pdq_records.append((overlap + (10 if exact else 0), record))
        pdq_records.sort(key=lambda value: value[0], reverse=True)

        page_limit = max(1, min(len(pdq_records), max_items))
        for _, record in pdq_records[:page_limit]:
            page_url = str(record.get("url") or "").strip()
            if not page_url:
                continue
            try:
                content_response = await client.get(page_url)
                content_response.raise_for_status()
                sections, page_metadata = _extract_nci_page(
                    content_response.text,
                    max_chars=60000,
                )
            except Exception as exc:
                content_failures.append(
                    f"{page_url}: {_clean_text(exc, max_chars=240)}"
                )
                sections = []
                page_metadata = {}
            slug = page_url.rstrip("/").rsplit("/", 1)[-1]
            page_title = (
                _clean_text(record.get("name"), max_chars=500)
                or _clean_text(record.get("title"), max_chars=500)
                or f"NCI PDQ: {query.disease}"
            )
            ranked_sections = sorted(
                sections,
                key=lambda section: (
                    {"diagnostic": 2, "therapeutic": 1}.get(
                        section["category"], 0
                    ),
                    len(section["text"]),
                ),
                reverse=True,
            )
            if not ranked_sections:
                ranked_sections = [
                    {
                        "heading": "Search summary",
                        "text": _clean_text(
                            record.get("description"), max_chars=10000
                        ),
                        "category": "general",
                    }
                ]
            for section_index, section in enumerate(ranked_sections, start=1):
                if len(items) >= max_items:
                    break
                excerpt = _clean_text(section["text"], max_chars=60000)
                if not excerpt:
                    continue
                category = section["category"]
                evidence_type = {
                    "diagnostic": "diagnostic_evidence_summary",
                    "therapeutic": "treatment_evidence_summary",
                }.get(category, "evidence_summary")
                heading = _clean_text(section["heading"], max_chars=500)
                items.append(
                    EvidenceItem(
                        evidence_id=f"nci-pdq:{slug}:section-{section_index}",
                        space_trial_id=query.space_trial_id,
                        trial_id=query.trial_id,
                        source=self.name,
                        evidence_type=evidence_type,
                        title=f"{page_title} — {heading}",
                        excerpt=excerpt,
                        url=page_url,
                        source_locator=f"NCI PDQ page {slug}, section {heading}",
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
                            "US Government text; page-specific third-party "
                            "material and reuse notices may apply"
                        ),
                        query=" | ".join(search_queries),
                        attributes={
                            "content_kind": (
                                "NCI PDQ health professional summary section"
                            ),
                            "evidence_category": category,
                            "section_heading": heading,
                            "is_clinical_practice_guideline": False,
                            "search_content_type": str(
                                record.get("contentType") or ""
                            ),
                        },
                    )
                )
            if len(items) >= max_items:
                break
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
                f"Retrieved {len(items)} disease-relevant NCI PDQ sections."
                + (
                    " Some PDQ page fetches failed: "
                    + "; ".join(content_failures)
                    if content_failures
                    else ""
                )
                if items
                else (
                    "No disease-relevant NCI PDQ health-professional page "
                    "matched the disease query."
                )
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


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _extract_jats_sections(xml_text: str) -> list[dict[str, str]]:
    """Extract heading-aware prose blocks from a Europe PMC JATS article."""

    root = ET.fromstring(xml_text)
    body = root.find(".//body")
    if body is None:
        return []
    sections: list[dict[str, str]] = []

    def visit(section: ET.Element, parents: list[str]) -> None:
        title_node = section.find("./title")
        title = _clean_text(
            " ".join(title_node.itertext()) if title_node is not None else "",
            max_chars=1000,
        )
        path = [*parents, title] if title else parents
        paragraphs: list[str] = []
        for child in section:
            child_name = _local_xml_name(child.tag)
            if child_name == "sec":
                continue
            if child_name in {
                "p",
                "list",
                "disp-quote",
                "boxed-text",
                "table-wrap",
            }:
                value = _clean_text(" ".join(child.itertext()), max_chars=60000)
                if value and value not in paragraphs:
                    paragraphs.append(value)
        text = _clean_text("\n".join(paragraphs), max_chars=60000)
        heading = " > ".join(path) or "Article body"
        if text:
            sections.append(
                {
                    "heading": heading,
                    "text": text,
                    "category": _section_category(heading, text),
                }
            )
        for child in section:
            if _local_xml_name(child.tag) == "sec":
                visit(child, path)

    for child in body:
        if _local_xml_name(child.tag) == "sec":
            visit(child, [])
    return sections


def _low_value_guideline_section(heading: str) -> bool:
    return bool(
        re.search(
            r"\b(?:materials? and methods?|methodology|statistical analysis|"
            r"references|acknowledgements?|funding|conflicts? of interest|"
            r"author contributions?)\b",
            heading,
            re.IGNORECASE,
        )
    )


def _permissive_europe_pmc_license(value: Any) -> str:
    normalized = re.sub(r"[\s_-]+", " ", str(value or "").casefold()).strip()
    return normalized if normalized in {"cc by", "cc0", "cc zero"} else ""


class EuropePMCOpenGuidelinesSource:
    """Permissively licensed guideline/consensus full text from Europe PMC."""

    name = "europe_pmc_open_guidelines"

    async def fetch(
        self,
        query: TrialSpaceQuery,
        *,
        client: httpx.AsyncClient,
        max_items: int,
        settings: Mapping[str, Any],
    ) -> tuple[list[EvidenceItem], list[SourceNotice]]:
        configured_languages = settings.get("europe_pmc_languages", ["eng", "en"])
        allowed_languages = {
            str(value).casefold()
            for value in (
                configured_languages
                if isinstance(configured_languages, (list, tuple, set))
                else [configured_languages]
            )
            if str(value).strip()
        }
        guideline_filter = (
            "(TITLE:guideline OR TITLE:consensus OR TITLE:recommendation* OR "
            'PUB_TYPE:"Practice Guideline" OR PUB_TYPE:"Consensus Statement") '
            "AND (diagnos* OR stag* OR imaging OR biopsy OR pathology OR "
            "biomarker OR molecular OR treatment) "
            'AND (LICENSE:"cc by" OR LICENSE:"cc0") AND OPEN_ACCESS:Y'
        )
        disease_phrase = query.disease.replace('"', " ").strip()
        search_queries = [
            f'(\"{disease_phrase}\") AND '
            f"{guideline_filter}"
        ]
        required_biomarker_phrase = query.biomarkers_required.replace(
            '"', " "
        ).strip()
        if required_biomarker_phrase:
            search_queries.insert(
                0,
                f'(\"{disease_phrase}\") AND '
                f'(\"{required_biomarker_phrase}\") AND {guideline_filter}',
            )
        records_by_pmcid: dict[str, Mapping[str, Any]] = {}
        for search_query in search_queries:
            response = await client.get(
                f"{EUROPE_PMC_API}/search",
                params={
                    "query": search_query,
                    "format": "json",
                    "resultType": "core",
                    "pageSize": max(10, min(max_items * 5, 50)),
                },
            )
            response.raise_for_status()
            payload = response.json()
            result_list = (
                payload.get("resultList", {})
                if isinstance(payload, Mapping)
                else {}
            )
            records = (
                result_list.get("result", [])
                if isinstance(result_list, Mapping)
                else []
            )
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, Mapping):
                    continue
                pmcid = str(record.get("pmcid") or "").strip()
                if pmcid:
                    records_by_pmcid.setdefault(pmcid, record)

        disease_tokens = _significant_tokens(query.disease)
        generic_biomarker_tokens = {
            "alteration",
            "expression",
            "mutation",
            "negative",
            "positive",
            "rearrangement",
            "status",
        }
        biomarker_tokens = _significant_tokens(query.biomarkers_required).difference(
            generic_biomarker_tokens
        )
        required_overlap = min(2, len(disease_tokens))
        candidates: list[tuple[int, Mapping[str, Any], str]] = []
        rejected_license_count = 0
        for record in records_by_pmcid.values():
            license_name = _permissive_europe_pmc_license(record.get("license"))
            if not license_name:
                rejected_license_count += 1
                continue
            pmcid = str(record.get("pmcid") or "").strip()
            if not pmcid or str(record.get("inEPMC") or "").upper() != "Y":
                continue
            language = str(record.get("language") or "").casefold().strip()
            if language and allowed_languages and language not in allowed_languages:
                continue
            title = _clean_text(record.get("title"), max_chars=1000)
            publication_types_raw = record.get("pubTypeList") or {}
            publication_types = (
                publication_types_raw.get("pubType", [])
                if isinstance(publication_types_raw, Mapping)
                else []
            )
            publication_types = [str(item) for item in publication_types]
            guideline_text = f"{title} {' '.join(publication_types)}".casefold()
            if not any(
                signal in guideline_text
                for signal in ("guideline", "consensus", "recommendation")
            ):
                continue
            searchable = " ".join(
                [
                    title,
                    _strip_html(record.get("abstractText"), max_chars=12000),
                    str(record.get("keywordList") or ""),
                ]
            )
            overlap = len(
                disease_tokens.intersection(_significant_tokens(searchable))
            )
            exact = query.disease.casefold() in searchable.casefold()
            if disease_tokens and not exact and overlap < required_overlap:
                continue
            score = overlap + (10 if exact else 0)
            score += 5 if "practice guideline" in guideline_text else 0
            score += 4 if "consensus" in guideline_text else 0
            score += sum(term in searchable.casefold() for term in _DIAGNOSTIC_TERMS)
            biomarker_overlap = biomarker_tokens.intersection(
                _significant_tokens(searchable)
            )
            score += 20 * len(biomarker_overlap)
            narrow_biomarker_title = bool(
                re.search(
                    r"(?:\bwith\b.{0,100}\b(?:mutation|alteration|fusion|exon)"
                    r"|\b(?:mutant|positive)[ -])",
                    title,
                    re.IGNORECASE,
                )
            )
            if biomarker_tokens and narrow_biomarker_title and not biomarker_overlap:
                score -= 30
            candidates.append((score, record, license_name))
        candidates.sort(key=lambda value: value[0], reverse=True)

        passage_candidates: list[
            tuple[int, Mapping[str, Any], str, dict[str, str]]
        ] = []
        fetch_failures: list[str] = []
        for article_score, record, license_name in candidates[: max_items * 2]:
            pmcid = str(record.get("pmcid") or "").strip()
            try:
                full_text_response = await client.get(
                    f"{EUROPE_PMC_API}/{pmcid}/fullTextXML"
                )
                full_text_response.raise_for_status()
                sections = _extract_jats_sections(full_text_response.text)
            except Exception as exc:
                fetch_failures.append(
                    f"{pmcid}: {_clean_text(exc, max_chars=240)}"
                )
                continue
            relevant = [
                section
                for section in sections
                if section["category"] != "general"
                and not _low_value_guideline_section(section["heading"])
            ]
            relevant.sort(
                key=lambda section: (
                    section["category"] == "diagnostic",
                    sum(
                        term in (
                            f"{section['heading']} {section['text']}".casefold()
                        )
                        for term in _DIAGNOSTIC_TERMS
                    ),
                    "recommend" in section["text"].casefold(),
                    len(section["text"]),
                ),
                reverse=True,
            )
            for section in relevant[:2]:
                passage_candidates.append(
                    (article_score, record, license_name, section)
                )

        passage_candidates.sort(
            key=lambda value: (
                value[3]["category"] == "diagnostic",
                value[0],
                len(value[3]["text"]),
            ),
            reverse=True,
        )
        items: list[EvidenceItem] = []
        for index, (article_score, record, license_name, section) in enumerate(
            passage_candidates[:max_items], start=1
        ):
            pmcid = str(record.get("pmcid") or "").strip()
            pmid = str(record.get("pmid") or "").strip()
            doi = str(record.get("doi") or "").strip()
            title = _clean_text(record.get("title"), max_chars=1000)
            publication_types_raw = record.get("pubTypeList") or {}
            publication_types = (
                publication_types_raw.get("pubType", [])
                if isinstance(publication_types_raw, Mapping)
                else []
            )
            category = section["category"]
            items.append(
                EvidenceItem(
                    evidence_id=f"europe-pmc:{pmcid}:passage-{index}",
                    space_trial_id=query.space_trial_id,
                    trial_id=query.trial_id,
                    source=self.name,
                    evidence_type=(
                        "diagnostic_guideline_full_text"
                        if category == "diagnostic"
                        else "therapeutic_guideline_full_text"
                    ),
                    title=f"{title} — {section['heading']}",
                    excerpt=_clean_text(section["text"], max_chars=60000),
                    url=f"https://europepmc.org/articles/{pmcid}",
                    source_locator=f"{pmcid}, section {section['heading']}",
                    published_at=str(
                        record.get("firstPublicationDate")
                        or record.get("pubYear")
                        or ""
                    ),
                    retrieved_at=_now(),
                    jurisdiction="Publication-specific; inspect authoring body",
                    license="CC BY" if license_name == "cc by" else "CC0",
                    query=" | ".join(search_queries),
                    attributes={
                        "evidence_category": category,
                        "section_heading": section["heading"],
                        "publication_types": list(publication_types),
                        "pmcid": pmcid,
                        "pmid": pmid,
                        "doi": doi,
                        "license_verified_from": "Europe PMC core metadata",
                        "source_relevance_score": article_score,
                        "is_clinical_practice_guideline": any(
                            "practice guideline" in str(value).casefold()
                            for value in publication_types
                        ),
                    },
                )
            )
        if items:
            status = "partial" if fetch_failures else "ok"
            message = (
                f"Retrieved {len(items)} diagnostic/therapeutic passages from "
                "permissively licensed Europe PMC guideline or consensus full text."
            )
        else:
            status = "empty"
            message = (
                "No disease-relevant guideline or consensus full text with an "
                "allowlisted CC BY/CC0 license was retrieved."
            )
        if rejected_license_count:
            message += (
                f" Rejected {rejected_license_count} result(s) without an "
                "allowlisted license."
            )
        if fetch_failures:
            message += " Full-text fetch failures: " + "; ".join(fetch_failures)
        return items, [SourceNotice(source=self.name, status=status, message=message)]


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
        disease_clause = f'("{query.disease}"[Title/Abstract])'
        guideline_clause = (
            '("Practice Guideline"[Publication Type] OR '
            '"Guideline"[Publication Type] OR '
            '"Consensus Development Conference"[Publication Type] OR '
            'guideline[Title] OR consensus[Title] OR recommendations[Title])'
        )
        facet_terms = {
            "diagnostic_workup": (
                "(diagnosis[Title/Abstract] OR diagnostic[Title/Abstract] OR "
                "workup[Title/Abstract] OR staging[Title/Abstract] OR "
                "imaging[Title/Abstract] OR biopsy[Title/Abstract] OR "
                "pathology[Title/Abstract])"
            ),
            "molecular_testing": (
                "(molecular[Title/Abstract] OR genomic[Title/Abstract] OR "
                "biomarker[Title/Abstract] OR testing[Title/Abstract] OR "
                "assay[Title/Abstract] OR specimen[Title/Abstract])"
            ),
            "treatment_guidance": (
                "(treatment[Title/Abstract] OR therapy[Title/Abstract] OR "
                '"standard of care"[Title/Abstract] OR '
                "management[Title/Abstract])"
            ),
        }
        facet_queries = {
            facet: f"{disease_clause} AND {guideline_clause} AND {terms}"
            for facet, terms in facet_terms.items()
        }
        common_params: dict[str, Any] = {
            "db": "pubmed",
            "tool": "matchminer-ai",
        }
        if email:
            common_params["email"] = email
        api_key = os.environ.get("NCBI_API_KEY", "").strip()
        if api_key:
            common_params["api_key"] = api_key
        ids: list[str] = []
        facets_by_id: dict[str, list[str]] = {}
        retmax = max(
            max_items,
            int(settings.get("pubmed_retmax", max_items)),
        )
        for facet, search_query in facet_queries.items():
            search_response = await client.get(
                f"{NCBI_EUTILS_URL}/esearch.fcgi",
                params={
                    **common_params,
                    "term": search_query,
                    "retmode": "json",
                    "retmax": max(4, min(retmax, 50)),
                    "sort": "relevance",
                },
            )
            search_response.raise_for_status()
            payload = search_response.json()
            search_result = (
                payload.get("esearchresult", {})
                if isinstance(payload, Mapping)
                else {}
            )
            facet_ids = (
                search_result.get("idlist", [])
                if isinstance(search_result, Mapping)
                else []
            )
            for value in facet_ids:
                pmid = str(value).strip()
                if not pmid:
                    continue
                facets_by_id.setdefault(pmid, []).append(facet)
                if pmid not in ids:
                    ids.append(pmid)
            if not api_key:
                await asyncio.sleep(0.34)
        ids = ids[: max(max_items * 3, max_items)]
        if not ids:
            return [], [
                SourceNotice(
                    source=self.name,
                    status="empty",
                    message="PubMed returned no matching citations.",
                )
            ]
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
        ranked_items: list[tuple[int, EvidenceItem]] = []
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
                continue
            publication_types = [
                _pubmed_text(node)
                for node in article.findall(".//PublicationType")
                if _pubmed_text(node)
            ]
            facets = facets_by_id.get(pmid, [])
            facet_priority = [
                facet
                for facet in (
                    "diagnostic_workup",
                    "molecular_testing",
                    "treatment_guidance",
                )
                if facet in facets
            ]
            primary_facet = facet_priority[0] if facet_priority else "general"
            evidence_type = {
                "diagnostic_workup": "diagnostic_guideline_abstract",
                "molecular_testing": "molecular_testing_guideline_abstract",
                "treatment_guidance": "therapeutic_guideline_abstract",
            }.get(primary_facet, "literature_abstract")
            publication_type_text = " ".join(publication_types).casefold()
            searchable = f"{title} {excerpt}".casefold()
            rank_score = 0
            rank_score += 12 if "practice guideline" in publication_type_text else 0
            rank_score += 8 if "guideline" in publication_type_text else 0
            rank_score += 7 if "consensus" in publication_type_text else 0
            rank_score += 5 if "guideline" in title.casefold() else 0
            rank_score += 5 if "consensus" in title.casefold() else 0
            rank_score += 3 if "recommendation" in title.casefold() else 0
            rank_score += 2 * len(facets)
            rank_score += len(
                _significant_tokens(query.disease).intersection(
                    _significant_tokens(searchable)
                )
            )
            ranked_items.append(
                (
                    rank_score,
                    EvidenceItem(
                    evidence_id=f"pubmed:{pmid}",
                    space_trial_id=query.space_trial_id,
                    trial_id=query.trial_id,
                    source=self.name,
                    evidence_type=evidence_type,
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
                    query=" | ".join(facet_queries[facet] for facet in facets),
                    attributes={
                        "publication_types": publication_types,
                        "journal": _pubmed_text(article.find(".//Journal/Title")),
                        "search_facets": facets,
                        "ranking_score": rank_score,
                        "is_clinical_practice_guideline": (
                            "practice guideline" in publication_type_text
                        ),
                    },
                    ),
                )
            )
        ranked_items.sort(key=lambda value: value[0], reverse=True)
        items = [item for _, item in ranked_items[:max_items]]
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
    "europe_pmc_open_guidelines": EuropePMCOpenGuidelinesSource(),
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
    "EUROPE_PMC_API",
    "NCI_SEARCH_API",
    "NCBI_EUTILS_URL",
    "CIViCSource",
    "FDAClinicalSource",
    "EuropePMCOpenGuidelinesSource",
    "NCIPDQSource",
    "PubMedSource",
    "SOURCE_ADAPTERS",
    "SourceAdapter",
    "resolve_sources",
]
