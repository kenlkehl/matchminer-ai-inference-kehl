"""ClinicalTrials.gov record lookup: GoodOption catalog studies, trial-space extraction documents and
complete eligibility criteria. Requests carry only an NCT ID, never patient context."""

from __future__ import annotations

import html
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
CLINICAL_TRIALS_STUDY = "https://clinicaltrials.gov/study"
NCT_ID_PATTERN = re.compile(r"^NCT\d{8}$", re.IGNORECASE)


@dataclass(frozen=True)
class TrialEligibilityCriteria:
    """Current complete eligibility criteria fetched from ClinicalTrials.gov."""

    nct_id: str
    eligibility_criteria: str
    source_url: str
    fetched_at_utc: str
    last_update_post_date: str = ""


@dataclass(frozen=True)
class TrialCtgovDocument:
    """ClinicalTrials.gov fields used by trial-space extraction."""

    nct_id: str
    trial_title: str
    brief_summary: str
    detailed_description: str
    eligibility_criteria: str
    source_url: str
    fetched_at_utc: str
    last_update_post_date: str = ""


def normalize_nct_id(value: Any) -> str:
    """Validate and normalize a ClinicalTrials.gov identifier."""

    nct_id = str(value or "").strip().upper()
    if not NCT_ID_PATTERN.fullmatch(nct_id):
        raise ValueError(f"Invalid ClinicalTrials.gov identifier: {value!r}")
    return nct_id


def normalize_nct_reference(value: Any) -> str:
    """Normalize an NCT ID or an official ClinicalTrials.gov study URL."""

    raw_value = str(value or "").strip()
    if NCT_ID_PATTERN.fullmatch(raw_value):
        return normalize_nct_id(raw_value)
    if raw_value.casefold().startswith(
        ("clinicaltrials.gov/", "www.clinicaltrials.gov/")
    ):
        raw_value = f"https://{raw_value}"
    parsed = urlparse(raw_value)
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or hostname not in {
        "clinicaltrials.gov",
        "www.clinicaltrials.gov",
    }:
        raise ValueError(
            "Enter an NCT ID or a ClinicalTrials.gov study URL."
        )
    for segment in parsed.path.split("/"):
        if NCT_ID_PATTERN.fullmatch(segment):
            return normalize_nct_id(segment)
    raise ValueError("ClinicalTrials.gov URL does not contain a valid NCT ID.")


def _clean_text(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return f"{text[: max_chars - 1].rstrip()}…"
    return text


def _clean_multiline_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def extract_trial_eligibility_criteria(study: Mapping[str, Any]) -> str:
    """Extract complete eligibility text from one API v2 study payload."""

    protocol = study.get("protocolSection") or {}
    if not isinstance(protocol, Mapping):
        raise ValueError("ClinicalTrials.gov study is missing protocolSection.")
    eligibility = protocol.get("eligibilityModule") or {}
    if not isinstance(eligibility, Mapping):
        raise ValueError("ClinicalTrials.gov study is missing eligibilityModule.")
    criteria = _clean_multiline_text(eligibility.get("eligibilityCriteria"))
    if not criteria:
        raise ValueError(
            "ClinicalTrials.gov does not provide complete eligibility criteria "
            "for this study."
        )
    return criteria


def _last_update_post_date(study: Mapping[str, Any]) -> str:
    protocol = study.get("protocolSection") or {}
    if not isinstance(protocol, Mapping):
        return ""
    status = protocol.get("statusModule") or {}
    if not isinstance(status, Mapping):
        return ""
    value = status.get("lastUpdatePostDateStruct") or {}
    if isinstance(value, Mapping):
        value = value.get("date")
    return _clean_text(value, max_chars=40)


def extract_trial_ctgov_document(
    nct_id: str,
    study: Mapping[str, Any],
) -> TrialCtgovDocument:
    """Wrangle an API v2 study into the trial summarization input fields."""

    normalized_id = normalize_nct_id(nct_id)
    protocol = study.get("protocolSection") or {}
    if not isinstance(protocol, Mapping):
        raise ValueError("ClinicalTrials.gov study is missing protocolSection.")
    identification = protocol.get("identificationModule") or {}
    description = protocol.get("descriptionModule") or {}
    if not isinstance(identification, Mapping):
        identification = {}
    if not isinstance(description, Mapping):
        description = {}
    title = _clean_multiline_text(
        identification.get("officialTitle") or identification.get("briefTitle")
    )
    brief_summary = _clean_multiline_text(description.get("briefSummary"))
    detailed_description = _clean_multiline_text(
        description.get("detailedDescription")
    )
    return TrialCtgovDocument(
        nct_id=normalized_id,
        trial_title=title,
        brief_summary=brief_summary,
        detailed_description=detailed_description,
        eligibility_criteria=extract_trial_eligibility_criteria(study),
        source_url=f"{CLINICAL_TRIALS_STUDY}/{normalized_id}",
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        last_update_post_date=_last_update_post_date(study),
    )


def _fetch_trial_study_sync(
    nct_reference: str,
    *,
    client: httpx.Client | None,
    timeout: float,
) -> tuple[str, Mapping[str, Any]]:
    normalized_id = normalize_nct_reference(nct_reference)
    owns_client = client is None
    resolved_client = client or httpx.Client(
        timeout=max(1.0, float(timeout)),
        follow_redirects=True,
    )
    try:
        response = resolved_client.get(f"{CLINICAL_TRIALS_API}/{normalized_id}")
        if response.status_code == 404:
            raise ValueError(
                f"{normalized_id} was not found on ClinicalTrials.gov."
            )
        response.raise_for_status()
        study = response.json()
        if not isinstance(study, Mapping):
            raise ValueError(
                f"ClinicalTrials.gov returned invalid data for {normalized_id}."
            )
        return normalized_id, study
    finally:
        if owns_client:
            resolved_client.close()


def fetch_trial_ctgov_document(
    nct_reference: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
) -> TrialCtgovDocument:
    """Fetch and wrangle one NCT ID or ClinicalTrials.gov study URL."""

    normalized_id, study = _fetch_trial_study_sync(
        nct_reference,
        client=client,
        timeout=timeout,
    )
    return extract_trial_ctgov_document(normalized_id, study)


def fetch_trial_eligibility_criteria(
    nct_id: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
) -> TrialEligibilityCriteria:
    """Fetch current complete eligibility criteria for one NCT ID."""

    document = fetch_trial_ctgov_document(
        nct_id,
        client=client,
        timeout=timeout,
    )
    return TrialEligibilityCriteria(
        nct_id=document.nct_id,
        eligibility_criteria=document.eligibility_criteria,
        source_url=document.source_url,
        fetched_at_utc=document.fetched_at_utc,
        last_update_post_date=document.last_update_post_date,
    )


async def fetch_trial_study(
    nct_id: str,
    *,
    client: httpx.AsyncClient,
) -> Mapping[str, Any]:
    """Fetch one ClinicalTrials.gov API v2 study."""

    normalized_id = normalize_nct_id(nct_id)
    response = await client.get(f"{CLINICAL_TRIALS_API}/{normalized_id}")
    if response.status_code == 404:
        raise ValueError(f"{normalized_id} was not found on ClinicalTrials.gov.")
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, Mapping):
        raise ValueError(
            f"ClinicalTrials.gov returned invalid data for {normalized_id}."
        )
    return data


# Names before November 2026, when "registry" (ClinicalTrials.gov) became "ctgov".
TrialRegistryDocument = TrialCtgovDocument


def extract_trial_registry_document(*args, **kwargs):
    """Deprecated name of :func:`extract_trial_ctgov_document`."""
    warnings.warn("extract_trial_registry_document is deprecated; use extract_trial_ctgov_document",
                  DeprecationWarning, stacklevel=2)
    return extract_trial_ctgov_document(*args, **kwargs)


def fetch_trial_registry_document(*args, **kwargs):
    """Deprecated name of :func:`fetch_trial_ctgov_document`."""
    warnings.warn("fetch_trial_registry_document is deprecated; use fetch_trial_ctgov_document",
                  DeprecationWarning, stacklevel=2)
    return fetch_trial_ctgov_document(*args, **kwargs)


__all__ = [
    "CLINICAL_TRIALS_API",
    "CLINICAL_TRIALS_STUDY",
    "NCT_ID_PATTERN",
    "TrialCtgovDocument",
    "TrialEligibilityCriteria",
    "TrialRegistryDocument",
    "extract_trial_ctgov_document",
    "extract_trial_eligibility_criteria",
    "extract_trial_registry_document",
    "fetch_trial_ctgov_document",
    "fetch_trial_eligibility_criteria",
    "fetch_trial_registry_document",
    "fetch_trial_study",
    "normalize_nct_id",
    "normalize_nct_reference",
]
