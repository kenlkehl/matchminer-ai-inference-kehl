"""Data models shared by clinical trial-space contextualization sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


EVIDENCE_COLUMNS = [
    "evidence_id",
    "citation_label",
    "space_trial_id",
    "trial_id",
    "source",
    "evidence_type",
    "title",
    "excerpt",
    "url",
    "source_locator",
    "published_at",
    "updated_at",
    "retrieved_at",
    "jurisdiction",
    "license",
    "query",
    "attributes",
]


@dataclass(frozen=True)
class TrialSpaceQuery:
    """Trial-only disease context derived from a clinical-space summary."""

    space_trial_id: str
    trial_id: str
    clinical_space_summary: str
    disease: str
    histology: str = ""
    disease_burden: str = ""
    biomarkers_required: str = ""
    biomarkers_excluded: str = ""
    prior_treatment_required: str = ""
    prior_treatment_excluded: str = ""

    @property
    def disease_query(self) -> str:
        """Return a compact disease phrase suitable for public-source queries."""

        parts = [self.disease, self.histology]
        return " ".join(
            part
            for part in parts
            if part and part.casefold() not in {"na", "n/a", "any", "none"}
        ).strip()

    @property
    def biomarker_query(self) -> str:
        """Return biomarker criteria without any patient context."""

        parts = [self.biomarkers_required, self.biomarkers_excluded]
        return " ".join(
            part
            for part in parts
            if part and part.casefold() not in {"na", "n/a", "none"}
        ).strip()


@dataclass(frozen=True)
class EvidenceItem:
    """One normalized source record used for grounded synthesis."""

    evidence_id: str
    space_trial_id: str
    trial_id: str
    source: str
    evidence_type: str
    title: str
    excerpt: str
    url: str
    source_locator: str
    published_at: str = ""
    updated_at: str = ""
    retrieved_at: str = ""
    jurisdiction: str = ""
    license: str = ""
    query: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Return a DataFrame-ready record with a stable field order."""

        return asdict(self)


@dataclass(frozen=True)
class SourceNotice:
    """A source-level success, empty-result, or failure notice."""

    source: str
    status: str
    message: str


@dataclass
class TrialSpaceContextualizationResult:
    """Narrative contexts, normalized evidence, and run provenance."""

    contexts: pd.DataFrame
    evidence: pd.DataFrame
    metadata: dict[str, Any]


def evidence_frame(items: list[EvidenceItem]) -> pd.DataFrame:
    """Build a normalized evidence table, including for an empty result."""

    if not items:
        return pd.DataFrame(columns=EVIDENCE_COLUMNS)
    return pd.DataFrame([item.to_record() for item in items], columns=EVIDENCE_COLUMNS)


__all__ = [
    "EVIDENCE_COLUMNS",
    "EvidenceItem",
    "SourceNotice",
    "TrialSpaceContextualizationResult",
    "TrialSpaceQuery",
    "evidence_frame",
]
