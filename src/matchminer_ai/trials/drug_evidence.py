"""Versioned data models for patient-free GoodOption drug catalogs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


CATALOG_SCHEMA_VERSION = "good-option-catalog-v3"
ROLE_POLICY_VERSION = "ctgov-cancer-treatment-agent-screen-v3"
SYNTHESIS_SCHEMA_VERSION = "drug-evidence-summary-v3"
GOOD_OPTION_PROJECTION_VERSION = "good-option-drug-summary-v3"
HELP_ME_CHOOSE_PROJECTION_VERSION = "help-me-choose-drug-summary-v3"
GOOD_OPTION_PROMPT_VERSION = "good-option-patient-drug-summaries-v9"
GOOD_OPTION_INPUT_VERSION = "patient-plus-drug-summary-v2-four-logit"
GOOD_OPTION_LABEL_SCHEMA_VERSION = "good-option-four-binary-per-drug-v9"

RUBRIC_CRITERIA = (
    "disease_type_benefit",
    "common_biomarker_in_disease",
    "patient_biomarker_targeted",
    "biomarker_targeted_benefit",
)

DRUG_ROLES = frozenset(
    {"investigational", "control", "background", "supportive", "uncertain"}
)
SCOREABLE_ROLES = frozenset({"investigational", "uncertain"})
TERMINAL_RESEARCH_STATUSES = frozenset({"complete", "blocked"})


@dataclass(frozen=True)
class DrugIdentity:
    """One canonical active entity shared across trials."""

    drug_id: str
    preferred_name: str
    ncit_code: str = ""
    aliases: tuple[str, ...] = ()
    definition: str = ""

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["aliases_json"] = json.dumps(record.pop("aliases"), ensure_ascii=False)
        return record


@dataclass(frozen=True)
class TrialDrugAssignment:
    """A canonical drug's role in one ClinicalTrials.gov study."""

    trial_id: str
    drug_id: str
    preferred_name: str
    registry_name: str
    intervention_type: str
    role: str
    role_confidence: str
    scoreable: bool
    arm_labels: tuple[str, ...] = ()
    arm_types: tuple[str, ...] = ()
    role_rationale: str = ""

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["arm_labels_json"] = json.dumps(
            record.pop("arm_labels"), ensure_ascii=False
        )
        record["arm_types_json"] = json.dumps(
            record.pop("arm_types"), ensure_ascii=False
        )
        return record


@dataclass(frozen=True)
class EvidencePassage:
    """A bounded drug-only evidence passage with an internal source ledger."""

    evidence_id: str
    drug_id: str
    facet: str
    source: str
    source_type: str
    title: str
    passage: str
    url: str
    source_locator: str
    published_at: str = ""
    retrieved_at: str = ""
    license: str = ""
    query: str = ""
    content_sha256: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["attributes_json"] = json.dumps(
            record.pop("attributes"), ensure_ascii=False, sort_keys=True
        )
        return record


@dataclass(frozen=True)
class ResearchAttempt:
    """One source request attempt, including successful empty searches."""

    drug_id: str
    facet: str
    source: str
    query: str
    attempt: int
    status: str
    started_at: str
    finished_at: str
    result_count: int = 0
    error_type: str = ""
    error_message: str = ""
    retry_after_seconds: float | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DrugSummary:
    """Structured synthesis plus clean task-specific narrative projections."""

    drug_id: str
    preferred_name: str
    ncit_code: str
    research_status: str
    synthesis_status: str
    structured_facts: Mapping[str, Any]
    good_option_summary: str
    help_me_choose_summary: str
    evidence_count: int
    technical_failures: tuple[str, ...] = ()

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["structured_facts_json"] = json.dumps(
            record.pop("structured_facts"), ensure_ascii=False, sort_keys=True
        )
        record["technical_failures_json"] = json.dumps(
            record.pop("technical_failures"), ensure_ascii=False
        )
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "DrugSummary":
        facts = record.get("structured_facts", record.get("structured_facts_json", {}))
        failures = record.get(
            "technical_failures", record.get("technical_failures_json", ())
        )
        if isinstance(facts, str):
            facts = json.loads(facts or "{}")
        if isinstance(failures, str):
            failures = json.loads(failures or "[]")
        return cls(
            drug_id=str(record.get("drug_id") or ""),
            preferred_name=str(record.get("preferred_name") or ""),
            ncit_code=str(record.get("ncit_code") or ""),
            research_status=str(record.get("research_status") or ""),
            synthesis_status=str(record.get("synthesis_status") or ""),
            structured_facts=(facts if isinstance(facts, Mapping) else {}),
            good_option_summary=str(record.get("good_option_summary") or ""),
            help_me_choose_summary=str(record.get("help_me_choose_summary") or ""),
            evidence_count=int(record.get("evidence_count") or 0),
            technical_failures=tuple(str(value) for value in (failures or ())),
        )


@dataclass(frozen=True)
class ParsedGoodOptionResult:
    """Code-validated LLM output for one patient-trial assessment."""

    score: float = math.nan
    points: int = 0
    max_points: int = 0
    drug_count: int = 0
    status: str = "parse_failed"
    patient_disease_type: str = ""
    drug_assessments: tuple[dict[str, Any], ...] = ()
    uncertainties: tuple[str, ...] = ()
    parse_error: str = ""


@dataclass
class GoodOptionCatalog:
    """Loaded, validated Parquet catalog bundle."""

    path: Path
    manifest: dict[str, Any]
    trial_registry: pd.DataFrame
    trial_drug_index: pd.DataFrame
    drug_summaries: pd.DataFrame
    drug_evidence: pd.DataFrame
    drug_research_attempts: pd.DataFrame
    trial_intervention_screening: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def compatibility_id(self) -> str:
        return str(self.manifest.get("compatibility_id") or "")

    def assignments_for_trial(
        self, trial_id: str, *, scoreable_only: bool = False
    ) -> tuple[TrialDrugAssignment, ...]:
        normalized = str(trial_id or "").strip().upper()
        frame = self.trial_drug_index.loc[
            self.trial_drug_index["trial_id"].astype(str).eq(normalized)
        ]
        if scoreable_only:
            frame = frame.loc[frame["scoreable"].astype(bool)]
        assignments: list[TrialDrugAssignment] = []
        for record in frame.to_dict(orient="records"):
            labels = record.get("arm_labels_json", "[]")
            types = record.get("arm_types_json", "[]")
            assignments.append(
                TrialDrugAssignment(
                    trial_id=normalized,
                    drug_id=str(record.get("drug_id") or ""),
                    preferred_name=str(record.get("preferred_name") or ""),
                    registry_name=str(record.get("registry_name") or ""),
                    intervention_type=str(record.get("intervention_type") or ""),
                    role=str(record.get("role") or ""),
                    role_confidence=str(record.get("role_confidence") or ""),
                    scoreable=bool(record.get("scoreable")),
                    arm_labels=tuple(json.loads(labels or "[]")),
                    arm_types=tuple(json.loads(types or "[]")),
                    role_rationale=str(record.get("role_rationale") or ""),
                )
            )
        return tuple(assignments)

    def summary_for_drug(self, drug_id: str) -> DrugSummary | None:
        frame = self.drug_summaries.loc[
            self.drug_summaries["drug_id"].astype(str).eq(str(drug_id))
        ]
        if frame.empty:
            return None
        return DrugSummary.from_record(frame.iloc[0].to_dict())

    def scoreable_summaries_for_trial(self, trial_id: str) -> tuple[DrugSummary, ...]:
        summaries: list[DrugSummary] = []
        for assignment in self.assignments_for_trial(trial_id, scoreable_only=True):
            summary = self.summary_for_drug(assignment.drug_id)
            if summary is not None:
                summaries.append(summary)
        return tuple(summaries)

    def trial_status(self, trial_id: str) -> str:
        normalized = str(trial_id).strip().upper()
        registry_row = self.trial_registry.loc[
            self.trial_registry["trial_id"].astype(str).eq(normalized)
        ]
        if registry_row.empty:
            return "missing_catalog_trial"
        if str(registry_row.iloc[0].get("registry_status") or "") != "ok":
            return "trial_registry_blocked"
        assignments = self.assignments_for_trial(trial_id, scoreable_only=True)
        if not assignments:
            return "no_scoreable_drug"
        for assignment in assignments:
            summary = self.summary_for_drug(assignment.drug_id)
            if summary is None:
                return "missing_drug_summary"
            if (
                summary.research_status != "complete"
                or summary.synthesis_status != "ok"
            ):
                return "drug_research_blocked"
        return "ok"


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "DRUG_ROLES",
    "DrugIdentity",
    "DrugSummary",
    "EvidencePassage",
    "GOOD_OPTION_INPUT_VERSION",
    "GOOD_OPTION_LABEL_SCHEMA_VERSION",
    "GOOD_OPTION_PROJECTION_VERSION",
    "GOOD_OPTION_PROMPT_VERSION",
    "GoodOptionCatalog",
    "HELP_ME_CHOOSE_PROJECTION_VERSION",
    "ParsedGoodOptionResult",
    "ROLE_POLICY_VERSION",
    "RUBRIC_CRITERIA",
    "ResearchAttempt",
    "SCOREABLE_ROLES",
    "SYNTHESIS_SCHEMA_VERSION",
    "TERMINAL_RESEARCH_STATUSES",
    "TrialDrugAssignment",
]
