"""Trial-space parsing and patient-data boundary checks."""

from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from .models import TrialSpaceQuery


REQUIRED_CONTEXT_COLUMNS = {
    "space_trial_id",
    "trial_id",
    "clinical_space_summary",
}
_PATIENT_COLUMN_MARKERS = (
    "patient",
    "pseudo_mrn",
    "medical_record",
    "cancer_history",
    "exclusion_criteria_evidence",
    "note_text",
    "clinical_note",
)
_FIELD_LABELS = (
    "Age range allowed",
    "Age",
    "Sex allowed",
    "Sex",
    "Cancer type allowed",
    "Histology allowed",
    "Cancer burden allowed",
    "Prior treatment required",
    "Prior treatment excluded",
    "Biomarkers required",
    "Biomarkers excluded",
)
_FIELD_PATTERN = re.compile(
    r"(?P<label>"
    + "|".join(re.escape(label) for label in _FIELD_LABELS)
    + r")\s*:\s*(?P<value>.*?)(?=(?:"
    + "|".join(re.escape(label) for label in _FIELD_LABELS)
    + r")\s*:|$)",
    flags=re.IGNORECASE,
)


def validate_trial_only_input(clinical_spaces: pd.DataFrame) -> None:
    """Reject missing trial fields and patient-bearing columns before retrieval."""

    if not isinstance(clinical_spaces, pd.DataFrame):
        raise TypeError("clinical_spaces must be a pandas DataFrame.")
    missing = REQUIRED_CONTEXT_COLUMNS.difference(clinical_spaces.columns)
    if missing:
        raise ValueError(
            "contextualize_trial_spaces requires columns: "
            f"{', '.join(sorted(REQUIRED_CONTEXT_COLUMNS))}. Missing: "
            f"{', '.join(sorted(missing))}."
        )
    patient_columns = [
        str(column)
        for column in clinical_spaces.columns
        if any(
            marker in str(column).casefold()
            for marker in _PATIENT_COLUMN_MARKERS
        )
    ]
    if patient_columns:
        raise ValueError(
            "Trial-space contextualization accepts trial-only input. Remove "
            "patient-bearing columns before retrieval: "
            + ", ".join(sorted(patient_columns))
            + "."
        )
    if clinical_spaces["space_trial_id"].isna().any():
        raise ValueError("space_trial_id values must be non-null.")
    normalized_ids = clinical_spaces["space_trial_id"].astype(str).str.strip()
    if normalized_ids.eq("").any():
        raise ValueError("space_trial_id values must be non-empty.")
    if normalized_ids.duplicated().any():
        duplicates = sorted(normalized_ids[normalized_ids.duplicated()].unique())
        raise ValueError(
            "contextualize_trial_spaces requires one row per space_trial_id. "
            "Duplicate values: "
            + ", ".join(duplicates)
            + "."
        )


def parse_clinical_space_summary(summary: Any) -> dict[str, str]:
    """Parse the stable labeled trial-space format without another LLM call."""

    text = re.sub(r"\s+", " ", str(summary or "")).strip()
    fields: dict[str, str] = {}
    for match in _FIELD_PATTERN.finditer(text):
        label = match.group("label").casefold()
        value = match.group("value").strip(" .")
        fields[label] = value
    return fields


def _field(fields: Mapping[str, str], name: str, fallback: str = "") -> str:
    return str(fields.get(name.casefold(), fallback)).strip()


def _row_text(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_trial_space_query(row: Mapping[str, Any]) -> TrialSpaceQuery:
    """Derive a source query from one trial-only clinical-space row."""

    summary = _row_text(row, "clinical_space_summary")
    fields = parse_clinical_space_summary(summary)
    disease = _field(fields, "Cancer type allowed")
    histology = _field(fields, "Histology allowed")
    if not disease:
        raise ValueError(
            f"Clinical space {_row_text(row, 'space_trial_id')!r} does not contain a "
            "'Cancer type allowed:' field."
        )
    return TrialSpaceQuery(
        space_trial_id=_row_text(row, "space_trial_id"),
        trial_id=_row_text(row, "trial_id"),
        clinical_space_summary=summary,
        disease=disease,
        histology=histology,
        disease_burden=_field(fields, "Cancer burden allowed"),
        biomarkers_required=_field(fields, "Biomarkers required"),
        biomarkers_excluded=_field(fields, "Biomarkers excluded"),
        prior_treatment_required=_field(fields, "Prior treatment required"),
        prior_treatment_excluded=_field(fields, "Prior treatment excluded"),
    )


__all__ = [
    "REQUIRED_CONTEXT_COLUMNS",
    "build_trial_space_query",
    "parse_clinical_space_summary",
    "validate_trial_only_input",
]
