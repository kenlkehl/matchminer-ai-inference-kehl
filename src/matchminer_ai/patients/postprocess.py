"""Patient postprocessing helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import re

import pandas as pd

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


logger = logging.getLogger(__name__)


OTHER_HISTORY_HEADING = "Other relevant history:"
_HISTORY_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?"
    r"(?:Other relevant history|Boilerplate(?: conditions)?)"
    r"(?:\*\*)?[ \t]*:?[ \t]*(?:\*\*)?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_patient_summary(text: str, other_history: str | None = None) -> str:
    """One complete summary; accept historical headings without losing content.

    Missing history is marked undocumented, never inferred to be negative.
    Empty/failed summaries stay empty/failed for upstream validation.
    """
    value = text.strip()
    if not value or value.startswith("ERROR:"):
        return value
    match = _HISTORY_HEADING.search(value)
    cancer = value[:match.start()].strip() if match else value
    history = value[match.end():].strip() if match else ""
    extra = other_history.strip() if isinstance(other_history, str) else ""
    if extra and extra != history:
        history = "\n".join(part for part in (history, extra) if part)
    return cancer + "\n\n" + OTHER_HISTORY_HEADING + "\n" + (history or "Not documented.")


def _split_boilerplate_section(text: str, boilerplate_marker: str) -> tuple[str, str]:
    """Split generated text at the line containing the boilerplate marker."""
    lines = text.splitlines()
    split_idx = next(
        (idx for idx, line in enumerate(lines) if boilerplate_marker in line),
        -1,
    )
    if split_idx == -1:
        cleaned = text.strip()
        return cleaned, cleaned or "None"

    main_part = "\n".join(lines[:split_idx]).strip()
    boilerplate_part = "\n".join(lines[split_idx + 1 :]).strip() or "None"
    return main_part, boilerplate_part


def parse_boilerplate(df: pd.DataFrame, boilerplate_marker: str) -> pd.DataFrame:
    """Split final patient summary output into summary and boilerplate portions."""
    df = df.copy()
    summary_source = df["patient_answer_text"].fillna("").astype(str)
    cleaned_summary = summary_source.str.strip()
    split_parts = cleaned_summary.apply(
        lambda text: _split_boilerplate_section(str(text), boilerplate_marker)
    )
    df["cancer_history_summary"] = split_parts.apply(lambda parts: parts[0])
    df["general_exclusion_criteria_evidence"] = split_parts.apply(
        lambda parts: parts[1]
    )
    return df


def clean_bad_data(
    df: pd.DataFrame,
    *,
    summary_column: str = "cancer_history_summary",
) -> tuple[pd.DataFrame, set[str]]:
    """Remove empty or non-informative summaries and return their patient IDs."""
    cleaned = df.copy()
    initial_patient_ids = set(cleaned["patient_id"].astype(str))
    cleaned[summary_column] = cleaned[summary_column].fillna("").astype(str)

    cleaned = cleaned[cleaned[summary_column] != ""]
    cleaned = cleaned[~cleaned[summary_column].str.startswith("No information")]

    retained_patient_ids = set(cleaned["patient_id"].astype(str))
    removed_patient_ids = initial_patient_ids - retained_patient_ids
    logger.info(
        "Filtered %d patient summary row(s) as non-informative.",
        len(removed_patient_ids),
    )
    return cleaned, removed_patient_ids


def postprocess_patient_summaries(
    df: pd.DataFrame,
    config: MMAIConfig,
) -> tuple[pd.DataFrame, set[str]]:
    """Parse patient summaries, remove non-informative rows, and return their IDs."""
    patient_config = dict(config.patient)
    output_format = patient_config.get("output_format", "combined")
    if output_format == "combined":
        cleaned = df.copy()
        cleaned["patient_summary"] = cleaned["patient_answer_text"].fillna("").map(normalize_patient_summary)
        cleaned = cleaned.drop(columns=["cancer_history_summary", "general_exclusion_criteria_evidence"], errors="ignore")
    elif output_format == "legacy_split":
        normalized = df.copy()
        normalized["patient_answer_text"] = normalized["patient_answer_text"].fillna("").map(normalize_patient_summary)
        cleaned = parse_boilerplate(normalized, OTHER_HISTORY_HEADING)
    else:
        raise ValueError("patient.output_format must be combined or legacy_split")
    summary_column = (
        "patient_summary" if output_format == "combined" else "cancer_history_summary"
    )
    cleaned, removed_patient_ids = clean_bad_data(cleaned, summary_column=summary_column)
    if not config.debug_mode:
        cleaned = cleaned.drop(columns=["patient_answer_text"], errors="ignore")
    cleaned = cleaned.drop(columns=["finish_reason"], errors="ignore")
    return cleaned, removed_patient_ids
