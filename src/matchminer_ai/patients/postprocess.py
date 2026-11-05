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


def split_patient_summary(text: str) -> tuple[str, str]:
    """Return a summary's key section and its boilerplate section.

    The key section is the cancer history (cancer type, histology, extent,
    treatment, biomarkers) that TrialSpace retrieval and TrialChecker read; the
    boilerplate section is the text after ``Other relevant history:``
    (comorbidities, performance status, organ function, other cancers and other
    general-eligibility evidence) that BoilerplateChecker reads. Historical
    headings are accepted, and a missing section is ``Not documented.``.
    """
    value = normalize_patient_summary(text if isinstance(text, str) else "")
    if not value or value.startswith("ERROR:"):
        return value, value
    key, _, boilerplate = value.partition("\n\n" + OTHER_HISTORY_HEADING + "\n")
    return key.strip(), boilerplate.strip() or "Not documented."


def add_summary_sections(df: pd.DataFrame, column: str = "patient_summary") -> pd.DataFrame:
    """Add ``cancer_history_summary`` (key) and ``general_exclusion_criteria_evidence``
    (boilerplate) columns split from a complete ``patient_summary`` column."""
    df = df.copy()
    sections = [split_patient_summary(text) for text in df[column].fillna("").astype(str)]
    df["cancer_history_summary"] = [key for key, _ in sections]
    df["general_exclusion_criteria_evidence"] = [boilerplate for _, boilerplate in sections]
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
    """Normalize patient summaries, add their key and boilerplate sections, and
    remove non-informative rows, returning the removed patient IDs."""
    cleaned = df.copy()
    cleaned["patient_summary"] = cleaned["patient_answer_text"].fillna("").map(normalize_patient_summary)
    cleaned = add_summary_sections(cleaned)
    cleaned, removed_patient_ids = clean_bad_data(cleaned, summary_column="cancer_history_summary")
    if not config.debug_mode:
        cleaned = cleaned.drop(columns=["patient_answer_text"], errors="ignore")
    cleaned = cleaned.drop(columns=["finish_reason"], errors="ignore")
    return cleaned, removed_patient_ids
