"""Patient postprocessing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
import re

import pandas as pd

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


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


def postprocess_patient_summaries(
    df: pd.DataFrame,
    config: MMAIConfig,
) -> pd.DataFrame:
    """Postprocess final serial patient summaries into clean outputs."""
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
    if not config.debug_mode:
        cleaned = cleaned.drop(columns=["patient_answer_text"], errors="ignore")
    cleaned = cleaned.drop(columns=["finish_reason"], errors="ignore")
    return cleaned
