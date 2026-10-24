"""Shared prompt and strict response contract for treatment/cohort extraction."""

from __future__ import annotations

import json
import re
from typing import Any

from matchminer_ai.llm.prompts import load_prompt_text

COHORT_PROMPT = "trial.cohorts.user.txt"
COHORT_SCHEMA_VERSION = "trial-treatment-cohorts-v1"


def load_cohort_prompt(filename: str = COHORT_PROMPT) -> str:
    return load_prompt_text(filename)


def build_cohort_messages(
    trial_text: str, *, prompt: str | None = None,
) -> list[dict[str, str]]:
    """Format once so braces in the source document remain literal."""
    template = load_cohort_prompt() if prompt is None else prompt
    return [{"role": "user", "content": template.format(trial_text=trial_text)}]


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_cohort_response(text: str, finish_reason: str = "stop") -> dict[str, Any]:
    """Validate all three terminal states; malformed output is never rejection."""
    if finish_reason != "stop":
        raise ValueError(f"Incomplete trial cohort response: {finish_reason}")
    try:
        result = json.loads(text, object_pairs_hook=_unique_keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Trial cohort response must be valid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("Trial cohort response must be a JSON object")
    treatment = result.get("cancer_treatment_trial")
    if treatment == "no":
        expected = ["cancer_treatment_trial"]
    elif treatment == "yes":
        multiple = result.get("multiple_cohorts")
        if multiple not in ("yes", "no"):
            raise ValueError('multiple_cohorts must be the string "yes" or "no"')
        expected = ["cancer_treatment_trial", "multiple_cohorts"]
        if multiple == "yes":
            expected.append("cohorts")
            cohorts = result.get("cohorts")
            if not isinstance(cohorts, list) or len(cohorts) < 2:
                raise ValueError("cohorts must contain at least two eligibility strings")
            if not all(isinstance(c, str) and c.strip() for c in cohorts):
                raise ValueError("Every cohort must be a nonempty eligibility string")
            normalized = [re.sub(r"\s+", " ", c).strip().casefold() for c in cohorts]
            if len(normalized) != len(set(normalized)):
                raise ValueError("Duplicate cohort eligibility descriptions")
    else:
        raise ValueError('cancer_treatment_trial must be the string "yes" or "no"')
    if list(result) != expected:
        raise ValueError(f"Expected exactly these fields in this order: {expected}")
    return result
