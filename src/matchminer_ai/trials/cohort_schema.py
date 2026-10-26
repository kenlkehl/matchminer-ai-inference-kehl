"""Shared prompt and strict response contract for treatment/cohort extraction."""

from __future__ import annotations

import json
import re
from typing import Any

from matchminer_ai.llm.prompts import load_prompt_text

COHORT_PROMPT = "trial.cohorts.user.txt"
COHORT_SCHEMA_VERSION = "trial-treatment-cohorts-v3"


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


def unwrap_json_fence(text: str) -> str:
    """Remove only a complete outer JSON fence, without repairing its payload."""
    fenced = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```",
        text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    ) if isinstance(text, str) else None
    return fenced.group(1) if fenced else text


def parse_cohort_response(
    text: str, finish_reason: str = "stop", *,
    schema_version: str = COHORT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Validate all three terminal states; malformed output is never rejection."""
    if schema_version not in ("trial-treatment-cohorts-v1", "trial-treatment-cohorts-v2", COHORT_SCHEMA_VERSION):
        raise ValueError(f"Unsupported trial cohort schema: {schema_version}")
    if finish_reason != "stop":
        raise ValueError(f"Incomplete trial cohort response: {finish_reason}")
    try:
        # Only unwrap a complete outer JSON fence; do not repair JSON or search
        # inside prose. Keep the original response intact for debug metadata.
        result = json.loads(unwrap_json_fence(text), object_pairs_hook=_unique_keys)
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
                raise ValueError("cohorts must contain at least two cohort entries")
            if schema_version == COHORT_SCHEMA_VERSION:
                fields = {"cohort_name", "cohort_specific_eligibility_criteria"}
                for cohort in cohorts:
                    if not isinstance(cohort, dict) or set(cohort) != fields:
                        raise ValueError("Every cohort must be an object with exactly cohort_name and cohort_specific_eligibility_criteria")
                    if not all(isinstance(cohort[k], str) and cohort[k].strip() for k in fields):
                        raise ValueError("Cohort names and eligibility criteria must be nonempty strings")
                descriptions = [c["cohort_specific_eligibility_criteria"] for c in cohorts]
            else:
                if not all(isinstance(c, str) and c.strip() for c in cohorts):
                    raise ValueError("Every historical cohort must be a nonempty eligibility string")
                descriptions = cohorts
            # Names cannot make otherwise identical eligibility populations distinct.
            normalized = [re.sub(r"\s+", " ", c).strip().casefold() for c in descriptions]
            if len(normalized) != len(set(normalized)):
                raise ValueError("Duplicate cohort eligibility descriptions")
            if schema_version != "trial-treatment-cohorts-v1":
                expected.append("common_eligibility_criteria")
                common = result.get("common_eligibility_criteria")
                if not isinstance(common, str) or (common and not common.strip()):
                    raise ValueError(
                        "common_eligibility_criteria must be a criteria string "
                        'or exactly "" when no shared criteria are established'
                    )
    else:
        raise ValueError('cancer_treatment_trial must be the string "yes" or "no"')
    if list(result) != expected:
        raise ValueError(f"Expected exactly these fields in this order: {expected}")
    return result
