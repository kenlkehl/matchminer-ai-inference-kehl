"""Shared prompt and strict response contract for treatment/cohort extraction."""

from __future__ import annotations

import json
import re
from typing import Any

from matchminer_ai.llm.prompts import load_prompt_text

COHORT_PROMPT = "trial.cohorts.user.txt"
# Registry text and full protocols follow one contract: a single-cohort answer
# carries that cohort's extracted criteria instead of deferring to the registry
# eligibility field, so both sources apply the same enrollment-only, US-only
# and formatting rules. v3 (named cohorts, no single-cohort criteria) remains
# readable for historical checkpoints.
COHORT_SCHEMA_VERSION = "trial-registry-cohorts-v1"
LEGACY_COHORT_SCHEMA_VERSION = "trial-treatment-cohorts-v3"
PROTOCOL_COHORT_PROMPT = "trial.cohorts.protocol.user.txt"
PROTOCOL_COHORT_SCHEMA_VERSION = "trial-protocol-cohorts-v2"
_NAMED_SCHEMAS = (LEGACY_COHORT_SCHEMA_VERSION, COHORT_SCHEMA_VERSION, PROTOCOL_COHORT_SCHEMA_VERSION)
_SINGLE_CRITERIA_SCHEMAS = (COHORT_SCHEMA_VERSION, PROTOCOL_COHORT_SCHEMA_VERSION)

# No minLength: under vLLM 0.31 structured outputs a minLength string admits no
# JSON escape, so v1 criteria lost every line break and double quote. The
# parser rejects empty names and criteria instead.
_TEXT = {"type": "string"}
_COHORT = {"type": "object", "additionalProperties": False,
           "required": ["cohort_name", "cohort_specific_eligibility_criteria"],
           "properties": {"cohort_name": _TEXT, "cohort_specific_eligibility_criteria": _TEXT}}
# Pass as a structured-output JSON schema with either source: on long
# multi-cohort protocols Nemotron otherwise appended a stray closing brace,
# which the strict parser rightly rejects. The parser still validates.
COHORT_JSON_SCHEMA = {"anyOf": [
    {"type": "object", "additionalProperties": False, "required": ["cancer_treatment_trial"],
     "properties": {"cancer_treatment_trial": {"const": "no"}}},
    {"type": "object", "additionalProperties": False,
     "required": ["cancer_treatment_trial", "multiple_cohorts", "eligibility_criteria"],
     "properties": {"cancer_treatment_trial": {"const": "yes"}, "multiple_cohorts": {"const": "no"},
                    "eligibility_criteria": _TEXT}},
    {"type": "object", "additionalProperties": False,
     "required": ["cancer_treatment_trial", "multiple_cohorts", "cohorts", "common_eligibility_criteria"],
     "properties": {"cancer_treatment_trial": {"const": "yes"}, "multiple_cohorts": {"const": "yes"},
                    "cohorts": {"type": "array", "minItems": 2, "items": _COHORT},
                    "common_eligibility_criteria": {"type": "string"}}},
]}
PROTOCOL_COHORT_JSON_SCHEMA = COHORT_JSON_SCHEMA


def structured_cohort_response_format(name: str) -> dict[str, Any]:
    """OpenAI-compatible response_format constraining answers to COHORT_JSON_SCHEMA."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": COHORT_JSON_SCHEMA}}


def load_cohort_prompt(filename: str = COHORT_PROMPT) -> str:
    return load_prompt_text(filename)


def build_cohort_messages(
    trial_text: str, *, prompt: str | None = None,
) -> list[dict[str, str]]:
    """Format once so braces in the source document remain literal."""
    template = load_cohort_prompt() if prompt is None else prompt
    return [{"role": "user", "content": template.format(trial_text=trial_text)}]


def build_protocol_cohort_messages(
    protocol_text: str, *, prompt: str | None = None,
) -> list[dict[str, str]]:
    """Full protocol text (for example, OCR output) in one user message."""
    template = load_cohort_prompt(PROTOCOL_COHORT_PROMPT) if prompt is None else prompt
    return [{"role": "user", "content": template.format(protocol_text=protocol_text)}]


# Under a JSON grammar a thinking model that is still revising can only write its
# revision into the strings (one registry answer added a cohort whose criteria
# were "placeholder"). Such a string is never real criteria; reject it so the
# trial is retried.
_PLACEHOLDER = re.compile(r"\W*(?:placeholder|tbd|todo|to be determined|\.\.\.|…)\W*", re.IGNORECASE)


def _require_real_criteria(field, value):
    if isinstance(value, str) and value.strip() and _PLACEHOLDER.fullmatch(value):
        raise ValueError(f"{field} is a placeholder, not eligibility criteria")


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
    if schema_version not in ("trial-treatment-cohorts-v1", "trial-treatment-cohorts-v2", *_NAMED_SCHEMAS):
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
        if multiple == "no" and schema_version in _SINGLE_CRITERIA_SCHEMAS:
            expected.append("eligibility_criteria")
            criteria = result.get("eligibility_criteria")
            if not isinstance(criteria, str) or not criteria.strip():
                raise ValueError("eligibility_criteria must be a nonempty string for a single-cohort trial")
            _require_real_criteria("eligibility_criteria", criteria)
        if multiple == "yes":
            expected.append("cohorts")
            cohorts = result.get("cohorts")
            if not isinstance(cohorts, list) or len(cohorts) < 2:
                raise ValueError("cohorts must contain at least two cohort entries")
            if schema_version in _NAMED_SCHEMAS:
                fields = {"cohort_name", "cohort_specific_eligibility_criteria"}
                for cohort in cohorts:
                    if not isinstance(cohort, dict) or set(cohort) != fields:
                        raise ValueError("Every cohort must be an object with exactly cohort_name and cohort_specific_eligibility_criteria")
                    if not all(isinstance(cohort[k], str) and cohort[k].strip() for k in fields):
                        raise ValueError("Cohort names and eligibility criteria must be nonempty strings")
                    _require_real_criteria("cohort_specific_eligibility_criteria",
                                           cohort["cohort_specific_eligibility_criteria"])
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
                _require_real_criteria("common_eligibility_criteria", common)
    else:
        raise ValueError('cancer_treatment_trial must be the string "yes" or "no"')
    if list(result) != expected:
        raise ValueError(f"Expected exactly these fields in this order: {expected}")
    return result
