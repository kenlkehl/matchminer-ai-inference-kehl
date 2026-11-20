"""Strict JSON helpers shared by the cohort steps, the decision model's labels and the match labels."""

from __future__ import annotations

import re

# Under a JSON grammar a thinking model that is still revising can only write its
# revision into the strings (one ctgov answer added a cohort whose criteria
# were "placeholder"). Such a string is never real criteria; reject it so the
# trial is retried.
_PLACEHOLDER = re.compile(r"\W*(?:placeholder|tbd|todo|to be determined|\.\.\.|…)\W*", re.IGNORECASE)


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
