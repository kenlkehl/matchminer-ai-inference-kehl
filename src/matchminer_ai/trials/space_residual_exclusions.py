"""Space-specific residual exclusions pilot; this name is not sent to the LLM.

Mirrored as training's trial_space_residual_exclusions.py. Only the task request,
original trial text, and completed space enter the prompt; no extraction primer
or field-classification instructions are included.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

VERSION = "space-specific-residual-exclusions-v1"


def load_residual_exclusion_prompts(directory):
    return {"request": (Path(directory) / "trial.space_residual_exclusions.txt").read_text()}


def build_residual_exclusion_messages(trial_text, space_text, prompts):
    for label, text in [("trial text", trial_text), ("completed space", space_text)]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Nonempty {label} required")
    return [{"role": "user", "content": (
        prompts["request"].rstrip()
        + "\n\nTRIAL DOCUMENT:\n" + trial_text
        + "\nEND TRIAL DOCUMENT\n\nCOMPLETED TRIAL SPACE:\n" + space_text
    )}]


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_residual_exclusions(text, finish_reason="stop"):
    if finish_reason not in {"stop", "end_turn", "eos"}:
        raise ValueError(f"Incomplete exclusion response: {finish_reason}")
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```", text.strip(), re.I | re.S)
    value = json.loads(fenced.group(1) if fenced else text, object_pairs_hook=_unique_keys)
    if not isinstance(value, dict) or set(value) != {"exclusions"} or not isinstance(value["exclusions"], list):
        raise ValueError('Expected exactly {"exclusions": ["criterion", ...]}')
    for item in value["exclusions"]:
        if not isinstance(item, str) or not item.strip() or any(c in item for c in "\n\r"):
            raise ValueError("Each exclusion must be a nonempty single-line string")
    if len(set(value["exclusions"])) != len(value["exclusions"]):
        raise ValueError("Duplicate exclusion entries")
    return value


def residual_exclusion_retry_messages(messages, final, error):
    return [*messages,
            {"role": "assistant", "content": final},
            {"role": "user", "content": (
                "The response failed JSON validation: " + str(error)
                + '. Return the complete corrected {"exclusions": ["criterion", ...]} object, with no text outside JSON. Keep each exclusion as one nonempty single-line string.'
            )}]
