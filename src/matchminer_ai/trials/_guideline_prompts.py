"""Versioned prompts. Disease-state extraction is always performed by the endpoint."""

import json

from ._guideline_schema import FIELDS
from .prompt_builder import load_prompt_text

PROMPT_VERSION = "nccn-trialspace-v13-source-excerpts"

FIELD_PLACEMENT_RULES = load_prompt_text("guideline.field_placement_rules.txt").rstrip(
    "\n"
)

SPECIFICITY_RULES = load_prompt_text("guideline.specificity_rules.txt").rstrip("\n")

BREAST_RECEPTOR_RULES = load_prompt_text("guideline.breast_receptor_rules.txt").rstrip(
    "\n"
)


def population_rules(guideline_title):
    rules = FIELD_PLACEMENT_RULES + "\n" + SPECIFICITY_RULES
    if "breast" in guideline_title.casefold():
        rules += "\n" + BREAST_RECEPTOR_RULES
    return rules


SYSTEM = load_prompt_text("guideline.system.txt").rstrip("\n")


def messages(task, payload, schema):
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": task
            + "\n\nRequired JSON schema:\n"
            + json.dumps(schema)
            + "\n\n"
            + payload,
        },
    ]


EXTRACT_TASK = load_prompt_text("guideline.extract.txt").rstrip("\n")

DETAIL_TASK = load_prompt_text("guideline.detail.txt").rstrip("\n")


def trialspace_contract():
    return ". ".join(f"{label}: <{key}>" for key, label in FIELDS.items()) + "."
