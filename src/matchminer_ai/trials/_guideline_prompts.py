"""Versioned prompts. Disease-state extraction is always performed by the endpoint."""

import json

from ._guideline_schema import FIELDS
from .prompt_builder import load_prompt_text

PROMPT_VERSION = "nccn-trialspace-v14-field-contract"
PDQ_PROMPT_VERSION = "pdq-trialspace-v3-trial-prompt-reference"

PDQ_TRIAL_PROMPT_FILES = ("trial.user.primer.txt", "trial.user.question.txt")
PDQ_PROMPT_FILES = (
    "pdq.system.txt",
    "pdq.trial_space_reference.txt",
    *PDQ_TRIAL_PROMPT_FILES,
)

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
PDQ_SYSTEM = (
    load_prompt_text("pdq.system.txt").rstrip("\n")
    + "\n\n"
    + load_prompt_text("pdq.trial_space_reference.txt").format(
        trial_primer=load_prompt_text(PDQ_TRIAL_PROMPT_FILES[0]),
        trial_question=load_prompt_text(PDQ_TRIAL_PROMPT_FILES[1]),
    ).rstrip("\n")
)


def source_system(guideline):
    if guideline.metadata.get("source_kind") == "nci_pdq":
        return PDQ_SYSTEM
    return SYSTEM


def source_messages(messages, guideline):
    """Adapt source semantics without altering any NCCN prompt or checkpoint."""
    if guideline.metadata.get("source_kind") != "nci_pdq":
        return messages
    result = [dict(m) for m in messages]
    if result[0]["role"] == "system":
        result[0]["content"] = (
            source_system(guideline)
            if result[0]["content"] == SYSTEM
            else result[0]["content"] + "\n\n" + source_system(guideline)
        )
    else:
        result.insert(0, {"role": "system", "content": source_system(guideline)})
    return result


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
