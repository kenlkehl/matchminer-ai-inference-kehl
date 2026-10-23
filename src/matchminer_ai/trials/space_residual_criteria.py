"""Bounded, opt-in residual extraction and core additions; no production writes.

Mirrored into training. The request callback owns transport, retries, and audit
files; this module owns the shared prompt contract and per-space loop.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

if __package__:
    from .space_refinement import FIELDS, validate_space
else:
    from trial_space_refinement import FIELDS, validate_space

VERSION = "space-residual-criteria-loop-v1"
DEFAULT_MAX_REFINEMENTS = 3


def core_space_primer(primer):
    """Retain authored core rules verbatim, omitting the boilerplate task."""
    boundary = 'After you output the trial spaces, output a newline, then the text "Boilerplate exclusions:"'
    if primer.count(boundary) != 1:
        raise ValueError("Cannot locate unique boilerplate section in authored primer")
    core = primer.split(boundary)[0]
    announcement = [line for line in core.splitlines(keepends=True)
                    if line.startswith('Once you define the list of trial spaces,')]
    if len(announcement) != 1:
        raise ValueError("Cannot locate unique boilerplate task announcement")
    return core.replace(announcement[0], "").rstrip()


def load_residual_criteria_prompts(directory):
    directory = Path(directory)
    return {
        "core_primer": core_space_primer((directory / "trial.user.primer.txt").read_text()),
        "extract": (directory / "trial.space_residual_criteria.txt").read_text(),
        "add": (directory / "trial.space_residual_additions.txt").read_text(),
    }


def build_residual_criteria_messages(trial_text, space, prompts, additions=None):
    validate_space(space)
    if not isinstance(trial_text, str) or not trial_text.strip():
        raise ValueError("Trial document must be nonempty")
    template = prompts["extract" if additions is None else "add"]
    if template.count("{{core_primer}}") != 1:
        raise ValueError("Expected exactly one core primer placeholder")
    content = template.replace("{{core_primer}}", prompts["core_primer"]).rstrip()
    content += "\n\nTRIAL DOCUMENT:\n" + trial_text + "\nEND TRIAL DOCUMENT"
    if additions is not None:
        content += "\n\nPROPOSED MISSING CORE REQUIREMENTS:\n" + json.dumps(additions, ensure_ascii=False)
    content += "\n\nCURRENT COMPLETED SPACE:\n" + json.dumps(space, ensure_ascii=False)
    return [{"role": "user", "content": content}]


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _json_response(text, finish_reason):
    if finish_reason not in {"stop", "end_turn", "eos"}:
        raise ValueError(f"Incomplete residual response: {finish_reason}")
    fence = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```", text.strip(), re.I | re.S)
    return json.loads(fence.group(1) if fence else text, object_pairs_hook=_unique_keys)


def parse_residual_criteria(text, trial_text, finish_reason="stop"):
    value = _json_response(text, finish_reason)
    if not isinstance(value, dict) or set(value) != {"criteria"} or not isinstance(value["criteria"], list):
        raise ValueError("Expected exactly a criteria array")
    seen = set()
    def quote_text(text):
        # CT.gov escapes comparison operators/brackets in otherwise plain text.
        # Rendering those punctuation escapes does not change the quoted words.
        text = re.sub(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~])", r"\1", text)
        text = text.replace("≤", "<=").replace("≥", ">=").replace("=<", "<=")
        return " ".join(text.split())

    source = quote_text(trial_text)
    for item in value["criteria"]:
        keys = {"kind", "criterion", "source_quote", "belongs_in_space", "field", "reason"}
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("Each criterion must contain exactly kind, criterion, source_quote, belongs_in_space, field, reason")
        if item["kind"] not in ("inclusion", "exclusion"):
            raise ValueError("kind must be inclusion or exclusion")
        for key in ("criterion", "source_quote", "reason"):
            if not isinstance(item[key], str) or not item[key].strip() or any(c in item[key] for c in "\r\n"):
                raise ValueError(f"{key} must be a nonempty single-line string")
        if type(item["belongs_in_space"]) is not bool:
            raise ValueError("belongs_in_space must be a JSON boolean")
        if item["belongs_in_space"]:
            if item["field"] not in tuple(key for key, _ in FIELDS):
                raise ValueError("A core criterion must name one of the seven fields")
        elif item["field"] is not None:
            raise ValueError("A non-core criterion must have field=null")
        quote = quote_text(item["source_quote"]).rstrip(".")
        if not quote or quote not in source:
            raise ValueError("source_quote must be a verbatim passage from the trial document; unmatched quote: "
                             + repr(item["source_quote"]))
        identity = " ".join(item["criterion"].casefold().split())
        if identity in seen:
            raise ValueError("Duplicate remaining criterion")
        seen.add(identity)
    return value


def parse_residual_additions(text, finish_reason="stop"):
    value = _json_response(text, finish_reason)
    # A complete bare seven-field object is unambiguous. Normalize the omitted
    # wrapper rather than spend another model call regenerating valid values.
    if isinstance(value, dict) and set(value) == {key for key, _ in FIELDS}:
        value = {"space": value}
    if not isinstance(value, dict) or set(value) != {"space"}:
        raise ValueError("Expected exactly one complete space object")
    validate_space(value["space"])
    return value


def residual_criteria_retry_messages(messages, final, error):
    return [*messages, {"role": "assistant", "content": final},
            {"role": "user", "content": "The response failed validation: " + str(error)
             + ". Return the complete corrected JSON object under the same contract. No Markdown or extra prose."}]


def _identity(space):
    # Ignore superficial whitespace/case/punctuation changes when detecting loops.
    return tuple(" ".join(space[key].casefold().split()).rstrip(".") for key, _ in FIELDS)


async def extract_residual_criteria(trial_text, original_space, prompts, request,
                                   max_refinements=DEFAULT_MAX_REFINEMENTS):
    """At most N additions and N+1 checks; return usable data even at the cap.

    request(stage_name, messages, parser) must return one validated response.
    A stopped loop keeps the latest valid space and *all* criteria from its last
    check, including those still marked core. It never creates a review queue or
    misrepresents a bounded stop as convergence.
    """
    if type(max_refinements) is not int or max_refinements < 0:
        raise ValueError("max_refinements must be a nonnegative integer")
    validate_space(original_space)
    current = dict(original_space)
    versions, checks = [current], []
    seen = {_identity(current)}
    for round_number in range(max_refinements + 1):
        checked = await request(
            f"check_{round_number:02d}",
            build_residual_criteria_messages(trial_text, current, prompts),
            lambda text, finish: parse_residual_criteria(text, trial_text, finish),
        )
        criteria = checked["criteria"]
        checks.append({"space": current, "criteria": criteria})
        additions = [item for item in criteria if item["belongs_in_space"]]
        if not additions:
            stop_reason = "converged"
            break
        if round_number == max_refinements:
            stop_reason = "refinement_limit"
            break
        refined = await request(
            f"add_{round_number + 1:02d}",
            build_residual_criteria_messages(trial_text, current, prompts, additions),
            parse_residual_additions,
        )
        candidate = refined["space"]
        identity = _identity(candidate)
        if identity in seen:
            stop_reason = "no_change" if identity == _identity(current) else "cycle"
            break
        current = candidate
        versions.append(current)
        seen.add(identity)
    return {"space": current, "residual_criteria": criteria,
            "converged": stop_reason == "converged", "stop_reason": stop_reason,
            "refinements_applied": len(versions) - 1, "checks": checks,
            "space_versions": versions, "max_refinements": max_refinements}
