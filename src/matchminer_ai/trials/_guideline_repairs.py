"""Bounded LLM repairs of citations and page accounting, preserving clinical fields."""

import copy
import json
from dataclasses import replace

from matchminer_ai._storage import digest

from ._guideline_context import pack_messages
from ._guideline_schema import (
    DETAIL,
    EVIDENCE,
    EXTRACTION,
    STRINGS,
    arr,
    normalize_evidence_lists,
    obj,
    validate_evidence,
    validate_shape,
)
from ._guideline_sources import render_pages
from ._guideline_specificity import validate_decision_fields
from .prompt_builder import load_prompt_text

INDEX = {"type": "integer", "minimum": 0}
REPAIR = obj(
    {
        "evidence_replacements": arr(
            obj(
                {
                    "candidate_index": INDEX,
                    "kind": {
                        "type": "string",
                        "enum": ["state", "diagnostic_workup", "treatment_options"],
                    },
                    "option_index": INDEX,
                    "evidence": EVIDENCE,
                }
            )
        ),
        "coverage_replacements": EXTRACTION["properties"]["page_coverage"],
        "unresolved": STRINGS,
    }
)

TASK = load_prompt_text("guideline.repair.txt").rstrip("\n")


def validation_targets(value, pages, primary_ids, *, is_extraction):
    """Locate every deterministic grounding failure without making clinical judgments."""
    candidates = value["candidates"] if is_extraction else [value]
    errors, claimed = [], set()
    claimed.update(
        c["defining_branch"]["page_id"] for c in candidates if "defining_branch" in c
    )
    for index, candidate in enumerate(candidates):
        owners = [("state", 0, candidate)]
        owners += [
            (kind, option_index, option)
            for kind in ("diagnostic_workup", "treatment_options")
            for option_index, option in enumerate(candidate[kind])
        ]
        for kind, option_index, owner in owners:
            target = {
                "candidate_index": index,
                "candidate_name": candidate["name"],
                "candidate_space": candidate["space"],
                "kind": kind,
                "option_index": option_index,
            }
            if kind != "state":
                target["option"] = {
                    key: owner[key] for key in ("name", "conditions", "category")
                }
            if not owner["evidence"]:
                errors.append({**target, "error": "Evidence list is empty"})
            for evidence_index, item in enumerate(owner["evidence"]):
                try:
                    validate_evidence([item], pages)
                except ValueError as exc:
                    detail = {
                        **target,
                        "evidence_index": evidence_index,
                        "page_id": item["page_id"],
                        "rejected_line_ids": item["line_ids"],
                        "error": str(exc),
                    }
                    if item["page_id"] in pages:
                        lines = pages[item["page_id"]].text.splitlines()
                        detail["blank_line_ids"] = [
                            i
                            for i in item["line_ids"]
                            if i <= len(lines) and not lines[i - 1].strip()
                        ]
                        detail["nonexistent_line_ids"] = [
                            i for i in item["line_ids"] if i > len(lines)
                        ]
                    errors.append(detail)
                else:
                    claimed.add(item["page_id"])
    coverage = [
        {
            "page_id": row["page_id"],
            "error": "states_extracted has no valid state/option citation to this page",
        }
        for row in value.get("page_coverage", [])
        if row["disposition"] == "states_extracted" and row["page_id"] not in claimed
    ]
    return {"evidence_errors": errors, "coverage_errors": coverage}


def apply_repair(value, patch, primary_ids, *, is_extraction):
    validate_shape(patch, REPAIR)
    if patch["unresolved"]:
        raise ValueError(
            "Evidence repair unresolved: " + "; ".join(patch["unresolved"])
        )
    result = copy.deepcopy(value)
    candidates = result["candidates"] if is_extraction else [result]
    changed = set()
    for item in patch["evidence_replacements"]:
        index, kind, option_index = (
            item["candidate_index"],
            item["kind"],
            item["option_index"],
        )
        key = (index, kind, option_index)
        if key in changed or index >= len(candidates):
            raise ValueError("Duplicate or out-of-range citation replacement")
        changed.add(key)
        owner = candidates[index]
        if kind == "state":
            if option_index != 0:
                raise ValueError("State citation replacement requires option_index=0")
        else:
            if option_index >= len(owner[kind]):
                raise ValueError("Out-of-range option citation replacement")
            owner = owner[kind][option_index]
        owner["evidence"] = copy.deepcopy(item["evidence"])
    coverage = {row["page_id"]: row for row in patch["coverage_replacements"]}
    if len(coverage) != len(patch["coverage_replacements"]) or set(coverage) - set(
        primary_ids
    ):
        raise ValueError("Coverage replacements must be distinct PRIMARY pages")
    if coverage:
        if not is_extraction:
            raise ValueError("A detail response cannot replace extraction coverage")
        result["page_coverage"] = [
            coverage.get(row["page_id"], row) for row in result["page_coverage"]
        ]
    normalize_evidence_lists(result)
    return result


def repair_response(
    client,
    guideline,
    supplied,
    primary_ids,
    job,
    value,
    error,
    validator,
    context_chars=None,
):
    """Only schema-valid clinical responses with grounding errors are eligible for repair."""
    if error.startswith("Extraction ownership:") or not any(
        term in error.lower()
        for term in ("evidence", "source line", "states_extracted")
    ):
        raise ValueError(error)
    is_extraction = "candidates" in value
    validate_shape(value, EXTRACTION if is_extraction else DETAIL)
    candidates = value["candidates"] if is_extraction else [value]
    # Do not spend repair attempts on immutable clinical errors hidden by an
    # earlier citation failure. They require regeneration of the original task.
    for candidate in candidates:
        validate_decision_fields(candidate, guideline.metadata["title"])
    required = set(primary_ids)
    for candidate in candidates:
        owners = (
            [candidate]
            + candidate["diagnostic_workup"]
            + candidate["treatment_options"]
        )
        required.update(
            e["page_id"]
            for owner in owners
            for e in owner["evidence"]
            if e["page_id"] in supplied
        )
    limited = replace(guideline, pages=supplied)
    response_id = digest(value)
    targets = validation_targets(
        value, supplied, primary_ids, is_extraction=is_extraction
    )
    focus_ids = {
        item["page_id"]
        for items in targets.values()
        for item in items
        if "page_id" in item
    }
    focus_pages = sorted(
        (supplied[key] for key in focus_ids if key in supplied), key=lambda p: p.number
    )
    tail = load_prompt_text("guideline.repair_focus.txt").format(
        pages=render_pages(focus_pages, primary_ids),
        targets=json.dumps(targets, ensure_ascii=False),
    )
    payload = {
        "repair_version": "citation-and-coverage-v4-local-positions",
        "validation_error": error,
        "validation_targets": targets,
        "primary_page_ids": list(primary_ids),
        "previous_response": value,
    }
    context, omitted, messages = pack_messages(
        client,
        limited,
        required,
        " ".join(c["name"] for c in candidates),
        TASK,
        payload,
        REPAIR,
        context_chars,
        primary_ids,
        tail=tail,
        population_guidance=False,
    )
    repair_pages = {p.id: p for p in context}

    def validate_patch(patch):
        repaired = apply_repair(value, patch, primary_ids, is_extraction=is_extraction)
        # Original citations remain valid in the original prompt; new references must
        # additionally have been visible to this repair call.
        for item in patch["evidence_replacements"]:
            holder = {"evidence": copy.deepcopy(item["evidence"])}
            normalize_evidence_lists(holder)
            validate_evidence(holder["evidence"], repair_pages)
        validator(repaired)

    patch = client.complete(
        f"{job}-evidence-repair-{response_id[:12]}", messages, REPAIR, validate_patch
    )
    return apply_repair(value, patch, primary_ids, is_extraction=is_extraction)
