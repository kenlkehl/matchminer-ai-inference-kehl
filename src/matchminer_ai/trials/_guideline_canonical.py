"""Content-based consolidation. Internal record IDs never enter model prompts."""

import copy
import difflib
import json

from matchminer_ai._storage import atomic_json

from . import _guideline_prompts as prompts
from ._guideline_context import ContextBudgetError, pack_messages
from ._guideline_schema import (
    EVIDENCE,
    SPACE,
    STRING,
    STRINGS,
    arr,
    format_space,
    obj,
    validate_evidence,
    validate_shape,
)
from ._guideline_specificity import validate_decision_fields
from .prompt_builder import load_prompt_text

LEAN_STATE = obj(
    {"name": STRING, "space": SPACE, "evidence": EVIDENCE, "uncertainties": STRINGS}
)
CATALOG = obj(
    {
        "states": arr(LEAN_STATE),
        "context_only_topics": STRINGS,
        "uncertainties": STRINGS,
    }
)
SELECTION = obj(
    {"spaces": arr(SPACE), "context_only_topics": STRINGS, "uncertainties": STRINGS}
)
MAX_CANDIDATES_PER_BATCH = 100
VERSION = "clinical-content-v7-owned-branches"
CONTENT_VERSIONS = frozenset(
    {
        "clinical-content-v1",
        "clinical-content-v2-treatment-decision-states",
        "clinical-content-v3-explicit-line-fields",
        "clinical-content-v4-scoped-selection",
        "clinical-content-v5-evidence-scoped",
        "clinical-content-v6-field-semantics",
        VERSION,
    }
)

TASK = load_prompt_text("guideline.canonical.txt").rstrip("\n")

SELECT_TASK = load_prompt_text("guideline.selection.txt").rstrip("\n")


def clinical_candidate(candidate):
    """Explicit allowlist keeps internal keys out of every model-facing payload."""
    value = {
        "name": candidate["name"],
        "space": candidate["space"],
        "evidence": candidate["evidence"],
        "treatment_options": [
            {"name": item["name"], "conditions": item["conditions"]}
            for item in candidate["treatment_options"]
        ],
        "uncertainties": candidate["uncertainties"],
    }
    if "defining_branch" in candidate:
        value["defining_branch"] = candidate["defining_branch"]
    return value


def build_call(client, guideline, candidates, context_chars=None):
    descriptions = [clinical_candidate(c) for c in candidates]
    evidence = [
        item
        for c in candidates
        for item in [c, *c["diagnostic_workup"], *c["treatment_options"]]
    ]
    required = {e["page_id"] for item in evidence for e in item["evidence"]}
    required.update(
        c["defining_branch"]["page_id"] for c in candidates if "defining_branch" in c
    )
    return pack_messages(
        client,
        guideline,
        required,
        " ".join(c["name"] for c in candidates),
        TASK,
        {
            "guideline": guideline.metadata["title"],
            "version": guideline.metadata["version"],
            "catalog_version": VERSION,
        },
        CATALOG,
        context_chars,
        tail="FINAL TASK AND DISEASE-STATE DESCRIPTIONS:\n"
        + TASK
        + "\n"
        + json.dumps(descriptions, ensure_ascii=False),
        source_ids=required,
    )


def validate_catalog(value, pages, guideline_title=None):
    validate_shape(value, CATALOG)
    for state in value["states"]:
        format_space(state["space"])
        validate_evidence(state["evidence"], pages)
        if guideline_title is not None:
            validate_decision_fields(state, guideline_title)


def partition_calls(client, guideline, candidates, context_chars=None):
    planned = []

    def plan(items):
        try:
            packed = build_call(client, guideline, items, context_chars)
        except ContextBudgetError:
            if len(items) <= 1:
                raise
            middle = len(items) // 2
            plan(items[:middle])
            plan(items[middle:])
        else:
            planned.append((items, packed))

    for start in range(0, len(candidates), MAX_CANDIDATES_PER_BATCH):
        plan(candidates[start : start + MAX_CANDIDATES_PER_BATCH])
    return planned


def combine_identical(states):
    """Union exact-definition evidence in code; no model bookkeeping is needed."""
    result, positions = [], {}
    for state in states:
        key = format_space(state["space"]).casefold()
        if key not in positions:
            positions[key] = len(result)
            result.append(copy.deepcopy(state))
        else:
            target = result[positions[key]]
            for field in ("evidence", "uncertainties", "source_batch_numbers"):
                for item in state[field]:
                    if item not in target[field]:
                        target[field].append(copy.deepcopy(item))
    return result


def selected_states(value, states):
    validate_shape(value, SELECTION)
    known = {format_space(s["space"]).casefold(): s for s in states}
    unknown = [s for s in value["spaces"] if format_space(s).casefold() not in known]
    if unknown:
        # This is retry guidance only: never accept a fuzzy match or alter a definition.
        # Show actual field differences instead of repeating only the invalid output.
        # Keep guidance inside the client's 1200-character feedback allowance.
        guidance = []
        for space in unknown:
            key = format_space(space).casefold()
            closest = difflib.get_close_matches(key, known, n=1, cutoff=0)
            if not closest:
                break
            supplied = known[closest[0]]
            item = {
                "closest_supplied_name": supplied["name"],
                "field_differences": {
                    field: {
                        "returned": space[field],
                        "supplied": supplied["space"][field],
                    }
                    for field in space
                    if space[field].casefold() != supplied["space"][field].casefold()
                },
            }
            proposed = json.dumps(guidance + [item], ensure_ascii=False)
            if len(proposed) > 900:
                break
            guidance.append(item)
        raise ValueError(
            "Copy complete supplied clinical definitions without rewriting fields. "
            f"{len(unknown)} returned definitions do not match. Closest textual matches "
            "below are hints, not equivalent populations; select only unchanged supplied "
            "definitions. " + json.dumps(guidance, ensure_ascii=False)
        )
    output, seen = [], set()
    for space in value["spaces"]:
        key = format_space(space).casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(copy.deepcopy(known[key]))
    if not output:
        raise ValueError("No source-backed clinical definitions selected")
    return output


def build_selection_call(client, guideline, states):
    """A closed selection task needs the validated definitions, not another source extraction."""
    omitted = [p.id for p in guideline.primary]
    payload = {
        "guideline": guideline.metadata["title"],
        "catalog_version": VERSION,
        "source_backed_definitions": [
            {k: s[k] for k in LEAN_STATE["properties"]} for s in states
        ],
        "omitted_source_page_ids": omitted,
    }
    messages = prompts.messages(
        SELECT_TASK,
        json.dumps(payload, ensure_ascii=False)
        + "\n\nFINAL SELECTION CONTRACT:\n"
        + SELECT_TASK,
        SELECTION,
    )
    if not client.fits(messages):
        raise ContextBudgetError(
            "The complete selection catalog exceeds available input capacity; "
            "definitions and the output reserve will not be truncated"
        )
    return [], omitted, messages


def consolidate(
    client, guideline, candidates, output, workers, run_jobs, log, context_chars=None
):
    def metadata(packed):
        context, omitted, messages = packed
        return {
            "included_page_ids": [p.id for p in context],
            "omitted_page_ids": omitted,
            "prompt_tokens": client.count_tokens(messages),
            "reserved_output_tokens": client.config.max_tokens,
        }

    planned = partition_calls(client, guideline, candidates, context_chars)
    log(
        f"{guideline.disease}: consolidating clinical descriptions in {len(planned)} source-backed batches"
    )

    def local(key, data):
        members, packed = data
        pages = {p.id: p for p in packed[0]}
        value = client.complete(
            key,
            packed[2],
            CATALOG,
            lambda v: validate_catalog(v, pages, guideline.metadata["title"]),
        )
        return {
            "result": value,
            "context": metadata(packed),
            "input_candidate_ids": [c["candidate_id"] for c in members],
        }

    values, failures = run_jobs(
        [(f"catalog-content-{i:04d}", data) for i, data in enumerate(planned, 1)],
        workers,
        local,
    )
    atomic_json(
        output / "canonical_batches.json",
        {"version": VERSION, "batches": values, "failures": failures},
    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} catalog batches failed; rerun to resume saved calls"
        )
    states, topics, uncertainties = [], [], []
    for number, (key, value) in enumerate(sorted(values.items()), 1):
        states.extend(
            {**s, "source_batch_numbers": [number]} for s in value["result"]["states"]
        )
        topics.extend(value["result"]["context_only_topics"])
        uncertainties.extend(value["result"]["uncertainties"])
    states = combine_identical(states)
    if not states:
        raise ValueError("No source-backed disease-state definitions extracted")
    context_metadata = {
        "mode": VERSION,
        "batch_count": len(planned),
        "batches": {key: v["context"] for key, v in values.items()},
    }
    if len(planned) > 1:
        packed = build_selection_call(client, guideline, states)
        selected = client.complete(
            "catalog-select-content",
            packed[2],
            SELECTION,
            lambda v: selected_states(v, states),
        )
        atomic_json(
            output / "canonical_selection.json",
            {"available_states": states, "result": selected},
        )
        states = selected_states(selected, states)
        topics.extend(selected["context_only_topics"])
        uncertainties.extend(selected["uncertainties"])
        context_metadata["selection"] = metadata(packed)
    atomic_json(output / "canonical_context.json", context_metadata)
    return {
        "version": VERSION,
        "groups": states,
        "context_only_topics": topics,
        "uncertainties": uncertainties,
    }
