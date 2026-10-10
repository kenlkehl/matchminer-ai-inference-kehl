"""Content-based consolidation. Internal record IDs never enter model prompts."""

import copy
import difflib
import json

from matchminer_ai._storage import atomic_json, digest, read_json
from matchminer_ai.llm.structured import EndpointError

from . import _guideline_prompts as prompts
from ._guideline_batching import run_batches
from ._guideline_context import ContextBudgetError, pack_messages
from ._guideline_completeness import (
    CoverageError, RETRY_TASK, require_coverage, review_coverage, population,
)
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
from ._guideline_specificity import validate_decision_field_batch
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
    errors = []
    for state in value["states"]:
        try:
            format_space(state["space"])
        except ValueError as error:
            errors.append(f"{state['name']!r}: {error}")
        # Report every invalid citation, including later items in one state.
        # A blank-line citation must not hide another state's field errors.
        for evidence in [[item] for item in state["evidence"]] or [[]]:
            try:
                validate_evidence(evidence, pages)
            except ValueError as error:
                errors.append(
                    f"{state['name']!r}: {error}. Rejected citation: "
                    + json.dumps(evidence, ensure_ascii=False)
                )
    if guideline_title is not None:
        try:
            validate_decision_field_batch(value["states"], guideline_title)
        except ValueError as error:
            errors.append(str(error))
    if errors:
        raise ValueError("\n\n".join(errors))


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


def combine_identical(states, *, preserve_names=False):
    """Union exact-definition evidence in code; no model bookkeeping is needed."""
    result, positions = [], {}
    for state in states:
        key = format_space(state["space"]).casefold()
        if key not in positions:
            positions[key] = len(result)
            result.append(copy.deepcopy(state))
            if preserve_names:
                result[-1]["population_names"] = list(dict.fromkeys(
                    [state["name"], *state.get("population_names", [])]
                ))
        else:
            target = result[positions[key]]
            if preserve_names:
                for name in [state["name"], *state.get("population_names", [])]:
                    if name not in target["population_names"]:
                        target["population_names"].append(name)
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
            {**{k: s[k] for k in LEAN_STATE["properties"]},
             **({"population_names": s["population_names"]} if "population_names" in s else {})}
            for s in states
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
    messages = prompts.source_messages(messages, guideline)
    if not client.fits(messages):
        raise ContextBudgetError(
            "The complete selection catalog exceeds available input capacity; "
            "definitions and the output reserve will not be truncated"
        )
    return [], omitted, messages


def consolidate(
    client, guideline, candidates, output, workers, run_jobs, log, context_chars=None,
    *, coverage_repairs=(),
):
    # Persisted final findings are retry guidance only; the same source-backed
    # generation, local coverage and full final coverage guards still run.
    saved_coverage = output / "canonical_coverage.json"
    if not coverage_repairs and saved_coverage.exists():
        previous = read_json(saved_coverage)
        if (previous.get("input_sha256") == digest([population(c) for c in candidates])
            and any(r["status"] != "represented" for r in previous["reviews"])):
            coverage_repairs = (previous,)

    def covered_complete(key, messages, schema, validate, inputs, states_from):
        def guarded(value):
            validate(value)
            log(f"{guideline.disease}: checking population coverage for {key}")
            report = review_coverage(client, inputs, states_from(value))
            atomic_json(output / "population_coverage" / f"{key}.json", report)
            require_coverage(report)

        try:
            return client.complete(key, messages, schema, guarded, reuse_exhausted=True)
        except CoverageError as error:
            # An older accepted checkpoint is revalidated before the client's
            # retry loop. Preserve it as evidence and request a corrected full
            # catalog under a distinct request fingerprint, on the same endpoint.
            return client.complete(
                key + "-restore-populations",
                [*messages, {"role": "user", "content": RETRY_TASK + "\n\n" + str(error)}],
                schema, guarded,
            )

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
        messages = packed[2]
        member_ids = {c["candidate_id"] for c in members}
        for report in coverage_repairs:
            findings = [
                row for candidate, row in zip(candidates, report["reviews"])
                if candidate["candidate_id"] in member_ids and row["status"] != "represented"
            ]
            if findings:
                messages = [*messages, {"role": "user", "content":
                    load_prompt_text("guideline.canonical_restore.txt")
                    + json.dumps(findings, ensure_ascii=False)}]
        if messages is not packed[2]:
            if not client.fits(messages):
                raise ContextBudgetError(
                    "Final coverage repair exceeds input budget; source and output reserve are unchanged"
                )
            key += "-final-coverage-" + digest(messages)[:16]
            packed = (packed[0], packed[1], messages)
        def generate(job, call):
            return covered_complete(
                job, call, CATALOG,
                lambda v: validate_catalog(v, pages, guideline.metadata["title"]),
                members, lambda v: v["states"],
            )
        try:
            value = generate(key, packed[2])
        except EndpointError as error:
            if len(members) != 1 or "exhausted" not in str(error) or "mixes AND and OR" not in str(error):
                raise
            # A singleton cannot split further. New focused feedback identifies
            # the entire offending group even beyond the old 500-character hint.
            focused = [*packed[2], {"role": "user", "content":
                load_prompt_text("guideline.boolean_retry.txt") + "\n\n" + str(error)}]
            if not client.fits(focused):
                raise ContextBudgetError("Focused Boolean repair exceeds source input budget")
            value = generate(key + "-focused-grouping-" + digest(focused)[:16], focused)
            packed = (packed[0], packed[1], focused)
        return {
            "result": value,
            "context": metadata(packed),
            "input_candidate_ids": [c["candidate_id"] for c in members],
        }

    values, failures = run_batches(
        planned, guideline, candidates, output, workers, run_jobs, local,
        lambda members: build_call(client, guideline, members, context_chars), log,
        coverage_repairs=coverage_repairs,
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
    states = combine_identical(states, preserve_names=True)
    if not states:
        raise ValueError("No source-backed disease-state definitions extracted")
    context_metadata = {
        "mode": VERSION,
        "batch_count": len(values),
        "batches": {key: v["context"] for key, v in values.items()},
    }
    if len(values) > 1:
        packed = build_selection_call(client, guideline, states)
        try:
            selected = covered_complete(
                "catalog-select-content", packed[2], SELECTION,
                lambda v: selected_states(v, states), states,
                lambda v: selected_states(v, states),
            )
        except EndpointError as error:
            if not selection_retention_allowed(str(error)):
                raise
            # Selection is optional deduplication of already validated content.
            # If it repeatedly rewrites/drops populations, retain ALL definitions
            # rather than accepting a bad selection. Full final coverage below
            # is mandatory. This is not presented as an accepted LLM response.
            warning = "Cross-batch deduplication could not preserve every validated definition; retained all exact-distinct definitions. Semantic duplicates may remain."
            uncertainties.append(warning)
            atomic_json(output / "canonical_selection.json", {
                "mode": "retain-validated-definitions", "available_states": states,
                "preserve_population_names": True,
                "reason": str(error), "warning": warning,
            })
            log(f"{guideline.disease}: retaining all validated definitions after selection failure")
        else:
            atomic_json(output / "canonical_selection.json", {
                "available_states": states, "result": selected, "preserve_population_names": True,
            })
            states = selected_states(selected, states)
            topics.extend(selected["context_only_topics"])
            uncertainties.extend(selected["uncertainties"])
        context_metadata["selection"] = metadata(packed)
    # Check the final catalog, too: cross-batch selection must not undo local
    # coverage. Identical review requests reuse the client's validated cache.
    coverage = review_coverage(client, candidates, states)
    atomic_json(output / "canonical_coverage.json", coverage)
    try:
        require_coverage(coverage)
    except CoverageError:
        # Final reviews can catch distinctions that intermediate reviews missed.
        # A retry must revisit the originating source batches, not repeat the
        # unchanged final review or ask closed selection to invent definitions.
        atomic_json(output / "population_coverage" / ("final-" + digest(coverage) + ".json"), coverage)
        if (
            len(coverage_repairs) >= client.config.attempts
            or any(digest(previous) == digest(coverage) for previous in coverage_repairs)
        ):
            raise
        log(f"{guideline.disease}: restoring source batches after final population-coverage failure")
        return consolidate(
            client, guideline, candidates, output, workers, run_jobs, log, context_chars,
            coverage_repairs=(*coverage_repairs, coverage),
        )
    atomic_json(output / "canonical_context.json", context_metadata)
    return {
        "version": VERSION,
        "preserve_population_names": True,
        "groups": states,
        "context_only_topics": topics,
        "uncertainties": uncertainties,
    }


def selection_retention_allowed(error):
    """Only a failed deduplication contract permits retention, not a failed source stage."""
    return "exhausted" in error and any(marker in error for marker in (
        "Copy complete supplied clinical definitions without rewriting fields",
        "Catalog omitted or broadened",
    ))
