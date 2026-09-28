"""Repair invalid excerpts independently without regenerating clinical records."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from matchminer_ai._storage import atomic_json, digest
from matchminer_ai.llm.structured import EndpointError

from ._guideline_context import pack_messages
from ._guideline_quotes import QUOTED_DETAIL, QUOTED_EVIDENCE, resolve_excerpt
from ._guideline_quote_selection import (
    PROMPT as SELECTION_PROMPT,
    audit_selection,
    select_literal_passages,
)
from ._guideline_schema import STRINGS, obj, validate_shape
from ._guideline_sources import render_pages
from .prompt_builder import load_prompt_text

VERSION = "focused-excerpt-repair-v1"
PROMPT_FILES = ("guideline.quote_repair.txt", SELECTION_PROMPT)
REPAIR = obj({"evidence": QUOTED_EVIDENCE, "unresolved": STRINGS})


def owners(record):
    return [record, *record["diagnostic_workup"], *record["treatment_options"]]


def repair_assertion(record, owner_index, evidence_index, diagnostic):
    owner = owners(record)[owner_index]
    return {
        "population": {"name": record["name"], "space": record["space"]},
        "assertion": {k: v for k, v in owner.items() if k != "evidence"},
        "existing_evidence": owner["evidence"],
        "rejected_evidence": owner["evidence"][evidence_index],
        "validation_error": diagnostic,
    }


def targets(record, pages):
    """Find invalid references; positions remain code-side, never in model prompts."""
    result = []
    for owner_index, owner in enumerate(owners(record)):
        for evidence_index, ref in enumerate(owner["evidence"]):
            try:
                if ref["page_id"] not in pages:
                    raise ValueError("The cited page was not supplied")
                resolve_excerpt(pages[ref["page_id"]], ref["source_text"])
            except ValueError as exc:
                result.append((owner_index, evidence_index, str(exc)))
    return result


def validate_replacement(value, pages):
    validate_shape(value, REPAIR)
    if value["unresolved"] or not value["evidence"]:
        raise ValueError(
            "Citation remains unsupported: " + "; ".join(value["unresolved"])
        )
    for ref in value["evidence"]:
        if ref["page_id"] not in pages:
            raise ValueError("Replacement cites an unsupplied page")
        resolve_excerpt(pages[ref["page_id"]], ref["source_text"])


def apply_replacements(original, replacements):
    """Apply all patches to original positions; preserve valid citations and every field."""
    result = copy.deepcopy(original)
    original_owners, result_owners = owners(original), owners(result)
    patches = {}
    for item in replacements:
        key = (item["owner_index"], item["evidence_index"])
        if key in patches or not 0 <= key[0] < len(original_owners):
            raise ValueError("Duplicate or unknown excerpt repair owner")
        if not 0 <= key[1] < len(original_owners[key[0]]["evidence"]):
            raise ValueError("Unknown excerpt repair position")
        patches[key] = item["result"]["evidence"]
    for i, (before, after) in enumerate(zip(original_owners, result_owners)):
        after["evidence"] = [
            copy.deepcopy(ref)
            for j, old_ref in enumerate(before["evidence"])
            for ref in patches.get((i, j), [old_ref])
        ]
    return result


def repair_quoted_response(
    client,
    guideline,
    pages,
    job,
    value,
    error,
    validator,
    *,
    output,
    notify=lambda _: None,
):
    # Clinical/schema/immutable-space errors require the original generation task.
    # The pipeline validator checks those before raising this quote-only diagnostic.
    if not error.startswith("Correct every invalid excerpt in the draft:"):
        raise ValueError(error)
    validate_shape(value, QUOTED_DETAIL)
    pending = targets(value, pages)
    if not pending:
        raise ValueError(error)
    original_owners = owners(value)
    notify(f"{job}: repairing {len(pending)} invalid source excerpts")

    def repair(target):
        i, j, diagnostic = target
        owner = original_owners[i]
        rejected = owner["evidence"][j]
        focus_ids = {rejected["page_id"]} & pages.keys()
        required = focus_ids | {
            ref["page_id"]
            for ref in [*value["evidence"], *owner["evidence"]]
            if ref["page_id"] in pages
        }
        assertion = repair_assertion(value, i, j, diagnostic)
        tail = (
            "AFFECTED SOURCE PAGES:\n"
            + render_pages(
                sorted((pages[p] for p in focus_ids), key=lambda p: p.number), ()
            )
            + "\nFIXED ASSERTION AND REJECTED CITATION:\n"
            + json.dumps(assertion, ensure_ascii=False)
        )
        # Start with whole pages cited by this assertion and its population. If
        # those cannot support a repair, all originally supplied pages remain
        # available in one bounded fallback. Neither call reduces output headroom.
        scopes = (
            [required, set(pages)]
            if required and required != set(pages)
            else [set(pages)]
        )
        selection = None
        for scope in scopes:
            context, _, messages = pack_messages(
                client,
                replace(
                    guideline,
                    pages={p: page for p, page in pages.items() if p in scope},
                ),
                required,
                owner["name"],
                load_prompt_text(PROMPT_FILES[0]),
                {"repair_version": VERSION},
                REPAIR,
                tail=tail,
                population_guidance=False,
                system_message=load_prompt_text("guideline.citation_system.txt"),
            )
            supplied = {p.id: p for p in context}
            repair_job = "quote-repair-" + digest(messages)[:20]
            try:
                patch = client.complete(
                    repair_job,
                    messages,
                    REPAIR,
                    lambda v: validate_replacement(v, supplied),
                    reuse_exhausted=True,
                )
            except EndpointError:
                if scope != set(pages):
                    notify(f"{job}: expanding source context for {owner['name']}")
            else:
                break
        else:
            notify(f"{job}: selecting literal source passages for {owner['name']}")
            patch, selection = select_literal_passages(
                client, guideline, pages, assertion
            )
            repair_job = selection["job"]
        notify(f"{job}: repaired citation for {owner['name']}")
        return {
            "owner_index": i,
            "evidence_index": j,
            "job": repair_job,
            "result": patch,
            "result_sha256": digest(patch),
            **({"selection": selection} if selection is not None else {}),
        }

    # Calls share the same endpoint semaphore and full context/output settings.
    unique = {}
    for target in pending:
        i, j, diagnostic = target
        owner = original_owners[i]
        key = digest(
            {"owner": owner, "rejected": owner["evidence"][j], "error": diagnostic}
        )
        unique.setdefault(key, []).append(target)
    with ThreadPoolExecutor(
        max_workers=min(len(unique), client.config.max_concurrent_requests)
    ) as pool:
        patches = list(pool.map(repair, [group[0] for group in unique.values()]))
    replacements = [
        {**patch, "owner_index": i, "evidence_index": j}
        for patch, group in zip(patches, unique.values())
        for i, j, _ in group
    ]
    result = apply_replacements(value, replacements)
    validator(result)
    atomic_json(
        output / "quote_repairs" / (digest(value) + ".json"),
        {
            "version": VERSION,
            "original_sha256": digest(value),
            "result_sha256": digest(result),
            "replacements": replacements,
        },
    )
    return result


def audit_repair(original, repaired, receipt, accepted_repairs, pages):
    """Tie every replacement to a validated, raw-provider-backed repair response."""
    if (
        receipt.get("version") != VERSION
        or receipt.get("original_sha256") != digest(original)
        or receipt.get("result_sha256") != digest(repaired)
    ):
        raise ValueError("Excerpt repair receipt differs from its clinical record")
    diagnostics = {(i, j): error for i, j, error in targets(original, pages)}
    expected = set(diagnostics)
    replacements = receipt["replacements"]
    if {(r["owner_index"], r["evidence_index"]) for r in replacements} != expected:
        raise ValueError(
            "Excerpt repair must account for exactly the invalid citations"
        )
    for item in replacements:
        validate_replacement(item["result"], pages)
        if digest(item["result"]) != item["result_sha256"]:
            raise ValueError("Excerpt repair lacks an accepted model response")
        if "selection" in item:
            i, j = item["owner_index"], item["evidence_index"]
            if item["job"] != item["selection"]["job"]:
                raise ValueError("Passage selection job changed")
            audit_selection(
                item["selection"],
                item["result"],
                repair_assertion(original, i, j, diagnostics[(i, j)]),
                accepted_repairs,
                pages,
            )
        elif (item["job"], item["result_sha256"]) not in accepted_repairs:
            raise ValueError("Excerpt repair lacks an accepted model response")
    if apply_replacements(original, replacements) != repaired:
        raise ValueError(
            "Excerpt repair changed unapproved clinical fields or citations"
        )
