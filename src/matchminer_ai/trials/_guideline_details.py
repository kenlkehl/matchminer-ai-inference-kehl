"""Keep the fixed population and tentative source menus adjacent to the final task."""

import json

from . import _guideline_prompts as prompts
from ._guideline_canonical import LEAN_STATE, VERSION, clinical_candidate
from ._guideline_context import pack_messages
from ._guideline_schema import format_space
from ._guideline_quotes import QUOTED_DETAIL


def candidate_menus(group, candidates):
    """Page overlap supplies review material, never an inferred applicability mapping."""
    pages = {e["page_id"] for e in group["evidence"]}
    selected, seen = [], set()
    for candidate in candidates:
        candidate_pages = {e["page_id"] for e in candidate["evidence"]}
        if "defining_branch" in candidate:
            candidate_pages.add(candidate["defining_branch"]["page_id"])
        if not pages.intersection(candidate_pages):
            continue
        value = clinical_candidate(candidate)
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            selected.append(value)
            seen.add(key)
    return selected


def build_detail_call(client, guideline, group, candidates, context_chars=None):
    payload = {
        "guideline": guideline.metadata["title"],
        "version": guideline.metadata["version"],
        "catalog_version": VERSION,
    }
    clinical = {
        "canonical_group": {k: group[k] for k in LEAN_STATE["properties"]},
        "tentative_candidates": candidate_menus(group, candidates),
    }
    tail = (
        "FINAL FIXED POPULATION AND CANDIDATE MENUS TO REVIEW:\n"
        + json.dumps(clinical, ensure_ascii=False)
        + "\n\n"
        + prompts.DETAIL_TASK
    )
    return pack_messages(
        client,
        guideline,
        {e["page_id"] for e in group["evidence"]},
        format_space(group["space"]),
        prompts.DETAIL_TASK,
        payload,
        QUOTED_DETAIL,
        context_chars,
        tail=tail,
    )
