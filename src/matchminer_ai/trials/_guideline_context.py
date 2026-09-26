"""Pack whole pages against the actual endpoint chat tokenizer and reserved output budget."""

import json

from . import _guideline_prompts as prompts
from ._guideline_sources import render_pages, select_context


class ContextBudgetError(ValueError):
    """Mandatory source pages and payload cannot fit while preserving the output reserve."""


def pack_messages(
    client,
    guideline,
    required_ids,
    query,
    task,
    payload,
    schema,
    context_chars=None,
    primary_ids=(),
    *,
    tail="",
    population_guidance=True,
    source_ids=None,
):
    required_ids = set(required_ids)
    if source_ids is not None and not required_ids <= set(source_ids):
        raise ValueError("Source scope must contain every required evidence page")
    ordered, _ = select_context(
        guideline, required_ids, query, context_chars, priority_order=True
    )
    if source_ids is not None:
        ordered = [p for p in ordered if p.id in source_ids]

    def build(count):
        selected = sorted(ordered[:count], key=lambda p: p.number)
        present = {p.id for p in selected}
        omitted = [p.id for p in guideline.primary if p.id not in present]
        data = {**payload, "omitted_source_page_ids": omitted}
        rules = (
            "\n\nFINAL POPULATION RULES:\n"
            + prompts.population_rules(guideline.metadata["title"])
            if population_guidance
            else ""
        )
        messages = prompts.messages(
            task,
            json.dumps(data, ensure_ascii=False)
            + "\n"
            + render_pages(selected, primary_ids)
            + ("\n\n" + tail if tail else "")
            + rules,
            schema,
        )
        return selected, omitted, messages

    # Usually the entire small/medium guideline fits. Do not impose arbitrary character caps.
    all_pages, omitted, full = build(len(ordered))
    if client.fits(full):
        return all_pages, omitted, full
    low, high = len(required_ids), len(ordered)
    _, _, minimum = build(low)
    if not client.fits(minimum):
        raise ContextBudgetError(
            "Required evidence and instructions exceed input budget with the configured "
            "output reserve; reduce guideline.packet_pages or use a larger-context endpoint"
        )
    # Whole-page, priority-prefix packing: never cut a page, quote, or disease branch.
    while low + 1 < high:
        middle = (low + high) // 2
        _, _, candidate = build(middle)
        if client.fits(candidate):
            low = middle
        else:
            high = middle
    return build(low)
