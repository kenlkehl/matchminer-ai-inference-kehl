"""Bounded source-address feedback for failed extraction, without editing its output."""

import json

from ._guideline_schema import EXTRACTION


def source_address_retry_allowed(error):
    """An invalid address is distinct from a branch belonging to a different owner."""
    return (
        "exhausted" in error
        and "Extraction ownership: invalid defining_branch:" in error
        and any(term in error for term in (
            "nonexistent source line IDs",
            "evidence may not consist entirely of blank lines",
            "cite 1-12 distinct source line numbers per evidence item",
        ))
    )


def owned_address_schema(primary):
    """Constrain addressing, never the number or clinical definitions of populations."""
    # The base schema deliberately shares STRING/EVIDENCE dictionaries. A JSON
    # round trip breaks those aliases so page enums cannot constrain names,
    # clinical fields or citations to supporting context pages.
    schema = json.loads(json.dumps(EXTRACTION))
    page_ids = [p.id for p in primary]
    line_ids = sorted({
        i for p in primary for i, line in enumerate(p.text.splitlines(), 1)
        if line.strip()
    })
    if not page_ids or not line_ids:
        raise ValueError("Owned source has no nonblank lines for an address retry")
    fields = schema['properties']['candidates']['items']['properties']['defining_branch']['properties']
    fields['page_id']['enum'] = page_ids
    fields['line_ids']['items']['enum'] = line_ids
    # With several primary pages this is the union of real addresses. Ordinary
    # extraction validation still checks the actual page/line combination.
    coverage = schema['properties']['page_coverage']
    coverage.update(minItems=len(page_ids), maxItems=len(page_ids))
    coverage['items']['properties']['page_id']['enum'] = page_ids
    return schema
