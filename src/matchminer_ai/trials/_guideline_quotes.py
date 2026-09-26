"""Resolve final citations from exact source excerpts instead of model-selected offsets."""

import copy
import re

from ._guideline_schema import (
    STATE,
    STRING,
    arr,
    materialize_evidence,
    obj,
    validate_shape,
    validate_state,
)

VERSION = "source-excerpts-v1"
QUOTED_EVIDENCE = arr(obj({"page_id": STRING, "source_text": STRING}))
QUOTED_DETAIL = copy.deepcopy(STATE)
QUOTED_DETAIL["properties"]["evidence"] = QUOTED_EVIDENCE
for _kind in ("diagnostic_workup", "treatment_options"):
    QUOTED_DETAIL["properties"][_kind]["items"]["properties"]["evidence"] = (
        QUOTED_EVIDENCE
    )


def resolve_excerpt(page, excerpt):
    """Match whitespace only; never fuzzy-match words or guess an ambiguous location."""
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise ValueError("Citation source_text must be a nonempty verbatim excerpt")
    pattern = re.compile(r"\s+".join(re.escape(word) for word in excerpt.split()))
    matches = list(pattern.finditer(page.text))
    if not matches:
        raise ValueError(
            f"{page.id}: source_text does not occur verbatim on this supplied page: "
            f"{excerpt[:240]!r}. Copy exact source words; do not paraphrase, add ellipses, "
            "or join text across the other PDF column. Use separate excerpts for wrapped lines."
        )
    if len(matches) != 1:
        raise ValueError(
            f"{page.id}: source_text occurs {len(matches)} times; include more surrounding "
            "source words to identify the intended branch unambiguously."
        )
    match = matches[0]
    first = page.text.count("\n", 0, match.start()) + 1
    last = page.text.count("\n", 0, match.end() - 1) + 1
    if last - first >= 12:
        raise ValueError(f"{page.id}: split excerpts spanning more than 12 lines")
    return list(range(first, last + 1)), match.group()


def _owners(state):
    yield state
    for kind in ("diagnostic_workup", "treatment_options"):
        yield from state[kind]


def materialize_quoted_state(state, pages):
    """Derive addresses and displayed quotations in code, preserving all clinical fields."""
    validate_shape(state, QUOTED_DETAIL)
    result = copy.deepcopy(state)
    quotes = []
    errors = []
    for owner in _owners(result):
        refs, owner_quotes = [], []
        for number, item in enumerate(owner["evidence"], 1):
            try:
                if item["page_id"] not in pages:
                    raise ValueError(f"Unknown or unsupplied evidence page: {item['page_id']}")
                ids, quote = resolve_excerpt(pages[item["page_id"]], item["source_text"])
            except ValueError as exc:
                errors.append(f"{owner['name']} excerpt {number}: {exc}")
                continue
            refs.append({"page_id": item["page_id"], "line_ids": ids})
            owner_quotes.append(quote)
        owner["evidence"] = refs
        quotes.append(owner_quotes)
    if errors:
        raise ValueError("Correct every invalid excerpt in the draft:\n" + "\n".join(errors))
    validate_state(result, pages)
    result = materialize_evidence(result, pages)
    for owner, owner_quotes in zip(_owners(result), quotes):
        for evidence, quote in zip(owner["evidence"], owner_quotes):
            # Full original lines remain in source_lines for provenance. Display
            # the verified excerpt so neighboring PDF columns are not interleaved.
            evidence["quote"] = quote
    return result
