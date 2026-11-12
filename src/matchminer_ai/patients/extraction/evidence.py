"""Ground model-written evidence quotes in the source note text, in code.

A quote is accepted only when its words occur in the source in order. It may
differ from the source only in layout: whitespace and line breaks, letter case,
markdown emphasis and heading marks (``*``, ``_``, backticks, ``#``), list
markers at the start of a line, and typographic quotes and dashes. A JSON
answer cannot reproduce a note's layout reliably (its line breaks, bullets and
bold markup), and the model routinely drops it. Words are never fuzzy-matched,
and the record keeps the source's own text at the matched offsets, never the
model's copy.
"""

from __future__ import annotations

import re
from functools import lru_cache

#: Layout characters a quote may leave out.
_LAYOUT = frozenset("*_`#")
#: List markers at the start of a line ("- ", "* ", "• ", "1. ", "a) ").
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*+•·▪◦]|\d{1,3}[.)]|[a-zA-Z][.)])[ \t]+")
_TYPOGRAPHIC = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "‐": "-", "‑": "-", "−": "-", " ": " ",
})


@lru_cache(maxsize=256)
def _layout_free(text: str) -> tuple[str, tuple[int, ...]]:
    """``text`` without layout, lowercased, with one space per whitespace run,
    and the original position of each character kept."""
    markers = {m.start(): m.end() for m in _LIST_MARKER.finditer(text)}
    characters: list[str] = []
    positions: list[int] = []
    space = False
    index = 0
    translated = text.translate(_TYPOGRAPHIC)
    while index < len(text):
        if index in markers and markers[index] > index:
            index, space = markers[index], True
            continue
        character = translated[index]
        if character.isspace():
            space = True
        elif character not in _LAYOUT:
            if space and characters:
                characters.append(" ")
                positions.append(index)
            space = False
            characters.append(character.lower())
            positions.append(index)
        index += 1
    return "".join(characters), tuple(positions)


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """``(start, end)`` of the first occurrence of ``quote`` in ``text``, or None.

    An exact match wins; otherwise the quote's words must occur in order with
    only layout differences (see the module docstring). The span runs from the
    first to the last matched character of the source.
    """
    stripped = quote.strip()
    if not stripped:
        return None
    start = text.find(stripped)
    if start >= 0:
        return start, start + len(stripped)
    needle, _ = _layout_free(stripped)
    if not needle:
        return None
    haystack, positions = _layout_free(text)
    found = haystack.find(needle)
    if found < 0:
        return None
    return positions[found], positions[found + len(needle) - 1] + 1


def evidence_entry(note: dict, text: str, offset: int, span: tuple[int, int]) -> dict:
    """The record's evidence item for ``span`` of ``text``, a part of ``note``
    starting at character ``offset`` of the note's own text."""
    start, end = span
    return {
        "note_id": note["note_id"],
        "note_date": note["note_date"],
        "quote": text[start:end],
        "start": offset + start,
        "end": offset + end,
    }


__all__ = ["evidence_entry", "locate_quote"]
