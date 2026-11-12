"""Ground model-written evidence quotes in the source note text, in code.

A quote is accepted only when it occurs in the source: exactly, or with
different whitespace (line breaks and repeated spaces in clinical text are
common, and the JSON grammar makes the model write them as spaces). Words are
never fuzzy-matched, and the record keeps the source's own text at the matched
offsets, never the model's copy.
"""

from __future__ import annotations

import re


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """``(start, end)`` of the first occurrence of ``quote`` in ``text``.

    Matches exactly first, then with any run of whitespace in the quote matching
    any run of whitespace in the text. Returns None when it does not occur.
    """
    stripped = quote.strip()
    if not stripped:
        return None
    start = text.find(stripped)
    if start >= 0:
        return start, start + len(stripped)
    pattern = r"\s+".join(re.escape(word) for word in stripped.split())
    match = re.search(pattern, text)
    return (match.start(), match.end()) if match else None


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
