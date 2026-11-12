"""Note-table normalization and exact-text chunking for structured extraction.

Notes keep their original text: nothing is deduplicated, reflowed or decoded
from tokens, so every evidence quote is located in, and offset into, the note
it came from.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pandas as pd

#: Tokens allowed for each note header in a patient chunk.
HEADER_TOKENS = 32


def normalize_notes(notes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validated notes, sorted by patient and date, with code-derived ids.

    ``notes`` needs ``patient_id``, ``note_text`` and ``note_date``; ``note_id``
    and ``note_type`` are used when present. Rows without text are dropped. A
    missing or blank date stays unknown (None); an unparseable one is an error.
    Dates become ``YYYY-MM-DD``. Without a ``note_id`` column each note gets a
    stable id from a hash of its patient, date, type and text, and exact
    duplicate notes are kept once. Returns the notes and counts of dropped rows.
    """
    if not isinstance(notes, pd.DataFrame):
        raise TypeError("notes must be a pandas DataFrame")
    missing = [c for c in ("patient_id", "note_text", "note_date") if c not in notes]
    if missing:
        raise ValueError(f"notes must include columns {missing}")
    frame = notes.copy()
    if frame.patient_id.isna().any() or frame.patient_id.astype(str).str.strip().eq("").any():
        raise ValueError("Every note needs a patient_id.")
    frame["patient_id"] = frame.patient_id.astype(str)
    has_text = frame.note_text.map(lambda t: isinstance(t, str) and bool(t.strip()))
    counts = {"notes_without_text": int((~has_text).sum())}
    frame = frame.loc[has_text].copy()
    if "note_type" not in frame:
        frame["note_type"] = None
    if any(not pd.isna(t) and not isinstance(t, str) for t in frame.note_type):
        raise ValueError("note_type must contain strings or missing values.")
    frame["note_type"] = [
        None if pd.isna(t) or not str(t).strip() else str(t).strip() for t in frame.note_type
    ]
    supplied = frame.note_date.notna() & frame.note_date.astype(str).str.strip().ne("")
    parsed = pd.to_datetime(frame.note_date.where(supplied), errors="coerce", format="mixed")
    if (parsed.isna() & supplied).any():
        raise ValueError("Invalid note_date; leave unavailable dates empty.")
    frame["note_date"] = [None if pd.isna(d) else d.strftime("%Y-%m-%d") for d in parsed]
    if "note_id" in frame:
        ids = frame.note_id
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
            raise ValueError("note_id must be present for every note when supplied.")
        frame["note_id"] = ids.astype(str).str.strip()
        if frame.note_id.duplicated().any():
            raise ValueError("note_id must be unique.")
        counts["duplicate_notes"] = 0
    else:
        frame["note_id"] = [
            "note-" + hashlib.sha256(
                "\x1f".join([p, d or "", t or "", x]).encode("utf-8")
            ).hexdigest()[:16]
            for p, d, t, x in zip(
                frame.patient_id, frame.note_date, frame.note_type, frame.note_text,
                strict=True,
            )
        ]
        before = len(frame)
        frame = frame.drop_duplicates("note_id")
        counts["duplicate_notes"] = before - len(frame)
    frame["_order"] = range(len(frame))
    frame = frame.sort_values(
        ["patient_id", "note_date", "_order"], kind="stable", na_position="last"
    ).drop(columns="_order")
    return frame.reset_index(drop=True), counts


def patient_note_list(group: pd.DataFrame) -> list[dict]:
    """One patient's normalized notes as dicts numbered from 1 in date order."""
    return [
        {
            "note_number": number,
            "note_id": row.note_id,
            "note_date": row.note_date,
            "note_type": row.note_type,
            "text": row.note_text,
        }
        for number, row in enumerate(group.itertuples(index=False), 1)
    ]


def note_fragments(notes: Sequence[dict], tokenizer, size: int, overlap: int) -> list[dict]:
    """Split notes longer than ``size`` tokens into overlapping fragments.

    Fragments are slices of the original text at token boundaries (tokenizer
    character offsets), with ``offset`` the fragment's start in its note.
    """
    if not 0 <= overlap < size:
        raise ValueError("chunk overlap must be at least 0 and less than the chunk size")
    fragments = []
    for note in notes:
        offsets = tokenizer(
            note["text"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        if len(offsets) <= size:
            fragments.append({**note, "offset": 0, "tokens": len(offsets)})
            continue
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            left = 0 if start == 0 else offsets[start][0]
            right = len(note["text"]) if end == len(offsets) else offsets[end][0]
            fragments.append(
                {**note, "text": note["text"][left:right], "offset": left, "tokens": end - start}
            )
            if end == len(offsets):
                break
    return fragments


def pack_fragments(fragments: Sequence[dict], size: int) -> list[list[dict]]:
    """Pack fragments in order into chunks of at most ``size`` tokens (headers
    included); a fragment never spans two chunks."""
    chunks, chunk, count = [], [], 0
    for fragment in fragments:
        tokens = fragment["tokens"] + HEADER_TOKENS
        if chunk and count + tokens > size:
            chunks.append(chunk)
            chunk, count = [], 0
        chunk.append(fragment)
        count += tokens
    if chunk:
        chunks.append(chunk)
    return chunks


def note_header(note: dict) -> str:
    """The header the prompt shows above a note (or note fragment)."""
    parts = [f"Note {note['note_number']}", note["note_date"] or "date unknown"]
    if note.get("note_type"):
        parts.append(f"type: {note['note_type']}")
    if note.get("offset"):
        parts.append("continued")
    return "[" + " | ".join(parts) + "]"


def render_chunk(chunk: Sequence[dict]) -> str:
    """A chunk's notes with their headers, in order."""
    return "\n\n".join(f"{note_header(f)}\n{f['text']}" for f in chunk)


__all__ = [
    "normalize_notes",
    "note_fragments",
    "note_header",
    "pack_fragments",
    "patient_note_list",
    "render_chunk",
]
