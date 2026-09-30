"""Normalize one patient's optional structured notes and navigation context."""

from dataclasses import dataclass
import json

import pandas as pd

from .workup import _notes


def history_from_notes(source):
    parts, spans, cursor = [], [], 0
    for note in source:
        kind = (
            " | type: " + json.dumps(note["note_type"], ensure_ascii=False)
            if note.get("note_type") is not None
            else ""
        )
        header = f"\n[Note {note['note_number']} | {note['note_date'] or 'date unavailable'}{kind}]\n"
        parts.extend([header, note["text"]])
        start = cursor + len(header)
        cursor = start + len(note["text"])
        spans.append({**note, "start": start, "end": cursor})
    return "".join(parts), spans


def source_excerpts(excerpts, spans):
    """Split original ranges at note boundaries; discard generated headers."""
    result = []
    for item in excerpts:
        for note in spans:
            start, end = (
                max(item["start"], note["start"]),
                min(item["end"], note["end"]),
            )
            if start < end:
                result.append(
                    {
                        "start": start,
                        "end": end,
                        "quote": note["text"][
                            start - note["start"] : end - note["start"]
                        ],
                        "note_number": note["note_number"],
                        "note_date": note["note_date"],
                        **(
                            {"note_type": note["note_type"]}
                            if note.get("note_type")
                            else {}
                        ),
                    }
                )
    return result


@dataclass
class NoteRecord:
    history: str
    spans: list[dict]
    patient_summary: str | None
    structured: bool
    text_supplied: bool

    def describe(self):
        dates = [n["note_date"] for n in self.spans if n["note_date"]]
        types = sorted({n["note_type"] for n in self.spans if n.get("note_type")})
        return {
            "rows": len(self.spans),
            "columns": ["note_number", "note_date", "note_type", "note_text"],
            "structured_input": self.structured,
            "date_range": [min(dates), max(dates)] if dates else None,
            "note_types": types[:50],
            "note_types_omitted": len(types) > 50,
            "undated_notes": sum(n["note_date"] is None for n in self.spans),
        }


def prepare_record(
    history=None,
    notes=None,
    patient_summary=None,
    *,
    max_bytes=32_000_000,
    max_summary_chars=100_000,
):
    if isinstance(history, pd.DataFrame):
        if notes is not None:
            raise ValueError("Supply the notes DataFrame only once.")
        notes, history = history, None
    if isinstance(notes, str):
        if history is not None and history != notes:
            raise ValueError("Supply concatenated text only once.")
        history, notes = notes, None
    if history is not None and not isinstance(history, str):
        raise TypeError("history must be concatenated text or a notes DataFrame.")
    if notes is not None and not isinstance(notes, pd.DataFrame):
        raise TypeError("notes must be a single-patient pandas DataFrame.")
    if patient_summary is not None and (
        not isinstance(patient_summary, str) or len(patient_summary) > max_summary_chars
    ):
        raise ValueError("patient_summary must be text within max_summary_chars.")
    patient_summary = (
        patient_summary if patient_summary and patient_summary.strip() else None
    )
    structured = notes is not None
    if structured:
        source = _notes(notes)
        full, spans = history_from_notes(source)
    elif history is not None and history.strip():
        # Preserve legacy string offsets exactly, including embedded headers.
        full = history
        spans = [
            {
                "note_number": 1,
                "note_date": None,
                "note_type": None,
                "text": history,
                "start": 0,
                "end": len(history),
            }
        ]
    else:
        raise ValueError("Supply nonempty notes or full-text history.")
    if len(full.encode("utf-8")) > max_bytes:
        raise ValueError("Supply notes/full-text history within max_history_bytes.")
    if any(len(n.get("note_type") or "") > 1000 for n in spans):
        raise ValueError("Note types must be at most 1000 characters.")
    return NoteRecord(full, spans, patient_summary, structured, history is not None)
