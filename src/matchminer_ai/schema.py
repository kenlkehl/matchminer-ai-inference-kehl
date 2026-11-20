"""Column names at the boundary between training's tables and the package's runtime tables.

Training keeps research-era names; the public stage functions use clearer ones. These maps are the one place both
sides agree on (AGENTS.md, "Training-to-inference schema boundary"):

==========================  ======================================
training                    runtime
==========================  ======================================
``pseudo_mrn``              ``patient_id``
``patient_summary``         ``cancer_history_summary``
``patient_boilerplate_text``  ``general_exclusion_criteria_evidence``
``nct_id``                  ``trial_id``
``this_space``              ``clinical_space_summary``
``trial_boilerplate_text``  ``general_exclusion_criteria``
==========================  ======================================

Synthetic notes (``all_synthetic_notes.parquet``) name a patient's notes ``pseudo_mrn`` / ``synthetic_note`` /
``date`` / ``event_type``; runtime notes are ``patient_id`` / ``note_text`` / ``note_date`` / ``note_type``. A
synthetic table may also carry its own ``patient_id`` (the generator's UUID): the synthetic layout is recognized by
``pseudo_mrn``, which then keys the patient.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

TRAINING_TO_RUNTIME = {
    "pseudo_mrn": "patient_id",
    "patient_summary": "cancer_history_summary",
    "patient_boilerplate_text": "general_exclusion_criteria_evidence",
    "nct_id": "trial_id",
    "this_space": "clinical_space_summary",
    "trial_boilerplate_text": "general_exclusion_criteria",
}
SYNTHETIC_NOTES_TO_RUNTIME = {
    "pseudo_mrn": "patient_id",
    "synthetic_note": "note_text",
    "date": "note_date",
    "event_type": "note_type",
}
RUNTIME_NOTE_COLUMNS = ("patient_id", "note_text", "note_date", "note_type")


def with_runtime_columns(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """A copy of ``frame`` that also has the runtime name of each training column present, where that runtime
    column is absent (training columns are kept). ``columns`` limits it to some training names."""
    out = frame.copy()
    for training in columns or TRAINING_TO_RUNTIME:
        runtime = TRAINING_TO_RUNTIME[training]
        if runtime not in out.columns and training in out.columns:
            out[runtime] = out[training]
    return out


def note_columns(names: Iterable[str]) -> tuple[str, dict[str, str]]:
    """For a notes table with these column names: its patient key column and ``{source: runtime}`` for its layout's
    note columns (synthetic if it has ``pseudo_mrn``, else runtime). Columns are not checked for presence."""
    if "pseudo_mrn" in set(names):
        return "pseudo_mrn", dict(SYNTHETIC_NOTES_TO_RUNTIME)
    return "patient_id", {c: c for c in RUNTIME_NOTE_COLUMNS}


__all__ = ["RUNTIME_NOTE_COLUMNS", "SYNTHETIC_NOTES_TO_RUNTIME", "TRAINING_TO_RUNTIME", "note_columns",
           "with_runtime_columns"]
