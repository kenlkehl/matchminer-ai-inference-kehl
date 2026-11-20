import pandas as pd

from matchminer_ai.schema import note_columns, with_runtime_columns


def test_runtime_columns_are_added_without_replacing_existing_ones():
    frame = pd.DataFrame(dict(pseudo_mrn=["1"], nct_id=["NCT1"], patient_summary=["history"], trial_id=["T"]))
    out = with_runtime_columns(frame)
    assert out.patient_id.tolist() == ["1"] and out.cancer_history_summary.tolist() == ["history"]
    assert out.trial_id.tolist() == ["T"] and "pseudo_mrn" in out
    assert "patient_id" not in frame


def test_synthetic_notes_are_keyed_by_pseudo_mrn_even_with_a_generator_uuid():
    key, mapping = note_columns(["patient_id", "pseudo_mrn", "synthetic_note", "date", "event_type"])
    assert key == "pseudo_mrn" and mapping["synthetic_note"] == "note_text"
    assert note_columns(["patient_id", "note_text"]) == (
        "patient_id", {c: c for c in ("patient_id", "note_text", "note_date", "note_type")})
