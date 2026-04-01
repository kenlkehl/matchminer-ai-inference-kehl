"""Patient summarization logic."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from transformers import AutoTokenizer

from mmai.backends import get_backend
from mmai.config import MMAIConfig, load_default_preset

from .postprocess import postprocess_patient_summaries
from .prepare import prepare_patient_notes
from .prompt_builder import get_serial_patient_prompt


def validate_existing_summaries(
    existing_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize existing patient summary state."""
    required_columns = ["patient_id", "patient_summary"]
    missing = [
        column
        for column in required_columns
        if column not in existing_summaries.columns
    ]
    if missing:
        raise ValueError(
            "existing summaries input must include columns "
            "'patient_id' and 'patient_summary'. Missing: "
            f"{', '.join(missing)}"
        )

    normalized = existing_summaries.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["patient_summary"] = normalized["patient_summary"].where(
        normalized["patient_summary"].notna(),
        None,
    )
    return normalized.drop_duplicates(subset=["patient_id"], keep="last")


def _build_existing_summary_lookup(
    existing_summaries: pd.DataFrame | None,
) -> dict[str, str | None]:
    if existing_summaries is None:
        return {}
    normalized = validate_existing_summaries(existing_summaries)
    return cast(
        dict[str, str | None],
        normalized.set_index("patient_id")["patient_summary"].to_dict(),
    )


def _build_rounds(prepared_chunks: pd.DataFrame) -> list[pd.DataFrame]:
    """Organize patient chunks into rounds by chunk index."""
    if prepared_chunks.empty:
        return []
    rounds: list[pd.DataFrame] = []
    ordered = prepared_chunks.sort_values(["chunk_index", "patient_id"]).reset_index(
        drop=True
    )
    for _, group in ordered.groupby("chunk_index", sort=True):
        rounds.append(group.reset_index(drop=True))
    return rounds


def summarize_patient_notes(
    notes: pd.DataFrame,
    config: MMAIConfig | None = None,
    *,
    existing_summaries: pd.DataFrame | None = None,
    return_qc: bool = False,
) -> (
    tuple[pd.DataFrame, dict[str, Any]]
    | tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]
):
    """
    Summarize longitudinal patient notes using serial chunk-based updates.

    Parameters
    ----------
    notes : pd.DataFrame
        Note-level input. One row per note.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        note_text : str
            Full note text.
        note_date : str or datetime
            Date of the note.
    existing_summaries : pd.DataFrame, optional
        Optional patient-level prior summaries used as the starting state for
        serial updates.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        patient_summary : str
            Existing full patient summary text to update.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this summarization step.
    """
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    patient_config = dict(resolved_config.patient)
    prompt_files = dict(patient_config["prompt_files"])
    primer_filename = prompt_files["primer"]
    question_filename = prompt_files["question"]

    # Convert note-level input into patient-level metadata plus chunk-level
    # rows. The chunk rows are what drive the serial summarization loop.
    tokenizer = AutoTokenizer.from_pretrained(patient_config["model_name"])
    prepared_patients, prepared_chunks = prepare_patient_notes(
        notes,
        tokenizer,
        chunk_size=int(patient_config["chunk_size"]),
        chunk_overlap=int(patient_config["chunk_overlap"]),
    )
    existing_summary_lookup = _build_existing_summary_lookup(existing_summaries)
    rounds = _build_rounds(prepared_chunks)

    backend = get_backend(resolved_config.backend)
    # This dict holds the latest available summary for each patient. If the
    # caller provided an existing summary, that is used for round 1; after each
    # round, the newly generated summary overwrites the prior one.
    current_summaries = {
        patient_id: summary for patient_id, summary in existing_summary_lookup.items()
    }
    model_metadata: dict[str, Any] = {}

    # Round N contains the Nth chunk for every patient that still has one.
    # Processing by rounds ensures each patient's next chunk sees the most
    # recent summary generated from prior chunks.
    for round_df in rounds:
        messages_list: list[list[dict[str, str]]] = []
        round_patient_ids: list[str] = []
        for _, row in round_df.iterrows():
            patient_id = str(row["patient_id"])
            round_patient_ids.append(patient_id)
            messages_list.append(
                get_serial_patient_prompt(
                    prior_summary=current_summaries.get(patient_id),
                    first_date=str(row["first_date"]),
                    last_date=str(row["last_date"]),
                    chunk_text=str(row["chunk_text"]),
                    tokenizer=tokenizer,
                    max_model_len=int(patient_config["max_model_len"]),
                    primer_filename=primer_filename,
                    question_filename=question_filename,
                    margin_tokens=int(patient_config["prompt_margin_tokens"]),
                    model_name=str(patient_config["model_name"]),
                )
            )

        summaries, round_model_metadata, _finish_reasons = backend.generate_llm_outputs(
            messages_list=messages_list,
            llm_config=patient_config,
            model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
        )
        if not model_metadata:
            model_metadata = round_model_metadata
        # Persist each round's output so it becomes the prior summary for the
        # next chunk from that same patient.
        for patient_id, summary in zip(round_patient_ids, summaries, strict=False):
            current_summaries[patient_id] = summary

    # Collapse the running patient state back to one final row per patient,
    # then do postprocessing and QC report generation.
    final_rows = prepared_patients.copy()
    final_rows["original_patient_summary"] = final_rows["patient_id"].map(
        current_summaries
    )
    final_rows = final_rows.dropna(subset=["original_patient_summary"]).copy()

    final_rows, noninformative_summary_qc_artifact = postprocess_patient_summaries(
        final_rows, resolved_config
    )

    metadata = {
        "config_snapshot": resolved_config.raw,
        "model_metadata": model_metadata,
    }

    from mmai._qc.patients import patient_summary_qc_report

    qc_report = patient_summary_qc_report(
        final_rows,
        noninformative_summary_qc_artifact=noninformative_summary_qc_artifact,
        config=resolved_config,
    )
    if return_qc:
        return final_rows, metadata, qc_report
    return final_rows, metadata


__all__ = [
    "summarize_patient_notes",
    "validate_existing_summaries",
]
