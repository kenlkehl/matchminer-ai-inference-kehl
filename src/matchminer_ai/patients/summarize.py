"""Patient summarization logic."""

from __future__ import annotations

import os
from typing import Any, cast

import pandas as pd
from transformers import AutoTokenizer

from matchminer_ai.llm.backends import (
    build_summarization_runtime_config,
    get_summarization_backend,
)
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.prompt_rendering import Prompt

from .postprocess import postprocess_patient_summaries
from .prepare import prepare_patient_notes, validate_note_inputs
from .prompt_builder import (
    PromptWorkItem,
    build_prompt_worker,
    prep_prompt_pool,
    shutdown_prompt_pool,
)


def validate_existing_summaries(
    existing_summaries: pd.DataFrame,
    *,
    boilerplate_marker: str = "Boilerplate conditions:",
) -> pd.DataFrame:
    """Validate and normalize existing patient summary state."""
    required_columns = ["patient_id", "last_note_date"]
    missing = [
        column
        for column in required_columns
        if column not in existing_summaries.columns
    ]
    if missing:
        raise ValueError(
            "existing summaries input must include columns "
            "'patient_id' and 'last_note_date'. Missing: "
            f"{', '.join(missing)}"
        )

    normalized = existing_summaries.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)

    has_combined_column = "patient_summary_with_boilerplate" in normalized.columns
    package_split_columns = [
        "cancer_history_summary",
        "general_exclusion_criteria_evidence",
    ]
    training_split_columns = [
        "patient_summary",
        "patient_boilerplate_text",
    ]
    has_package_split_columns = all(
        column in normalized.columns for column in package_split_columns
    )
    has_training_split_columns = all(
        column in normalized.columns for column in training_split_columns
    )
    has_legacy_combined_column = (
        "patient_summary" in normalized.columns and not has_training_split_columns
    )
    if (
        not has_combined_column
        and not has_package_split_columns
        and not has_training_split_columns
        and not has_legacy_combined_column
    ):
        raise ValueError(
            "existing summaries input must include either "
            "'patient_summary_with_boilerplate', legacy 'patient_summary', or "
            "'cancer_history_summary' and 'general_exclusion_criteria_evidence', "
            "or 'patient_summary' and 'patient_boilerplate_text'."
        )

    if has_combined_column:
        normalized["patient_summary_with_boilerplate"] = normalized[
            "patient_summary_with_boilerplate"
        ].where(
            normalized["patient_summary_with_boilerplate"].notna(),
            None,
        )
    elif has_package_split_columns:
        cancer_summary = normalized["cancer_history_summary"].fillna("").astype(str)
        boilerplate = normalized["general_exclusion_criteria_evidence"].fillna(
            ""
        ).astype(str)
        normalized["patient_summary_with_boilerplate"] = (
            cancer_summary.str.strip()
            + "\n\n"
            + boilerplate_marker
            + "\n"
            + boilerplate.str.strip()
        ).str.strip()
    elif has_training_split_columns:
        cancer_summary = normalized["patient_summary"].fillna("").astype(str)
        boilerplate = normalized["patient_boilerplate_text"].fillna("").astype(str)
        normalized["patient_summary_with_boilerplate"] = (
            cancer_summary.str.strip()
            + "\n\n"
            + boilerplate_marker
            + "\n"
            + boilerplate.str.strip()
        ).str.strip()
    else:
        normalized["patient_summary_with_boilerplate"] = normalized[
            "patient_summary"
        ].where(
            normalized["patient_summary"].notna(),
            None,
        )

    parsed_dates = pd.to_datetime(normalized["last_note_date"], errors="coerce")
    invalid_dates = normalized.loc[parsed_dates.isna(), "patient_id"].astype(str)
    if not invalid_dates.empty:
        raise ValueError(
            "existing summaries input has invalid or missing last_note_date for "
            f"patient_id(s): {', '.join(sorted(invalid_dates.unique()))}"
        )
    normalized["_last_note_date"] = parsed_dates.dt.normalize()
    normalized["last_note_date"] = parsed_dates.dt.date.astype(str)

    normalized["patient_summary_with_boilerplate"] = normalized[
        "patient_summary_with_boilerplate"
    ].where(
        normalized["patient_summary_with_boilerplate"].notna(),
        None,
    )
    return normalized.drop_duplicates(subset=["patient_id"], keep="last")[
        [
            "patient_id",
            "last_note_date",
            "_last_note_date",
            "patient_summary_with_boilerplate",
        ]
    ]


def _empty_existing_summary_state() -> pd.DataFrame:
    """Return an empty canonical existing-summary table."""
    return pd.DataFrame(
        columns=[
            "patient_id",
            "last_note_date",
            "_last_note_date",
            "patient_summary_with_boilerplate",
        ]
    )


def _filter_notes_newer_than_existing_summaries(
    notes: pd.DataFrame,
    existing_summary_state: pd.DataFrame,
) -> pd.DataFrame:
    """Keep all new-patient notes and only newer notes for existing patients."""
    normalized_notes = validate_note_inputs(notes)
    if normalized_notes.empty or existing_summary_state.empty:
        return normalized_notes

    cutoff_by_patient = existing_summary_state.set_index("patient_id")[
        "_last_note_date"
    ].to_dict()
    note_dates = normalized_notes["note_date"].dt.normalize()
    cutoff_dates = pd.to_datetime(normalized_notes["patient_id"].map(cutoff_by_patient))
    keep = cutoff_dates.isna() | (note_dates > cutoff_dates)
    return normalized_notes.loc[keep].copy()


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


def _build_prompt_list(
    round_df: pd.DataFrame,
    *,
    current_summaries: dict[str, str | None],
    prompt_pool: Any,
    n_prompt_workers: int,
) -> tuple[list[Prompt], list[str]]:
    work_items: list[PromptWorkItem] = []
    round_patient_ids: list[str] = []
    for row_idx, (_, row) in enumerate(round_df.iterrows()):
        patient_id = str(row["patient_id"])
        round_patient_ids.append(patient_id)
        work_items.append(
            PromptWorkItem(
                row_idx=row_idx,
                prior_summary_text=current_summaries.get(patient_id),
                first_date=str(row["first_date"]),
                last_date=str(row["last_date"]),
                chunk_text=str(row["chunk_text"]),
            )
        )

    chunksize = max(1, len(work_items) // (n_prompt_workers * 4)) if work_items else 1
    return (
        list(prompt_pool.map(build_prompt_worker, work_items, chunksize=chunksize)),
        round_patient_ids,
    )


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
        last_note_date : str or datetime
            Date of the last note included in the existing summary.
        patient_summary_with_boilerplate : str, optional
            Existing full patient summary text, including the boilerplate
            conditions section. Legacy ``patient_summary`` is also accepted.
        cancer_history_summary : str, optional
            Existing cancer history summary. Required only when
            ``patient_summary_with_boilerplate`` is not provided.
        general_exclusion_criteria_evidence : str, optional
            Existing boilerplate/exclusion evidence. Required only when
            ``patient_summary_with_boilerplate`` is not provided.
        patient_summary : str, optional
            Existing cancer history summary when paired with
            ``patient_boilerplate_text``; otherwise accepted as a legacy
            combined-summary alias.
        patient_boilerplate_text : str, optional
            Existing boilerplate/exclusion evidence from training-style output.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this summarization step.
    """
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    patient_config = dict(resolved_config.patient)
    runtime_patient_config = build_summarization_runtime_config(
        "patient",
        patient_config,
        config=resolved_config,
    )
    boilerplate_marker = str(patient_config["boilerplate_marker"])
    existing_summary_state = (
        validate_existing_summaries(
            existing_summaries,
            boilerplate_marker=boilerplate_marker,
        )
        if existing_summaries is not None
        else _empty_existing_summary_state()
    )
    filtered_notes = _filter_notes_newer_than_existing_summaries(
        notes,
        existing_summary_state,
    )

    prepared_patients = pd.DataFrame(columns=["patient_id", "last_note_date"])
    rounds: list[pd.DataFrame] = []
    if not filtered_notes.empty:
        # Convert note-level input into patient-level metadata plus chunk-level
        # rows. The chunk rows are what drive the serial summarization loop.
        tokenizer = AutoTokenizer.from_pretrained(
            patient_config["model_name"],
            trust_remote_code=True,
        )
        prepared_patients, prepared_chunks = prepare_patient_notes(
            filtered_notes,
            tokenizer,
            chunk_size=int(patient_config["chunk_size"]),
            chunk_overlap=int(patient_config["chunk_overlap"]),
        )
        rounds = _build_rounds(prepared_chunks)

    # This dict holds the latest available summary for each patient. If the
    # caller provided an existing summary, that is used for round 1; after each
    # round, the newly generated summary overwrites the prior one.
    current_summaries = cast(
        dict[str, str | None],
        existing_summary_state.set_index("patient_id")[
            "patient_summary_with_boilerplate"
        ].to_dict(),
    )
    current_raw_outputs: dict[str, str] = {}
    current_reasoning_outputs: dict[str, str] = {}
    generated_patient_ids: set[str] = set()
    model_metadata: dict[str, Any] = {}
    prompt_pool = None

    # Round N contains the Nth chunk for every patient that still has one.
    # Processing by rounds ensures each patient's next chunk sees the most
    # recent summary generated from prior chunks.
    n_prompt_workers = max(
        1,
        int(patient_config.get("prompt_build_workers", min(os.cpu_count() or 4, 32))),
    )
    try:
        if rounds:
            backend = get_summarization_backend(resolved_config)
            prompt_pool = prep_prompt_pool(
                patient_config=runtime_patient_config,
                n_workers=n_prompt_workers,
            )

        for round_df in rounds:
            prompt_list, round_patient_ids = _build_prompt_list(
                round_df,
                current_summaries=current_summaries,
                prompt_pool=prompt_pool,
                n_prompt_workers=n_prompt_workers,
            )
            generation = backend.generate_llm_outputs(
                prompt_list=prompt_list,
                llm_config=runtime_patient_config,
                model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
            )
            if not model_metadata:
                model_metadata = generation.model_metadata
            summaries = generation.final_outputs
            has_raw_outputs = bool(generation.raw_outputs)
            raw_outputs = generation.raw_outputs if has_raw_outputs else summaries
            reasoning_outputs = generation.reasoning_outputs
            # Persist each round's final summary, not the reasoning trace, so
            # it becomes the prior summary for the next patient chunk.
            for patient_id, summary, raw_output, reasoning in zip(
                round_patient_ids,
                summaries,
                raw_outputs,
                reasoning_outputs,
                strict=False,
            ):
                current_summaries[patient_id] = str(summary)
                generated_patient_ids.add(patient_id)
                if has_raw_outputs:
                    current_raw_outputs[patient_id] = str(raw_output)
                current_reasoning_outputs[patient_id] = str(reasoning)
    finally:
        if prompt_pool is not None:
            shutdown_prompt_pool(prompt_pool)

    # Collapse generated patient state and unchanged prior summaries back to one
    # final row per patient, then do postprocessing and QC report generation.
    generated_rows = prepared_patients[
        prepared_patients["patient_id"].isin(generated_patient_ids)
    ].copy()
    generated_rows["original_patient_summary"] = generated_rows["patient_id"].map(
        current_summaries
    )
    if resolved_config.debug_mode:
        # These columns preserve final-round debug traces without feeding them
        # back into the serial patient summary state.
        if current_raw_outputs:
            generated_rows["final_round_patient_summary_raw_output"] = generated_rows[
                "patient_id"
            ].map(current_raw_outputs)
        generated_rows["final_round_patient_summary_reasoning"] = generated_rows[
            "patient_id"
        ].map(current_reasoning_outputs)
    generated_rows = generated_rows.dropna(subset=["original_patient_summary"]).copy()

    generated_rows, noninformative_summary_qc_artifact = postprocess_patient_summaries(
        generated_rows,
        resolved_config,
    )
    pass_through_state = existing_summary_state[
        ~existing_summary_state["patient_id"].isin(generated_patient_ids)
    ].copy()
    pass_through_rows = pd.DataFrame(
        columns=[
            "patient_id",
            "last_note_date",
            "original_patient_summary",
        ]
    )
    if not pass_through_state.empty:
        pass_through_rows = pass_through_state[
            [
                "patient_id",
                "last_note_date",
                "patient_summary_with_boilerplate",
            ]
        ].rename(
            columns={
                "patient_summary_with_boilerplate": "original_patient_summary",
            }
        )
        pass_through_rows, _ = postprocess_patient_summaries(
            pass_through_rows,
            resolved_config,
            drop_noninformative=False,
        )

    final_rows = pd.concat(
        [generated_rows, pass_through_rows],
        ignore_index=True,
        sort=False,
    )

    metadata = {
        "config_snapshot": config_snapshot(resolved_config),
        "model_metadata": model_metadata,
    }

    if return_qc:
        from matchminer_ai._qc.patients import patient_summary_qc_report

        qc_report = patient_summary_qc_report(
            final_rows,
            noninformative_summary_qc_artifact=noninformative_summary_qc_artifact,
            config=resolved_config,
        )
        return final_rows, metadata, qc_report
    return final_rows, metadata


__all__ = [
    "summarize_patient_notes",
    "validate_existing_summaries",
]
