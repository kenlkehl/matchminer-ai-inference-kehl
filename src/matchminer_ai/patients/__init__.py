"""Patient summarization workflows."""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset

from .compression import (
    NoteCompressionError,
    NoteCompressionProgress,
    compress_patient_note,
    compress_patient_notes,
)
from .colbert import (
    ColBERTConfig,
    ColBERTPatientIndex,
    colbert_notes_fingerprint,
    encode_patient_notes_colbert,
    load_colbert_patient_index,
    retrieve_patient_chunks_colbert,
    save_colbert_patient_index,
)
from .full_screen import (
    FullPatientScreenError,
    FullPatientScreenProgress,
    full_patient_screen,
)
from .note_search_qa import (
    NoteSearchLimits,
    NoteSearchLLMConfig,
    answer_patient_question_batch,
    answer_patient_questions,
)
from .pdf import (
    PatientPDFInput,
    PatientPDFProgress,
    concatenate_patient_note_pdfs,
)
from .raw_note_qa import (
    RawPatientNoteQAProgress,
    RawPatientNoteQuestionError,
    answer_question_with_raw_patient_notes,
)
from .structure import structure_patient_summaries, structure_patient_summary
from .summarize import summarize_patient_notes
from .workup import review_patient_workup
from .workup_colbert import review_patient_workup_with_colbert
from .workup_search import review_patient_workup_with_note_search
from .workup_search_review import WorkupSearchReviewConfig


def summarize_patients(
    notes: pd.DataFrame | PatientPDFInput,
    *,
    config: MMAIConfig | None = None,
    existing_summaries: pd.DataFrame | None = None,
    return_metadata: bool = False,
    return_qc: bool = False,
    patient_id: str = "pdf-patient",
    pdf_progress_callback: PatientPDFProgress | None = None,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, dict]
    | tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, dict, pd.DataFrame]
):
    """
    Summarize longitudinal patient notes into a cancer history summary and
    evidence related to general clinical trial exclusion criteria.

    Parameters
    ----------
    notes : pd.DataFrame, path-like, or sequence of path-like values
        Note-level input with one row per note, or one/more ordered local PDF
        paths for a single patient. PDF pages are converted locally with
        embedded-text extraction and RapidOCR fallback, then combined into one
        long patient-note string before serial summarization.

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
    return_metadata : bool, optional
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this run.
    patient_id : str, optional
        Patient identifier assigned when ``notes`` contains PDF path(s). Ignored
        for DataFrame input.
    pdf_progress_callback : callable or None, optional
        For PDF input, called after each processed page as
        ``callback(document_number, document_count, page_number, page_count,
        method)``.

    Returns
    -------
    pd.DataFrame
        Patient-level DataFrame. One row per patient.

        Columns
        -------
        patient_id : str
            Original patient identifier.
        cancer_history_summary : str
            Summary of the patient's cancer history.
        general_exclusion_criteria_evidence : str
            Summary of conditions / findings that correspond to common
            clinical trial exclusion criteria.

        Debug Columns
        (Available only if pipeline initialized with debug_mode=True)
        -------------------------------------------------------------
        patient_answer_text : str
            Text the package treated as the LLM answer and used for
            postprocessing.
        patient_reasoning_text : str
            Optional separate reasoning trace returned by the backend or
            extracted by the configured reasoning parser.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict.
    tuple[pd.DataFrame, pd.DataFrame]
        When return_qc is True, returns the DataFrame plus a QC report DataFrame.
    tuple[pd.DataFrame, dict, pd.DataFrame]
        When return_metadata and return_qc are True, returns the DataFrame,
        metadata dict, and QC report DataFrame.

    """
    logger = logging.getLogger(__name__)
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    if not isinstance(notes, pd.DataFrame):
        normalized_patient_id = str(patient_id).strip()
        if not normalized_patient_id:
            raise ValueError("patient_id must not be empty for PDF input.")
        long_note = concatenate_patient_note_pdfs(
            notes,
            progress_callback=pdf_progress_callback,
        )
        notes = pd.DataFrame(
            [
                {
                    "patient_id": normalized_patient_id,
                    "note_text": long_note,
                    # PDFs can contain many internally dated records. Do not
                    # invent a clinical note date from a filename or mtime.
                    "note_date": pd.NaT,
                }
            ]
        )

    required_columns = [
        "patient_id",
        "note_text",
        "note_date",
    ]
    missing = [col for col in required_columns if col not in notes.columns]
    if missing:
        raise ValueError(
            "summarize_patients requires columns "
            f"{', '.join(missing)} in the input DataFrame."
        )

    logger.info("Preparing serial patient summarization for %d notes.", len(notes))
    summary_result = summarize_patient_notes(
        notes,
        config=resolved_config,
        existing_summaries=existing_summaries,
        return_qc=return_qc,
    )
    if return_qc:
        summaries, metadata, qc_report = cast(
            tuple[pd.DataFrame, dict, pd.DataFrame],
            summary_result,
        )
    else:
        summaries, metadata = cast(
            tuple[pd.DataFrame, dict],
            summary_result,
        )
        qc_report = None
    logger.info("Patient summarization complete. Produced %d rows.", len(summaries))

    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "patient_summarizer": metadata["model_metadata"],
            },
        }
        if return_qc:
            return summaries, metadata_payload, qc_report
        return summaries, metadata_payload
    if return_qc:
        return summaries, qc_report
    return summaries


__all__ = [
    "ColBERTConfig",
    "ColBERTPatientIndex",
    "colbert_notes_fingerprint",
    "encode_patient_notes_colbert",
    "load_colbert_patient_index",
    "retrieve_patient_chunks_colbert",
    "save_colbert_patient_index",
    "review_patient_workup_with_colbert",
    "FullPatientScreenError",
    "FullPatientScreenProgress",
    "NoteCompressionError",
    "NoteCompressionProgress",
    "NoteSearchLLMConfig",
    "NoteSearchLimits",
    "PatientPDFInput",
    "PatientPDFProgress",
    "RawPatientNoteQAProgress",
    "RawPatientNoteQuestionError",
    "WorkupSearchReviewConfig",
    "answer_patient_question_batch",
    "answer_patient_questions",
    "answer_question_with_raw_patient_notes",
    "compress_patient_note",
    "compress_patient_notes",
    "concatenate_patient_note_pdfs",
    "full_patient_screen",
    "review_patient_workup",
    "review_patient_workup_with_note_search",
    "structure_patient_summaries",
    "structure_patient_summary",
    "summarize_patient_notes",
    "summarize_patients",
]
