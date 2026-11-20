"""Patient summarization workflows."""

from __future__ import annotations

import logging
from pathlib import Path
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
from .extraction import (
    extract_document_records,
    extract_patient_records,
    extraction_items,
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
from .summary_quality import check_patient_summary
from .summarize import serial_trace, summarize_patient_notes
from .workup import review_patient_workup
from .workup_queries import (
    check_workup_with_extractions,
    compile_workup_queries,
    evaluate_workup_query,
    validate_workup_query,
)
from .workup_search import review_patient_workup_with_note_search
from .workup_sql import (
    build_extraction_database,
    check_workup_with_sql,
    compile_workup_sql,
    write_extraction_database,
)
from .workup_search_review import WorkupSearchReviewConfig


def summarize_patients(
    notes: pd.DataFrame | PatientPDFInput,
    *,
    config: MMAIConfig | None = None,
    existing_summaries: pd.DataFrame | None = None,
    checkpoint_dir: str | Path | None = None,
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
    Summarize longitudinal notes into one complete patient_summary, ending
    with an Other relevant history section.

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
        serial updates. Patients with no usable new notes retain their existing
        summary unchanged (normalized, never dropped as non-informative).

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        last_note_date : str or datetime, optional
            Date of the last note the summary includes. When given (for every
            row), only notes dated after it are summarized onto the summary, or,
            when that note is within 10 days of today, notes from 10 days before
            it, so notes that arrive late are not missed. Without it, every
            supplied note is summarized onto the summary: pass only new notes.
        The summary, in the first of these layouts present:
        ``patient_summary_with_boilerplate`` (a complete summary);
        ``cancer_history_summary`` and ``general_exclusion_criteria_evidence``
        (this function's output, so its result can be passed back with new
        notes); ``patient_summary`` and ``patient_boilerplate_text`` (training
        output); or ``patient_summary`` (a complete summary).
    checkpoint_dir : str or pathlib.Path, optional
        Directory used to save progress during patient summarization: each
        chunk's result is saved as soon as it is final. Reusing the same
        directory on retry continues each patient from their last saved chunk.
        The caller is responsible for supplying a directory specific to this
        input and run.
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
        Patient-level DataFrame. One row per patient. Patients whose summary
        is empty or begins with "No information", or whose summary still fails
        after ``patient.summary_quality_max_attempts`` attempts, are omitted
        and counted in the QC report.

        Columns
        -------
        patient_id : str
            Original patient identifier.
        last_note_date : str or None
            Date (YYYY-MM-DD) of the latest dated note the summary includes: its
            newest summarized note, or its existing summary's ``last_note_date``.
            Pass it back with the next notes to summarize only newer ones.
        patient_summary : str
            Cancer history followed by ``Other relevant history:`` and a
            newline containing relevant comorbidities, performance status,
            inactive cancers and other eligibility evidence.
        cancer_history_summary : str
            The summary's key section: the cancer history before ``Other
            relevant history:``. TrialSpace retrieval and the decision model's match questions read it.
        general_exclusion_criteria_evidence : str
            The summary's boilerplate section: the text after ``Other relevant
            history:`` (``Not documented.`` when absent). The decision model's boilerplate question
            reads it.

        Debug Columns
        (Available only if pipeline initialized with debug_mode=True)
        -------------------------------------------------------------
        patient_answer_text : str
            Text the package treated as the LLM answer and used for
            postprocessing. Reflects only each patient's final chunk;
            reasoning and answer text from earlier chunks are not retained
            here. If checkpoint_dir was supplied, per-chunk values can be
            found in its chunk journal.
        patient_reasoning_text : str
            Optional separate reasoning trace returned by the backend or
            extracted by the configured reasoning parser. Reflects only the
            final chunk for each patient; see
            patient_answer_text above.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict:
        ``package``, ``config_snapshot``, ``model_metadata`` and ``summary_quality``
        (attempt and rejection counts of the summary quality checks).
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
        checkpoint_dir=checkpoint_dir,
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
                "summarize_patients": metadata["model_metadata"],
            },
        }
        if "summary_quality" in metadata:
            # Attempt and rejection counts of the summary quality checks.
            metadata_payload["summary_quality"] = metadata["summary_quality"]
        if return_qc:
            return summaries, metadata_payload, qc_report
        return summaries, metadata_payload
    if return_qc:
        return summaries, qc_report
    return summaries


__all__ = [
    "NoteCompressionError",
    "NoteCompressionProgress",
    "NoteSearchLLMConfig",
    "NoteSearchLimits",
    "PatientPDFInput",
    "PatientPDFProgress",
    "WorkupSearchReviewConfig",
    "answer_patient_question_batch",
    "answer_patient_questions",
    "build_extraction_database",
    "check_patient_summary",
    "check_workup_with_extractions",
    "check_workup_with_sql",
    "compile_workup_queries",
    "compile_workup_sql",
    "compress_patient_note",
    "compress_patient_notes",
    "concatenate_patient_note_pdfs",
    "extract_document_records",
    "extract_patient_records",
    "evaluate_workup_query",
    "extraction_items",
    "review_patient_workup",
    "review_patient_workup_with_note_search",
    "serial_trace",
    "summarize_patient_notes",
    "summarize_patients",
    "validate_workup_query",
    "write_extraction_database",
]
