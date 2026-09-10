"""Trial summarization APIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import MMAIConfig, config_snapshot

from .eligibility import (
    TrialSpaceCriteriaExtractionError,
    TrialSpaceCriteriaProgress,
    extract_trial_space_eligibility_criteria,
)
from .structure import structure_trial_space

if TYPE_CHECKING:
    import pandas as pd


def summarize_trials(
    trials: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    return_qc: bool = False,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, dict]
    | tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, dict, pd.DataFrame]
):
    """
    Summarize clinical trials into clinical spaces and general exclusion criteria.

    Parameters
    ----------
    trials : pd.DataFrame
        Trial-level input. One row per trial.

        Expected columns
        ----------------
        trial_id : str
            Unique trial identifier.
        trial_title : str
            Human-readable trial title.
        brief_summary : str
            Brief textual summary of the trial.
        eligibility_criteria : str
            Full eligibility criteria text for the trial.
    return_metadata : bool, optional
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this run.

    Returns
    -------
    pd.DataFrame
        Clinical-space-level DataFrame. One row per clinical space per trial.

        Columns
        -------
        space_trial_id : str
            Unique identifier for a specific trial + clinical space combination.
        trial_id : str
            Original trial identifier (copied through from input).
        clinical_space_number : int
            Zero-based index of the clinical space within the trial. Newly
            generated space_trial_id values start at <trial_id>-0.
        clinical_space_summary : str
            Summary of the clinical space (disease context, line of therapy, etc).
        general_exclusion_criteria : str
            General trial-level exclusion criteria text extracted for this space.

        Debug Columns
        (Available only if pipeline initialized with debug_mode=True)
        -------------------------------------------------------------
        trial_input_text : str
            Concatenation of trial_title + brief_summary + eligibility_criteria.
            This is the raw input text fed into the LLM.
        trial_answer_text : str
            Text the package treated as the LLM answer and used for
            postprocessing.
        trial_reasoning_text : str
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
    from matchminer_ai.config import MMAIConfig, load_default_preset

    from .postprocess import postprocess_trial_summaries
    from .summarize import run_llm_summarization

    logger = logging.getLogger(__name__)
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    required_columns = {
        "trial_id",
        "trial_title",
        "brief_summary",
        "eligibility_criteria",
    }
    missing = required_columns.difference(trials.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"summarize_trials requires columns: {', '.join(sorted(required_columns))}. "
            f"Missing: {missing_list}."
        )

    logger.info("Starting trial summarization for %d trials.", len(trials))
    (
        trials_with_summaries,
        metadata,
        truncated_llm_qc_artifact,
        failed_llm_qc_artifact,
    ) = run_llm_summarization(trials, resolved_config)
    logger.info("Completed LLM summarization. Beginning postprocessing.")
    # Capture unfiltered spaces for QC before keyword filtering.
    result, unfiltered_spaces = postprocess_trial_summaries(
        trials_with_summaries,
        resolved_config,
    )
    logger.info("Postprocessing complete. Produced %d rows.", len(result))

    # Build QC report
    from matchminer_ai._qc.trials import trial_qc_report

    qc_report = trial_qc_report(
        result,
        trial_source=trials,
        unfiltered_spaces=unfiltered_spaces,
        truncated_llm_qc_artifact=truncated_llm_qc_artifact,
        failed_llm_qc_artifact=failed_llm_qc_artifact,
        config=resolved_config,
    )

    # Depending on flags, decide what to return
    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "trial_summarizer": metadata["model_metadata"],
            },
        }
        if return_qc:
            return result, metadata_payload, qc_report
        return result, metadata_payload
    if return_qc:
        return result, qc_report
    return result


__all__ = [
    "TrialSpaceCriteriaExtractionError",
    "TrialSpaceCriteriaProgress",
    "extract_trial_space_eligibility_criteria",
    "structure_trial_space",
    "summarize_trials",
]


# Resolve optional drug research/scoring APIs without eager cross-stage imports.
_GOOD_OPTION_EXPORTS = {
    "CATALOG_CHECKPOINT_SCHEMA_VERSION": "trials.drug_catalog",
    "INTERVENTION_EXCLUSION_CATEGORIES": "trials.drug_catalog",
    "INTERVENTION_SCREENING_DISPOSITIONS": "trials.drug_catalog",
    "ROLE_PROMPT_VERSION": "trials.drug_catalog",
    "SYNTHESIS_PROMPT_VERSION": "trials.drug_catalog",
    "build_good_option_catalog": "trials.drug_catalog",
    "build_intervention_screening_messages": "trials.drug_catalog",
    "build_role_resolution_messages": "trials.drug_catalog",
    "build_synthesis_messages": "trials.drug_catalog",
    "load_good_option_catalog": "trials.drug_catalog",
    "validate_good_option_catalog": "trials.drug_catalog",
    "CATALOG_SCHEMA_VERSION": "trials.drug_evidence",
    "DrugIdentity": "trials.drug_evidence",
    "DrugSummary": "trials.drug_evidence",
    "EvidencePassage": "trials.drug_evidence",
    "GoodOptionCatalog": "trials.drug_evidence",
    "ResearchAttempt": "trials.drug_evidence",
    "TrialDrugAssignment": "trials.drug_evidence",
    "DDGSWebProvider": "trials.drug_research",
    "DrugEvidenceSource": "trials.drug_research",
    "GeneralWebProvider": "trials.drug_research",
    "ResearchSettings": "trials.drug_research",
    "build_facet_query": "trials.drug_research",
    "default_sources": "trials.drug_research",
    "research_drug": "trials.drug_research",
}
__all__ += list(_GOOD_OPTION_EXPORTS)


def __getattr__(name: str):
    if name in _GOOD_OPTION_EXPORTS:
        from importlib import import_module

        value = getattr(
            import_module(f"matchminer_ai.{_GOOD_OPTION_EXPORTS[name]}"), name
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
