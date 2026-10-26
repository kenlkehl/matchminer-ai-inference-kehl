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
from .guidelines import audit_guideline_catalog, list_guidelines, summarize_guidelines, review_guideline_citations
from .guideline_catalog import load_guideline_catalog, get_guideline_considerations
from .pdq import prepare_pdq_library, summarize_pdq, summarize_pdq_collection

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
    Screen for cancer treatment and extract distinct eligibility populations.

    Parameters
    ----------
    trials : pd.DataFrame
        One row per trial with unique, nonempty ``trial_id`` and the text columns
        ``trial_title``, ``brief_summary``, and ``eligibility_criteria``.
    config : MMAIConfig, optional
        Defaults to the package preset, with ``trial.extraction_method=cohorts``.
    return_metadata : bool, optional
        Also return package/config/model provenance, the prompt hash, validated
        ``trial_decisions``, ``trial_rejections``, ``trial_errors``, and attempts.
    return_qc : bool, optional
        Also return QC separating treatment rejections, inference failures,
        and truncated responses.

    Returns
    -------
    pd.DataFrame or tuple
        With the default ``cohorts`` method, one row per accepted population:
        ``cohort_trial_id`` (``<trial_id>-cohort-<number>``), ``trial_id``,
        zero-based ``cohort_number``, ``cohort_name``, ``cohort_eligibility_criteria``,
        ``cohort_specific_eligibility_criteria``, ``common_eligibility_criteria``,
        ``cohort_criteria_source``, ``cancer_treatment_trial``, and
        ``multiple_cohorts``. Accepted single-population trials retain their
        original ``eligibility_criteria`` field verbatim; multi-population
        criteria join the specific and common strings from the LLM response in
        code, without the cohort name. Name and specific/common columns are null for single-population
        trials. The title, summary and eligibility text remain available together
        for screening and cohort detection. Missing/blank eligibility in an
        accepted single-population trial is recorded as an error, with no row.
        Rejected trials produce
        no rows and remain in metadata and DataFrame attrs. Invalid responses
        are failures, not rejections. Debug mode adds ``trial_input_text``,
        ``trial_answer_text``, and ``trial_reasoning_text`` and retains all
        responses in metadata.

        Optional results are ``(rows, metadata)``, ``(rows, qc)`` or
        ``(rows, metadata, qc)`` according to the flags.

        The explicitly selected legacy ``hierarchical`` and ``one_shot`` methods
        retain ``space_trial_id``, ``trial_id``, ``clinical_space_number``,
        ``clinical_space_summary``, and ``general_exclusion_criteria``. Full
        cohort criteria are not interchangeable with those model inputs or
        their saved embeddings.
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
    if resolved_config.trial.get("extraction_method", "cohorts") == "cohorts":
        from .cohorts import summarize_trial_cohorts

        result, metadata, qc_report = summarize_trial_cohorts(trials, resolved_config)
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            **{key: value for key, value in metadata.items() if key != "model_metadata"},
            "model_metadata": {"trial_summarizer": metadata["model_metadata"]},
        }
        if return_metadata and return_qc:
            return result, metadata_payload, qc_report
        if return_metadata:
            return result, metadata_payload
        if return_qc:
            return result, qc_report
        return result

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
            "trial_rejections": metadata.get("trial_rejections", {}),
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
    "summarize_guidelines",
    "review_guideline_citations",
    "list_guidelines",
    "audit_guideline_catalog",
    "load_guideline_catalog",
    "prepare_pdq_library",
    "summarize_pdq",
    "summarize_pdq_collection",
    "get_guideline_considerations",
]


# Resolve optional drug research/scoring APIs without eager cross-stage imports.
_GOOD_OPTION_EXPORTS = {
    "INTERVENTION_EXCLUSION_CATEGORIES": "trials.drug_catalog",
    "INTERVENTION_SCREENING_DISPOSITIONS": "trials.drug_catalog",
    "build_good_option_catalog": "trials.drug_catalog",
    "build_intervention_screening_messages": "trials.drug_catalog",
    "build_role_resolution_messages": "trials.drug_catalog",
    "load_good_option_catalog": "trials.drug_catalog",
    "validate_good_option_catalog": "trials.drug_catalog",
    "build_drug_class_messages": "trials.drug_catalog",
    "class_id_for": "trials.drug_catalog",
    "normalize_class_name": "trials.drug_catalog",
    "ClassSummary": "trials.drug_evidence",
    "DrugClass": "trials.drug_evidence",
    "DrugIdentity": "trials.drug_evidence",
    "DrugSummary": "trials.drug_evidence",
    "EvidencePassage": "trials.drug_evidence",
    "GoodOptionCatalog": "trials.drug_evidence",
    "ResearchAttempt": "trials.drug_evidence",
    "TrialClassEvidence": "trials.drug_evidence",
    "TrialDrugAssignment": "trials.drug_evidence",
    "DDGSWebProvider": "trials.drug_research",
    "DrugEvidenceSource": "trials.drug_research",
    "GeneralWebProvider": "trials.drug_research",
    "ResearchSettings": "trials.drug_research",
    "ResearchSubject": "trials.drug_research",
    "build_facet_query": "trials.drug_research",
    "default_sources": "trials.drug_research",
    "research_class": "trials.drug_research",
    "research_drug": "trials.drug_research",
    "research_subject": "trials.drug_research",
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
