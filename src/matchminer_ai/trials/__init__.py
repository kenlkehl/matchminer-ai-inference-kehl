"""Trial processing APIs.

Each trial step is its own function, so callers can run some and skip others:
:func:`extract_cohorts` (ctgov listings) or :func:`extract_protocol_cohorts` (full
protocols) screen for cancer treatment and extract each cohort's complete criteria;
:func:`merge_equivalent_cohorts` combines a trial's cohorts with essentially the same
criteria; :func:`remove_boilerplate_criteria` tags every criteria line inclusion or exclusion and
boilerplate or not, and separates key from boilerplate criteria.
:func:`process_trials` runs all of them, on ctgov listings or protocols, in that order.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import warnings

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import MMAIConfig, config_snapshot

from .eligibility import (
    TrialSpaceCriteriaExtractionError,
    TrialSpaceCriteriaProgress,
    extract_trial_space_eligibility_criteria,
)
from .cohort_merge import merge_equivalent_cohorts
from .cohort_boilerplate import remove_boilerplate_criteria
from .cohort_screen import screen_trials
from .protocol_cohorts import ProtocolCohortProgress, extract_protocol_cohorts
from .synthetic_protocols import SyntheticProtocolProgress, generate_synthetic_protocols
from .structure import structure_trial_space
from .guidelines import audit_guideline_catalog, list_guidelines, summarize_guidelines, review_guideline_citations
from .guideline_catalog import load_guideline_catalog, get_guideline_considerations
from .pdq import prepare_pdq_library, summarize_pdq, summarize_pdq_collection

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def extract_cohorts(
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

    Each cohort's criteria are copied verbatim, boilerplate included. Two optional steps take the
    returned rows, as for full protocols (:func:`extract_protocol_cohorts`):
    :func:`merge_equivalent_cohorts` and :func:`remove_boilerplate_criteria`, both with the default
    ``section="trial"`` (training uses both, in that order; :func:`process_trials` runs all three).
    Named ``summarize_trials`` before November 2026; that name remains as a deprecated alias.

    Parameters
    ----------
    trials : pd.DataFrame
        One row per trial with unique, nonempty ``trial_id`` and the text columns
        ``trial_title``, ``brief_summary``, and ``eligibility_criteria``.
    config : MMAIConfig, optional
        Defaults to the package preset, with ``trial.extraction_method=cohorts``.
    return_metadata : bool, optional
        Also return package/config/model provenance, the prompt hashes, validated
        ``trial_decisions``, ``trial_cohort_names``, ``trial_rejections``,
        ``trial_errors``, and attempts per step.
    return_qc : bool, optional
        Also return QC separating treatment rejections, inference failures,
        and truncated responses.

    Returns
    -------
    pd.DataFrame or tuple
        With the default ``cohorts`` method, one row per accepted population:
        ``cohort_trial_id`` (``<trial_id>-cohort-<number>``), ``trial_id``,
        zero-based ``cohort_number``, ``cohort_name``, ``cohort_eligibility_criteria``,
        ``cohort_criteria_source`` (``"ctgov"``; ``"registry"`` before November 2026), ``cancer_treatment_trial``, and
        ``multiple_cohorts``. Each listing's title, brief summary and eligibility
        criteria form one document that goes through the same three steps and
        prompts as :func:`extract_protocol_cohorts`
        (:mod:`matchminer_ai.trials.cohort_workflow`): a cancer-treatment screen,
        the names of its cohorts with distinct eligibility, and one request per
        cohort for that cohort's complete criteria, which become its row's
        ``cohort_eligibility_criteria``; the ctgov field is never copied
        verbatim. Rejected trials produce no rows and remain in metadata and
        DataFrame attrs. Invalid responses are failures, not rejections. Debug
        mode adds ``trial_input_text`` and retains every step's responses in
        metadata.

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
            f"extract_cohorts requires columns: {', '.join(sorted(required_columns))}. "
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
            "model_metadata": {"extract_cohorts": metadata["model_metadata"]},
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
                "extract_cohorts": metadata["model_metadata"],
            },
            "trial_rejections": metadata.get("trial_rejections", {}),
        }
        if return_qc:
            return result, metadata_payload, qc_report
        return result, metadata_payload
    if return_qc:
        return result, qc_report
    return result


def summarize_trials(*args, **kwargs):
    """Deprecated name of :func:`extract_cohorts` (renamed November 2026)."""
    warnings.warn("summarize_trials is deprecated; use extract_cohorts", DeprecationWarning, stacklevel=2)
    return extract_cohorts(*args, **kwargs)


def process_trials(
    trials: pd.DataFrame,
    *,
    source: str = "ctgov",
    config: MMAIConfig | None = None,
    output_dir: str | Path | None = None,
    text_output_dir: str | Path | None = None,
    progress_callback: ProtocolCohortProgress | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Run every trial step end to end: treatment screen, cohorts, merge and boilerplate.

    The steps are the public functions, in the order training uses them. ``source`` selects the
    extractor and the configuration section all three steps use:

    - ``"ctgov"`` (default): ClinicalTrials.gov listings through :func:`extract_cohorts`, with
      ``section="trial"``.
    - ``"protocol"``: full protocol documents through :func:`extract_protocol_cohorts`, with
      ``section="protocol_cohort_extraction"``.

    Each extractor runs a cancer-treatment screen, then the cohort names and each cohort's
    complete criteria; :func:`merge_equivalent_cohorts` makes a trial's cohorts with
    essentially the same criteria one row; :func:`remove_boilerplate_criteria` tags each criteria
    line as an inclusion or exclusion criterion and as boilerplate or not. Call the functions
    separately to run only some of them.

    Parameters
    ----------
    trials : pd.DataFrame
        For ``"ctgov"``, as for :func:`extract_cohorts`: unique ``trial_id`` and the text columns
        ``trial_title``, ``brief_summary`` and ``eligibility_criteria``. For ``"protocol"``, as
        for :func:`extract_protocol_cohorts`: unique ``trial_id`` and ``protocol_path`` (PDF,
        ``.txt`` or ``.md``).
    source : {"ctgov", "protocol"}, optional
        What each row is; see above.
    config : MMAIConfig, optional
        Defaults to the package preset. For ``"ctgov"`` its ``trial`` section must use the
        ``cohorts`` method.
    output_dir : str or Path, optional
        Saves each trial's results under ``<output_dir>/<trial_id>/`` (for protocols also the
        extraction) and reuses them while the inputs and prompts are unchanged.
    text_output_dir, progress_callback : optional
        Protocols only, as for :func:`extract_protocol_cohorts`: where to keep each document's
        extracted text, and a progress callback for the extraction.
    return_metadata : bool, optional
        Also return each step's metadata, keyed ``extraction``, ``merge`` and ``boilerplate``.

    Returns
    -------
    pd.DataFrame or tuple
        One row per merged cohort of each accepted trial: the extractor's columns
        (``cohort_eligibility_criteria`` keeps the complete criteria), ``merged_from``, and
        the criteria split into ``cohort_key_criteria`` (read by TrialSpace retrieval and the
        decision model's match questions) and ``cohort_boilerplate_criteria`` (read by its
        boilerplate question), and ``cohort_criteria_items``, every line with its tags (read by
        :func:`~matchminer_ai.matching.check_eligibility_criteria`).
        Rejected trials have no rows; their decisions are in the metadata and DataFrame attrs,
        as for the extractors. ``(rows, metadata)`` with ``return_metadata``.
    """
    from matchminer_ai.config import load_default_preset

    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    if source == "ctgov":
        if text_output_dir is not None or progress_callback is not None:
            raise ValueError("text_output_dir and progress_callback apply only to source='protocol'")
        if resolved_config.trial.get("extraction_method", "cohorts") != "cohorts":
            raise ValueError("process_trials requires trial.extraction_method=cohorts")
        section = "trial"
        cohorts, extraction = extract_cohorts(trials, config=resolved_config, return_metadata=True)
    elif source == "protocol":
        section = "protocol_cohort_extraction"
        cohorts, extraction = extract_protocol_cohorts(
            trials, config=resolved_config, text_output_dir=text_output_dir, output_dir=output_dir,
            return_metadata=True, progress_callback=progress_callback)
    else:
        raise ValueError("source must be 'ctgov' or 'protocol'")
    # The decisions stay in the metadata; pandas would deep-copy attrs into every group below.
    attrs, cohorts.attrs = dict(cohorts.attrs), {}
    merged, merging = merge_equivalent_cohorts(cohorts, config=resolved_config, section=section,
                                               output_dir=output_dir, return_metadata=True)
    result, cleaning = remove_boilerplate_criteria(merged, config=resolved_config, section=section,
                                                   output_dir=output_dir, return_metadata=True)
    result.attrs = attrs
    if return_metadata:
        return result, {"extraction": extraction, "merge": merging, "boilerplate": cleaning}
    return result


__all__ = [
    "TrialSpaceCriteriaExtractionError",
    "TrialSpaceCriteriaProgress",
    "extract_trial_space_eligibility_criteria",
    "ProtocolCohortProgress",
    "extract_protocol_cohorts",
    "merge_equivalent_cohorts",
    "remove_boilerplate_criteria",
    "screen_trials",
    "extract_cohorts",
    "process_trials",
    "SyntheticProtocolProgress",
    "generate_synthetic_protocols",
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
