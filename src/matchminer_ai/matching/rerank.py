"""Match quality scoring with the decision model's five match questions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from .decision_model import (
    attribute_decision, match_questions, match_state, match_state_with_ranges, oriented, run_decisions,
)
from .inference import TOKEN_ATTRIBUTION_CAVEAT

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


def _pair_inputs(candidate_pairs: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    """The trial identifier column, the cohorts' key criteria and the patients' key sections."""
    from matchminer_ai._texts import patient_key_texts, trial_id_column, trial_key_texts
    if "patient_id" not in candidate_pairs.columns:
        raise ValueError("candidate_pairs is missing required columns: patient_id")
    return trial_id_column(candidate_pairs), trial_key_texts(candidate_pairs), patient_key_texts(candidate_pairs)


def _scores(answers: list[dict[str, float]]) -> tuple[list[float], dict[str, list[float]]]:
    """The 0-1 match score per pair and each match label's P(yes).

    The score is the expected match score fraction, P(reasonable) x (1 + P(cancer type) + P(burden)
    + P(treatment history) + 2 P(biomarker)) / 6 (:func:`~matchminer_ai.matching.match_labels.match_score`).
    """
    from .match_labels import LABELS, MAX_SCORE, POINTS
    heads = {label: [row[label] for row in answers] for label in LABELS}
    scores = [row["reasonableness"] * (POINTS["reasonableness"] + sum(
        POINTS[label] * row[label] for label in LABELS if label != "reasonableness")) / MAX_SCORE for row in answers]
    return scores, heads


def _decide(candidate_pairs: pd.DataFrame, config: MMAIConfig) -> tuple[list[dict[str, float]], dict]:
    _, cohorts, patients = _pair_inputs(candidate_pairs)
    questions = match_questions()
    requests = [(match_state(patient, cohort), questions) for cohort, patient in zip(cohorts, patients, strict=True)]
    return run_decisions(requests, config=config)


def score_match_quality(
    candidate_pairs: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    filter_low_quality: bool = True,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate the clinical match quality of each candidate patient–cohort pair with the decision model.

    The decision model (:mod:`matchminer_ai.matching.decision_model`, configured by
    ``decision_model``) answers the five match questions about one state: the cohort's key criteria
    and the patient summary's key section (:func:`exclusion_criteria_check` handles the boilerplate).
    The questions are the match labels of :mod:`matchminer_ai.matching.match_labels`:
    ``reasonableness`` (yes: potentially eligible after review of these texts; no: clearly
    ineligible) and the cancer type, burden, treatment history and biomarker specificity answers.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        DataFrame of candidate patient–cohort pairs.

        Expected columns
        ----------------
        patient_id : str
            Patient identifier.
        cohort_trial_id : str
            Cohort identifier (historical ``space_trial_id`` is accepted).
        cancer_history_summary : str
            The patient summary's key section (or a complete ``patient_summary``, whose key
            section is taken).
        cohort_key_criteria : str
            The cohort's key criteria (else ``cohort_eligibility_criteria``, else historical
            ``clinical_space_summary``).

    config : MMAIConfig, optional
        MMAI configuration: ``decision_model`` (the model) and ``match_quality.score_cutoff``.
        Uses default preset when omitted.
    filter_low_quality : bool, default True
        If True, only return rows where ``match_quality_pass`` evaluates to True.
    return_metadata : bool, default False
        When True, also return a metadata dict containing the config snapshot
        and model metadata for this run.

    Returns
    -------
    pd.DataFrame
        Derived output table containing:

        Columns
        -------
        patient_id : str
            Patient identifier.
        cohort_trial_id or space_trial_id : str
            The input's trial identifier.
        match_quality_score : float
            0–1 match score, the expected match score fraction: P(reasonable) × (1 +
            P(cancer type) + P(burden) + P(treatment history) + 2 P(biomarker)) / 6.
        match_quality_pass : bool
            Whether the match quality score meets the configured cutoff.
        match_quality_<label> : float
            Each match question's P(yes), for reasonableness, cancer_type, burden,
            treatment_history and biomarker.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict.
    """
    trial_id, _, _ = _pair_inputs(candidate_pairs)
    resolved_config = config or load_default_preset()
    score_cutoff = float(resolved_config.raw.get("match_quality", {}).get("score_cutoff", 0.2))
    answers, model_metadata = _decide(candidate_pairs, resolved_config)

    # Convert the answers into a compact, derived result table.
    output = candidate_pairs[["patient_id", trial_id]].copy()
    confidence_scores, heads = _scores(answers)
    output["match_quality_score"] = confidence_scores
    output["match_quality_pass"] = [
        score >= score_cutoff for score in confidence_scores
    ]
    for label, values in heads.items():
        output[f"match_quality_{label}"] = values

    # Optionally keep only matches that pass the quality threshold.
    if filter_low_quality:
        keep_rows = output["match_quality_pass"]
        output = output.loc[keep_rows].copy()
    output = output.reset_index(drop=True)

    # Optionally return metadata for reproducibility/debugging.
    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "score_match_quality": model_metadata,
            },
        }
        return output, metadata_payload
    return output


def interpret_match_quality(
    candidate_pairs: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    top_k: int | None = 30,
    head: str = "reasonableness",
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Interpret one match question's decision for specific pairs with token attributions.

    Scores each pair as :func:`score_match_quality` does, then, for the match question ``head``
    (default ``reasonableness``), runs one gradient-enabled forward pass per pair over the same
    packed sequence: gradient times input embedding for the question's yes-minus-no logit at its
    answer slot. Token offsets are relative to the cohort key criteria (``cohort_key_criteria``
    source) and the patient key section (``patient_key_summary`` source). Positive normalized scores
    support the interpreted decision (``head`` yes when P(yes) >= 0.5, else no); negative scores
    oppose it.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        One or more specific pairs with the same required input columns as
        :func:`score_match_quality`.
    config : MMAIConfig, optional
        Configuration. Uses the default preset when omitted.
    top_k : int or None, default 30
        Retain the most important tokens per source field. Use ``None`` to
        return every retained input token.
    head : str, default "reasonableness"
        The match question to interpret.
    return_metadata : bool, default False
        Return the config snapshot and model metadata with the result.

    Returns
    -------
    pd.DataFrame
        One row per input pair with the scores, ``token_attributions``, input
        ``attribution_coverage``, and caveat.
    tuple[pd.DataFrame, dict]
        Result and reproducibility metadata when ``return_metadata`` is True.
    """
    from .match_labels import LABELS
    if head not in LABELS:
        raise ValueError(f"head must be one of {list(LABELS)}")
    trial_id, cohorts, patients = _pair_inputs(candidate_pairs)
    resolved_config = config or load_default_preset()
    score_cutoff = float(resolved_config.raw.get("match_quality", {}).get("score_cutoff", 0.2))
    answers, model_metadata = _decide(candidate_pairs, resolved_config)
    confidence_scores, heads = _scores(answers)
    questions = match_questions()
    explanations, targets = [], []
    for cohort, patient, probability in zip(cohorts, patients, heads[head], strict=True):
        state, ranges = match_state_with_ranges(patient, cohort)
        ranges = {"cohort_key_criteria": ranges["cohort_key_criteria"], "patient_key_summary": ranges["patient_key_summary"]}
        decision = probability >= 0.5
        explanations.append(oriented(attribute_decision(state, questions, head, ranges, config=resolved_config,
                                                        top_k=top_k), decision))
        targets.append(f"{head}_{'yes' if decision else 'no'}")

    output = candidate_pairs[["patient_id", trial_id]].copy()
    output["match_quality_score"] = confidence_scores
    output["match_quality_pass"] = [score >= score_cutoff for score in confidence_scores]
    for label, values in heads.items():
        output[f"match_quality_{label}"] = values
    output["attribution_method"] = [item["method"] for item in explanations]
    output["attribution_target"] = targets
    output["token_attributions"] = [item["token_attributions"] for item in explanations]
    output["attribution_coverage"] = [item["coverage"] for item in explanations]
    output["attribution_caveat"] = TOKEN_ATTRIBUTION_CAVEAT
    output = output.reset_index(drop=True)

    if return_metadata:
        return output, {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {"interpret_match_quality": model_metadata},
            "interpretability": {"method": "gradient_x_input", "top_k_per_source": top_k, "head": head},
        }
    return output


__all__ = ["interpret_match_quality", "score_match_quality"]
