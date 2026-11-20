"""Boilerplate-criteria check with the decision model's boilerplate question."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from .decision_model import (
    BOILERPLATE, attribute_decision, boilerplate_question, boilerplate_state, boilerplate_state_with_ranges,
    no_request_metadata, oriented, run_decisions,
)
from .inference import TOKEN_ATTRIBUTION_CAVEAT
from .rerank import _same_length, _text_list

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig
    import pandas as pd
else:
    import pandas as pd


def _pair_inputs(matches: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    """The trial identifier column, the patients' boilerplate sections and the cohorts' boilerplate criteria."""
    from matchminer_ai._texts import patient_boilerplate_texts, trial_boilerplate_texts
    if "patient_id" not in matches.columns:
        raise ValueError("matches is missing required columns: patient_id")
    trial_id = next((c for c in ("cohort_trial_id", "space_trial_id", "trial_id") if c in matches.columns), None)
    if trial_id is None:
        raise ValueError("matches needs cohort_trial_id, space_trial_id or trial_id")
    return trial_id, patient_boilerplate_texts(matches), trial_boilerplate_texts(matches)


def _decide(matches: pd.DataFrame, config: MMAIConfig) -> tuple[list[float], dict]:
    _, evidence, criteria = _pair_inputs(matches)
    question = {BOILERPLATE: boilerplate_question()}
    answers, metadata = run_decisions([(boilerplate_state(e, c), question) for e, c in zip(evidence, criteria, strict=True)],
                                      config=config)
    return [row[BOILERPLATE] for row in answers], metadata


def _pass_cutoff(config: MMAIConfig) -> float:
    return float(config.raw.get("exclusion_criteria", {}).get("pass_cutoff", 0.5))


def _pair_result(probability: float, cutoff: float) -> dict[str, Any]:
    passed = probability >= cutoff
    return {"exclusion_criteria_pass_probability": probability,
            "exclusion_score": probability if passed else 1.0 - probability,
            "exclusion_criteria_pass": passed}


def exclusion_criteria_check_pairs(
    *,
    general_exclusion_criteria_evidence: Sequence[str],
    cohort_boilerplate_criteria: Sequence[str],
    config: MMAIConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check patient-cohort text pairs with the decision model's boilerplate question, in input order.

    The service-layer form of :func:`exclusion_criteria_check`: plain text in, no identifiers or
    DataFrames, for callers that keep their own data types (such as a live matching service).
    ``exclusion_criteria_check`` is built on it. Pair ``i`` is
    ``(general_exclusion_criteria_evidence[i], cohort_boilerplate_criteria[i])``; to check one
    patient against many cohorts, repeat its evidence.

    Parameters
    ----------
    general_exclusion_criteria_evidence : sequence of str
        Each patient summary's boilerplate section (``general_exclusion_criteria_evidence``).
    cohort_boilerplate_criteria : sequence of str
        Each cohort's boilerplate criteria (``cohort_boilerplate_criteria``), the same length.
    config : MMAIConfig, optional
        ``decision_model`` (the model) and ``exclusion_criteria.pass_cutoff``. Uses the default
        preset when omitted.

    Returns
    -------
    tuple[list[dict], dict]
        One dict per pair with the columns of :func:`exclusion_criteria_check`:
        ``exclusion_criteria_pass_probability`` (P(yes), potentially eligible),
        ``exclusion_score`` (the probability of the predicted outcome) and
        ``exclusion_criteria_pass``; and the decision model's metadata. With no pairs the model
        is not loaded.

    Raises
    ------
    TypeError
        If either argument is a single str or holds a non-string value.
    ValueError
        If the two sequences differ in length.
    """
    evidence = _text_list("general_exclusion_criteria_evidence", general_exclusion_criteria_evidence)
    criteria = _text_list("cohort_boilerplate_criteria", cohort_boilerplate_criteria)
    _same_length(("general_exclusion_criteria_evidence", evidence), ("cohort_boilerplate_criteria", criteria))
    resolved_config = config or load_default_preset()
    cutoff = _pass_cutoff(resolved_config)
    if not evidence:
        return [], no_request_metadata(resolved_config)
    question = {BOILERPLATE: boilerplate_question()}
    answers, metadata = run_decisions(
        [(boilerplate_state(e, c), question) for e, c in zip(evidence, criteria, strict=True)], config=resolved_config)
    return [_pair_result(row[BOILERPLATE], cutoff) for row in answers], metadata


def _result(matches: pd.DataFrame, trial_id: str, probabilities: list[float], cutoff: float) -> pd.DataFrame:
    output = matches[["patient_id", trial_id]].copy()
    rows = [_pair_result(p, cutoff) for p in probabilities]
    for column in ("exclusion_criteria_pass_probability", "exclusion_score", "exclusion_criteria_pass"):
        output[column] = [row[column] for row in rows]
    return output


def exclusion_criteria_check(
    matches: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    filter_excluded: bool = False,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate with the decision model whether each candidate pair passes the boilerplate criteria.

    The decision model (:mod:`matchminer_ai.matching.decision_model`, configured by
    ``decision_model``) answers its boilerplate question about one state: the patient summary's
    boilerplate section and the cohort's boilerplate criteria (the general requirements most trials
    share, inclusion as well as exclusion; :func:`score_match_quality` handles the key criteria).
    **Yes means the patient is potentially eligible after review of these texts; no means the
    patient clearly fails at least one boilerplate criterion.** Missing information is not a
    failure.

    Parameters
    ----------
    matches : pd.DataFrame
        DataFrame of candidate patient–cohort pairs.

        Expected columns
        ----------------
        patient_id : str
            Patient identifier.
        cohort_trial_id : str
            Cohort identifier (historical ``space_trial_id`` or ``trial_id`` is accepted).
        cohort_boilerplate_criteria : str
            The cohort's boilerplate criteria (historical ``general_exclusion_criteria`` is accepted).
        general_exclusion_criteria_evidence : str
            The patient summary's boilerplate section (or a complete ``patient_summary``, whose
            boilerplate section is taken).

    config : MMAIConfig, optional
        MMAI configuration: ``decision_model`` (the model) and ``exclusion_criteria.pass_cutoff``
        (default 0.5). Uses default preset when omitted.
    filter_excluded : bool, default False
        If True, return only rows where ``exclusion_criteria_pass`` is True.
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
        cohort_trial_id, space_trial_id or trial_id : str
            The input's trial identifier.
        exclusion_criteria_pass_probability : float
            P(yes): the probability that the patient is potentially eligible, not clearly failing
            any boilerplate criterion.
        exclusion_score : float
            The probability of the predicted outcome: P(yes) for a pass, 1 - P(yes) for a fail.
        exclusion_criteria_pass : bool
            True when P(yes) is at least ``exclusion_criteria.pass_cutoff``.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus metadata.
    """
    trial_id, evidence, criteria = _pair_inputs(matches)
    resolved_config = config or load_default_preset()
    rows, model_metadata = exclusion_criteria_check_pairs(
        general_exclusion_criteria_evidence=evidence, cohort_boilerplate_criteria=criteria, config=resolved_config)
    output = matches[["patient_id", trial_id]].copy()
    for column in ("exclusion_criteria_pass_probability", "exclusion_score", "exclusion_criteria_pass"):
        output[column] = [row[column] for row in rows]
    if filter_excluded:
        output = output.loc[output["exclusion_criteria_pass"]].copy()
    output = output.reset_index(drop=True)
    if return_metadata:
        return output, {"package": package_metadata(), "config_snapshot": config_snapshot(resolved_config),
                        "model_metadata": {"exclusion_criteria_check": model_metadata}}
    return output


def interpret_exclusion_criteria(
    matches: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    top_k: int | None = 30,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Interpret specific boilerplate decisions with token attributions.

    Scores each pair as :func:`exclusion_criteria_check` does, then runs one gradient-enabled
    forward pass per pair: gradient times input embedding for the boilerplate question's
    yes-minus-no logit at its answer slot. Returned token offsets are relative to the patient's
    boilerplate section (``patient_boilerplate_summary`` source) and the cohort's boilerplate
    criteria (``cohort_boilerplate_criteria`` source). Positive normalized scores support the
    reported pass/fail decision; negative scores oppose it.

    Parameters
    ----------
    matches : pd.DataFrame
        One or more specific pairs with the same required input columns as
        :func:`exclusion_criteria_check`.
    config : MMAIConfig, optional
        Configuration. Uses the default preset when omitted.
    top_k : int or None, default 30
        Retain the most important tokens per source field. Use ``None`` to
        return every retained input token.
    return_metadata : bool, default False
        Return the config snapshot and model metadata with the result.

    Returns
    -------
    pd.DataFrame
        One row per input pair with the decision, ``token_attributions``, input
        ``attribution_coverage``, and caveat.
    tuple[pd.DataFrame, dict]
        Result and reproducibility metadata when ``return_metadata`` is True.
    """
    trial_id, evidence, criteria = _pair_inputs(matches)
    resolved_config = config or load_default_preset()
    cutoff = _pass_cutoff(resolved_config)
    probabilities, model_metadata = _decide(matches, resolved_config)
    output = _result(matches, trial_id, probabilities, cutoff)
    question = {BOILERPLATE: boilerplate_question()}
    explanations = []
    for patient, cohort, passed in zip(evidence, criteria, output["exclusion_criteria_pass"], strict=True):
        state, ranges = boilerplate_state_with_ranges(patient, cohort)
        explanations.append(oriented(attribute_decision(state, question, BOILERPLATE, ranges, config=resolved_config,
                                                        top_k=top_k), bool(passed)))
    output["attribution_method"] = [item["method"] for item in explanations]
    output["attribution_target"] = ["exclusion_criteria_pass" if passed else "exclusion_criteria_fail"
                                    for passed in output["exclusion_criteria_pass"]]
    output["token_attributions"] = [item["token_attributions"] for item in explanations]
    output["attribution_coverage"] = [item["coverage"] for item in explanations]
    output["attribution_caveat"] = TOKEN_ATTRIBUTION_CAVEAT
    output = output.reset_index(drop=True)
    if return_metadata:
        return output, {"config_snapshot": config_snapshot(resolved_config),
                        "model_metadata": {"interpret_exclusion_criteria": model_metadata},
                        "interpretability": {"method": "gradient_x_input", "top_k_per_source": top_k}}
    return output


__all__ = ["exclusion_criteria_check", "exclusion_criteria_check_pairs", "interpret_exclusion_criteria"]
