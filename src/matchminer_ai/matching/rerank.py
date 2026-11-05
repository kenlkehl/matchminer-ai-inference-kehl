"""Match quality scoring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from importlib import resources

import pandas as pd
import torch

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from .inference import (
    TOKEN_ATTRIBUTION_CAVEAT,
    attribute_checker_tokens,
    format_checker_prompt_with_ranges,
    run_checker,
)

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


def _load_match_quality_template(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _pair_inputs(candidate_pairs: pd.DataFrame) -> tuple[str, list[str], list[str]]:
    """The trial identifier column, the cohorts' key criteria and the patients' key sections."""
    from matchminer_ai._texts import patient_key_texts, trial_id_column, trial_key_texts
    if "patient_id" not in candidate_pairs.columns:
        raise ValueError("candidate_pairs is missing required columns: patient_id")
    return trial_id_column(candidate_pairs), trial_key_texts(candidate_pairs), patient_key_texts(candidate_pairs)


def _build_match_quality_prompts(
    candidate_pairs: pd.DataFrame,
    *,
    template: str,
) -> list[str]:
    _, cohorts, patients = _pair_inputs(candidate_pairs)
    return [template.format(cohort, patient) for cohort, patient in zip(cohorts, patients, strict=True)]


def _build_match_quality_prompts_with_ranges(
    candidate_pairs: pd.DataFrame,
    *,
    template: str,
) -> tuple[list[str], list[dict[str, tuple[int, int]]]]:
    """Build prompts plus source ranges used for token attribution."""
    prompts: list[str] = []
    ranges: list[dict[str, tuple[int, int]]] = []
    _, cohorts, patients = _pair_inputs(candidate_pairs)
    for cohort, patient in zip(cohorts, patients, strict=True):
        prompt, component_ranges = format_checker_prompt_with_ranges(
            template,
            [("cohort_key_criteria", cohort), ("patient_key_summary", patient)],
        )
        prompts.append(prompt)
        ranges.append(component_ranges)
    return prompts, ranges


def _scores(predictions: list) -> tuple[list[float], dict[str, list[float]] | None]:
    """The 0–1 match score per pair and, for a multi-head TrialChecker, each head's probability.

    A multi-head model has one sigmoid head per match label; its score is the expected match
    score fraction, P(reasonable) × (1 + P(cancer type) + P(burden) + P(treatment history)
    + 2 P(biomarker)) / 6. A single-output (regression) model's score is the sigmoid of its logit.
    """
    from .match_labels import LABELS, MAX_SCORE, POINTS
    rows = [{item.get("label", "LABEL_0"): float(item["score"]) for item in (p if isinstance(p, list) else [p])}
            for p in predictions]
    if rows and all(set(LABELS) <= set(row) for row in rows):
        heads = {label: [row[label] for row in rows] for label in LABELS}
        scores = [row["reasonableness"] * (POINTS["reasonableness"] + sum(
            POINTS[label] * row[label] for label in LABELS if label != "reasonableness")) / MAX_SCORE for row in rows]
        return scores, heads
    return [float(torch.sigmoid(torch.tensor(next(iter(row.values())))).item()) for row in rows], None


def score_match_quality(
    candidate_pairs: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    filter_low_quality: bool = True,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate the clinical match quality of each candidate patient–cohort pair with TrialChecker.

    TrialChecker reads key text only: the cohort's key criteria and the patient summary's key
    section (BoilerplateChecker handles the boilerplate). A multi-head TrialChecker has one head
    per match label (:mod:`matchminer_ai.matching.match_labels`): reasonableness and the cancer
    type, burden, treatment history and biomarker specificity answers.

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
        MMAI configuration containing match quality checker settings.
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
            0–1 match score: for a multi-head model the expected match score fraction,
            P(reasonable) × (1 + P(cancer type) + P(burden) + P(treatment history)
            + 2 P(biomarker)) / 6; for a single-output model the sigmoid of its logit.
        match_quality_pass : bool
            Whether the match quality score meets the configured cutoff.
        match_quality_<label> : float
            Multi-head models only: each head's probability, for reasonableness, cancer_type,
            burden, treatment_history and biomarker.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict.
    """
    trial_id, _, _ = _pair_inputs(candidate_pairs)

    # Resolve run config and build checker prompts from the configured template.
    resolved_config = config or load_default_preset()
    checker_config = dict(resolved_config.raw.get("match_quality", {}))
    prompt_file = str(checker_config["prompt_file"]).strip()
    score_cutoff = float(checker_config.get("score_cutoff", 0.2))

    template = _load_match_quality_template(prompt_file)
    prompts = _build_match_quality_prompts(candidate_pairs, template=template)

    # Run the backend text-classification model over all prompts.
    predictions, model_metadata = run_checker(
        prompts,
        checker_config=checker_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
        return_all_scores=True,
    )

    if len(predictions) != len(candidate_pairs):
        raise ValueError(
            "Checker returned a different number of predictions than input rows."
        )

    # Convert model outputs into a compact, derived result table.
    output = candidate_pairs[["patient_id", trial_id]].copy()
    confidence_scores, heads = _scores(predictions)
    output["match_quality_score"] = confidence_scores
    output["match_quality_pass"] = [
        score >= score_cutoff for score in confidence_scores
    ]
    for label, values in (heads or {}).items():
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
    """Interpret specific TrialChecker predictions with token attributions.

    This function reruns TrialChecker for each supplied patient–cohort pair and
    performs one gradient-enabled forward pass per row. The returned token
    offsets are relative to the original cohort key criteria
    (``cohort_key_criteria`` source) and patient key section
    (``patient_key_summary`` source). Positive normalized scores support the
    interpreted prediction; negative scores oppose it. A single-output model's
    pass/fail prediction is interpreted; for a multi-head model, ``head`` names
    the match label whose yes/no prediction is interpreted.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        One or more specific pairs with the same required input columns as
        :func:`score_match_quality`.
    config : MMAIConfig, optional
        Checker configuration. Uses the default preset when omitted.
    top_k : int or None, default 30
        Retain the most important tokens per source field. Use ``None`` to
        return every retained input token.
    return_metadata : bool, default False
        Return the config snapshot and model metadata with the result.

    Returns
    -------
    pd.DataFrame
        One row per input pair with the TrialChecker prediction,
        ``token_attributions``, input ``attribution_coverage``, and caveat.
    tuple[pd.DataFrame, dict]
        Result and reproducibility metadata when ``return_metadata`` is True.
    """
    trial_id, _, _ = _pair_inputs(candidate_pairs)

    resolved_config = config or load_default_preset()
    checker_config = dict(resolved_config.raw.get("match_quality", {}))
    prompt_file = str(checker_config["prompt_file"]).strip()
    score_cutoff = float(checker_config.get("score_cutoff", 0.2))
    template = _load_match_quality_template(prompt_file)
    prompts, component_ranges = _build_match_quality_prompts_with_ranges(
        candidate_pairs,
        template=template,
    )
    predictions, model_metadata = run_checker(
        prompts,
        checker_config=checker_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
        return_all_scores=True,
    )
    if len(predictions) != len(candidate_pairs):
        raise ValueError(
            "Checker returned a different number of predictions than input rows."
        )

    confidence_scores, heads = _scores(predictions)
    passes = [score >= score_cutoff for score in confidence_scores]
    if heads is not None:
        if head not in heads:
            raise ValueError(f"head must be one of {list(heads)}")
        decisions = [value >= 0.5 for value in heads[head]]
        labels, targets = [head] * len(decisions), [f"{head}_{'yes' if d else 'no'}" for d in decisions]
    else:
        decisions, labels = passes, [""] * len(passes)
        targets = ["match_quality_pass" if passed else "match_quality_fail" for passed in passes]
    explanations = attribute_checker_tokens(
        prompts,
        component_ranges=component_ranges,
        checker_config=checker_config,
        target_labels=labels,
        target_directions=[1.0 if decision else -1.0 for decision in decisions],
        top_k=top_k,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
    )

    output = candidate_pairs[["patient_id", trial_id]].copy()
    output["match_quality_score"] = confidence_scores
    output["match_quality_pass"] = passes
    for label, values in (heads or {}).items():
        output[f"match_quality_{label}"] = values
    output["attribution_method"] = [item["method"] for item in explanations]
    output["attribution_target"] = targets
    output["token_attributions"] = [
        item["token_attributions"] for item in explanations
    ]
    output["attribution_coverage"] = [item["coverage"] for item in explanations]
    output["attribution_caveat"] = TOKEN_ATTRIBUTION_CAVEAT
    output = output.reset_index(drop=True)

    if return_metadata:
        return output, {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {"interpret_match_quality": model_metadata},
            "interpretability": {
                "method": "gradient_x_input",
                "top_k_per_source": top_k,
            },
        }
    return output


__all__ = ["interpret_match_quality", "score_match_quality"]
