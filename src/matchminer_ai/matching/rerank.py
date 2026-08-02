"""Match quality scoring helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from importlib import resources

import pandas as pd
import torch

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


def _build_match_quality_prompts(
    candidate_pairs: pd.DataFrame,
    *,
    template: str,
) -> list[str]:
    patient_summaries = (
        candidate_pairs["cancer_history_summary"].fillna("").astype(str).tolist()
    )
    clinical_spaces = (
        candidate_pairs["clinical_space_summary"].fillna("").astype(str).tolist()
    )
    return [
        template.format(clinical_space, patient_summary)
        for patient_summary, clinical_space in zip(
            patient_summaries, clinical_spaces, strict=False
        )
    ]


def _build_match_quality_prompts_with_ranges(
    candidate_pairs: pd.DataFrame,
    *,
    template: str,
) -> tuple[list[str], list[dict[str, tuple[int, int]]]]:
    """Build prompts plus source ranges used for token attribution."""
    prompts: list[str] = []
    ranges: list[dict[str, tuple[int, int]]] = []
    patient_summaries = (
        candidate_pairs["cancer_history_summary"].fillna("").astype(str).tolist()
    )
    clinical_spaces = (
        candidate_pairs["clinical_space_summary"].fillna("").astype(str).tolist()
    )
    for patient_summary, clinical_space in zip(
        patient_summaries,
        clinical_spaces,
        strict=True,
    ):
        prompt, component_ranges = format_checker_prompt_with_ranges(
            template,
            [
                ("clinical_space_summary", clinical_space),
                ("cancer_history_summary", patient_summary),
            ],
        )
        prompts.append(prompt)
        ranges.append(component_ranges)
    return prompts, ranges


def score_match_quality(
    candidate_pairs: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    filter_low_quality: bool = True,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Evaluate the clinical match quality of each candidate patient-trial pair.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        DataFrame of candidate patient-trial pairs.

        Expected columns
        ----------------
        patient_id : str
            Patient identifier.
        space_trial_id : str
            Trial-space identifier.
        cancer_history_summary : str
            Patient summary text.
        clinical_space_summary : str
            Trial clinical-space summary text.

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
        space_trial_id : str
            Trial-space identifier.
        match_quality_score : float
            Model-generated confidence score for clinical match quality.
        match_quality_pass : bool
            Whether the match quality score meets the configured cutoff.
    tuple[pd.DataFrame, dict]
        When return_metadata is True, returns the DataFrame plus a metadata dict.
    """
    # Validate that candidate pair rows contain the text + ids needed for checker prompts.
    required = [
        "patient_id",
        "space_trial_id",
        "cancer_history_summary",
        "clinical_space_summary",
    ]
    missing = [col for col in required if col not in candidate_pairs.columns]
    if missing:
        raise ValueError(
            f"candidate_pairs is missing required columns: {', '.join(missing)}"
        )

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
    )

    if len(predictions) != len(candidate_pairs):
        raise ValueError(
            "Checker returned a different number of predictions than input rows."
        )

    # Convert model outputs into a compact, derived result table.
    output = candidate_pairs[["patient_id", "space_trial_id"]].copy()
    confidence_scores = [
        float(torch.sigmoid(torch.tensor(float(prediction["score"]))).item())
        for prediction in predictions
    ]
    output["match_quality_score"] = confidence_scores
    output["match_quality_pass"] = [
        score >= score_cutoff for score in confidence_scores
    ]

    # Optionally keep only matches that pass the quality threshold.
    if filter_low_quality:
        keep_rows = output["match_quality_pass"]
        output = output.loc[keep_rows].copy()
    output = output.reset_index(drop=True)

    # Optionally return metadata for reproducibility/debugging.
    if return_metadata:
        metadata_payload = {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "match_quality_checker": model_metadata,
            },
        }
        return output, metadata_payload
    return output


def interpret_match_quality(
    candidate_pairs: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    top_k: int | None = 30,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Interpret specific TrialChecker predictions with token attributions.

    This function reruns TrialChecker for each supplied patient-trial pair and
    performs one gradient-enabled forward pass per row. The returned token
    offsets are relative to the original ``cancer_history_summary`` and
    ``clinical_space_summary`` strings. Positive normalized scores support the
    reported pass/fail prediction; negative scores oppose it.

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
    required = [
        "patient_id",
        "space_trial_id",
        "cancer_history_summary",
        "clinical_space_summary",
    ]
    missing = [col for col in required if col not in candidate_pairs.columns]
    if missing:
        raise ValueError(
            f"candidate_pairs is missing required columns: {', '.join(missing)}"
        )

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
    )
    if len(predictions) != len(candidate_pairs):
        raise ValueError(
            "Checker returned a different number of predictions than input rows."
        )

    confidence_scores = [
        float(torch.sigmoid(torch.tensor(float(prediction["score"]))).item())
        for prediction in predictions
    ]
    passes = [score >= score_cutoff for score in confidence_scores]
    explanations = attribute_checker_tokens(
        prompts,
        component_ranges=component_ranges,
        checker_config=checker_config,
        target_labels=[str(prediction.get("label", "")) for prediction in predictions],
        target_directions=[1.0 if passed else -1.0 for passed in passes],
        top_k=top_k,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
    )

    output = candidate_pairs[["patient_id", "space_trial_id"]].copy()
    output["match_quality_score"] = confidence_scores
    output["match_quality_pass"] = passes
    output["attribution_method"] = [item["method"] for item in explanations]
    output["attribution_target"] = [
        "match_quality_pass" if passed else "match_quality_fail" for passed in passes
    ]
    output["token_attributions"] = [
        item["token_attributions"] for item in explanations
    ]
    output["attribution_coverage"] = [item["coverage"] for item in explanations]
    output["attribution_caveat"] = TOKEN_ATTRIBUTION_CAVEAT
    output = output.reset_index(drop=True)

    if return_metadata:
        return output, {
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {"match_quality_checker": model_metadata},
            "interpretability": {
                "method": "gradient_x_input",
                "top_k_per_source": top_k,
            },
        }
    return output


__all__ = ["interpret_match_quality", "score_match_quality"]
