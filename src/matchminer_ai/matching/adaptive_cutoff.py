"""Adaptive TrialSpace cutoff selection for trial-centric matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Callable, Literal

import pandas as pd

from .llm_checks import score_match_quality_with_llm
from .rerank import score_match_quality

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


CutoffCheckMethod = Literal["trial_checker", "llm"]
TrialCentricCutoffProgress = Callable[[int, int, str], None]

DEFAULT_TRIAL_CHECKER_CUTOFF_THRESHOLD = 0.20
DEFAULT_LLM_CUTOFF_THRESHOLD = 1.0
DEFAULT_CUTOFF_PATIENTS_PER_SIDE = 10
DEFAULT_INITIAL_CUTOFF_PROPORTION = 0.5
DEFAULT_STABILITY_INITIAL_CUTOFF_PROPORTIONS = (0.10, 0.25, 0.50, 0.75, 0.90)

_HISTORY_COLUMNS = [
    "iteration",
    "proposed_cutoff",
    "lower_bound_before",
    "upper_bound_before",
    "window_start_rank",
    "window_end_rank",
    "window_patient_count",
    "newly_scored_patient_count",
    "pass_count",
    "fail_count",
    "pass_fraction",
    "decision",
    "lower_bound_after",
    "upper_bound_after",
]

_STABILITY_COLUMNS = [
    "initial_cutoff_proportion",
    "initial_proposed_cutoff",
    "selected_cutoff",
    "selected_cutoff_fraction",
    "reasonable_consideration_count",
    "reasonable_consideration_fraction",
    "search_iteration_count",
    "probe_patient_count",
]


@dataclass(frozen=True)
class TrialCentricCutoffResult:
    """Result of an adaptive trial-centric cutoff search.

    ``cutoff`` is the number of leading TrialSpace-ranked patients that should
    proceed to the full match-quality and general-exclusion checks. The probe
    tables are diagnostics for the search, not eligibility determinations.
    """

    cutoff: int
    total_candidates: int
    check_method: CutoffCheckMethod
    score_threshold: float
    patients_per_side: int
    initial_cutoff_proportion: float
    scored_candidates: pd.DataFrame
    search_history: pd.DataFrame


@dataclass(frozen=True)
class TrialCentricCutoffStabilityResult:
    """Starting-position QA for an adaptive trial-centric cutoff search.

    TrialChecker scores are computed once for the complete ranking, then reused
    for every requested initial cutoff proportion. ``cutoff_runs`` compares the
    selected prefix and the number of threshold-passing patients inside that
    prefix. These are model-screening diagnostics, not eligibility results.
    """

    total_candidates: int
    score_threshold: float
    patients_per_side: int
    initial_cutoff_proportions: tuple[float, ...]
    cutoff_runs: pd.DataFrame
    scored_candidates: pd.DataFrame
    selected_cutoff_spread: int
    reasonable_consideration_count_spread: int


def _normalize_check_method(value: str) -> CutoffCheckMethod:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases: dict[str, CutoffCheckMethod] = {
        "trial_checker": "trial_checker",
        "trialchecker": "trial_checker",
        "llm": "llm",
        "llm_check": "llm",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "check_method must be either 'trial_checker' or 'llm'."
        ) from exc


def _resolve_score_threshold(
    check_method: CutoffCheckMethod,
    score_threshold: float | None,
) -> float:
    default = (
        DEFAULT_TRIAL_CHECKER_CUTOFF_THRESHOLD
        if check_method == "trial_checker"
        else DEFAULT_LLM_CUTOFF_THRESHOLD
    )
    threshold = default if score_threshold is None else float(score_threshold)
    maximum = 1.0 if check_method == "trial_checker" else 5.0
    if not math.isfinite(threshold) or not 0.0 <= threshold <= maximum:
        raise ValueError(
            f"score_threshold must be between 0 and {maximum:g} for "
            f"check_method='{check_method}'."
        )
    return threshold


def _validate_patients_per_side(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("patients_per_side must be a positive integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("patients_per_side must be a positive integer.") from exc
    if resolved < 1 or resolved != value:
        raise ValueError("patients_per_side must be a positive integer.")
    return resolved


def _validate_initial_cutoff_proportion(value: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_cutoff_proportion must be between 0 and 1.") from exc
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError("initial_cutoff_proportion must be between 0 and 1.")
    return resolved


def _prepare_ranked_candidates(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    required = [
        "patient_id",
        "space_trial_id",
        "rank",
        "cancer_history_summary",
        "clinical_space_summary",
    ]
    missing = [column for column in required if column not in candidate_pairs.columns]
    if missing:
        raise ValueError(
            "candidate_pairs is missing required columns: " + ", ".join(missing)
        )
    if candidate_pairs.empty:
        return candidate_pairs.copy().reset_index(drop=True)
    if candidate_pairs["space_trial_id"].nunique(dropna=False) != 1:
        raise ValueError(
            "candidate_pairs must contain the ranking for exactly one trial space."
        )
    if candidate_pairs["patient_id"].duplicated().any():
        raise ValueError("candidate_pairs must contain unique patient_id values.")

    numeric_rank = pd.to_numeric(candidate_pairs["rank"], errors="coerce")
    if numeric_rank.isna().any() or (numeric_rank <= 0).any():
        raise ValueError("rank must contain unique positive integer values.")
    integer_rank = numeric_rank.astype(int)
    if not (numeric_rank == integer_rank).all() or integer_rank.duplicated().any():
        raise ValueError("rank must contain unique positive integer values.")

    ranked = candidate_pairs.assign(rank=integer_rank).sort_values(
        "rank",
        kind="stable",
    )
    expected_ranks = list(range(1, len(ranked) + 1))
    if ranked["rank"].tolist() != expected_ranks:
        raise ValueError(
            "candidate_pairs must contain a complete TrialSpace ranking with "
            "contiguous ranks starting at 1."
        )
    return ranked.reset_index(drop=True)


def _empty_scored_candidates(check_method: CutoffCheckMethod) -> pd.DataFrame:
    score_column = (
        "match_quality_score"
        if check_method == "trial_checker"
        else "llm_match_quality_score"
    )
    return pd.DataFrame(
        columns=[
            "patient_id",
            "space_trial_id",
            "rank",
            score_column,
            "cutoff_probe_score",
            "cutoff_probe_pass",
            "cutoff_probe_iteration",
        ]
    )


def _run_cutoff_search(
    ranked: pd.DataFrame,
    *,
    method: CutoffCheckMethod,
    threshold: float,
    radius: int,
    starting_proportion: float,
    score_rows: Callable[[pd.DataFrame], pd.DataFrame],
    progress_callback: TrialCentricCutoffProgress | None = None,
) -> TrialCentricCutoffResult:
    """Run the adaptive search with either live or already-cached scores."""
    total_candidates = len(ranked)
    if total_candidates == 0:
        return TrialCentricCutoffResult(
            cutoff=0,
            total_candidates=0,
            check_method=method,
            score_threshold=threshold,
            patients_per_side=radius,
            initial_cutoff_proportion=starting_proportion,
            scored_candidates=_empty_scored_candidates(method),
            search_history=pd.DataFrame(columns=_HISTORY_COLUMNS),
        )

    score_column = (
        "match_quality_score"
        if method == "trial_checker"
        else "llm_match_quality_score"
    )
    score_cache: dict[int, float] = {}
    scored_records: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    lower_bound = 0
    upper_bound = total_candidates
    iteration = 0
    initial_cutoff = int(total_candidates * starting_proportion)
    largest_remaining_interval = max(
        initial_cutoff + 1,
        total_candidates - initial_cutoff,
    )
    maximum_iterations = 1 + math.ceil(math.log2(largest_remaining_interval))

    while lower_bound < upper_bound:
        iteration += 1
        proposed_cutoff = (
            initial_cutoff if iteration == 1 else (lower_bound + upper_bound) // 2
        )
        window_start = max(0, proposed_cutoff - radius)
        window_stop = min(total_candidates, proposed_cutoff + radius)
        window_positions = list(range(window_start, window_stop))
        unseen_positions = [
            position for position in window_positions if position not in score_cache
        ]

        if unseen_positions:
            checker_output = score_rows(ranked.iloc[unseen_positions].copy())
            if len(checker_output) != len(unseen_positions):
                raise ValueError(
                    "Cutoff checker returned a different number of rows than "
                    "the probe window."
                )
            if score_column not in checker_output.columns:
                raise ValueError(f"Cutoff checker output is missing '{score_column}'.")

            for position, (_, output_row) in zip(
                unseen_positions,
                checker_output.iterrows(),
                strict=True,
            ):
                score = float(output_row[score_column])
                passed = bool(math.isfinite(score) and score >= threshold)
                score_cache[position] = score
                record = output_row.to_dict()
                record.update(
                    {
                        "rank": int(ranked.iloc[position]["rank"]),
                        "cutoff_probe_score": score,
                        "cutoff_probe_pass": passed,
                        "cutoff_probe_iteration": iteration,
                    }
                )
                scored_records.append(record)

        pass_count = sum(
            math.isfinite(score_cache[position]) and score_cache[position] >= threshold
            for position in window_positions
        )
        window_count = len(window_positions)
        fail_count = window_count - pass_count
        lower_before = lower_bound
        upper_before = upper_bound
        if pass_count > fail_count:
            decision = "search_lower_ranks"
            lower_bound = min(upper_bound, proposed_cutoff + 1)
        else:
            decision = "search_higher_ranks"
            upper_bound = proposed_cutoff

        history.append(
            {
                "iteration": iteration,
                "proposed_cutoff": proposed_cutoff,
                "lower_bound_before": lower_before,
                "upper_bound_before": upper_before,
                "window_start_rank": window_start + 1,
                "window_end_rank": window_stop,
                "window_patient_count": window_count,
                "newly_scored_patient_count": len(unseen_positions),
                "pass_count": pass_count,
                "fail_count": fail_count,
                "pass_fraction": pass_count / window_count,
                "decision": decision,
                "lower_bound_after": lower_bound,
                "upper_bound_after": upper_bound,
            }
        )
        if progress_callback is not None:
            direction = (
                "down the ranking" if pass_count > fail_count else "toward the top"
            )
            progress_callback(
                iteration,
                maximum_iterations,
                (
                    f"Cutoff probe {iteration}: {pass_count}/{window_count} "
                    f"passed; searching {direction}."
                ),
            )

    scored_candidates = pd.DataFrame.from_records(scored_records)
    if scored_candidates.empty:
        scored_candidates = _empty_scored_candidates(method)
    else:
        scored_candidates = scored_candidates.sort_values(
            "rank",
            kind="stable",
        ).reset_index(drop=True)

    return TrialCentricCutoffResult(
        cutoff=lower_bound,
        total_candidates=total_candidates,
        check_method=method,
        score_threshold=threshold,
        patients_per_side=radius,
        initial_cutoff_proportion=starting_proportion,
        scored_candidates=scored_candidates,
        search_history=pd.DataFrame.from_records(history, columns=_HISTORY_COLUMNS),
    )


def find_trial_centric_cutoff(
    candidate_pairs: pd.DataFrame,
    *,
    check_method: str = "trial_checker",
    score_threshold: float | None = None,
    patients_per_side: int = DEFAULT_CUTOFF_PATIENTS_PER_SIDE,
    initial_cutoff_proportion: float = DEFAULT_INITIAL_CUTOFF_PROPORTION,
    config: MMAIConfig | None = None,
    progress_callback: TrialCentricCutoffProgress | None = None,
) -> TrialCentricCutoffResult:
    """Find how many TrialSpace-ranked patients should receive both checkers.

    The input must contain the complete patient ranking for one trial space,
    normally produced with ``generate_candidate_matches(..., k=None)`` and
    joined to the patient and trial summary text. By default the search starts
    halfway down the ranking; ``initial_cutoff_proportion`` can move that first
    probe when the expected qualifying fraction is substantially smaller or
    larger. At each proposed cutoff it scores the patients immediately above
    and below the boundary, then bisects toward the top when failures are the
    majority (or the window is tied) and toward the bottom when passes are the
    strict majority. Overlapping probe windows reuse prior scores.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Complete TrialSpace-ranked patient pairs for exactly one trial space.
        Required columns are ``patient_id``, ``space_trial_id``, ``rank``,
        ``cancer_history_summary``, and ``clinical_space_summary``.
    check_method : {"trial_checker", "llm"}, default "trial_checker"
        Use TrialChecker's 0-1 sigmoid score or the LLM checker's 0-5 score for
        the boundary probes.
    score_threshold : float, optional
        Score at or above which a probe patient passes. Defaults to 0.20 for
        TrialChecker and 1 for the LLM checker.
    patients_per_side : int, default 10
        Number of patients to probe on each side of a proposed cutoff. Windows
        are clipped at the top and bottom of the ranking.
    initial_cutoff_proportion : float, default 0.5
        Proportion of the complete ranking at which to place the first cutoff
        probe. Must be between 0 (the top) and 1 (the bottom). Later probes use
        the same bisection behavior regardless of this starting point.
    config : MMAIConfig, optional
        Package configuration passed to the selected checker.
    progress_callback : callable, optional
        Called after each iteration as ``(completed, maximum, message)``.

    Returns
    -------
    TrialCentricCutoffResult
        The selected cutoff, unique probe scores, and iteration diagnostics.

    Notes
    -----
    This heuristic assumes that checker pass rates generally decline with
    TrialSpace rank. It estimates a compute cutoff; it does not establish
    patient eligibility or replace review of the complete current protocol.
    """
    method = _normalize_check_method(check_method)
    threshold = _resolve_score_threshold(method, score_threshold)
    radius = _validate_patients_per_side(patients_per_side)
    starting_proportion = _validate_initial_cutoff_proportion(initial_cutoff_proportion)
    ranked = _prepare_ranked_candidates(candidate_pairs)

    def score_rows(rows: pd.DataFrame) -> pd.DataFrame:
        if method == "trial_checker":
            return score_match_quality(
                rows,
                config=config,
                filter_low_quality=False,
            )
        return score_match_quality_with_llm(rows, config=config)

    return _run_cutoff_search(
        ranked,
        method=method,
        threshold=threshold,
        radius=radius,
        starting_proportion=starting_proportion,
        score_rows=score_rows,
        progress_callback=progress_callback,
    )


def _validate_stability_proportions(
    values: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("initial_cutoff_proportions must contain numeric values.")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "initial_cutoff_proportions must contain numeric values."
        ) from exc
    if not raw_values:
        raise ValueError("initial_cutoff_proportions must not be empty.")
    resolved = tuple(_validate_initial_cutoff_proportion(value) for value in raw_values)
    if len(set(resolved)) != len(resolved):
        raise ValueError("initial_cutoff_proportions must not contain duplicates.")
    return resolved


def _score_complete_ranking_for_stability(
    ranked: pd.DataFrame,
    *,
    threshold: float,
    config: MMAIConfig | None,
) -> pd.DataFrame:
    output_columns = [
        "patient_id",
        "space_trial_id",
        "rank",
        "match_quality_score",
        "stability_score_pass",
    ]
    if ranked.empty:
        return pd.DataFrame(columns=output_columns)

    checker_output = score_match_quality(
        ranked,
        config=config,
        filter_low_quality=False,
    )
    required = ["patient_id", "space_trial_id", "match_quality_score"]
    missing = [column for column in required if column not in checker_output.columns]
    if missing:
        raise ValueError(
            "TrialChecker stability output is missing required columns: "
            + ", ".join(missing)
        )
    if len(checker_output) != len(ranked):
        raise ValueError(
            "TrialChecker returned a different number of rows than the complete "
            "stability ranking."
        )

    key_columns = ["patient_id", "space_trial_id"]
    if checker_output.duplicated(key_columns).any():
        raise ValueError("TrialChecker stability output contains duplicate pairs.")
    scored = ranked[key_columns + ["rank"]].merge(
        checker_output[required],
        on=key_columns,
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if (scored["_merge"] != "both").any():
        raise ValueError(
            "TrialChecker stability output does not match the complete ranking."
        )
    scored = scored.drop(columns="_merge")
    scored["match_quality_score"] = pd.to_numeric(
        scored["match_quality_score"],
        errors="coerce",
    )
    scored["stability_score_pass"] = [
        bool(math.isfinite(score) and score >= threshold)
        for score in scored["match_quality_score"]
    ]
    return scored[output_columns]


def assess_trial_centric_cutoff_stability(
    candidate_pairs: pd.DataFrame,
    *,
    initial_cutoff_proportions: tuple[float, ...] | list[float] = (
        DEFAULT_STABILITY_INITIAL_CUTOFF_PROPORTIONS
    ),
    score_threshold: float | None = None,
    patients_per_side: int = DEFAULT_CUTOFF_PATIENTS_PER_SIDE,
    config: MMAIConfig | None = None,
    progress_callback: TrialCentricCutoffProgress | None = None,
) -> TrialCentricCutoffStabilityResult:
    """Assess adaptive-cutoff sensitivity to its first boundary position.

    This QA helper runs TrialChecker over the complete ranked corpus exactly
    once, then replays the adaptive search against those cached scores for each
    requested starting proportion. It reports both the selected cutoff and the
    number of threshold-passing patients within the selected prefix. A stable
    result has small spreads across starts.

    Unlike :func:`find_trial_centric_cutoff`, this function intentionally scores
    every candidate. It is an offline diagnostic for synthetic or otherwise
    explicitly authorized data, not the compute-saving production path and not
    an eligibility assessment.

    Parameters
    ----------
    candidate_pairs : pd.DataFrame
        Complete TrialSpace-ranked patient pairs for one trial space, with the
        same required columns as :func:`find_trial_centric_cutoff`.
    initial_cutoff_proportions : sequence of float
        Unique first-probe positions between 0 and 1. Defaults to 10%, 25%,
        50%, 75%, and 90% of the complete ranking.
    score_threshold : float, optional
        TrialChecker score at or above which a patient passes for this QA run.
        Defaults to 0.20 on the 0-1 sigmoid scale.
    patients_per_side : int, default 10
        Number of patients on each side of every replayed boundary probe.
    config : MMAIConfig, optional
        Package configuration used for the one complete TrialChecker pass.
    progress_callback : callable, optional
        Called after each starting-position replay as
        ``(completed, total_starts, message)``.

    Returns
    -------
    TrialCentricCutoffStabilityResult
        Per-start diagnostics, one complete table of cached TrialChecker scores,
        and the cutoff and passing-count spreads.
    """
    proportions = _validate_stability_proportions(initial_cutoff_proportions)
    threshold = _resolve_score_threshold("trial_checker", score_threshold)
    radius = _validate_patients_per_side(patients_per_side)
    ranked = _prepare_ranked_candidates(candidate_pairs)
    total_candidates = len(ranked)
    scored = _score_complete_ranking_for_stability(
        ranked,
        threshold=threshold,
        config=config,
    )

    score_lookup = scored[["patient_id", "space_trial_id", "match_quality_score"]]

    def cached_score_rows(rows: pd.DataFrame) -> pd.DataFrame:
        return rows[["patient_id", "space_trial_id"]].merge(
            score_lookup,
            on=["patient_id", "space_trial_id"],
            how="left",
            validate="one_to_one",
            sort=False,
        )

    run_records: list[dict[str, object]] = []
    for run_number, proportion in enumerate(proportions, start=1):
        result = _run_cutoff_search(
            ranked,
            method="trial_checker",
            threshold=threshold,
            radius=radius,
            starting_proportion=proportion,
            score_rows=cached_score_rows,
        )
        selected = result.cutoff
        reasonable_count = int(
            scored.loc[
                scored["rank"] <= selected,
                "stability_score_pass",
            ].sum()
        )
        run_records.append(
            {
                "initial_cutoff_proportion": proportion,
                "initial_proposed_cutoff": int(total_candidates * proportion),
                "selected_cutoff": selected,
                "selected_cutoff_fraction": (
                    selected / total_candidates if total_candidates else 0.0
                ),
                "reasonable_consideration_count": reasonable_count,
                "reasonable_consideration_fraction": (
                    reasonable_count / selected if selected else 0.0
                ),
                "search_iteration_count": len(result.search_history),
                "probe_patient_count": len(result.scored_candidates),
            }
        )
        if progress_callback is not None:
            progress_callback(
                run_number,
                len(proportions),
                (
                    f"Stability start {proportion:.0%}: selected top "
                    f"{selected:,} with {reasonable_count:,} "
                    "threshold-passing patient(s)."
                ),
            )

    cutoff_runs = pd.DataFrame.from_records(run_records, columns=_STABILITY_COLUMNS)
    selected_cutoff_spread = int(
        cutoff_runs["selected_cutoff"].max() - cutoff_runs["selected_cutoff"].min()
    )
    reasonable_count_spread = int(
        cutoff_runs["reasonable_consideration_count"].max()
        - cutoff_runs["reasonable_consideration_count"].min()
    )
    return TrialCentricCutoffStabilityResult(
        total_candidates=total_candidates,
        score_threshold=threshold,
        patients_per_side=radius,
        initial_cutoff_proportions=proportions,
        cutoff_runs=cutoff_runs,
        scored_candidates=scored,
        selected_cutoff_spread=selected_cutoff_spread,
        reasonable_consideration_count_spread=reasonable_count_spread,
    )


__all__ = [
    "DEFAULT_CUTOFF_PATIENTS_PER_SIDE",
    "DEFAULT_INITIAL_CUTOFF_PROPORTION",
    "DEFAULT_LLM_CUTOFF_THRESHOLD",
    "DEFAULT_STABILITY_INITIAL_CUTOFF_PROPORTIONS",
    "DEFAULT_TRIAL_CHECKER_CUTOFF_THRESHOLD",
    "CutoffCheckMethod",
    "TrialCentricCutoffProgress",
    "TrialCentricCutoffResult",
    "TrialCentricCutoffStabilityResult",
    "assess_trial_centric_cutoff_stability",
    "find_trial_centric_cutoff",
]
