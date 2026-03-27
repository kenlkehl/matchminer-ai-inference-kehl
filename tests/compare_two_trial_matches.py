from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def summarize_match_consistency(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    trial_col: str = "trial_id",
    rank_col: str = "rank",
    score_col: str = "similarity_score",
    topk_score_source: str = "a",
    include_score_stratification: bool = True,
) -> Dict[str, Any]:
    """
    Summarize consistency between two patient->trial match result tables.

    Headline metrics returned:
    - full-set jaccard: mean, median
    - top-5 overlap fraction: mean, median
    - top-10 overlap fraction: mean, median
    - top-1 agreement: proportion

    Optional diagnostic returned:
    - top-5 overlap fraction stratified by mean top-5 score (low vs high)

    Parameters
    ----------
    df_a, df_b
        DataFrames containing patient-trial matches.
        Expected columns:
            - patient_col
            - trial_col
            - rank_col
        Optional:
            - score_col (required only for score stratification)

    patient_col, trial_col, rank_col, score_col
        Column names.

    topk_score_source
        Which dataframe to use for the top-5 score stratification.
        Must be one of: {"a", "b"}.

    include_score_stratification
        If True, compute the optional diagnostic:
        top-5 overlap fraction split by low vs high mean top-5 score.

    Returns
    -------
    dict
        Dictionary with:
        - "headline_metrics": compact summary for reporting
        - "per_patient_metrics": per-patient table
        - "score_stratification": optional stratified summary or None

    Notes
    -----
    - Top-k overlap fraction is defined as:
          |top-k_A ∩ top-k_B| / min(k, n_A, n_B)
      so patients with fewer than k returned trials are handled fairly.
    - Full-set Jaccard is:
          |A ∩ B| / |A ∪ B|
    - Top-1 agreement is 1 if the top-ranked trial is identical, else 0.
    - Duplicate patient-trial pairs within a dataframe are dropped,
      keeping the best-ranked occurrence.
    """

    if topk_score_source not in {"a", "b"}:
        raise ValueError("topk_score_source must be one of {'a', 'b'}")

    required_cols = {patient_col, trial_col, rank_col}
    missing_a = required_cols - set(df_a.columns)
    missing_b = required_cols - set(df_b.columns)

    if missing_a:
        raise ValueError(f"df_a is missing required columns: {sorted(missing_a)}")
    if missing_b:
        raise ValueError(f"df_b is missing required columns: {sorted(missing_b)}")

    if include_score_stratification:
        if score_col not in df_a.columns or score_col not in df_b.columns:
            raise ValueError(
                f"include_score_stratification=True requires '{score_col}' in both dataframes"
            )

    a = df_a.copy()
    b = df_b.copy()

    # Standardize types a bit for safer matching
    for d in (a, b):
        d[patient_col] = d[patient_col].astype(str)
        d[trial_col] = d[trial_col].astype(str)

    # Keep first occurrence after sorting by rank, so duplicates retain the best rank
    a = a.sort_values([patient_col, rank_col]).drop_duplicates(
        [patient_col, trial_col], keep="first"
    )
    b = b.sort_values([patient_col, rank_col]).drop_duplicates(
        [patient_col, trial_col], keep="first"
    )

    def _build_patient_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for pid, sub in df.groupby(patient_col):
            sub = sub.sort_values(rank_col).copy()

            ordered_trials = sub[trial_col].tolist()
            trial_set = set(ordered_trials)

            info: Dict[str, Any] = {
                "ordered_trials": ordered_trials,
                "trial_set": trial_set,
                "n_trials": len(ordered_trials),
            }

            if score_col in sub.columns:
                ordered_scores = sub[score_col].tolist()
                k_eff = min(5, len(ordered_scores))
                info["ordered_scores"] = ordered_scores
                info["mean_top5_score"] = (
                    float(np.mean(ordered_scores[:k_eff])) if k_eff > 0 else np.nan
                )
            else:
                info["ordered_scores"] = None
                info["mean_top5_score"] = np.nan

            out[pid] = info
        return out

    a_lookup = _build_patient_lookup(a)
    b_lookup = _build_patient_lookup(b)

    all_patients = sorted(set(a_lookup) | set(b_lookup))

    def _topk_overlap_fraction(
        ordered_a: list[str],
        ordered_b: list[str],
        k: int,
    ) -> float:
        n_a = len(ordered_a)
        n_b = len(ordered_b)

        denom = min(k, n_a, n_b)
        if denom == 0:
            return float("nan")

        top_a = set(ordered_a[: min(k, n_a)])
        top_b = set(ordered_b[: min(k, n_b)])
        overlap = len(top_a & top_b)
        return overlap / denom

    per_patient_rows = []

    empty_info = {
        "ordered_trials": [],
        "trial_set": set(),
        "n_trials": 0,
        "ordered_scores": None,
        "mean_top5_score": np.nan,
    }

    for pid in all_patients:
        info_a = a_lookup.get(pid, empty_info)
        info_b = b_lookup.get(pid, empty_info)

        ordered_a = info_a["ordered_trials"]
        ordered_b = info_b["ordered_trials"]
        set_a = info_a["trial_set"]
        set_b = info_b["trial_set"]

        union = set_a | set_b
        shared = set_a & set_b

        full_jaccard = len(shared) / len(union) if len(union) > 0 else np.nan

        top1_a = ordered_a[0] if ordered_a else None
        top1_b = ordered_b[0] if ordered_b else None
        top1_agreement = (
            int(top1_a == top1_b)
            if (top1_a is not None and top1_b is not None)
            else np.nan
        )

        row = {
            patient_col: pid,
            "n_trials_a": len(ordered_a),
            "n_trials_b": len(ordered_b),
            "full_set_jaccard": full_jaccard,
            "top5_overlap_fraction": _topk_overlap_fraction(ordered_a, ordered_b, 5),
            "top10_overlap_fraction": _topk_overlap_fraction(ordered_a, ordered_b, 10),
            "top1_agreement": top1_agreement,
            "mean_top5_score_a": info_a["mean_top5_score"],
            "mean_top5_score_b": info_b["mean_top5_score"],
        }
        per_patient_rows.append(row)

    per_patient_metrics = pd.DataFrame(per_patient_rows)

    headline_metrics = {
        "full_set_jaccard_mean": float(per_patient_metrics["full_set_jaccard"].mean()),
        "full_set_jaccard_median": float(
            per_patient_metrics["full_set_jaccard"].median()
        ),
        "top5_overlap_fraction_mean": float(
            per_patient_metrics["top5_overlap_fraction"].mean()
        ),
        "top5_overlap_fraction_median": float(
            per_patient_metrics["top5_overlap_fraction"].median()
        ),
        "top10_overlap_fraction_mean": float(
            per_patient_metrics["top10_overlap_fraction"].mean()
        ),
        "top10_overlap_fraction_median": float(
            per_patient_metrics["top10_overlap_fraction"].median()
        ),
        "top1_agreement_proportion": float(
            per_patient_metrics["top1_agreement"].mean()
        ),
        "n_patients": int(len(per_patient_metrics)),
    }

    score_stratification: Optional[Dict[str, Any]] = None

    if include_score_stratification:
        strat_col = f"mean_top5_score_{topk_score_source}"
        tmp = per_patient_metrics[
            [patient_col, "top5_overlap_fraction", strat_col]
        ].copy()
        tmp = tmp.dropna(subset=[strat_col, "top5_overlap_fraction"])

        if len(tmp) > 0:
            median_score = float(tmp[strat_col].median())
            tmp["score_group"] = np.where(tmp[strat_col] < median_score, "low", "high")

            grouped = (
                tmp.groupby("score_group", dropna=False)["top5_overlap_fraction"]
                .agg(["mean", "median", "count"])
                .reset_index()
            )

            # Ensure stable ordering
            grouped["score_group"] = pd.Categorical(
                grouped["score_group"],
                categories=["low", "high"],
                ordered=True,
            )
            grouped = grouped.sort_values("score_group")

            score_stratification = {
                "score_source": topk_score_source,
                "stratification_metric": strat_col,
                "median_cutpoint": median_score,
                "summary_table": grouped,
            }
        else:
            score_stratification = {
                "score_source": topk_score_source,
                "stratification_metric": strat_col,
                "median_cutpoint": np.nan,
                "summary_table": pd.DataFrame(
                    columns=["score_group", "mean", "median", "count"]
                ),
            }

    return {
        "headline_metrics": headline_metrics,
        "per_patient_metrics": per_patient_metrics,
        "score_stratification": score_stratification,
    }
