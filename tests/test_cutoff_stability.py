from __future__ import annotations

import pandas as pd
import pytest

from matchminer_ai.matching import assess_trial_centric_cutoff_stability


def _ranked_candidates(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"P{rank:03d}" for rank in range(1, count + 1)],
            "space_trial_id": ["T1-1"] * count,
            "rank": list(range(1, count + 1)),
            "cancer_history_summary": [
                f"Synthetic patient summary {rank}" for rank in range(1, count + 1)
            ],
            "clinical_space_summary": ["Synthetic trial space"] * count,
        }
    )


def test_stability_scores_complete_ranking_once_and_reuses_scores(monkeypatch):
    ranked = _ranked_candidates(20)
    calls: list[list[int]] = []

    def fake_score(rows, *, config, filter_low_quality):
        del config
        assert filter_low_quality is False
        ranks = rows["rank"].astype(int).tolist()
        calls.append(ranks)
        output = pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "match_quality_score": [0.9 if rank <= 7 else 0.1 for rank in ranks],
            }
        )
        return output.iloc[::-1].reset_index(drop=True)

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fake_score,
    )
    progress: list[tuple[int, int, str]] = []

    result = assess_trial_centric_cutoff_stability(
        ranked,
        initial_cutoff_proportions=[0.1, 0.5, 0.9],
        patients_per_side=2,
        progress_callback=lambda *items: progress.append(items),
    )

    assert calls == [list(range(1, 21))]
    assert result.cutoff_runs["selected_cutoff"].tolist() == [7, 7, 7]
    assert result.cutoff_runs["reasonable_consideration_count"].tolist() == [
        7,
        7,
        7,
    ]
    assert result.selected_cutoff_spread == 0
    assert result.reasonable_consideration_count_spread == 0
    assert result.scored_candidates["rank"].tolist() == list(range(1, 21))
    assert result.scored_candidates["stability_score_pass"].sum() == 7
    assert [completed for completed, _, _ in progress] == [1, 2, 3]
    assert all(total == 3 for _, total, _ in progress)


def test_stability_uses_requested_threshold_for_retained_pass_count(monkeypatch):
    ranked = _ranked_candidates(12)

    def fake_score(rows, *, config, filter_low_quality):
        del config, filter_low_quality
        return pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "match_quality_score": [0.5] * len(rows),
            }
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fake_score,
    )

    passing = assess_trial_centric_cutoff_stability(
        ranked,
        initial_cutoff_proportions=[0.25, 0.75],
        score_threshold=0.4,
        patients_per_side=2,
    )
    failing = assess_trial_centric_cutoff_stability(
        ranked,
        initial_cutoff_proportions=[0.25, 0.75],
        score_threshold=0.6,
        patients_per_side=2,
    )

    assert passing.cutoff_runs["reasonable_consideration_count"].tolist() == [
        12,
        12,
    ]
    assert failing.cutoff_runs["reasonable_consideration_count"].tolist() == [0, 0]


def test_stability_empty_ranking_does_not_load_trial_checker(monkeypatch):
    def fail_score(*args, **kwargs):
        del args, kwargs
        raise AssertionError("TrialChecker should not run for an empty ranking")

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fail_score,
    )

    result = assess_trial_centric_cutoff_stability(
        _ranked_candidates(0),
        initial_cutoff_proportions=[0.0, 1.0],
    )

    assert result.cutoff_runs["selected_cutoff"].tolist() == [0, 0]
    assert result.cutoff_runs["reasonable_consideration_count"].tolist() == [0, 0]
    assert result.scored_candidates.empty
    assert result.selected_cutoff_spread == 0


@pytest.mark.parametrize(
    "proportions",
    [[], [0.5, 0.5], [-0.01], [1.01], [float("nan")], "0.5"],
)
def test_stability_validates_starting_proportions(proportions):
    with pytest.raises(ValueError, match="initial_cutoff_proportion"):
        assess_trial_centric_cutoff_stability(
            _ranked_candidates(0),
            initial_cutoff_proportions=proportions,
        )
