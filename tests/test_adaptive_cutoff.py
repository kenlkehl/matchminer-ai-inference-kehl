from __future__ import annotations

import pandas as pd
import pytest

from matchminer_ai.matching import find_trial_centric_cutoff


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


def test_find_trial_centric_cutoff_bisects_and_caches_probe_scores(monkeypatch):
    ranked = _ranked_candidates(64)
    calls: list[list[int]] = []

    def fake_score(rows, *, config, filter_low_quality):
        del config
        assert filter_low_quality is False
        ranks = rows["rank"].astype(int).tolist()
        calls.append(ranks)
        return pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "match_quality_score": [0.9 if rank <= 20 else 0.1 for rank in ranks],
                "match_quality_pass": [rank <= 20 for rank in ranks],
            }
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fake_score,
    )

    progress: list[tuple[int, int, str]] = []
    result = find_trial_centric_cutoff(
        ranked,
        patients_per_side=10,
        progress_callback=lambda *items: progress.append(items),
    )

    assert result.cutoff == 20
    assert result.total_candidates == 64
    assert result.check_method == "trial_checker"
    assert result.score_threshold == pytest.approx(0.2)
    assert result.initial_cutoff_proportion == pytest.approx(0.5)
    assert result.search_history.iloc[0]["proposed_cutoff"] == 32
    assert len(progress) == len(result.search_history)
    probed_ranks = [rank for batch in calls for rank in batch]
    assert len(probed_ranks) == len(set(probed_ranks))
    assert result.scored_candidates["rank"].is_unique
    assert result.scored_candidates["cutoff_probe_pass"].dtype == bool


@pytest.mark.parametrize(
    ("score", "expected_cutoff"),
    [(0.9, 17), (0.1, 0)],
)
def test_find_trial_centric_cutoff_can_reach_both_ranking_edges(
    monkeypatch,
    score,
    expected_cutoff,
):
    ranked = _ranked_candidates(17)

    def fake_score(rows, *, config, filter_low_quality):
        del config, filter_low_quality
        return pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "match_quality_score": [score] * len(rows),
                "match_quality_pass": [score >= 0.2] * len(rows),
            }
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fake_score,
    )

    result = find_trial_centric_cutoff(ranked, patients_per_side=3)

    assert result.cutoff == expected_cutoff


@pytest.mark.parametrize(
    ("threshold", "expected_cutoff"),
    [(0.4, 9), (0.6, 0)],
)
def test_find_trial_centric_cutoff_applies_custom_pass_threshold(
    monkeypatch,
    threshold,
    expected_cutoff,
):
    ranked = _ranked_candidates(9)

    def fake_score(rows, *, config, filter_low_quality):
        del config, filter_low_quality
        return pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "match_quality_score": [0.5] * len(rows),
                "match_quality_pass": [True] * len(rows),
            }
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fake_score,
    )

    result = find_trial_centric_cutoff(
        ranked,
        score_threshold=threshold,
        patients_per_side=2,
    )

    assert result.cutoff == expected_cutoff
    assert result.score_threshold == threshold


def test_find_trial_centric_cutoff_starts_at_requested_ranking_proportion(
    monkeypatch,
):
    ranked = _ranked_candidates(100)

    def fake_score(rows, *, config, filter_low_quality):
        del config, filter_low_quality
        return pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "match_quality_score": [0.1] * len(rows),
                "match_quality_pass": [False] * len(rows),
            }
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fake_score,
    )

    result = find_trial_centric_cutoff(
        ranked,
        patients_per_side=3,
        initial_cutoff_proportion=0.1,
    )

    assert result.initial_cutoff_proportion == pytest.approx(0.1)
    assert result.search_history["proposed_cutoff"].iloc[:2].tolist() == [10, 5]
    assert result.cutoff == 0


@pytest.mark.parametrize(
    ("proportion", "score", "expected_cutoff", "first_cutoff"),
    [(0.0, 0.1, 0, 0), (1.0, 0.9, 12, 12)],
)
def test_find_trial_centric_cutoff_accepts_ranking_endpoint_starts(
    monkeypatch,
    proportion,
    score,
    expected_cutoff,
    first_cutoff,
):
    ranked = _ranked_candidates(12)

    def fake_score(rows, *, config, filter_low_quality):
        del config, filter_low_quality
        return pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "match_quality_score": [score] * len(rows),
                "match_quality_pass": [score >= 0.2] * len(rows),
            }
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fake_score,
    )

    result = find_trial_centric_cutoff(
        ranked,
        patients_per_side=2,
        initial_cutoff_proportion=proportion,
    )

    assert result.search_history.iloc[0]["proposed_cutoff"] == first_cutoff
    assert result.cutoff == expected_cutoff


def test_find_trial_centric_cutoff_uses_llm_scale_and_default_threshold(monkeypatch):
    ranked = _ranked_candidates(12)

    def fail_local(*args, **kwargs):
        del args, kwargs
        raise AssertionError("TrialChecker should not run for an LLM cutoff search")

    def fake_llm(rows, *, config):
        del config
        ranks = rows["rank"].astype(int).tolist()
        return pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "llm_match_quality_score": [1 if rank <= 4 else 0 for rank in ranks],
            }
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fail_local,
    )
    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality_with_llm",
        fake_llm,
    )

    result = find_trial_centric_cutoff(
        ranked,
        check_method="llm",
        patients_per_side=2,
    )

    assert result.cutoff == 4
    assert result.score_threshold == pytest.approx(1.0)
    assert "llm_match_quality_score" in result.scored_candidates


def test_find_trial_centric_cutoff_treats_a_tied_window_conservatively(monkeypatch):
    ranked = _ranked_candidates(20)

    def fake_score(rows, *, config, filter_low_quality):
        del config, filter_low_quality
        ranks = rows["rank"].astype(int).tolist()
        return pd.DataFrame(
            {
                "patient_id": rows["patient_id"].tolist(),
                "space_trial_id": rows["space_trial_id"].tolist(),
                "match_quality_score": [0.9 if rank <= 10 else 0.1 for rank in ranks],
                "match_quality_pass": [rank <= 10 for rank in ranks],
            }
        )

    monkeypatch.setattr(
        "matchminer_ai.matching.adaptive_cutoff.score_match_quality",
        fake_score,
    )

    result = find_trial_centric_cutoff(ranked, patients_per_side=10)

    first = result.search_history.iloc[0]
    assert first["pass_count"] == first["fail_count"] == 10
    assert first["decision"] == "search_higher_ranks"
    assert result.cutoff == 10


def test_find_trial_centric_cutoff_validates_complete_single_space_ranking():
    ranked = _ranked_candidates(3)
    ranked.loc[2, "rank"] = 4

    with pytest.raises(ValueError, match="complete TrialSpace ranking"):
        find_trial_centric_cutoff(ranked)

    ranked = _ranked_candidates(3)
    ranked.loc[2, "space_trial_id"] = "T2-1"
    with pytest.raises(ValueError, match="exactly one trial space"):
        find_trial_centric_cutoff(ranked)


@pytest.mark.parametrize(
    ("method", "threshold", "message"),
    [
        ("trial_checker", 1.1, "between 0 and 1"),
        ("llm", 5.1, "between 0 and 5"),
        ("unknown", 0.2, "check_method"),
    ],
)
def test_find_trial_centric_cutoff_validates_method_specific_thresholds(
    method,
    threshold,
    message,
):
    with pytest.raises(ValueError, match=message):
        find_trial_centric_cutoff(
            _ranked_candidates(0),
            check_method=method,
            score_threshold=threshold,
        )


@pytest.mark.parametrize("proportion", [-0.01, 1.01, float("nan")])
def test_find_trial_centric_cutoff_validates_initial_proportion(proportion):
    with pytest.raises(ValueError, match="initial_cutoff_proportion"):
        find_trial_centric_cutoff(
            _ranked_candidates(0),
            initial_cutoff_proportion=proportion,
        )
