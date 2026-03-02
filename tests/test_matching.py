import pandas as pd

from mmai.matching import generate_candidate_matches


def test_generate_candidate_matches_returns_top_k_per_query():
    """Return top-k matches per query sorted by similarity."""
    query_df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    corpus_df = pd.DataFrame(
        {
            "space_trial_id": ["T1-1", "T2-1", "T3-1"],
            "embedding": [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        }
    )

    result = generate_candidate_matches(query_df, corpus_df, k=2)

    assert list(result.columns[:3]) == [
        "patient_id",
        "space_trial_id",
        "similarity_score",
    ]
    assert (result["patient_id"].value_counts() == 2).all()
    p1 = result[result["patient_id"] == "P1"]["space_trial_id"].tolist()
    p2 = result[result["patient_id"] == "P2"]["space_trial_id"].tolist()
    assert p1[0] == "T1-1"
    assert p2[0] == "T3-1"


def test_generate_candidate_matches_returns_all_when_k_none():
    """Return all corpus rows per query when k is None, sorted by similarity."""
    query_df = pd.DataFrame(
        {
            "patient_id": ["P1"],
            "embedding": [[1.0, 0.0]],
        }
    )
    corpus_df = pd.DataFrame(
        {
            "space_trial_id": ["T1-1", "T2-1"],
            "embedding": [[0.0, 1.0], [1.0, 0.0]],
        }
    )

    result = generate_candidate_matches(query_df, corpus_df, k=None)

    assert result["space_trial_id"].tolist() == ["T2-1", "T1-1"]
