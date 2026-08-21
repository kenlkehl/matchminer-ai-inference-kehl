from __future__ import annotations

import pandas as pd
import pytest

from matchminer_ai.config import MMAIConfig
from matchminer_ai.paradigms import rank_patient_space_paradigms


def _config() -> MMAIConfig:
    return MMAIConfig(
        preset_name="test",
        debug_mode=False,
        trial={},
        patient={},
        local={},
        remote={},
        model_metadata_cache_dir=None,
        raw={
            "match_quality": {
                "model_name": "checker/test",
                "device": "cpu",
                "prompt_file": "match_quality_checker_template.txt",
                "score_cutoff": 0.2,
            }
        },
        embedding={
            "model_path": "trialspace/test",
            "device": "cpu",
            "prompt_file": "embedding.txt",
            "max_seq_length": 2500,
        },
    )


def _embedding_metadata(*, model_sha: str = "trialspace-sha") -> dict:
    return {
        "config_snapshot": {
            "embedding": {
                "model_path": "trialspace/test",
                "device": "cpu",
                "prompt_file": "embedding.txt",
                "max_seq_length": 2500,
            }
        },
        "model_metadata": {
            "embedding_model": {
                "model_name": "trialspace/test",
                "model_sha": model_sha,
            }
        },
    }


def _inputs():
    patients = pd.DataFrame(
        [
            {
                "patient_id": "patient-1",
                "cancer_history_summary": "Synthetic EGFR-mutated lung cancer",
            }
        ]
    )
    spaces = pd.DataFrame(
        [
            {
                "space_trial_id": "NCT1-1",
                "trial_id": "NCT1",
                "clinical_space_summary": "Closest TrialSpace text",
            },
            {
                "space_trial_id": "NCT2-1",
                "trial_id": "NCT2",
                "clinical_space_summary": "Best TrialChecker text",
            },
            {
                "space_trial_id": "NCT3-1",
                "trial_id": "NCT3",
                "clinical_space_summary": "Unmapped but retrievable text",
            },
        ]
    )
    memberships = pd.DataFrame(
        [
            {"space_trial_id": "NCT1-1", "paradigm_id": "paradigm-a"},
            {"space_trial_id": "NCT2-1", "paradigm_id": "paradigm-a"},
            {"space_trial_id": "NCT2-1", "paradigm_id": "paradigm-b"},
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "paradigm_id": "paradigm-a",
                "paradigm_label": "Reviewed paradigm A",
                "document_status": "reviewed_v2",
            },
            {
                "paradigm_id": "paradigm-b",
                "paradigm_label": "Caveated paradigm B",
                "document_status": "caveated_best_available",
            },
        ]
    )
    return patients, spaces, memberships, catalog


def _mock_models(monkeypatch):
    def fake_embed(df, *, entity_type, config, return_metadata):
        assert return_metadata is True
        if entity_type == "patient":
            embeddings = pd.DataFrame(
                [
                    {
                        "patient_id": patient_id,
                        "embedding": [1.0, 0.0],
                    }
                    for patient_id in df["patient_id"]
                ]
            )
        else:
            vectors = {
                "NCT1-1": [1.0, 0.0],
                "NCT2-1": [0.8, 0.2],
                "NCT3-1": [0.0, 1.0],
            }
            embeddings = pd.DataFrame(
                [
                    {
                        "space_trial_id": space_id,
                        "embedding": vectors[space_id],
                    }
                    for space_id in df["space_trial_id"]
                ]
            )
        return embeddings, _embedding_metadata()

    def fake_score(
        candidate_pairs,
        *,
        config,
        filter_low_quality,
        return_metadata,
    ):
        assert filter_low_quality is False
        assert return_metadata is True
        scores = {"NCT1-1": 0.8, "NCT2-1": 0.9, "NCT3-1": 0.1}
        output = candidate_pairs[["patient_id", "space_trial_id"]].copy()
        output["match_quality_score"] = output["space_trial_id"].map(scores)
        output["match_quality_pass"] = output["match_quality_score"] >= 0.2
        return output, {
            "model_metadata": {
                "match_quality_checker": {
                    "model_name": "checker/test",
                    "model_sha": "checker-sha",
                }
            }
        }

    monkeypatch.setattr(
        "matchminer_ai.paradigms.ranking.embed_for_matching",
        fake_embed,
    )
    monkeypatch.setattr(
        "matchminer_ai.paradigms.ranking.score_match_quality",
        fake_score,
    )


def test_rank_patient_space_paradigms_reranks_and_preserves_fanout(monkeypatch):
    _mock_models(monkeypatch)
    patients, spaces, memberships, catalog = _inputs()

    result = rank_patient_space_paradigms(
        patients,
        spaces,
        memberships,
        catalog,
        config=_config(),
        retrieval_k=3,
        top_space_count=2,
    )

    assert result.space_matches["space_trial_id"].tolist() == ["NCT2-1", "NCT1-1"]
    assert result.space_matches["space_rank"].tolist() == [1, 2]
    assert result.space_matches["trialspace_rank"].tolist() == [2, 1]
    assert result.space_matches["paradigm_count"].tolist() == [2, 1]
    assert result.space_matches["trial_id"].tolist() == ["NCT2", "NCT1"]

    mapped = result.space_paradigm_matches
    assert list(zip(mapped["space_trial_id"], mapped["paradigm_id"])) == [
        ("NCT2-1", "paradigm-a"),
        ("NCT2-1", "paradigm-b"),
        ("NCT1-1", "paradigm-a"),
    ]

    paradigms = result.paradigm_matches
    assert paradigms["paradigm_id"].tolist() == ["paradigm-a", "paradigm-b"]
    assert paradigms["paradigm_rank"].tolist() == [1, 2]
    assert paradigms["supporting_space_count"].tolist() == [2, 1]
    assert paradigms.loc[0, "supporting_space_ids"] == ("NCT2-1", "NCT1-1")
    assert paradigms.loc[1, "document_status"] == "caveated_best_available"
    assert result.metadata["model_metadata"]["embedding_model"]["model_sha"] == (
        "trialspace-sha"
    )
    assert result.metadata["trial_embeddings_reused"] is False


def test_rank_patient_space_paradigms_allows_shared_spaces_across_patients(
    monkeypatch,
):
    _mock_models(monkeypatch)
    patients, spaces, memberships, catalog = _inputs()
    patients = pd.concat(
        [
            patients,
            pd.DataFrame(
                [
                    {
                        "patient_id": "patient-2",
                        "cancer_history_summary": (
                            "Synthetic EGFR-mutated lung cancer"
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    result = rank_patient_space_paradigms(
        patients,
        spaces,
        memberships,
        catalog,
        config=_config(),
        retrieval_k=2,
        top_space_count=1,
    )

    assert list(
        result.space_matches[["patient_id", "space_trial_id"]].itertuples(
            index=False,
            name=None,
        )
    ) == [
        ("patient-1", "NCT2-1"),
        ("patient-2", "NCT2-1"),
    ]
    assert list(
        result.space_paradigm_matches[
            ["patient_id", "space_trial_id", "paradigm_id"]
        ].itertuples(index=False, name=None)
    ) == [
        ("patient-1", "NCT2-1", "paradigm-a"),
        ("patient-1", "NCT2-1", "paradigm-b"),
        ("patient-2", "NCT2-1", "paradigm-a"),
        ("patient-2", "NCT2-1", "paradigm-b"),
    ]


def test_rank_patient_space_paradigms_reuses_compatible_trial_embeddings(
    monkeypatch,
):
    patients, spaces, memberships, catalog = _inputs()
    calls = []

    def patient_embed(df, *, entity_type, config, return_metadata):
        calls.append(entity_type)
        assert entity_type == "patient"
        return (
            pd.DataFrame([{"patient_id": "patient-1", "embedding": [1.0, 0.0]}]),
            _embedding_metadata(),
        )

    def fake_score(
        candidate_pairs,
        *,
        config,
        filter_low_quality,
        return_metadata,
    ):
        output = candidate_pairs[["patient_id", "space_trial_id"]].copy()
        output["match_quality_score"] = 0.5
        output["match_quality_pass"] = True
        return output, {"model_metadata": {"match_quality_checker": {}}}

    monkeypatch.setattr(
        "matchminer_ai.paradigms.ranking.embed_for_matching",
        patient_embed,
    )
    monkeypatch.setattr(
        "matchminer_ai.paradigms.ranking.score_match_quality",
        fake_score,
    )
    cached = pd.DataFrame(
        [
            {"space_trial_id": "NCT3-1", "embedding": [0.0, 1.0]},
            {"space_trial_id": "NCT1-1", "embedding": [1.0, 0.0]},
            {"space_trial_id": "NCT2-1", "embedding": [0.8, 0.2]},
        ]
    )

    result = rank_patient_space_paradigms(
        patients,
        spaces,
        memberships,
        catalog,
        config=_config(),
        retrieval_k=2,
        top_space_count=1,
        trial_embeddings=cached,
        trial_embedding_metadata=_embedding_metadata(),
    )

    assert calls == ["patient"]
    assert result.metadata["trial_embeddings_reused"] is True
    assert result.trial_embeddings["space_trial_id"].tolist() == [
        "NCT1-1",
        "NCT2-1",
        "NCT3-1",
    ]


def test_rank_patient_space_paradigms_rejects_incompatible_cached_model(
    monkeypatch,
):
    _mock_models(monkeypatch)
    patients, spaces, memberships, catalog = _inputs()
    cached = pd.DataFrame(
        [
            {"space_trial_id": "NCT1-1", "embedding": [1.0, 0.0]},
            {"space_trial_id": "NCT2-1", "embedding": [0.8, 0.2]},
            {"space_trial_id": "NCT3-1", "embedding": [0.0, 1.0]},
        ]
    )

    with pytest.raises(
        ValueError, match="Patient and trial embeddings are incompatible"
    ):
        rank_patient_space_paradigms(
            patients,
            spaces,
            memberships,
            catalog,
            config=_config(),
            retrieval_k=2,
            top_space_count=1,
            trial_embeddings=cached,
            trial_embedding_metadata=_embedding_metadata(model_sha="different-sha"),
        )


def test_rank_patient_space_paradigms_validates_counts_and_mapping():
    patients, spaces, memberships, catalog = _inputs()

    with pytest.raises(ValueError, match="greater than or equal"):
        rank_patient_space_paradigms(
            patients,
            spaces,
            memberships,
            catalog,
            config=_config(),
            retrieval_k=1,
            top_space_count=2,
        )

    duplicate_membership = pd.concat([memberships, memberships.iloc[[0]]])
    with pytest.raises(ValueError, match="unique space_trial_id/paradigm_id"):
        rank_patient_space_paradigms(
            patients,
            spaces,
            duplicate_membership,
            catalog,
            config=_config(),
            retrieval_k=2,
            top_space_count=1,
        )


def test_rank_patient_space_paradigms_rejects_patient_columns_in_trial_table():
    patients, spaces, memberships, catalog = _inputs()
    spaces["patient_summary"] = "must not enter the trial corpus"

    with pytest.raises(ValueError, match="patient-bearing columns: patient_summary"):
        rank_patient_space_paradigms(
            patients,
            spaces,
            memberships,
            catalog,
            config=_config(),
            retrieval_k=2,
            top_space_count=1,
        )


def test_rank_patient_space_paradigms_does_not_force_below_cutoff_match(
    monkeypatch,
):
    _mock_models(monkeypatch)
    patients, spaces, memberships, catalog = _inputs()

    passing_only = rank_patient_space_paradigms(
        patients,
        spaces,
        memberships,
        catalog,
        config=_config(),
        retrieval_k=3,
        top_space_count=3,
    )
    diagnostic = rank_patient_space_paradigms(
        patients,
        spaces,
        memberships,
        catalog,
        config=_config(),
        retrieval_k=3,
        top_space_count=3,
        require_match_quality_pass=False,
    )

    assert passing_only.space_matches["space_trial_id"].tolist() == [
        "NCT2-1",
        "NCT1-1",
    ]
    assert diagnostic.space_matches["space_trial_id"].tolist() == [
        "NCT2-1",
        "NCT1-1",
        "NCT3-1",
    ]
    assert (
        passing_only.metadata["output_counts"]["trialchecker_passing_candidates"] == 2
    )
