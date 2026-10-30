import numpy as np
import pandas as pd
import pytest

from matchminer_ai.matching import retrieve_guideline_patients
from test_guideline_retrieval import catalog_frame as catalog_fixture

catalog_frame = catalog_fixture


@pytest.fixture
def stages(monkeypatch):
    calls = []

    def embed(frame, *, entity_type, **kwargs):
        id_col = "patient_id" if entity_type == "patient" else "space_trial_id"
        result = frame[[id_col]].copy()
        result["embedding"] = (
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]
            if entity_type == "patient"
            else [[1.0, 0.0]]
        )
        assert "note_text" not in frame
        return result, {"model_metadata": {"embed_for_matching": {"model": "same"}}}

    def score(frame, **kwargs):
        calls.append(frame.copy())
        assert kwargs["filter_low_quality"] is False
        result = frame[["patient_id", "space_trial_id"]].copy()
        result["match_quality_score"] = result.patient_id.map(
            {"first": 0.1, "second": 0.9, "third": 0.5}
        )
        result["match_quality_pass"] = result.match_quality_score >= 0.2
        return result.iloc[::-1], {
            "model_metadata": {"score_match_quality": {"model": "test"}}
        }

    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", embed)
    monkeypatch.setattr("matchminer_ai.matching.score_match_quality", score)
    return calls


def patients():
    return pd.DataFrame(
        {
            "patient_id": ["first", "second", "third"],
            "cancer_history_summary": ["Fabricated"] * 3,
            "note_text": ["Must not enter retrieval"] * 3,
        }
    )


def test_trial_is_query_patient_is_corpus_candidate_limit_and_threshold(
    catalog_frame, stages
):
    selected, meta = retrieve_guideline_patients(
        patients(), catalog_frame, space_trial_id="a", candidate_k=2
    )
    assert len(stages[0]) == 2 and set(stages[0].patient_id) == {"first", "second"}
    assert selected.patient_id.tolist() == ["second"]
    assert selected.retrieval_rank.tolist() == [2]
    assert meta["scored_count"] == 2 and meta["population_count"] == 3
    assert meta["space"]["diagnostic_workup"] == catalog_frame.iloc[0].diagnostic_workup


def test_zero_matches_and_candidate_cap(catalog_frame, stages):
    selected, meta = retrieve_guideline_patients(
        patients(),
        catalog_frame,
        space_trial_id="a",
        candidate_k=100,
        min_match_score=1,
    )
    assert selected.empty and meta["scored_count"] == 3
    selected, meta = retrieve_guideline_patients(
        patients(),
        catalog_frame,
        space_trial_id="a",
        candidate_k=100,
        min_match_score=None,
    )
    assert selected.patient_id.tolist() == ["second", "third", "first"]


@pytest.mark.parametrize(
    "options",
    [
        {"candidate_k": 0},
        {"candidate_k": True},
        {"min_match_score": np.nan},
        {"min_match_score": 2},
    ],
)
def test_bad_controls(catalog_frame, options):
    with pytest.raises(ValueError):
        retrieve_guideline_patients(
            patients(), catalog_frame, space_trial_id="a", **options
        )


def test_large_population_reports_batches_and_cancels_before_next_batch(
    catalog_frame, monkeypatch
):
    from matchminer_ai.cancellation import (
        CancellationToken,
        InferenceCancelled,
        cancellation_scope,
    )

    token = CancellationToken()
    calls, progress = [], []
    cohort = pd.DataFrame(
        {
            "patient_id": [str(i) for i in range(70)],
            "cancer_history_summary": ["Fabricated"] * 70,
        }
    )

    def embed(frame, **kwargs):
        calls.append(len(frame))
        result = frame[["patient_id"]].copy()
        result["embedding"] = [[1.0, 0.0]] * len(frame)
        return result, {"model_metadata": {"embed_for_matching": {"model": "same"}}}

    def update(message):
        progress.append(message)
        if message.startswith("Embedded 32/"):
            token.cancel()

    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", embed)
    with cancellation_scope(token), pytest.raises(InferenceCancelled):
        retrieve_guideline_patients(
            cohort, catalog_frame, space_trial_id="a", progress_callback=update
        )
    assert calls == [32]
    assert "Embedded 32/70 patient summaries" in progress


def test_multiple_embedding_and_checker_batches_cover_requested_candidates(
    catalog_frame, monkeypatch
):
    calls, updates = [], []
    cohort = pd.DataFrame(
        {
            "patient_id": [str(i) for i in range(70)],
            "cancer_history_summary": ["Fabricated"] * 70,
        }
    )

    def embed(frame, *, entity_type, **kwargs):
        key = "patient_id" if entity_type == "patient" else "space_trial_id"
        calls.append((entity_type, len(frame)))
        result = frame[[key]].copy()
        result["embedding"] = [[1.0, 0.0]] * len(frame)
        return result, {"model_metadata": {"embed_for_matching": {"model": "fixed"}}}

    def score(frame, **kwargs):
        calls.append(("checker", len(frame)))
        result = frame[["patient_id", "space_trial_id"]].copy()
        result["match_quality_score"] = result.patient_id.astype(int) / 70
        result["match_quality_pass"] = result.match_quality_score >= 0.2
        return result, {"model_metadata": {"score_match_quality": {"model": "fixed"}}}

    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", embed)
    monkeypatch.setattr("matchminer_ai.matching.score_match_quality", score)
    selected, metadata = retrieve_guideline_patients(
        cohort,
        catalog_frame,
        space_trial_id="a",
        candidate_k=40,
        min_match_score=None,
        progress_callback=updates.append,
    )
    assert calls == [
        ("patient", 32),
        ("patient", 32),
        ("patient", 6),
        ("trial", 1),
        ("checker", 32),
        ("checker", 8),
    ]
    assert len(selected) == 40 and selected.patient_id.is_unique
    assert selected.match_quality_score.is_monotonic_decreasing
    assert metadata["population_count"] == 70 and metadata["scored_count"] == 40
    assert "Embedded 70/70 patient summaries" in updates
    assert "TrialChecked 40/40 candidate patients" in updates


def test_precomputed_embeddings_match_computed_retrieval_without_re_embedding(
    catalog_frame, stages, monkeypatch
):
    from dataclasses import replace

    from matchminer_ai.matching import PatientEmbeddings, embed_guideline_patients

    identity = {"encoder": "fabricated-revision"}
    monkeypatch.setattr(
        "matchminer_ai.embedding.inference._embedding_cache_identity",
        lambda config, model: identity,
    )
    cached = embed_guideline_patients(patients())
    assert isinstance(cached, PatientEmbeddings)
    assert cached.patient_ids == ("first", "second", "third")
    assert cached.vectors.dtype == np.float32 and cached.vectors.shape == (3, 2)
    assert cached.encoder_identity == identity
    expected, _ = retrieve_guideline_patients(
        patients(), catalog_frame, space_trial_id="a", candidate_k=2
    )

    embedded = []
    monkeypatch.setattr(
        "matchminer_ai.embedding.embed_for_matching",
        lambda frame, *, entity_type, **kw: (
            embedded.append(entity_type),
            (
                frame[["space_trial_id"]].assign(embedding=[[1.0, 0.0]]),
                {"model_metadata": {"embed_for_matching": {"model": "same"}}},
            ),
        )[1],
    )
    selected, meta = retrieve_guideline_patients(
        patients(), catalog_frame, space_trial_id="a", candidate_k=2,
        patient_embeddings=cached,
    )
    assert embedded == ["trial"]
    assert meta["patient_embeddings"] == "precomputed"
    pd.testing.assert_frame_equal(selected, expected)

    # A subset of a cached population reuses its rows.
    subset, _ = retrieve_guideline_patients(
        patients().iloc[1:], catalog_frame, space_trial_id="a",
        candidate_k=2, min_match_score=None, patient_embeddings=cached,
    )
    assert set(subset.patient_id) == {"second", "third"}

    changed = patients()
    changed.loc[1, "cancer_history_summary"] = "Edited after embedding"
    missing = pd.concat([patients(), patients().iloc[:1].assign(patient_id="fourth")])
    for frame, embeddings, message in [
        (changed, cached, "changed after"),
        (missing, cached, "lack 1 patients"),
        (patients(), replace(cached, encoder_identity={"encoder": "other"}), "different"),
        (patients(), replace(cached, encoder_identity=None), "unverifiable"),
        (patients(), replace(cached, vectors=cached.vectors[:2]), "malformed"),
        (patients(), cached.vectors, "embed_guideline_patients"),
    ]:
        with pytest.raises(ValueError, match=message):
            retrieve_guideline_patients(
                frame, catalog_frame, space_trial_id="a", patient_embeddings=embeddings
            )
    with pytest.raises(ValueError, match="batch_size"):
        embed_guideline_patients(patients(), batch_size=0)


@pytest.fixture
def colbert(monkeypatch):
    from matchminer_ai.config import load_default_preset
    from matchminer_ai.embedding import late_interaction
    from matchminer_ai.embedding.embed import embed_for_matching
    from test_colbert_matching import Encoder

    encoder = Encoder()
    monkeypatch.setattr(late_interaction, "_model", lambda *args: encoder)
    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", embed_for_matching)
    config = load_default_preset()
    config.embedding = dict(
        backend="colbert", model_path="lightonai/GTE-ModernColBERT-v1", device="cpu"
    )
    return encoder, config


def test_colbert_retrieval_scores_patient_token_matrices(catalog_frame, stages, colbert):
    _, config = colbert
    selected, meta = retrieve_guideline_patients(
        patients(), catalog_frame, space_trial_id="a", candidate_k=2,
        min_match_score=None, config=config,
    )
    assert len(stages[0]) == 2 and meta["scored_count"] == 2
    assert selected.similarity_score.notna().all()
    assert set(selected.patient_id) <= {"first", "second", "third"}


def test_precomputed_colbert_tokens_match_computed_retrieval(
    catalog_frame, stages, colbert
):
    from dataclasses import replace

    from matchminer_ai.matching import embed_guideline_patients

    encoder, config = colbert
    cached = embed_guideline_patients(patients(), config=config)
    assert cached.vectors.dtype == object and len(cached.vectors) == 3
    assert all(tokens.ndim == 2 for tokens in cached.vectors)
    assert cached.encoder_identity["artifact_sha256"] == "test-weights"
    expected, _ = retrieve_guideline_patients(
        patients(), catalog_frame, space_trial_id="a", candidate_k=2,
        min_match_score=None, config=config,
    )
    encoder.calls.clear()
    selected, meta = retrieve_guideline_patients(
        patients(), catalog_frame, space_trial_id="a", candidate_k=2,
        min_match_score=None, config=config, patient_embeddings=cached,
    )
    assert len(encoder.calls) == 2  # Guideline query and document roles only.
    assert meta["patient_embeddings"] == "precomputed"
    pd.testing.assert_frame_equal(selected, expected)

    changed = patients()
    changed.loc[1, "cancer_history_summary"] = "Edited after embedding"
    broken = cached.vectors.copy()
    broken[0] = np.full((2, 3), np.nan, dtype=np.float16)
    for frame, embeddings, message in [
        (changed, cached, "changed after"),
        (patients(), replace(cached, encoder_identity={"artifact_sha256": "other"}), "different"),
        (patients(), replace(cached, vectors=broken), "invalid"),
        (patients(), replace(cached, model_metadata={"embedding_model": {}}), "malformed"),
    ]:
        with pytest.raises(ValueError, match=message):
            retrieve_guideline_patients(
                frame, catalog_frame, space_trial_id="a", config=config,
                patient_embeddings=embeddings,
            )
