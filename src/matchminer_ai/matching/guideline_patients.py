"""Trial-centric retrieval from a caller-supplied population of summaries."""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from matchminer_ai._storage import digest
from matchminer_ai.cancellation import check_cancelled
from matchminer_ai.config import load_default_preset
from .guidelines import _patient_frame


@dataclass(frozen=True)
class PatientEmbeddings:
    """Reusable TrialSpace vectors; row i of ``vectors`` belongs to ``patient_ids[i]``.

    ``summary_sha256`` binds each vector to its exact summary text and
    ``encoder_identity`` to the loaded encoder, so retrieval rejects stale or
    mixed vectors. Vectors derive from patient summaries: store them as privately
    as the summaries themselves.
    """

    patient_ids: tuple[str, ...]
    summary_sha256: tuple[str, ...]
    vectors: np.ndarray
    model_metadata: dict
    encoder_identity: dict | None


def _same(a, b):
    # Saved metadata round-trips through JSON (tuples become lists).
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str
    )


def _summary_digests(patients):
    return tuple(digest(text.encode()) for text in patients.cancer_history_summary)


def _embed_patients(patients, config, progress, batch_size):
    from matchminer_ai.embedding import embed_for_matching

    progress(f"Embedding {len(patients)} patient summaries with TrialSpace")
    vector_batches = []
    pm = None
    for start in range(0, len(patients), batch_size):
        check_cancelled()
        batch, batch_metadata = embed_for_matching(
            patients.iloc[start : start + batch_size],
            entity_type="patient",
            config=config,
            return_metadata=True,
        )
        if pm is not None and pm["model_metadata"] != batch_metadata["model_metadata"]:
            raise ValueError("Patient embedding model changed between batches.")
        pm = batch_metadata
        vector_batches.append(batch)
        progress(
            f"Embedded {min(start + batch_size, len(patients))}/{len(patients)} "
            "patient summaries"
        )
    pv = pd.concat(vector_batches, ignore_index=True)
    if pv.patient_id.astype(str).tolist() != patients.patient_id.tolist():
        raise ValueError("TrialSpace returned embeddings for different patients.")
    return pv, pm


def embed_guideline_patients(
    patient_summaries, *, config=None, batch_size=32, progress_callback=None
):
    """Embed a population once for repeated ``retrieve_guideline_patients`` calls.

    Uses the same TrialSpace configuration and summary text as retrieval. The
    result records each summary's SHA-256 and the verified encoder identity;
    ``encoder_identity`` is None when the encoder revision cannot be verified
    (for example a mutable local model directory), and retrieval then refuses to
    reuse the vectors. Nothing is written to disk here.
    """
    from matchminer_ai.embedding.inference import _embedding_cache_identity

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    config = config or load_default_preset()
    patients = _patient_frame(patient_summaries)
    pv, pm = _embed_patients(
        patients, config, progress_callback or (lambda _: None), batch_size
    )
    vectors = np.asarray(pv.embedding.tolist(), dtype=np.float32)
    if vectors.ndim != 2 or not vectors.size or not np.isfinite(vectors).all():
        raise ValueError("TrialSpace returned invalid embeddings.")
    return PatientEmbeddings(
        patient_ids=tuple(patients.patient_id),
        summary_sha256=_summary_digests(patients),
        vectors=vectors,
        model_metadata=pm["model_metadata"],
        encoder_identity=_embedding_cache_identity(
            config.embedding, pm["model_metadata"]["embedding_model"]
        ),
    )


def _precomputed_patients(patients, embeddings):
    """Select verified rows for these patients; never re-embeds or guesses."""
    if not isinstance(embeddings, PatientEmbeddings):
        raise ValueError("patient_embeddings must come from embed_guideline_patients.")
    vectors = np.asarray(embeddings.vectors)
    count = len(embeddings.patient_ids)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != count
        or len(embeddings.summary_sha256) != count
        or len(set(embeddings.patient_ids)) != count
    ):
        raise ValueError("Precomputed patient embeddings are malformed.")
    index = {pid: row for row, pid in enumerate(embeddings.patient_ids)}
    missing = sum(pid not in index for pid in patients.patient_id)
    if missing:
        raise ValueError(
            f"Precomputed embeddings lack {missing} patients; rebuild them."
        )
    rows = np.fromiter((index[pid] for pid in patients.patient_id), dtype=np.int64)
    if any(
        embeddings.summary_sha256[row] != text
        for row, text in zip(rows, _summary_digests(patients))
    ):
        raise ValueError(
            "Patient summaries changed after their embeddings were made; rebuild them."
        )
    selected = vectors[rows]
    if not np.isfinite(selected).all():
        raise ValueError("Precomputed patient embeddings are invalid.")
    pv = pd.DataFrame(
        {"patient_id": patients.patient_id.to_numpy(), "embedding": list(selected)}
    )
    return pv, {"model_metadata": embeddings.model_metadata}


def retrieve_guideline_patients(
    patient_summaries,
    catalog,
    *,
    space_trial_id,
    candidate_k=50,
    min_match_score=0.2,
    config=None,
    progress_callback=None,
    patient_embeddings=None,
):
    """Retrieve then TrialChecker-score patients for one finalized guideline space.

    Returns (selected patients, provenance). candidate_k controls the number of
    highest-cosine patient summaries scored, capped at the population size.
    min_match_score=None retains every scored candidate. Scores are prioritization
    signals, not eligibility decisions. No raw notes or remote LLM are used.
    Patient vectors are made in this call unless ``patient_embeddings`` from
    ``embed_guideline_patients`` covers every patient with unchanged summary text
    and the same verified encoder used for the guideline space. None are saved.
    """
    from matchminer_ai.embedding import embed_for_matching
    from matchminer_ai.matching import generate_candidate_matches, score_match_quality
    from matchminer_ai.trials import get_guideline_considerations

    if type(candidate_k) is not int or candidate_k < 1:
        raise ValueError("candidate_k must be a positive integer.")
    if min_match_score is not None and (
        isinstance(min_match_score, bool)
        or not np.isfinite(min_match_score)
        or not 0 <= min_match_score <= 1
    ):
        raise ValueError("min_match_score must be between zero and one, or None.")
    config = config or load_default_preset()
    patients = _patient_frame(patient_summaries)
    progress = progress_callback or (lambda _: None)
    spaces, source = get_guideline_considerations(
        catalog, space_trial_id=space_trial_id, return_metadata=True
    )
    if len(spaces) != 1:
        raise ValueError("Select exactly one finalized guideline space.")
    check_cancelled()
    if patient_embeddings is None:
        pv, pm = _embed_patients(patients, config, progress, 32)
    else:
        progress(f"Reusing precomputed TrialSpace embeddings for {len(patients)} patients")
        pv, pm = _precomputed_patients(patients, patient_embeddings)
    check_cancelled()
    progress("Embedding the selected guideline population with TrialSpace")
    sv, sm = embed_for_matching(
        spaces[["space_trial_id", "clinical_space_summary"]],
        entity_type="trial",
        config=config,
        return_metadata=True,
    )
    if not _same(
        pm["model_metadata"]["embedding_model"], sm["model_metadata"]["embedding_model"]
    ):
        raise ValueError("Patient and guideline embedding models differ.")
    if patient_embeddings is not None:
        from matchminer_ai.embedding.inference import _embedding_cache_identity

        current = _embedding_cache_identity(
            config.embedding, sm["model_metadata"]["embedding_model"]
        )
        if current is None or not _same(patient_embeddings.encoder_identity, current):
            raise ValueError(
                "Precomputed patient embeddings came from a different or unverifiable "
                "TrialSpace encoder; rebuild them."
            )
    colbert = "query_embedding" in sv
    for frame in (pv, sv):
        if colbert:
            from .late_interaction import _contract, _vectors

            _contract(frame)
            _vectors(frame.embedding)
            continue
        matrix = np.asarray(frame.embedding.tolist(), dtype=float)
        if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
            raise ValueError("TrialSpace returned invalid embeddings.")
    # Query is the trial space; the corpus is the full patient population.
    candidates = generate_candidate_matches(
        sv, pv, k=candidate_k, **({"config": config} if colbert else {})
    ).rename(columns={"rank": "retrieval_rank"})
    pairs = candidates.merge(patients, on="patient_id", validate="one_to_one").merge(
        spaces[["space_trial_id", "clinical_space_summary"]],
        on="space_trial_id",
        validate="many_to_one",
    )
    check_cancelled()
    progress(f"TrialChecker scoring {len(pairs)} candidate patients")
    score_batches = []
    metadata = None
    for start in range(0, len(pairs), 32):
        check_cancelled()
        batch, batch_metadata = score_match_quality(
            pairs.iloc[start : start + 32],
            config=config,
            filter_low_quality=False,
            return_metadata=True,
        )
        if (
            metadata is not None
            and metadata["model_metadata"] != batch_metadata["model_metadata"]
        ):
            raise ValueError("TrialChecker model changed between batches.")
        metadata = batch_metadata
        score_batches.append(batch)
        progress(
            f"TrialChecked {min(start + 32, len(pairs))}/{len(pairs)} candidate patients"
        )
    scores = pd.concat(score_batches, ignore_index=True)
    keys = ["patient_id", "space_trial_id"]
    if len(scores) != len(pairs) or scores.duplicated(keys).any():
        raise ValueError("TrialChecker must return one score per candidate.")
    ranked = candidates.merge(
        scores[keys + ["match_quality_score", "match_quality_pass"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if not np.isfinite(
        pd.to_numeric(ranked.match_quality_score, errors="coerce")
    ).all():
        raise ValueError("TrialChecker returned invalid scores.")
    ranked = ranked.sort_values(
        ["match_quality_score", "similarity_score", "patient_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    selected = (
        ranked
        if min_match_score is None
        else ranked.loc[ranked.match_quality_score >= min_match_score]
    )
    selected = selected.merge(patients, on="patient_id", validate="one_to_one")
    check_cancelled()
    progress(f"Selected {len(selected)} patients from {len(ranked)} scored candidates")
    return selected, {
        "catalog": source,
        "space": spaces.iloc[0].to_dict(),
        "model_metadata": {**pm["model_metadata"], **metadata["model_metadata"]},
        "patient_embeddings": "computed" if patient_embeddings is None else "precomputed",
        "candidate_k": candidate_k,
        "scored_count": len(ranked),
        "population_count": len(patients),
        "selected_count": len(selected),
        "min_match_score": min_match_score,
        "candidates": ranked.to_dict("records"),
    }
