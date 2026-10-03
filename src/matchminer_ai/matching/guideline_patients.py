"""Trial-centric retrieval from a caller-supplied population of summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from matchminer_ai.cancellation import check_cancelled
from matchminer_ai.config import load_default_preset
from .guidelines import _patient_frame


def retrieve_guideline_patients(
    patient_summaries,
    catalog,
    *,
    space_trial_id,
    candidate_k=50,
    min_match_score=0.2,
    config=None,
    progress_callback=None,
):
    """Retrieve then TrialChecker-score patients for one finalized guideline space.

    Returns (selected patients, provenance). candidate_k controls the number of
    highest-cosine patient summaries scored, capped at the population size.
    min_match_score=None retains every scored candidate. Scores are prioritization
    signals, not eligibility decisions. No raw notes or remote LLM are used.
    All vectors are made with the same resolved model in this call; none are saved.
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
    progress(f"Embedding {len(patients)} patient summaries with TrialSpace")
    vector_batches = []
    pm = None
    for start in range(0, len(patients), 32):
        check_cancelled()
        batch, batch_metadata = embed_for_matching(
            patients.iloc[start : start + 32],
            entity_type="patient",
            config=config,
            return_metadata=True,
        )
        if pm is not None and pm["model_metadata"] != batch_metadata["model_metadata"]:
            raise ValueError("Patient embedding model changed between batches.")
        pm = batch_metadata
        vector_batches.append(batch)
        progress(
            f"Embedded {min(start + 32, len(patients))}/{len(patients)} patient summaries"
        )
    pv = pd.concat(vector_batches, ignore_index=True)
    check_cancelled()
    progress("Embedding the selected guideline population with TrialSpace")
    sv, sm = embed_for_matching(
        spaces[["space_trial_id", "clinical_space_summary"]],
        entity_type="trial",
        config=config,
        return_metadata=True,
    )
    if (
        pm["model_metadata"]["embedding_model"]
        != sm["model_metadata"]["embedding_model"]
    ):
        raise ValueError("Patient and guideline embedding models differ.")
    for frame in (pv, sv):
        matrix = np.asarray(frame.embedding.tolist(), dtype=float)
        if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
            raise ValueError("TrialSpace returned invalid embeddings.")
    # Query is the trial space; the corpus is the full patient population.
    candidates = generate_candidate_matches(sv, pv, k=candidate_k).rename(
        columns={"rank": "retrieval_rank"}
    )
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
        "candidate_k": candidate_k,
        "scored_count": len(ranked),
        "population_count": len(patients),
        "selected_count": len(selected),
        "min_match_score": min_match_score,
        "candidates": ranked.to_dict("records"),
    }
