"""Retrieve stored guideline considerations with TrialSpace and TrialChecker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset

if TYPE_CHECKING:
    import pandas as pd


def _patient_frame(patient_summaries):
    import pandas as pd

    if not isinstance(patient_summaries, pd.DataFrame):
        raise TypeError("patient_summaries must be a DataFrame.")
    required = ["patient_id", "cancer_history_summary"]
    if any(key not in patient_summaries for key in required):
        raise ValueError(
            "patient_summaries requires patient_id and cancer_history_summary."
        )
    patients = patient_summaries[required].copy()
    if patients.empty or patients["patient_id"].isna().any():
        raise ValueError("Supply at least one patient with a non-null patient_id.")
    patients["patient_id"] = patients["patient_id"].astype(str)
    if (
        patients["patient_id"].duplicated().any()
        or patients["patient_id"].str.strip().eq("").any()
    ):
        raise ValueError("Patient identifiers must be unique and nonempty.")
    if not all(
        isinstance(s, str) and s.strip() for s in patients["cancer_history_summary"]
    ):
        raise ValueError("Patient summaries must be nonempty strings.")
    return patients


def retrieve_guideline_considerations(
    patient_summaries: pd.DataFrame,
    catalog: str | Path | pd.DataFrame | list[str | Path],
    *,
    top_n: int = 5,
    candidate_k: int | None = 20,
    config: MMAIConfig | None = None,
    embedding_cache_dir: str | Path | None = None,
    return_metadata: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Return considerations for each patient's highest TrialChecker-ranked spaces.

    Parameters
    ----------
    patient_summaries : pd.DataFrame
        One row per patient, with ``patient_id`` and ``cancer_history_summary``.
        Only these columns enter model inference. No raw-note summarization occurs.
    catalog : path, DataFrame, or list of paths
        Complete guideline records accepted by ``trials.load_guideline_catalog``.
        Unchanged files reuse validated records in the loader's process-local
        cache; changed catalog/status/audit files are reloaded and validated.
    top_n : int, default 5
        Maximum spaces returned per patient, ranked by TrialChecker score.
    candidate_k : int or None, default 20
        TrialSpace nearest neighbors to rerank per patient. Must be >= top_n,
        or None to score every catalog space. A smaller catalog returns fewer rows.
    config : MMAIConfig, optional
        Uses existing ``embedding`` and ``raw['match_quality']`` settings and
        their bundled prompts. Both models run locally through the public stage
        APIs; no generative LLM or public-source retrieval is used.
    return_metadata : bool, optional
        Also return config, catalog identity, both model metadata records, and
        retrieval/reranking parameters and counts.
    embedding_cache_dir : path, optional
        Persist guideline vectors outside the code repository for reuse across
        calls and process restarts. Keys cover exact text and the loaded encoder
        revision, tokenizer, prompt, sequence cutoff, and inference libraries.
        Patient text and patient vectors are never written to this cache.
    progress_callback : callable, optional
        Receives preparation, embedding, retrieval, scoring, and completion updates.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, dict]
        One row per selected patient-space pair, including ``patient_id``,
        ``rank`` (TrialChecker order), ``retrieval_rank`` (TrialSpace order),
        ``similarity_score``, ``match_quality_score``, ``match_quality_pass``, and
        the complete stored catalog record, including both consideration menus.

    Notes
    -----
    The two public model stages are ``embed_for_matching`` and
    ``score_match_quality``, with ``generate_candidate_matches`` between them.
    Catalog embeddings are computed once per call for all supplied patients,
    or reused from the optional verified cache. Only missing/changed text is
    embedded. Patient embeddings always use the same model configuration.
    Ranking is at space level, not disease/trial level.
    All retrieved candidates are scored with ``filter_low_quality=False`` so a
    cutoff cannot silently remove the requested top N. Low scores remain flagged
    by ``match_quality_pass``. Ties use similarity, then the space ID. A score is
    a prioritization signal, not a probability of clinical appropriateness.
    Stored menus and caveats are returned unchanged, without asserting that every
    option applies to the individual patient. Source and clinician review remain
    necessary, including checking prior treatment, biomarkers, and missing facts.
    """
    import pandas as pd

    from matchminer_ai.embedding import embed_for_matching
    from matchminer_ai.matching import generate_candidate_matches, score_match_quality
    from matchminer_ai.trials import load_guideline_catalog

    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    if candidate_k is not None and (
        type(candidate_k) is not int or candidate_k < top_n
    ):
        raise ValueError("candidate_k must be >= top_n or None.")
    resolved = load_default_preset() if config is None else config
    if not isinstance(resolved, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")

    def progress(message):
        logging.getLogger(__name__).info(message)
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Guideline retrieval progress callback failed"
                )

    patients = _patient_frame(patient_summaries)
    spaces, catalog_metadata = load_guideline_catalog(
        catalog, return_metadata=True, progress_callback=progress
    )
    progress(f"Embedding {len(patients)} patient summaries with TrialSpace")
    patient_vectors, patient_metadata = embed_for_matching(
        patients, entity_type="patient", config=resolved, return_metadata=True
    )
    patient_model = patient_metadata["model_metadata"]["embedding_model"]
    matrix = np.asarray(patient_vectors["embedding"].tolist(), dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("TrialSpace returned invalid or non-finite embeddings.")
    cache_metadata = {"enabled": False}
    if embedding_cache_dir is not None:
        from ._guideline_embedding_cache import cached_guideline_embeddings

        space_vectors, cache_metadata = cached_guideline_embeddings(
            spaces,
            config=resolved,
            patient_model=patient_model,
            dimension=matrix.shape[1],
            directory=embedding_cache_dir,
            progress=progress,
            embed=embed_for_matching,
        )
        space_model = patient_model
    else:
        progress(
            f"Embedding {len(spaces)} guideline spaces with the same TrialSpace model"
        )
        space_vectors, space_metadata = embed_for_matching(
            spaces[["space_trial_id", "clinical_space_summary"]],
            entity_type="trial",
            config=resolved,
            return_metadata=True,
        )
        space_model = space_metadata["model_metadata"]["embedding_model"]
    if patient_model != space_model:
        raise ValueError(
            "Patient and guideline embedding model metadata differ; refusing to mix embeddings."
        )
    for vectors in (patient_vectors, space_vectors):
        matrix = np.asarray(vectors["embedding"].tolist(), dtype=float)
        if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
            raise ValueError("TrialSpace returned invalid or non-finite embeddings.")
    progress("Retrieving nearest guideline spaces by TrialSpace cosine similarity")
    candidates = generate_candidate_matches(
        patient_vectors, space_vectors, k=candidate_k
    )
    candidates = candidates.rename(columns={"rank": "retrieval_rank"})
    pairs = candidates.merge(patients, on="patient_id", validate="many_to_one").merge(
        spaces[["space_trial_id", "clinical_space_summary"]],
        on="space_trial_id",
        validate="many_to_one",
    )
    progress(f"Scoring {len(pairs)} patient-space pairs with TrialChecker")
    scores, checker_metadata = score_match_quality(
        pairs, config=resolved, filter_low_quality=False, return_metadata=True
    )
    keys = ["patient_id", "space_trial_id"]
    if scores.duplicated(keys).any() or len(scores) != len(pairs):
        raise ValueError(
            "TrialChecker did not return exactly one score per candidate pair."
        )
    scored = candidates.merge(scores, on=keys, validate="one_to_one", how="left")
    if not np.isfinite(
        pd.to_numeric(scored["match_quality_score"], errors="coerce")
    ).all():
        raise ValueError("TrialChecker returned missing or non-finite scores.")
    scored = scored.sort_values(
        ["patient_id", "match_quality_score", "similarity_score", "space_trial_id"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    scored["rank"] = scored.groupby("patient_id", sort=False).cumcount() + 1
    selected = scored.loc[scored["rank"] <= top_n]
    result = selected.merge(spaces, on="space_trial_id", validate="many_to_one")
    # Preserve the caller's patient order; within each patient retain checker order.
    order = {pid: i for i, pid in enumerate(patients["patient_id"])}
    result = result.sort_values(
        "patient_id", key=lambda s: s.map(order), kind="stable"
    ).reset_index(drop=True)
    metadata = {
        "package": package_metadata(),
        "config_snapshot": config_snapshot(resolved),
        "model_metadata": {
            "embedding_model": patient_model,
            "match_quality_checker": checker_metadata["model_metadata"][
                "match_quality_checker"
            ],
        },
        "catalog": catalog_metadata,
        "guideline_embedding_cache": cache_metadata,
        "retrieval": {
            "candidate_k": candidate_k,
            "top_n": top_n,
            "patients": len(patients),
            "scored_pairs": len(scored),
            "returned_pairs": len(result),
            "ranked_by": "match_quality_score",
            "filter_low_quality": False,
            "score_cutoff": resolved.raw["match_quality"].get("score_cutoff", 0.2),
        },
    }
    progress(f"Returned considerations for {len(result)} ranked patient-space pairs")
    return (result, metadata) if return_metadata else result
