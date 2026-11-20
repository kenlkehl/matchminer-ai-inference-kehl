"""Candidate match generation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _stack_embeddings(values: pd.Series) -> torch.Tensor:
    vectors = []
    for value in values:
        if isinstance(value, np.ndarray):
            vectors.append(value.astype(float))
            continue
        if isinstance(value, list):
            vectors.append(np.asarray(value, dtype=float))
            continue
        raise ValueError("Embeddings must be list or numpy array values.")
    if not vectors:
        return torch.empty((0, 0), dtype=torch.float32)
    if any(vec.ndim != 1 for vec in vectors):
        raise ValueError("Embeddings must be vectors.")
    dim = max(vec.size for vec in vectors)
    if any(vec.size != dim for vec in vectors):
        raise ValueError("Embedding vectors must share the same dimension.")
    return torch.tensor(np.vstack(vectors).astype(float), dtype=torch.float32)


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    return F.normalize(matrix, p=2, dim=1)


def _resolve_id_column(df: pd.DataFrame, label: str) -> str:
    candidates = [col for col in ("patient_id", "cohort_trial_id", "space_trial_id") if col in df.columns]
    if not candidates:
        raise ValueError(f"{label} must include patient_id, cohort_trial_id or space_trial_id")
    if len(candidates) > 1:
        raise ValueError(
            f"{label} must include only one identifier column; found {candidates}"
        )
    id_col = candidates[0]
    if df[id_col].isna().any() or df[id_col].duplicated().any():
        raise ValueError(f"{label} must have unique {id_col} values")
    return id_col


def rank_similarity(
    query_embeddings,
    corpus_embeddings,
    *,
    k: int | None = 20,
) -> list[list[tuple[int, float]]]:
    """Rank corpus embeddings by cosine similarity for each query embedding.

    The service-layer form of :func:`generate_candidate_matches`: plain embedding vectors in,
    positions out, for callers that keep their own identifiers and data types (such as a live
    matching service). ``generate_candidate_matches`` is built on it.

    Parameters
    ----------
    query_embeddings, corpus_embeddings : Sequence[array-like]
        Query-side and corpus-side vectors (lists or 1-D arrays) of one TrialSpace model.
    k : int | None, default 20
        Matches to return per query; None returns every corpus vector.

    Returns
    -------
    list[list[tuple[int, float]]]
        Per query, ``(corpus_index, similarity_score)`` pairs into ``corpus_embeddings``, by
        similarity descending. Ties keep corpus order, so the ranking is the same on every run
        and device.
    """
    if k is not None and (type(k) is not int or k <= 0):
        raise ValueError("k must be a positive integer or None")
    # Parse embeddings and ensure consistent vector dimensionality.
    query_matrix = _stack_embeddings(query_embeddings)
    corpus_matrix = _stack_embeddings(corpus_embeddings)
    if query_matrix.numel() == 0 or corpus_matrix.numel() == 0:
        return [[] for _ in range(len(query_embeddings))]
    similarity = _normalize_rows(query_matrix) @ _normalize_rows(corpus_matrix).T
    top_k = similarity.shape[1] if k is None else min(k, similarity.shape[1])
    results: list[list[tuple[int, float]]] = []
    for scores in similarity:
        # A stable sort breaks ties by corpus order; torch.topk guarantees no order at all.
        ranked = torch.argsort(scores, descending=True, stable=True)[:top_k]
        results.append([(int(index), float(scores[int(index)])) for index in ranked])
    return results


def generate_candidate_matches(
    query_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    *,
    k: int | None = 20,
    config=None,
) -> pd.DataFrame:
    """Return top-k corpus matches per query, preserving entity IDs and rank.

    Inputs are TrialSpace tables from ``embed_for_matching``, ranked by cosine
    similarity. Reverse the arguments for the other retrieval direction.
    ``k=None`` returns all corpus entities. ``config`` is accepted for call
    compatibility and unused. Results include both identifiers,
    ``similarity_score``, and one-based rank.
    """
    # Validate required schema for IDs and embeddings.
    query_id_col = _resolve_id_column(query_df, "query_df")
    corpus_id_col = _resolve_id_column(corpus_df, "corpus_df")
    if "embedding" not in query_df.columns:
        raise ValueError("query_df must include embedding")
    if "embedding" not in corpus_df.columns:
        raise ValueError("corpus_df must include embedding")
    if k is not None and (type(k) is not int or k <= 0):
        raise ValueError("k must be a positive integer or None")

    query_ids = query_df[query_id_col].astype(str).tolist()
    corpus_ids = corpus_df[corpus_id_col].astype(str).tolist()
    ranked = rank_similarity(query_df["embedding"], corpus_df["embedding"], k=k)
    records: list[dict[str, object]] = [
        {
            query_id_col: query_id,
            corpus_id_col: corpus_ids[corpus_index],
            "similarity_score": score,
            "rank": rank,
        }
        for query_id, matches in zip(query_ids, ranked)
        for rank, (corpus_index, score) in enumerate(matches, start=1)
    ]
    if not records:
        return pd.DataFrame(columns=[query_id_col, corpus_id_col, "similarity_score", "rank"])
    return pd.DataFrame.from_records(records)
