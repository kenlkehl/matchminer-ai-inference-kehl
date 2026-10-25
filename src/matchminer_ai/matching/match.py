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
        raise ValueError("Sequence embeddings must be vectors; ColBERT matrices require role/contract columns")
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


def generate_candidate_matches(
    query_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    *,
    k: int | None = 20,
    config=None,
) -> pd.DataFrame:
    """Return top-k corpus matches per query, preserving entity IDs and rank.

    Inputs are tables from ``embed_for_matching``. ColBERT tables use batched
    mean-MaxSim over query-role and document-role token matrices; their model,
    prompt and context contracts must agree. Reverse the arguments for the other
    retrieval direction. Historical vector tables use cosine similarity.

    ``k=None`` returns all corpus entities. ``config.embedding`` controls the
    scoring device, pair/query batch sizes, and resident corpus cache budget.
    Larger corpora are streamed without changing scores or truncating tokens.
    Results include both identifiers, ``similarity_score``, and one-based rank.
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

    if any(column in frame for frame in (query_df, corpus_df)
           for column in ("query_embedding", "embedding_contract")):
        from .late_interaction import colbert_matches
        return colbert_matches(query_df, corpus_df, query_id_col=query_id_col,
            corpus_id_col=corpus_id_col, k=k, settings=config.embedding if config else {})

    # Parse embeddings and ensure consistent vector dimensionality.
    query_embeddings = _stack_embeddings(query_df["embedding"])
    corpus_embeddings = _stack_embeddings(corpus_df["embedding"])
    if query_embeddings.numel() == 0 or corpus_embeddings.numel() == 0:
        return pd.DataFrame(
            columns=[query_id_col, corpus_id_col, "similarity_score", "rank"]
        )

    # Normalize embeddings and compute cosine similarity matrix.
    query_norm = _normalize_rows(query_embeddings)
    corpus_norm = _normalize_rows(corpus_embeddings)
    similarity = query_norm @ corpus_norm.T

    # Extract top-k corpus matches per query.
    query_ids = query_df[query_id_col].astype(str).tolist()
    corpus_ids = corpus_df[corpus_id_col].astype(str).tolist()

    records: list[dict[str, object]] = []
    num_corpus = similarity.shape[1]
    top_k = num_corpus if k is None else min(k, num_corpus)
    for row_idx, query_id in enumerate(query_ids):
        scores = similarity[row_idx]
        if top_k == 0:
            continue
        if top_k == num_corpus:
            ranked_idx = torch.argsort(scores, descending=True)
        else:
            ranked_idx = torch.topk(scores, k=top_k, largest=True).indices
        for rank, corpus_idx in enumerate(ranked_idx, start=1):
            corpus_idx_int = int(corpus_idx)
            records.append(
                {
                    query_id_col: query_id,
                    corpus_id_col: corpus_ids[corpus_idx_int],
                    "similarity_score": float(scores[corpus_idx_int]),
                    "rank": int(rank),
                }
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result

    return result
