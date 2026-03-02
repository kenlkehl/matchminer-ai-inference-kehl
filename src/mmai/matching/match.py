"""Candidate match generation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _parse_embedding(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    text = str(value).strip()
    if not text:
        return np.asarray([], dtype=float)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return np.fromstring(text, sep=" ", dtype=float)


def _stack_embeddings(values: pd.Series) -> np.ndarray:
    vectors = [_parse_embedding(value) for value in values]
    if not vectors:
        return np.empty((0, 0), dtype=float)
    dim = max(vec.size for vec in vectors)
    if any(vec.size != dim for vec in vectors):
        raise ValueError("Embedding vectors must share the same dimension.")
    return np.vstack(vectors).astype(float)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def _prepare_context_columns(
    df: pd.DataFrame,
    *,
    id_col: str,
) -> pd.DataFrame:
    columns = [col for col in df.columns if col not in {id_col, "embedding"}]
    return df[columns].copy()


def generate_candidate_matches(
    query_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    *,
    k: int | None = 20,
    query_id_col: str,
    corpus_id_col: str,
) -> pd.DataFrame:
    """
    Generate top-k candidate matches by ranking corpus items for each query item.

    Parameters
    ----------
    query_df : pd.DataFrame
        Query-side DataFrame with embeddings.
        Must contain:
            - <query_id_col>: str
            - embedding : array-like
    corpus_df : pd.DataFrame
        Corpus-side DataFrame with embeddings.
        Must contain:
            - <corpus_id_col>: str
            - embedding : array-like
    k : int | None, default 20
        Number of top matches to return per query entity. If None, return all
        corpus items per query (sorted by similarity).
    query_id_col : str
        Column name for the query entity identifier.
    corpus_id_col : str
        Column name for the corpus entity identifier.

    Returns
    -------
    pd.DataFrame
        Ranked top-k match pairs including:
            - <query_id_col>
            - <corpus_id_col>
            - similarity_score : float
        plus any additional columns from the input DataFrames for context.
    """
    # Validate required schema for IDs and embeddings.
    if query_id_col not in query_df.columns:
        raise ValueError(f"query_df must include {query_id_col}")
    if corpus_id_col not in corpus_df.columns:
        raise ValueError(f"corpus_df must include {corpus_id_col}")
    if "embedding" not in query_df.columns:
        raise ValueError("query_df must include embedding")
    if "embedding" not in corpus_df.columns:
        raise ValueError("corpus_df must include embedding")
    if k is not None and k <= 0:
        raise ValueError("k must be a positive integer or None")

    # Parse embeddings and ensure consistent vector dimensionality.
    query_embeddings = _stack_embeddings(query_df["embedding"])
    corpus_embeddings = _stack_embeddings(corpus_df["embedding"])
    if query_embeddings.size == 0 or corpus_embeddings.size == 0:
        return pd.DataFrame(columns=[query_id_col, corpus_id_col, "similarity_score"])

    # Normalize embeddings and compute cosine similarity matrix.
    query_norm = _normalize_rows(query_embeddings)
    corpus_norm = _normalize_rows(corpus_embeddings)
    similarity = query_norm @ corpus_norm.T

    # Extract top-k corpus matches per query.
    query_ids = query_df[query_id_col].astype(str).tolist()
    corpus_ids = corpus_df[corpus_id_col].astype(str).tolist()

    records: list[dict[str, Any]] = []
    top_k = similarity.shape[1] if k is None else min(k, similarity.shape[1])
    for row_idx, query_id in enumerate(query_ids):
        scores = similarity[row_idx]
        if top_k == 0:
            continue
        if top_k == similarity.shape[1]:
            ranked_idx = np.argsort(-scores)
        else:
            candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
            ranked_idx = candidate_idx[np.argsort(-scores[candidate_idx])]
        for corpus_idx in ranked_idx:
            records.append(
                {
                    query_id_col: query_id,
                    corpus_id_col: corpus_ids[corpus_idx],
                    "similarity_score": float(scores[corpus_idx]),
                }
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result

    # Attach context columns without duplicating ids/embedding columns.
    query_context = _prepare_context_columns(query_df, id_col=query_id_col)
    corpus_context = _prepare_context_columns(corpus_df, id_col=corpus_id_col)

    if not query_context.empty:
        result = result.merge(
            pd.concat([query_df[[query_id_col]].astype(str), query_context], axis=1),
            on=query_id_col,
            how="left",
        )
    if not corpus_context.empty:
        result = result.merge(
            pd.concat([corpus_df[[corpus_id_col]].astype(str), corpus_context], axis=1),
            on=corpus_id_col,
            how="left",
        )
    return result
