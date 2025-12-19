"""Embedding stubs."""

from __future__ import annotations

from .embed import run_embedding_step


def embed_for_matching(*args: object, **kwargs: object) -> None:
    """Phase-level embedding (stub)."""
    raise NotImplementedError("Embedding phase not implemented in skeleton.")


__all__ = [
    "embed_for_matching",
    "run_embedding_step",
]
