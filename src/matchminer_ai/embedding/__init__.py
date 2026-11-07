"""Embedding APIs."""

from __future__ import annotations

from .embed import embed_for_matching, embed_texts
from .encoder import EncoderSettings, encoder_identity, open_encoder

__all__ = ["EncoderSettings", "embed_for_matching", "embed_texts", "encoder_identity", "open_encoder"]
