"""Embedding step for trial/patient matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from .inference import generate_embeddings

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


def _resolve_texts(df: pd.DataFrame, entity_type: Literal["patient", "trial"]) -> list[str]:
    """Retrieval reads only key text: a patient summary's key section and a cohort's key criteria."""
    from matchminer_ai._texts import patient_key_texts, trial_key_texts
    if entity_type == "patient":
        return patient_key_texts(df)
    if entity_type == "trial":
        return trial_key_texts(df)
    raise ValueError("entity_type must be one of: 'patient', 'trial'")


def _resolve_id_columns(entity_type: Literal["patient", "trial"]) -> list[str]:
    if entity_type == "patient":
        return ["patient_id"]
    if entity_type == "trial":
        return ["space_trial_id"]
    raise ValueError("entity_type must be one of: 'patient', 'trial'")


def embed_texts(
    texts: list[str],
    *,
    config: MMAIConfig | None = None,
    entity_type: Literal["patient", "trial"] | None = None,
) -> tuple[list[list[float]], dict]:
    """Embed summary strings in input order and return vectors plus model metadata.

    Accepts patient or trial text without identifiers or DataFrames. Uses the
    default preset when config is omitted and reuses the cached inference model.
    ``entity_type`` selects the TrialSpace model's patient or cohort prompt when
    it has them. Model metadata includes model_name and model_sha.
    """
    resolved_config = config or load_default_preset()
    embedding_config = dict(getattr(resolved_config, "embedding", {}))
    return generate_embeddings(
        texts,
        embedding_config=embedding_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
        entity_type=entity_type,
    )


def embedding_model_metadata(model_metadata: dict) -> dict:
    """Return encoder metadata from an ``embed_for_matching`` ``model_metadata``.

    Payloads saved before 0.6 recorded it under ``embedding_model``.
    """
    if "embed_for_matching" in model_metadata:
        return model_metadata["embed_for_matching"]
    return model_metadata["embedding_model"]


def embed_for_matching(
    df: pd.DataFrame,
    *,
    entity_type: Literal["patient", "trial"],
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Encode key patient/cohort text with TrialSpace for bidirectional retrieval.

    Retrieval reads key text only, never boilerplate. Patient rows require
    ``patient_id`` and the summary's key section, ``cancer_history_summary``
    (when only a complete ``patient_summary`` is given, its key section is taken:
    the text before ``Other relevant history:``). Trial rows require
    ``cohort_trial_id`` and ``cohort_key_criteria`` (the cohort's criteria without
    its boilerplate, from ``remove_boilerplate_criteria``); without the boilerplate
    step, ``cohort_eligibility_criteria`` is embedded, and historical
    ``space_trial_id`` / ``clinical_space_summary`` remains accepted.

    Returns the identifier and one embedding vector per row. ``config.embedding``
    selects the checkpoint, device and context. A TrialSpace model that saves
    patient and cohort prompts gets them for each side. With
    ``return_metadata=True``, also return the configuration and model fingerprint.
    See docs/api/embedding.md.
    """
    summaries = _resolve_texts(df, entity_type)

    id_cols = _resolve_id_columns(entity_type)
    if entity_type == "trial" and "cohort_trial_id" in df.columns:
        id_cols = ["cohort_trial_id"]
    missing = [col for col in id_cols if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns for {entity_type}: {missing}")

    resolved_config = config or load_default_preset()
    output = df.copy()
    embeddings, model_metadata = embed_texts(summaries, config=resolved_config, entity_type=entity_type)
    output["embedding"] = embeddings
    result = output[id_cols + ["embedding"]].copy()
    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "embed_for_matching": model_metadata,
            },
        }
        return result, metadata_payload
    return result
