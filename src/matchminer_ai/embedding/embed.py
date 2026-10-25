"""Embedding step for trial/patient matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import config_snapshot, load_default_preset
from .inference import generate_embeddings

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


def _resolve_text_column(entity_type: Literal["patient", "trial"]) -> str:
    if entity_type == "patient":
        return "cancer_history_summary"
    if entity_type == "trial":
        return "clinical_space_summary"
    raise ValueError("entity_type must be one of: 'patient', 'trial'")


def _resolve_id_columns(entity_type: Literal["patient", "trial"]) -> list[str]:
    if entity_type == "patient":
        return ["patient_id"]
    if entity_type == "trial":
        return ["space_trial_id"]
    raise ValueError("entity_type must be one of: 'patient', 'trial'")


def embed_for_matching(
    df: pd.DataFrame,
    *,
    entity_type: Literal["patient", "trial"],
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Encode full patient/cohort text for bidirectional retrieval.

    Patient rows require ``patient_id`` and ``patient_summary``; historical
    ``cancer_history_summary`` plus exclusion history is also accepted. Trial
    rows require ``cohort_trial_id`` and complete ``cohort_eligibility_criteria``;
    historical ``space_trial_id`` / ``clinical_space_summary`` remains accepted.

    The default ColBERT backend returns the identifier, ``embedding`` (document
    token matrix), ``query_embedding`` (instruction plus text with query marker),
    and ``embedding_contract`` (JSON provenance). Retain all columns when saving.
    Encoding is batched and never silently truncates text. Explicit sequence
    configurations return the historical identifier and one embedding vector.

    ``config.embedding`` selects the checkpoint, backend, device, context and
    batch sizes. With ``return_metadata=True``, also return the configuration and
    model fingerprint. See docs/api/embedding.md for configuration examples.
    """
    text_col = _resolve_text_column(entity_type)
    if entity_type == "patient" and "patient_summary" in df.columns:
        text_col = "patient_summary"
    if entity_type == "trial" and "cohort_eligibility_criteria" in df.columns:
        text_col = "cohort_eligibility_criteria"
    if text_col not in df.columns:
        raise ValueError(f"df is missing required column for {entity_type}: {text_col}")

    id_cols = _resolve_id_columns(entity_type)
    if entity_type == "trial" and "cohort_trial_id" in df.columns:
        id_cols = ["cohort_trial_id"]
    missing = [col for col in id_cols if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns for {entity_type}: {missing}")

    resolved_config = config or load_default_preset()
    embedding_config = dict(getattr(resolved_config, "embedding", {}))

    output = df.copy()

    summaries = output[text_col].fillna("").astype(str).tolist()
    backend = embedding_config.get("backend", "sequence")
    if backend == "colbert":
        from .late_interaction import encode_matching, token_series
        if output[id_cols[0]].isna().any() or output[id_cols[0]].duplicated().any():
            raise ValueError("ColBERT input identifiers must be present and unique")
        if any(not text.strip() for text in summaries):
            raise ValueError("ColBERT requires nonempty complete input text")
        if entity_type == "patient":
            from matchminer_ai.patients.postprocess import normalize_patient_summary
            other_col = next((c for c in ("other_relevant_history",
                "general_exclusion_criteria_evidence", "patient_boilerplate_text") if c in output), None)
            other = output[other_col].fillna("").astype(str).tolist() if other_col else [""] * len(output)
            summaries = [normalize_patient_summary(text, history) for text, history in zip(summaries, other)]
        queries, documents, signature, model_metadata = encode_matching(
            summaries, entity_type=entity_type, settings=embedding_config)
        output["query_embedding"] = token_series(queries, output.index)
        output["embedding"] = token_series(documents, output.index)
        output["embedding_contract"] = signature
        result = output[id_cols + ["embedding", "query_embedding", "embedding_contract"]].copy()
    elif backend == "sequence":
        embeddings, model_metadata = generate_embeddings(
            summaries,
            embedding_config=embedding_config,
            model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
        )
        output["embedding"] = embeddings
        result = output[id_cols + ["embedding"]].copy()
    else:
        raise ValueError("embedding.backend must be 'colbert' or 'sequence'")
    if return_metadata:
        metadata_payload = {
            "package": package_metadata(),
            "config_snapshot": config_snapshot(resolved_config),
            "model_metadata": {
                "embedding_model": model_metadata,
            },
        }
        return result, metadata_payload
    return result
