"""Patient summarization logic."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mmai.backends import get_backend
from mmai.config import MMAIConfig, load_default_preset

from .postprocess import postprocess_patient_summaries
from .prompt_builder import get_filled_patient_prompt


def _truncate_patient_texts(
    patient_texts: list[str],
    *,
    model_name: str,
    text_token_threshold: int,
) -> list[str]:
    """Truncate overly long patient texts using the model tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    truncated: list[str] = []
    for patient_text in patient_texts:
        patient_text_tokens = tokenizer(
            patient_text, add_special_tokens=False
        ).input_ids
        if len(patient_text_tokens) > text_token_threshold:
            first_part = patient_text_tokens[: text_token_threshold // 2]
            last_part = patient_text_tokens[-text_token_threshold // 2 :]
            patient_text = (
                tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)
            )
        truncated.append(patient_text)
    return truncated


def summarize_from_relevant_sentences(
    df: pd.DataFrame,
    config: MMAIConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Summarize patient text extracted from relevant sentences.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level input with `patient_long_text`.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        patient_long_text : str
            Concatenated relevant note text for the patient.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Patient-level summaries and metadata about the run.
    """
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    patient_config = dict(resolved_config.patient)
    model_name = patient_config["model_name"]
    prompt_files = dict(patient_config["prompt_files"])
    primer_filename = prompt_files["primer"]
    question_filename = prompt_files["question"]

    patient_long_text_col = "patient_long_text"
    text_token_threshold = int(patient_config["text_token_threshold"])

    df = df.copy()
    df = df[df[patient_long_text_col] != ""]
    df = df.dropna(subset=[patient_long_text_col])

    patient_texts = df[patient_long_text_col].astype(str).tolist()
    truncated_texts = _truncate_patient_texts(
        patient_texts,
        model_name=model_name,
        text_token_threshold=text_token_threshold,
    )

    messages_list = [
        get_filled_patient_prompt(patient_text, primer_filename, question_filename)
        for patient_text in truncated_texts
    ]

    backend = get_backend(resolved_config.backend)
    summaries, model_metadata = backend.generate_llm_outputs(
        messages_list=messages_list,
        trial_config=patient_config,
        model_metadata_cache_dir=resolved_config.model_metadata_cache_dir,
    )

    df["original_patient_summary"] = summaries
    df = postprocess_patient_summaries(df, resolved_config)
    metadata = {
        "config_snapshot": {"patient": patient_config},
        "model_metadata": model_metadata,
    }
    return df, metadata


__all__ = ["summarize_from_relevant_sentences"]
