"""Trial summarization logic."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

import pandas as pd

from matchminer_ai._qc.trials import build_qc_artifact
from matchminer_ai.llm.backends import (
    build_summarization_runtime_config,
    get_summarization_backend,
)
from matchminer_ai.llm.prompt_rendering import build_prompt_list

from .prompt_builder import build_trial_text, get_filled_trial_prompt

if TYPE_CHECKING:
    from matchminer_ai.config import MMAIConfig


def _backend_outputs_or(
    backend: Any,
    attr_name: str,
    fallback: list[str],
) -> list[str]:
    value = getattr(backend, attr_name, None)
    if isinstance(value, list) and len(value) == len(fallback):
        return value
    return fallback


def summarize_trials_multi_cohort(
    trial_texts: list[str],
    backend: Any,
    *,
    trial_config: dict[str, Any],
    primer_filename: str,
    question_filename: str,
    model_metadata_cache_dir: str | None = None,
) -> tuple[list[str], dict[str, Any], list[str]]:
    """Summarize trials using the configured backend."""
    messages_list = [
        get_filled_trial_prompt(text, primer_filename, question_filename)
        for text in trial_texts
    ]
    prompt_list = build_prompt_list(messages_list, llm_config=trial_config)
    return cast(
        tuple[list[str], dict[str, Any], list[str]],
        backend.generate_llm_outputs(
            prompt_list=prompt_list,
            llm_config=trial_config,
            model_metadata_cache_dir=model_metadata_cache_dir,
        ),
    )


def run_llm_summarization(
    trials_to_process: pd.DataFrame, config: MMAIConfig
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, object]]:
    """Run LLM-based trial summarization."""
    trial_config = dict(config.trial)
    runtime_trial_config = build_summarization_runtime_config(
        "trial",
        trial_config,
        config=config,
    )
    prompt_files = dict(trial_config["prompt_files"])
    primer_filename = prompt_files["primer"]
    question_filename = prompt_files["question"]

    backend = get_summarization_backend(config)

    trials_with_summaries = trials_to_process.copy()
    trials_with_summaries["trial_text"] = build_trial_text(trials_to_process)
    trial_summaries, model_metadata, finish_reasons = summarize_trials_multi_cohort(
        trials_with_summaries["trial_text"].tolist(),
        backend,
        trial_config=runtime_trial_config,
        primer_filename=primer_filename,
        question_filename=question_filename,
        model_metadata_cache_dir=config.model_metadata_cache_dir,
    )

    trials_with_summaries["space_output_no_reasoning"] = trial_summaries
    trials_with_summaries["space_reasoning_and_output"] = _backend_outputs_or(
        backend,
        "last_raw_outputs",
        trial_summaries,
    )
    trials_with_summaries["space_reasoning"] = _backend_outputs_or(
        backend,
        "last_reasoning_outputs",
        ["" for _summary in trial_summaries],
    )
    trial_ids = trials_with_summaries["trial_id"].astype(str).tolist()
    truncated_llm_qc_artifact = build_qc_artifact(
        metric="trials_truncated_llm_response",
        ids=[
            trial_id
            for trial_id, reason in zip(trial_ids, finish_reasons, strict=False)
            if str(reason) == "length"
        ],
        denominator=len(trial_ids),
    )
    metadata = {
        "config_snapshot": {"trial": trial_config},
        "model_metadata": model_metadata,
    }
    return trials_with_summaries, metadata, truncated_llm_qc_artifact
