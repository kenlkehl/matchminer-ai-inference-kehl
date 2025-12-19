"""Trial-related stubs."""

from __future__ import annotations

from .postprocess import postprocess_trial_summaries
from .prompt_builder import build_trial_summary_prompt
from .summarize import run_llm_summarization


def summarize_trials(*args: object, **kwargs: object) -> None:
    """Phase-level trial summarization (stub)."""
    raise NotImplementedError("Trial phase summarization not implemented in skeleton.")


__all__ = [
    "build_trial_summary_prompt",
    "postprocess_trial_summaries",
    "run_llm_summarization",
    "summarize_trials",
]
