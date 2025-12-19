"""Patient-related stubs."""

from __future__ import annotations

from .postprocess import postprocess_patient_summaries
from .prompt_builder import build_patient_summary_prompt
from .summarize import run_llm_summarization
from .tagging import tag_patient_notes


def summarize_patients(*args: object, **kwargs: object) -> None:
    """Phase-level patient summarization (stub)."""
    raise NotImplementedError("Patient phase summarization not implemented in skeleton.")


__all__ = [
    "build_patient_summary_prompt",
    "postprocess_patient_summaries",
    "run_llm_summarization",
    "summarize_patients",
    "tag_patient_notes",
]
