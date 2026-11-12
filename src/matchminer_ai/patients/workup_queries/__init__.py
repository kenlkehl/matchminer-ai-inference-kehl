"""Extraction-based workup checks: guideline requirements as queries over extracted items.

A research review aid beside the note-reading workup reviews
(:func:`~matchminer_ai.patients.review_patient_workup`): an LLM compiles each
guideline workup requirement once into a declarative JSON query
(:func:`compile_workup_queries`; its only power is returning the query), code
validates it, and code evaluates it deterministically on every patient's
structured extraction records (:func:`check_workup_with_extractions`), with no
I/O and no model calls. Results are met, not met (not documented), unknown or
not applicable, each with the matching items, their dates and evidence quotes,
and a machine-made explanation. They need human review against the source notes.
"""

from .compile import (
    compile_contract,
    compile_workup_queries,
    default_extensions,
    render_compile_prompt,
    workup_query_cache_key,
)
from .evaluate import (
    APPLICABILITY,
    NOTICE,
    REASONS,
    STATUSES,
    aggregate_extraction_workup,
    check_workup_with_extractions,
    evaluate_workup_query,
    patient_data,
)
from .language import (
    QUERY_LANGUAGE_VERSION,
    QueryError,
    check_workup_query,
    query_response_schema,
    render_query_field_guide,
    validate_workup_query,
)

__all__ = [
    "APPLICABILITY",
    "NOTICE",
    "QUERY_LANGUAGE_VERSION",
    "QueryError",
    "REASONS",
    "STATUSES",
    "aggregate_extraction_workup",
    "check_workup_query",
    "check_workup_with_extractions",
    "compile_contract",
    "compile_workup_queries",
    "default_extensions",
    "evaluate_workup_query",
    "patient_data",
    "query_response_schema",
    "render_compile_prompt",
    "render_query_field_guide",
    "validate_workup_query",
    "workup_query_cache_key",
]
