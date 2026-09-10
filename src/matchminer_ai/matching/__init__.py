"""Matching helpers."""

from __future__ import annotations

from .adaptive_cutoff import (
    DEFAULT_CUTOFF_PATIENTS_PER_SIDE,
    DEFAULT_INITIAL_CUTOFF_PROPORTION,
    DEFAULT_LLM_CUTOFF_THRESHOLD,
    DEFAULT_STABILITY_INITIAL_CUTOFF_PROPORTIONS,
    DEFAULT_TRIAL_CHECKER_CUTOFF_THRESHOLD,
    TrialCentricCutoffResult,
    TrialCentricCutoffStabilityResult,
    assess_trial_centric_cutoff_stability,
    find_trial_centric_cutoff,
)
from .exclusion_check import exclusion_criteria_check, interpret_exclusion_criteria
from .llm_checks import exclusion_criteria_check_with_llm
from .llm_checks import score_match_quality_with_llm
from .match import generate_candidate_matches
from .rerank import interpret_match_quality, score_match_quality

__all__ = [
    "DEFAULT_CUTOFF_PATIENTS_PER_SIDE",
    "DEFAULT_INITIAL_CUTOFF_PROPORTION",
    "DEFAULT_LLM_CUTOFF_THRESHOLD",
    "DEFAULT_STABILITY_INITIAL_CUTOFF_PROPORTIONS",
    "DEFAULT_TRIAL_CHECKER_CUTOFF_THRESHOLD",
    "TrialCentricCutoffResult",
    "TrialCentricCutoffStabilityResult",
    "assess_trial_centric_cutoff_stability",
    "exclusion_criteria_check",
    "exclusion_criteria_check_with_llm",
    "generate_candidate_matches",
    "find_trial_centric_cutoff",
    "interpret_exclusion_criteria",
    "interpret_match_quality",
    "score_match_quality",
    "score_match_quality_with_llm",
]


# Resolve optional drug research/scoring APIs without eager cross-stage imports.
_GOOD_OPTION_EXPORTS = {
    "build_good_option_checker_text": "matching.good_options",
    "build_good_option_messages": "matching.good_options",
    "evaluate_good_options": "matching.good_options",
    "parse_good_option_response": "matching.good_options",
    "score_good_options": "matching.good_options",
    "score_good_options_with_llm": "matching.good_options",
    "GOOD_OPTION_INPUT_VERSION": "trials.drug_evidence",
    "GOOD_OPTION_LABEL_SCHEMA_VERSION": "trials.drug_evidence",
    "GOOD_OPTION_PROMPT_VERSION": "trials.drug_evidence",
    "RUBRIC_CRITERIA": "trials.drug_evidence",
    "ParsedGoodOptionResult": "trials.drug_evidence",
}
__all__ += list(_GOOD_OPTION_EXPORTS)


def __getattr__(name: str):
    if name in _GOOD_OPTION_EXPORTS:
        from importlib import import_module

        value = getattr(
            import_module(f"matchminer_ai.{_GOOD_OPTION_EXPORTS[name]}"), name
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
