"""Source-grounded clinical context for MatchMiner-AI trial spaces."""

from .api import (
    DEFAULT_SOURCES,
    contextualize_trial_spaces,
    personalize_trial_space_context,
)
from .models import (
    EvidenceItem,
    SourceNotice,
    TrialSpaceContextualizationResult,
    TrialSpaceQuery,
)

__all__ = [
    "DEFAULT_SOURCES",
    "EvidenceItem",
    "SourceNotice",
    "TrialSpaceContextualizationResult",
    "TrialSpaceQuery",
    "contextualize_trial_spaces",
    "personalize_trial_space_context",
]
