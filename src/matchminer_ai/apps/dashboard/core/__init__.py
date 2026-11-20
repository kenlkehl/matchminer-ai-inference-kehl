"""What the dashboard matches with: the synthetic corpus and its text search, the TrialSpace rounds and vectors, the
query encoder (and its worker), the decision model, the cancer-type patterns, the caches and background jobs.

Moved from the former ``mmai_app`` package (removed 2026-11-17; ``mmai_dash`` is the only interface), unchanged but for
dropping its command line and the settings of its removed features, so cache keys, fingerprints and the
decision-model selection are the same. Research use only; scores are screening signals, not eligibility determinations.
"""
