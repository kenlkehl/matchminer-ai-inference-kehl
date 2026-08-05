# Candidate Matching

::: matchminer_ai.matching.match
    options:
      members:
        - generate_candidate_matches

## Adaptive trial-centric cutoff

::: matchminer_ai.matching.adaptive_cutoff
    options:
      members:
        - TrialCentricCutoffResult
        - find_trial_centric_cutoff

Rank the entire patient corpus, attach the package-format patient and trial
summary columns, then select the leading prefix reported by the search:

```python
from matchminer_ai.matching import (
    find_trial_centric_cutoff,
    generate_candidate_matches,
)

ranking = generate_candidate_matches(trial_embeddings, patient_embeddings, k=None)
candidate_pairs = (
    ranking.merge(patient_summaries, on="patient_id")
    .merge(trial_spaces, on="space_trial_id")
)
cutoff = find_trial_centric_cutoff(
    candidate_pairs,
    check_method="trial_checker",  # or "llm"
    score_threshold=0.20,           # use 1 for the default LLM threshold
    patients_per_side=10,
    initial_cutoff_proportion=0.10, # start 10% down when few patients qualify
)
pairs_for_both_checkers = candidate_pairs.sort_values("rank", kind="stable").iloc[
    : cutoff.cutoff
]
```

LLM probes send the probe patient and trial summaries to the configured LLM
backend. Use only endpoints authorized for the sensitivity of those inputs.
