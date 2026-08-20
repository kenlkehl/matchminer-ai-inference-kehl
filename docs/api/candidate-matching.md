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
        - TrialCentricCutoffStabilityResult
        - find_trial_centric_cutoff
        - assess_trial_centric_cutoff_stability

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

To test whether the adaptive result depends on the first boundary position,
run the offline stability helper on synthetic or otherwise explicitly
authorized data:

```python
from matchminer_ai.matching import assess_trial_centric_cutoff_stability

stability = assess_trial_centric_cutoff_stability(
    candidate_pairs,
    initial_cutoff_proportions=(0.10, 0.25, 0.50, 0.75, 0.90),
    score_threshold=0.20,
    patients_per_side=10,
)
print(stability.cutoff_runs)
```

This diagnostic scores the complete ranking once with TrialChecker and reuses
those scores for every start. `selected_cutoff_spread` measures allocation
sensitivity; `reasonable_consideration_count_spread` measures how many
threshold-passing patients that sensitivity adds or omits. These are research
QA signals, not patient eligibility determinations.
