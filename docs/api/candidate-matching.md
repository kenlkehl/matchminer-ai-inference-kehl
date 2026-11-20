# Candidate Matching

`generate_candidate_matches(query_df, corpus_df, k=20)` ranks TrialSpace vectors from
[`embed_for_matching`](embedding.md) by cosine similarity over the entire supplied corpus.

- **Inputs:** each table has exactly one identifier column (`patient_id`, `cohort_trial_id` or the historical
  `space_trial_id`), with unique, non-null values, and an `embedding` column of equal-length vectors.
- **Direction:** pass patients first for the cohorts of each patient (patient-centric), and cohorts first for the
  patients of each cohort (trial-centric).
- **`k`:** a positive integer, or `None` for the full ranking.
- **Output:** one row per query and retrieved item, with both identifiers (as strings), `similarity_score` and
  one-based `rank`.
- **`config`:** accepted for call compatibility and unused.

Vectors from different TrialSpace models cannot be mixed; changing models requires embedding both sides again. For
exact search over large matrices on several devices, see `matching.retrieval.score_batches` on the
[embedding](embedding.md#large-encoding-jobs) page.

::: matchminer_ai.matching.match
    options:
      members:
        - generate_candidate_matches

## Adaptive trial-centric cutoff

For one cohort, `find_trial_centric_cutoff` estimates how many of its top-ranked patients are worth sending to the
full checks, without scoring the whole ranking.

1. The first probe is at `initial_cutoff_proportion` of the ranking (default 0.5). Use a smaller value when few
   patients are expected to qualify.
2. At each proposed cutoff, the `patients_per_side` patients (default 10) above and below it are scored.
3. When passes are a strict majority of the window, the search moves down the ranking; otherwise (including a tie)
   it moves toward the top. Later probes bisect, and overlapping windows reuse earlier scores.

| `check_method` | Probe score | Default `score_threshold` | Allowed thresholds |
| --- | --- | --- | --- |
| `"decision_model"` (default) | `match_quality_reasonableness`, the decision model's P(reasonable), from `score_match_quality` | `match_quality.reasonableness_cutoff` (0.5) | 0-1 |
| `"llm"` | `llm_match_quality_score` (0-6, -1 when unparsed) from `score_match_quality_with_llm` | 1 | 0-6 |

A probe patient passes when its score is at least the threshold. With the defaults that is the checks' own pass
call: `match_quality_pass` for the decision model, and a reasonable answer from the LLM (any score of 1 or more).

The result (`TrialCentricCutoffResult`) has `cutoff`, the number of leading patients to check, and the diagnostics
`scored_candidates` (every probed patient with its scores and `cutoff_probe_score`, `cutoff_probe_pass` and
`cutoff_probe_iteration`) and `search_history` (one row per probe).

The input is the complete ranking of every patient for one cohort, with contiguous ranks from 1, joined to the
patient and cohort text: `patient_id`, `rank`, `cohort_trial_id`, `cohort_key_criteria` and `cancer_history_summary`
(the historical `space_trial_id` and `clinical_space_summary` are also accepted).

```python
from matchminer_ai.matching import find_trial_centric_cutoff, generate_candidate_matches

cohort = cohort_vectors[cohort_vectors["cohort_trial_id"] == "NCT01234567-cohort-1"]
ranking = generate_candidate_matches(cohort, patient_vectors, k=None)    # every patient, ranked
candidate_pairs = ranking.merge(patients, on="patient_id").merge(cohorts, on="cohort_trial_id")

result = find_trial_centric_cutoff(candidate_pairs, config=config, initial_cutoff_proportion=0.10)
pairs_to_check = candidate_pairs.sort_values("rank").head(result.cutoff)
```

The search assumes that pass rates generally decline with TrialSpace rank. It estimates a compute cutoff; it does not
establish eligibility. With `check_method="llm"`, the probe patients' key sections go to the configured LLM
endpoint, so use only an endpoint authorized for that data.

To test whether the result depends on the first probe position, run the offline stability helper on synthetic or
otherwise authorized data. It scores the complete ranking once with `score_match_quality`, then replays the search
from each start; a patient passes when P(reasonable) is at least `score_threshold` (default
`match_quality.reasonableness_cutoff`):

```python
from matchminer_ai.matching import assess_trial_centric_cutoff_stability

stability = assess_trial_centric_cutoff_stability(
    candidate_pairs,
    initial_cutoff_proportions=(0.10, 0.25, 0.50, 0.75, 0.90),
    config=config,
)
print(stability.cutoff_runs)
```

`cutoff_runs` has one row per start. `selected_cutoff_spread` is the range of the selected cutoffs across starts, and
`reasonable_consideration_count_spread` the range of the number of threshold-passing patients inside them. These are
research QA signals, not eligibility determinations.

::: matchminer_ai.matching.adaptive_cutoff
    options:
      members:
        - TrialCentricCutoffResult
        - TrialCentricCutoffStabilityResult
        - find_trial_centric_cutoff
        - assess_trial_centric_cutoff_stability
