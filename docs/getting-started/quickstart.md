# Quickstart

The example notebook provides a runnable walkthrough using sample input data:

[Open `examples/run_examples.ipynb` on GitHub](https://github.com/dfci/matchminer-ai-inference/blob/main/examples/run_examples.ipynb)

Most external workflows should start with these import paths:

```python
from matchminer_ai import load_preset
from matchminer_ai.trials import extract_cohorts
from matchminer_ai.patients import summarize_patients
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import (
    exclusion_criteria_check,
    exclusion_criteria_check_with_llm,
    generate_candidate_matches,
    score_match_quality,
    score_match_quality_with_llm,
)
```

Before starting, review the [data requirements](requirements.md#data) for the
trial and patient input tables expected by the workflow.

By default, `extract_cohorts` now screens for cancer treatment and returns full
eligibility criteria per distinct cohort. Rejected trials are recorded in
metadata:

```python
cohorts, metadata = extract_cohorts(trials, return_metadata=True)
print(cohorts[["cohort_trial_id", "cohort_eligibility_criteria"]])
print(metadata["trial_rejections"])
```

Cohort criteria include boilerplate. The optional
`trials.remove_boilerplate_criteria` step adds each cohort's key criteria
(`cohort_key_criteria`, read by TrialSpace retrieval and the decision model's match questions) and
boilerplate criteria (`cohort_boilerplate_criteria`, read by its boilerplate
question) and every line with its inclusion/exclusion and boilerplate tags
(`cohort_criteria_items`, read by `check_eligibility_criteria`); patient summaries carry the matching key and boilerplate
sections. See the [trial API](../api/trials.md#key-and-boilerplate-criteria)
and the [embedding API](../api/embedding.md).

The default `0526` TrialSpace and checker models were trained on clinical-space
summaries and separate general exclusions, which the example notebook also
uses. They accept cohort key and boilerplate criteria but have not been
evaluated on them. To run the established workflow, select its legacy
extraction mode explicitly:

```python
config = load_preset("default")
config.trial["extraction_method"] = "hierarchical"
trial_spaces = extract_cohorts(trials, config=config)
```

The patient-centric workflow with those legacy trial spaces has these main steps:

1. Turn trial descriptions and eligibility criteria into structured trial
   summaries with `extract_cohorts`.
2. Turn patient notes into cancer history summaries with `summarize_patients`.
3. Convert the trial and patient summaries into embeddings with
   `embed_for_matching` so they can be compared.
4. Find likely patient-trial pairs with `generate_candidate_matches`.
5. Score how reasonable each patient-trial match looks with either
   `score_match_quality` or `score_match_quality_with_llm`.
6. Screen matched patient-trial pairs for possible exclusion criteria with
   either `exclusion_criteria_check` or `exclusion_criteria_check_with_llm`.

