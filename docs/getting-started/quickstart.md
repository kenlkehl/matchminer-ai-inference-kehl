# Quickstart

The example notebook provides a runnable walkthrough using sample input data:

[Open `examples/run_examples.ipynb` on GitHub](https://github.com/dfci/matchminer-ai-inference/blob/main/examples/run_examples.ipynb)

Most external workflows should start with these import paths:

```python
from matchminer_ai import load_preset
from matchminer_ai.trials import summarize_trials
from matchminer_ai.patients import summarize_patients
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import (
    exclusion_criteria_check,
    exclusion_criteria_check_with_llm,
    generate_candidate_matches,
    score_match_quality,
    score_match_quality_with_llm,
)
from matchminer_ai.paradigms import rank_patient_space_paradigms
```

Before starting, review the [data requirements](requirements.md#data) for the
trial and patient input tables expected by the workflow.

By default, `summarize_trials` now screens for cancer treatment and returns full
eligibility criteria per distinct cohort. Rejected trials are recorded in
metadata:

```python
cohorts, metadata = summarize_trials(trials, return_metadata=True)
print(cohorts[["cohort_trial_id", "cohort_eligibility_criteria"]])
print(metadata["trial_rejections"])
```

See the [trial API](../api/trials.md) for the cohort schema. The existing matching
models and the example notebook still use clinical-space summaries and separate
general exclusions. To run that workflow, select its legacy extraction mode
explicitly; full cohort criteria are not a drop-in replacement for its model
inputs:

```python
config = load_preset("default")
config.trial["extraction_method"] = "hierarchical"
trial_spaces = summarize_trials(trials, config=config)
```

The patient-centric workflow with those legacy trial spaces has these main steps:

1. Turn trial descriptions and eligibility criteria into structured trial
   summaries with `summarize_trials`.
2. Turn patient notes into cancer history summaries with `summarize_patients`.
3. Convert the trial and patient summaries into embeddings with
   `embed_for_matching` so they can be compared.
4. Find likely patient-trial pairs with `generate_candidate_matches`.
5. Score how reasonable each patient-trial match looks with either
   `score_match_quality` or `score_match_quality_with_llm`.
6. Screen matched patient-trial pairs for possible exclusion criteria with
   either `exclusion_criteria_check` or `exclusion_criteria_check_with_llm`.

When an exact space-to-paradigm membership graph is available, the optional
`rank_patient_space_paradigms` workflow can retrieve and TrialChecker-rerank a
wider trial-space pool, retain the leading spaces, and roll their IDs up to
one-line paradigm descriptors. The package does not bundle the paradigm
catalog; callers retain responsibility for its version, provenance, and caveat
status.
