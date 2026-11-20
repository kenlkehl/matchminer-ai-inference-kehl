# Quickstart

This page runs the matching workflow end to end, from a table of trials and a table of clinical notes to scored
patient-cohort pairs. The [example notebook](example-notebook.md) runs the same steps on the sample data in
`examples/data/`.

## Inputs

| Table | One row per | Columns |
| --- | --- | --- |
| Trials | trial | `trial_id`, `trial_title`, `brief_summary`, `eligibility_criteria` (ClinicalTrials.gov fields; see [requirements](requirements.md#data)) |
| Notes | clinical note | `patient_id`, `note_text`, `note_date` |

To start from full protocols instead of listings, give each trial a `protocol_path` (PDF, `.txt` or `.md`) and
pass `source="protocol"` to `process_trials`.

## Configure

```python
from matchminer_ai import load_preset

config = load_preset("default")
config.embedding["model_path"] = "/path/to/trialspace"    # your TrialSpace checkpoint
```

- **LLM.** By default the LLM steps run `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` in-process with vLLM.
  To use a running OpenAI-compatible server instead, set `config.remote["enabled"] = True` and
  `config.remote["server_urls"] = ["http://localhost:8000/v1"]`. The server must serve the preset's model, or you
  change `model_name`. [Choosing an inference setup](../user-guide/local-vs-remote.md) compares the two.
- **TrialSpace.** The preset's default `embedding.model_path` is a local checkpoint that is not published, so point
  it at your own. Embedding fails rather than fall back to a different model.
- **Decision model.** It runs in-process on a GPU (`decision_model.device`). It is the pinned base `LiquidAI/d1-2B`
  unless `decision_model.model_name` names a fine-tuned checkpoint.

Only send clinical text to an endpoint approved for it.

## 1. Trials to cohorts

```python
import pandas as pd
from matchminer_ai.trials import process_trials

trials = pd.read_csv("trials.csv")
cohorts, trial_metadata = process_trials(trials, config=config, output_dir="runs/trials", return_metadata=True)
```

`process_trials` runs four steps:

1. It screens each trial for cancer treatment.
2. It lists the trial's cohorts and extracts each cohort's complete criteria.
3. It merges cohorts whose criteria are equivalent.
4. It tags every criteria line as inclusion or exclusion and as key or boilerplate.

It returns one row per cohort:

| Column | Content |
| --- | --- |
| `cohort_trial_id`, `trial_id`, `cohort_number`, `cohort_name` | Identifiers; `cohort_trial_id` is `<trial_id>-cohort-<n>` |
| `cohort_eligibility_criteria` | The cohort's complete criteria |
| `cohort_key_criteria` | The criteria that define the population, read by TrialSpace and the match questions |
| `cohort_boilerplate_criteria` | The general criteria most trials share, read by the boilerplate question |
| `cohort_criteria_items` | Every line with its tags, read by the line-by-line check |

Trials that are not cancer treatment trials have no rows. Their decisions are in
`trial_metadata["extraction"]["trial_rejections"]`.

With `output_dir`, the merge and line-tagging results are saved per trial and reused while the inputs and prompts
are unchanged; for protocols the extraction is saved too, while ctgov listings are extracted again on every run.
The steps are also public functions on their own: `extract_cohorts`, `merge_equivalent_cohorts` and
`remove_boilerplate_criteria`. See [trials](../api/trials.md#saved-results-and-reuse).

## 2. Notes to patient summaries

```python
from matchminer_ai.patients import summarize_patients

notes = pd.read_csv("notes.csv")
patients = summarize_patients(notes, config=config, checkpoint_dir="runs/patients")
```

Each patient's notes are summarized serially.

- **Preparation:** the notes are sorted by date, repeated sentences are removed, and the record is split into
  overlapping chunks.
- **Updates:** each chunk updates the running summary.
- **Result:** one row per patient with `patient_summary` and its two sections. `cancer_history_summary` is the key
  section; `general_exclusion_criteria_evidence` is the evidence for boilerplate criteria.
- **Failures:** a patient whose summary keeps failing the quality checks is left out and counted in the QC report
  (`return_qc=True`).
- **Resuming:** `checkpoint_dir` saves each chunk as it finishes, so an interrupted run resumes.

`summarize_patients` also accepts one patient's PDF files. See [patients](../api/patients.md).

## 3. Embed and retrieve

```python
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import generate_candidate_matches

patient_vectors = embed_for_matching(patients, entity_type="patient", config=config)
cohort_vectors = embed_for_matching(cohorts, entity_type="trial", config=config)

candidates = generate_candidate_matches(patient_vectors, cohort_vectors, k=20)    # cohorts for each patient
# generate_candidate_matches(cohort_vectors, patient_vectors, k=50)              # patients for each cohort
```

TrialSpace encodes patients' cancer histories and cohorts' key criteria into one vector space. Retrieval ranks by
cosine similarity and returns both identifiers, `similarity_score` and `rank`. Vectors from different TrialSpace
models must never be compared. See [embedding](../api/embedding.md) and
[candidate matching](../api/candidate-matching.md).

## 4. Score the candidates

The checks read the pair's texts, so join the summaries and cohorts onto the candidates first:

```python
from matchminer_ai.matching import check_eligibility_criteria, exclusion_criteria_check, score_match_quality

pairs = candidates.merge(patients, on="patient_id").merge(cohorts, on="cohort_trial_id")
keys = ["patient_id", "cohort_trial_id"]

quality = score_match_quality(pairs, config=config)
pairs = pairs.merge(quality, on=keys)

boilerplate = exclusion_criteria_check(pairs, config=config)
failing_lines = check_eligibility_criteria(pairs, config=config, filter_failing=True)
```

| Function | Question | Main outputs |
| --- | --- | --- |
| `score_match_quality` | Five yes/no match questions: reasonableness, cancer type, disease burden, treatment history, biomarkers | `match_quality_score` for ranking; `match_quality_pass`, true when the reasonableness answer is at least 0.5; `match_quality_<question>` |
| `exclusion_criteria_check` | Does the patient clearly fail a boilerplate criterion? | `exclusion_criteria_pass_probability`, `exclusion_criteria_pass` |
| `check_eligibility_criteria` | For each tagged line: does the patient clearly fail it? | One row per line with `criterion_pass_probability` and `criterion_clearly_fails` |

- **Filtering:** by default `score_match_quality` returns only the pairs that pass (`filter_low_quality=False`
  keeps them all).
- **LLM versions:** `score_match_quality_with_llm`, `exclusion_criteria_check_with_llm` and
  `check_eligibility_criteria_with_llm` ask an LLM instead and return a reason with each answer.
- **Interpretation:** these scores rank and screen pairs for human review; they are not eligibility decisions. See
  [match quality](../api/match-quality.md), [exclusion criteria](../api/exclusion-criteria.md) and
  [criteria check](../api/criteria-check.md).
