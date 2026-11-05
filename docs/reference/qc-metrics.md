# QC metrics

Patient and trial summarization can produce QC reports. These reports are meant
to help you review how a summarization run behaved before using the outputs
downstream.

To request a QC report, pass `return_qc=True` to `summarize_patients` or
`extract_cohorts`.

QC reports summarize how the current pipeline behaves, given the current
prompts, filters, and postprocessing rules. Metrics are returned as rows with
`metric`, `value`, `denominator`, `percent`, and `ids`.

## Patient summarization QC

Returned by `summarize_patients(..., return_qc=True)`.

- `patients_dropped_noninformative_summary`: summaries removed because they
  were empty or began with `No information`.
- `patients_failed_inference`: patients removed because LLM inference failed
  or the summary was still rejected after
  `patient.summary_quality_max_attempts` attempts.
- `patients_exclusion_criteria_not_extracted`: summaries whose boilerplate
  section (`general_exclusion_criteria_evidence`, the text after
  `Other relevant history:`) is empty, `Not documented.`, or identical to the
  key section.
- `patients_missing_keyword:<keyword>`: key sections (`cancer_history_summary`)
  missing an expected keyword.
- `patients_exceed_embedding_token_limit`: key sections whose
  embedding-tokenized length, with the TrialSpace patient prompt, exceeds
  `embedding.max_seq_length`, the configured truncation cutoff used during
  embedding generation.

## Trial summarization QC

Returned by `extract_cohorts(..., return_qc=True)`.

The default `cohorts` method reports:

- `trials_rejected_cancer_treatment`: valid negative treatment decisions; these
  trials produce no cohort rows and remain in `trial_rejections` metadata.
- `trials_failed_inference`: responses that remain invalid or incomplete after
  the configured attempts. These are errors, not treatment rejections.
- `trials_truncated_llm_response`: trials whose final attempt ended at its
  token limit. These are also counted as inference failures.

Each denominator is the number of input trials. Full eligibility cohorts do not
run the legacy keyword, boilerplate-separation, or embedding-length checks.

Explicit legacy `hierarchical` and `one_shot` extraction reports:

- `trials_missing_in_output`: trials present in the input but not represented
  in the output after summarization/postprocessing.
- `trials_failed_inference`: trials whose LLM inference finished with an error
  after retries were exhausted. These trials are removed before
  postprocessing.
- `trials_truncated_llm_response`: trials where the LLM stopped due to max
  token length.
- `spaces_per_trial_min|median|max`: min/median/max number of spaces per trial.
- `trials_with_non_distinct_spaces`: trials with duplicate space numbers or
  duplicate space text.
- `spaces_dropped_missing_keyword:<keyword>`: trial spaces dropped due to missing a required keyword.
- `trials_exclusion_criteria_not_extracted`: trials whose exclusion criteria was not extracted.
- `spaces_exceed_embedding_token_limit`: trial spaces whose embedding-tokenized
  length exceeds `embedding.max_seq_length`, the configured truncation cutoff
  used during embedding generation.
