# QC metrics

Patient summarization, cohort extraction and guideline catalog builds can return a QC report: mechanical checks of how
a run behaved, for review before its outputs are used downstream. The checks do not establish clinical correctness.

| Function | Request |
| --- | --- |
| `summarize_patients` | `return_qc=True` |
| `extract_cohorts` | `return_qc=True` (`process_trials` does not return QC) |
| `summarize_guidelines`, `summarize_pdq` | `return_qc=True` |

The report is a DataFrame with one row per metric:

| Column | Content |
| --- | --- |
| `metric` | Metric name |
| `value` | Count of flagged items, or the statistic itself for distribution metrics |
| `denominator` | Items checked |
| `percent` | `value / denominator * 100`; 0 when the denominator is 0 |
| `ids` | Sorted IDs of the flagged items |

## Patient summarization

IDs are `patient_id`s.

| Metric | Flags |
| --- | --- |
| `patients_dropped_noninformative_summary` | Summaries removed because their key section was empty or began with `No information` |
| `patients_failed_inference` | Patients removed because a chunk's LLM request failed, or its summary was still rejected after `patient.summary_quality_max_attempts` attempts (default 3) |
| `patients_summary_quality_flagged` | Returned summaries with any quality finding |
| `patients_summary_quality_flagged:<code>` | Returned summaries with that finding; one row per code present in the run |
| `patients_exceed_embedding_token_limit` | Key sections (`cancer_history_summary`) longer than `embedding.max_seq_length` TrialSpace tokens, prompt included; embedding truncates them |
| `patients_exclusion_criteria_not_extracted` | Summaries whose other relevant history (`general_exclusion_criteria_evidence`) is empty, `Not documented.`, or identical to the key section |
| `patients_missing_keyword:<keyword>` | Key sections missing a field heading: `Age`, `Sex`, `Cancer type`, `Histology`, `Current extent`, `Biomarkers`, `Treatment history` |

The quality findings come from `matchminer_ai.patients.summary_quality.check_patient_summary`, which reads the
summary text only. Findings marked *review* are ambiguous signals; the others are strong evidence of malformed
output.

| Code | Finding |
| --- | --- |
| `empty_or_failed` | Empty, or marked as a generation error |
| `self_editing_language` | Possible first-person drafting or self-correction (*review*) |
| `prompt_commentary` | Commentary about the prompt, formatting or answer construction |
| `reasoning_delimiter` | A raw model or reasoning delimiter |
| `duplicate_cancer_section` | A cancer section whose five fields repeat an earlier one |
| `repeated_cancer_fields` | Age, cancer type, histology and biomarkers repeat (*review*: two primaries can share them) |
| `repeated_long_paragraph` | A paragraph of at least 40 words repeated verbatim (*review*) |

Every finding except `repeated_cancer_fields` also makes the summarizer regenerate the chunk, and a patient whose
last attempt still has one is counted in `patients_failed_inference` instead. Those codes therefore appear here
mainly for supplied summaries (`existing_summaries`) passed through unchanged.

The token-limit check loads the TrialSpace tokenizer from `embedding.model_path`.

## Cohort extraction

Returned by `extract_cohorts` with the default `cohorts` method. IDs are `trial_id`s, and each denominator is the
number of input trials.

| Metric | Flags |
| --- | --- |
| `trials_rejected_cancer_treatment` | Trials the screen judged not to be cancer treatment trials. They have no cohort rows; their decisions are in the `trial_rejections` metadata. |
| `trials_failed_inference` | Trials with a step whose answer was still invalid or incomplete after `trial.cohorts.max_attempts` attempts. These are errors, not rejections. |
| `trials_truncated_llm_response` | Trials with at least one response, in any step, that stopped at its token limit. Such a response is invalid and retried, so the trial may still succeed. |

### Legacy trial-space extraction

The legacy `hierarchical` and `one_shot` methods (`trial.extraction_method`), which write trial spaces instead of
cohorts, report a different set. IDs are `trial_id`s, or `space_trial_id`s for the space metrics.

| Metric | Flags |
| --- | --- |
| `trials_missing_in_output` | Input trials with no space in the output, or a blank space summary |
| `trials_truncated_llm_response` | Trials whose response stopped at its token limit |
| `trials_failed_inference` | Trials whose LLM request still failed after retries; removed before postprocessing |
| `spaces_exceed_embedding_token_limit` | Space summaries longer than `embedding.max_seq_length` TrialSpace tokens, prompt included |
| `spaces_per_trial_min`, `spaces_per_trial_median`, `spaces_per_trial_max` | The statistic itself (no denominator or IDs) |
| `trials_with_non_distinct_spaces` | Trials with duplicate space numbers or duplicate space text |
| `spaces_dropped_missing_keyword:<keyword>` | Spaces dropped for missing a required field: `Age`, `Sex`, `Cancer primary site`, `Histology`, `Disease state`, `Prior treatment requirements`, `Biomarker requirements` (older field names are accepted) |
| `trials_exclusion_criteria_not_extracted` | Trials with a space whose `general_exclusion_criteria` is empty or `None` |

## Guideline catalogs

Returned by `summarize_guidelines` and `summarize_pdq`. IDs are source page IDs for the first metric and
`space_trial_id`s for the others.

| Metric | Flags |
| --- | --- |
| `guideline_uncertain_source_pages` | Extracted source pages whose coverage was marked uncertain |
| `guideline_spaces_with_uncertainties` | Spaces with recorded `uncertainties` |
| `guideline_spaces_with_omitted_context` | Spaces whose requests left out some of the guideline's pages to fit the context (`omitted_source_page_ids`) |
| `guideline_spaces_without_treatment_options` | Spaces with no treatment options |
