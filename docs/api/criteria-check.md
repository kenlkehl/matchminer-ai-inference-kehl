# Criteria Check

`check_eligibility_criteria` asks the decision model
([match quality](match-quality.md#decision-model)) about **each line** of a
cohort's eligibility criteria.

!!! important "Polarity"
    **Yes (`criterion_pass_probability`) means the patient does not clearly fail
    the line**: it is met, its information is missing, or the patient could meet
    it by waiting. **No means the patient clearly fails it: an inclusion line
    clearly not met, or an exclusion line clearly met.** A low
    `criterion_pass_probability` flags a likely failing line for review,
    whichever list the line is on.

Each row of the input is one candidate pair with:

- `patient_id`;
- `cohort_trial_id` (or historical `space_trial_id`);
- `cancer_history_summary` and `general_exclusion_criteria_evidence`, the
  summary's two sections (or one complete `patient_summary`);
- `cohort_criteria_items`, the cohort's tagged lines from
  [`remove_boilerplate_criteria`](trials.md#key-and-boilerplate-criteria): a JSON
  list of `number`, `heading`, `text`, `criterion_type` (`inclusion` or
  `exclusion`) and `boilerplate`.

For each pair the model reads one state, the complete patient summary followed
by the cohort's complete criteria numbered in their original order under their
inclusion and exclusion labels, and one question per line. The question opens
*Is this patient potentially eligible for this cohort after review of the
patient summary and the one criterion named below? If the patient is
potentially eligible, answer yes; if the patient is clearly ineligible because
of this criterion, answer no*, gives the rules (benefit of the doubt, waitable
washouts are not failures, a sub-item belongs to the criterion above it, a line
that only introduces others fails only when one of them does) and ends by
restating the line, for example `[12] Exclusion criterion: Active hepatitis B.`
Lines whose type never validated are skipped and counted in the metadata's
`unchecked_lines`; identical summary and criteria are asked once.

The result has one row per pair and line: `criterion_number`,
`criterion_type`, `boilerplate`, `criterion_text`,
`criterion_pass_probability` and `criterion_clearly_fails`
(`criterion_pass_probability < criteria_check.fail_cutoff`, 0.5 by default).
`filter_failing=True` keeps only flagged lines.

These are screening signals for human review, not eligibility determinations.
The decision model runs locally by default; patient text reaches only the
configured model.

::: matchminer_ai.matching.criteria_check

## LLM check

`check_eligibility_criteria_with_llm` is the LLM alternative, with the same
inputs and the same polarity: **yes means the patient does not clearly fail the
line; no means the patient clearly fails it.** It sends **one request per tagged
line** with the per-criterion teacher prompt the decision model's question 7 is
trained on, `criterion_check.labels.user.txt` around the decision model's own
criterion state and question (`decision_model.criterion_teacher_messages`; the
`criteria` label contract, `labeling.contract_hash("criteria")`). The answer is a
JSON object with `answer` (yes/no), `criterion_status` (`met`, `not_met`,
`insufficient_information` or `not_applicable`, as the line binds this patient)
and `reason`. On a remote backend it is decoded under
`decision_model.CRITERION_LABEL_JSON_SCHEMA`
(`llm_criteria_check.structured_outputs`), and it is always validated by
`decision_model.parse_criterion_label`: "no" needs `not_met` on an inclusion
line and `met` on an exclusion line, and a response stopped before its end is
invalid. An invalid answer is asked again with the validation error, up to
`llm_criteria_check.max_attempts` (3). A pair's lines go through the
`RequestQueue`, `chunk_size × chunks_in_flight` (64) in flight, sharing the
prompt prefix; identical summary and criteria are asked once.

The `llm_criteria_check` section uses the training teacher's settings:
Nemotron 3.5 Super, thinking on, 32,768 output tokens and `sampling_profile:
auto`, so the served model's own sampling profile applies.

The result has one row per pair and checkable line, with the decision-model
check's line columns (`criterion_number`, `criterion_type`, `boilerplate`,
`criterion_text`) and `llm_criterion_answer`, `llm_criterion_clearly_fails`,
`llm_criterion_status`, `llm_criterion_reason` (None or empty when no attempt
validated), `llm_criterion_finish_reason`, `llm_criterion_attempts` and
`llm_criterion_parse_status` (`parsed` or `parse_failed: <error>`); with
`debug_mode`, the last attempt's `llm_criterion_answer_text` and
`llm_criterion_reasoning_text`. `filter_failing=True` keeps the lines answered
no. An optional `progress(done, total)` callback reports each line's final
answer. With `return_metadata=True` the metadata gives the `criteria_contract`
and, under `model_metadata["check_eligibility_criteria_with_llm"]`, the
endpoint's metadata with `lines`, `requests`, `retries`, `parse_failures`,
`unchecked_lines` and `structured_outputs`.

The patient's complete summary reaches the configured LLM endpoint: use an
endpoint authorized for the sensitivity of the data.

::: matchminer_ai.matching.llm_checks
    options:
      members:
        - check_eligibility_criteria_with_llm
