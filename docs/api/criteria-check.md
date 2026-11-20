# Criteria Check

`check_eligibility_criteria` asks the decision model ([match quality](match-quality.md#decision-model)) about **each
line** of a cohort's eligibility criteria, and `check_eligibility_criteria_with_llm` asks an LLM.

!!! important "Polarity"
    **Yes (`criterion_pass_probability`) means the patient does not clearly fail the line**: it is met, its
    information is missing, the patient could meet it by waiting, or it does not bind this patient. **No means the
    patient clearly fails it: an inclusion line clearly not met, or an exclusion line clearly met.** A low
    `criterion_pass_probability` flags a likely failing line for review, whichever list the line is on.

Each input row is one candidate pair with:

| Column | Content |
| --- | --- |
| `patient_id` | Patient identifier |
| `cohort_trial_id` (or historical `space_trial_id`) | Cohort identifier |
| `cancer_history_summary` and `general_exclusion_criteria_evidence` | The summary's two sections; or one complete `patient_summary` |
| `cohort_criteria_items` | The cohort's tagged lines from [`remove_boilerplate_criteria`](trials.md#key-and-boilerplate-criteria): a JSON list of `number`, `heading`, `text`, `criterion_type` (`inclusion` or `exclusion`) and `boilerplate` |

## Decision-model check

For each pair the model reads one state and one question per line.

- **State:** the complete patient summary, then the cohort's complete criteria, numbered in their original order
  under their inclusion and exclusion labels. The state also gives the rules for how a line binds the patient
  through its role: a sub-item belongs to the line above it; an alternative fails only when the patient clearly fails
  every alternative of its group; a line introducing others fails when they fail together; a requirement for only
  some patients fails only when it clearly applies; a standalone requirement fails a patient who lacks what it
  requires.
- **Question:** *Is this patient potentially eligible for this cohort after review of the patient summary and the
  one criterion named below? If the patient is potentially eligible, answer yes; if the patient is clearly
  ineligible because of this criterion, answer no.* It adds the answer rules (benefit of the doubt; missing
  information and waitable requirements, such as washouts, are not failures) and ends by restating the line, for
  example `[12] Exclusion criterion: Active hepatitis B.`

Lines without a validated inclusion or exclusion type, or with empty text, are skipped and counted in the
metadata's `unchecked_lines`. Pairs with identical summary and criteria are asked once.

| Output column | Content |
| --- | --- |
| `patient_id`, and `cohort_trial_id` or `space_trial_id` | The pair |
| `criterion_number`, `criterion_type`, `boilerplate`, `criterion_text` | The line |
| `criterion_pass_probability` | P(yes) |
| `criterion_clearly_fails` | `criterion_pass_probability < criteria_check.fail_cutoff` (0.5) |

The result has one row per pair and checked line. `filter_failing=True` keeps only flagged lines.

These are screening signals for human review, not eligibility determinations. The decision model runs locally;
patient text reaches only the configured model.

::: matchminer_ai.matching.criteria_check

## LLM check

`check_eligibility_criteria_with_llm` takes the same inputs and has the same polarity: **yes means the patient does
not clearly fail the line; no means the patient clearly fails it.**

- **Requests:** one per checkable line, with the per-criterion teacher prompt the decision model is trained on:
  `criterion_check.labels.user.txt` around the decision model's own state and question
  (`decision_model.criterion_teacher_messages`). Its label contract is `labeling.contract_hash("criteria")`.
- **Answer:** a JSON object with `answer` (`yes` or `no`), `criterion_status` (`met`, `not_met`,
  `insufficient_information` or `not_applicable`, as the line binds this patient) and `reason`.
- **Validation:** on a remote backend the answer is decoded under `decision_model.CRITERION_LABEL_JSON_SCHEMA`
  (`llm_criteria_check.structured_outputs`, default true). Every answer is validated by
  `decision_model.parse_criterion_label`: "no" needs `not_met` on an inclusion line and `met` on an exclusion line,
  and a response stopped before its end is invalid. An invalid answer is asked again with the validation error, up
  to `llm_criteria_check.max_attempts` (3) requests per line.
- **Concurrency:** on a remote backend, `chunk_size × chunks_in_flight` requests (64) are in flight through the
  `RequestQueue`, and a pair's lines share the prompt prefix. A local backend gets batches of `chunk_size`. Pairs
  with identical summary and criteria are asked once.
- **Model:** the `llm_criteria_check` section uses the training teacher's settings:
  `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, thinking on, 32,768 output tokens, `reasoning_effort: xhigh`
  and `sampling_profile: auto`, so the served model's own sampling profile applies.

| Output column | Content |
| --- | --- |
| `patient_id`, and `cohort_trial_id` or `space_trial_id` | The pair |
| `criterion_number`, `criterion_type`, `boilerplate`, `criterion_text` | The line, as in the decision-model check |
| `llm_criterion_answer` | `yes` or `no`; None when no attempt validated |
| `llm_criterion_clearly_fails` | True when the answer is no; None when no attempt validated |
| `llm_criterion_status` | The line's `criterion_status`; None when no attempt validated |
| `llm_criterion_reason` | The model's explanation; empty when no attempt validated |
| `llm_criterion_finish_reason` | The last attempt's finish reason (`stop` when complete) |
| `llm_criterion_attempts` | Requests sent for the line |
| `llm_criterion_parse_status` | `parsed`, or `parse_failed: <the last validation error>` |
| `llm_criterion_answer_text`, `llm_criterion_reasoning_text` | The last attempt's answer and reasoning; with `debug_mode` only |

`filter_failing=True` keeps the lines answered no. An optional `progress(done, total)` callback reports each line's
final answer. With `return_metadata=True` the metadata gives the `criteria_contract` and, under
`model_metadata["check_eligibility_criteria_with_llm"]`, the endpoint's metadata with `lines`, `requests`, `retries`,
`parse_failures`, `unchecked_lines` and `structured_outputs`.

The patient's complete summary reaches the configured LLM endpoint: use an endpoint authorized for the sensitivity
of the data.

::: matchminer_ai.matching.llm_checks
    options:
      members:
        - check_eligibility_criteria_with_llm
