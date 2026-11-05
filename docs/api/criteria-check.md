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
