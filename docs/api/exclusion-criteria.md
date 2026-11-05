# Exclusion Criteria

BoilerplateChecker (`exclusion_criteria_check`) and its LLM alternative
(`exclusion_criteria_check_with_llm`) read **boilerplate text only**: the
general requirements most cancer trials share, inclusion requirements as well
as exclusions. Key criteria are the [match-quality](match-quality.md) stage's.
Each row is one candidate pair with:

- `patient_id`;
- `cohort_trial_id` (or historical `space_trial_id` or `trial_id`);
- `cohort_boilerplate_criteria` from `remove_boilerplate_criteria` (or
  historical `general_exclusion_criteria`);
- `general_exclusion_criteria_evidence`, the patient summary's boilerplate
  section (or a complete `patient_summary`, whose text after
  `Other relevant history:` is taken).

Cohorts from `extract_cohorts` need the boilerplate step
(`trials.remove_boilerplate_criteria`) before this stage; see
[key and boilerplate criteria](trials.md#key-and-boilerplate-criteria).

BoilerplateChecker is a two-class model (NEGATIVE/POSITIVE): `exclusion_score`
is the probability of its predicted label, and `exclusion_criteria_pass` is
True when that label is NEGATIVE. Inputs are truncated at
`exclusion_criteria.max_length` (8192) and padded to the longest input of each
batch (`exclusion_criteria.batch_size`, default 8). `interpret_exclusion_criteria`
token offsets refer to the `patient_boilerplate_summary` and
`cohort_boilerplate_criteria` sources.

The LLM prompt (`llm_exclusion_criteria.user.txt`) asks whether the patient
clearly fails any of the cohort's boilerplate criteria: clearly meets an
exclusion or clearly does not meet an inclusion requirement. Missing or
undocumented information is not a failure, nor is a requirement the patient
could meet by waiting, such as a washout period. It returns
`llm_exclusion_criteria_pass` (None when the answer cannot be parsed).

These are screening signals for human review, not eligibility determinations.
The LLM check sends the patient's boilerplate section to the configured LLM
endpoint; use only an endpoint authorized for that data.

::: matchminer_ai.matching.exclusion_check
    options:
      members:
        - exclusion_criteria_check
        - interpret_exclusion_criteria

::: matchminer_ai.matching.llm_checks
    options:
      members:
        - exclusion_criteria_check_with_llm
