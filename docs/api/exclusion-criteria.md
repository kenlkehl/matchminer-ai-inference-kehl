# Exclusion Criteria

The decision model's boilerplate question (`exclusion_criteria_check`) and its LLM alternative
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

The decision model ([match quality](match-quality.md#decision-model)) answers one
question about the patient's boilerplate section and the cohort's boilerplate
criteria: *Is this patient potentially eligible for this cohort after review of
the patient's other relevant history and these boilerplate criteria? If the
patient is potentially eligible, answer yes; if the patient is clearly
ineligible, answer no.* **Yes means potentially eligible; no means the patient
clearly fails at least one boilerplate criterion** (clearly meets an exclusion
or clearly does not meet an inclusion requirement); missing information and
waitable washouts are not failures. `exclusion_criteria_pass_probability` is
P(yes), `exclusion_criteria_pass` is P(yes) ≥ `exclusion_criteria.pass_cutoff`
(0.5), and `exclusion_score` is the probability of the predicted outcome (P(yes)
for a pass, 1 − P(yes) for a fail). `interpret_exclusion_criteria` token offsets
refer to the `patient_boilerplate_summary` and `cohort_boilerplate_criteria`
sources. For a line-by-line check, see [criteria check](criteria-check.md).

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
