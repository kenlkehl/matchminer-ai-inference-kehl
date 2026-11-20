# Exclusion Criteria

`exclusion_criteria_check` asks the decision model its boilerplate question about each candidate pair, and
`exclusion_criteria_check_with_llm` asks an LLM. Both read **boilerplate text only**: the general requirements most
cancer trials share, inclusion requirements as well as exclusions. Key criteria are the
[match-quality](match-quality.md) check's; for a line-by-line check, see [criteria check](criteria-check.md).

Each input row is one candidate pair with:

| Column | Content |
| --- | --- |
| `patient_id` | Patient identifier |
| `cohort_trial_id` (or historical `space_trial_id` or `trial_id`) | Cohort identifier |
| `cohort_boilerplate_criteria` (or historical `general_exclusion_criteria`) | The cohort's boilerplate criteria, from `remove_boilerplate_criteria` |
| `general_exclusion_criteria_evidence` | The patient summary's boilerplate section; else a complete `patient_summary`, whose text after `Other relevant history:` is taken |

`process_trials` runs the boilerplate step. Cohorts from `extract_cohorts` alone need `trials.remove_boilerplate_criteria`
first; see [key and boilerplate criteria](trials.md#key-and-boilerplate-criteria).

## Decision-model check

The decision model ([match quality](match-quality.md#decision-model)) reads one state, the patient's boilerplate
section and the cohort's boilerplate criteria, and answers: *Is this patient potentially eligible for this cohort
after review of the patient's other relevant history and these boilerplate criteria? If the patient is potentially
eligible, answer yes; if the patient is clearly ineligible, answer no.*

**Yes means potentially eligible; no means the patient clearly fails at least one boilerplate criterion**: clearly
meets an exclusion, or clearly does not meet an inclusion requirement. Missing information and requirements the
patient could meet by waiting, such as washout periods, are not failures.

| Output column | Content |
| --- | --- |
| `patient_id`, and the input's cohort identifier | The pair |
| `exclusion_criteria_pass_probability` | P(yes) |
| `exclusion_score` | The probability of the predicted outcome: P(yes) for a pass, 1 − P(yes) for a fail |
| `exclusion_criteria_pass` | P(yes) ≥ `exclusion_criteria.pass_cutoff` (0.5) |

Every pair is returned unless `filter_excluded=True`, which keeps only passing pairs.

`interpret_exclusion_criteria` adds gradient-times-input token attributions for the reported decision
(`attribution_target` `exclusion_criteria_pass` or `exclusion_criteria_fail`), with token offsets in the
`patient_boilerplate_summary` and `cohort_boilerplate_criteria` sources. Attributions are local model sensitivities
for debugging and review, not clinical rationales.

## LLM check

The prompt (`llm_exclusion_criteria.prompt_file`, `llm_exclusion_criteria.user.txt`) asks the LLM to go through the
cohort's boilerplate criteria one at a time and decide whether the patient clearly fails any of them, under the same
rules: missing information and waitable requirements are not failures. It ends with "Yes!" (clearly excluded) or
"No!".

| Output column | Content |
| --- | --- |
| `llm_exclusion_criteria_pass` | True for "No!" (not excluded), False for "Yes!"; None when the answer cannot be parsed |
| `llm_exclusion_criteria_answer_text`, `llm_exclusion_criteria_reasoning_text` | With `debug_mode` only |
| `llm_exclusion_criteria_parse_status` | With `debug_mode` only: `parsed`, `parsed_fallback` (a yes or no without "!" at the end) or `parse_failed` |

These are screening signals for human review, not eligibility determinations. The LLM check sends the patient's
boilerplate section to the configured LLM endpoint; use only an endpoint authorized for that data.

::: matchminer_ai.matching.exclusion_check
    options:
      members:
        - exclusion_criteria_check
        - interpret_exclusion_criteria

::: matchminer_ai.matching.llm_checks
    options:
      members:
        - exclusion_criteria_check_with_llm
