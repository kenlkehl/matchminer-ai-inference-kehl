# Match Quality

The decision model's match questions (`score_match_quality`) and their LLM alternative
(`score_match_quality_with_llm`) read **key text only**: the patient summary's
key section and the cohort's key criteria. Boilerplate is the
[exclusion-criteria](exclusion-criteria.md) stage's. Each row is one candidate
pair with:

- `patient_id`;
- `cohort_trial_id` (or historical `space_trial_id`);
- `cancer_history_summary`, the patient's key section (or a complete
  `patient_summary`, whose text before `Other relevant history:` is taken);
- `cohort_key_criteria` (else `cohort_eligibility_criteria` when the
  boilerplate step did not run, else historical `clinical_space_summary`).

## Match labels

One LLM prompt, `match_quality.labels.system.txt`
(`matchminer_ai.matching.match_labels`, schema `trial-match-labels-v1`), asks
five yes/no questions about one patient's key section and one cohort's key
criteria. Each answer carries an `evidence_status` (`supported`,
`contradicted`, `insufficient_information` or `not_applicable`) and a `reason`.

| Label | Question | Points |
| --- | --- | --- |
| `reasonableness` | No clear evidence that the patient fails any of the key criteria. Missing information is not a mismatch. | 1 (gate) |
| `cancer_type` | The cohort specifically names a cancer type or histology matching the patient's active cancer. | 1 |
| `burden` | A specific stage or disease-burden requirement is documented as satisfied. | 1 |
| `treatment_history` | A specific prior-treatment requirement is documented as satisfied. | 1 |
| `biomarker` | A specific required biomarker condition is documented as satisfied. | 2 |

`match_score` gives 0–6: 0 unless the patient is reasonable, otherwise 1 plus
the points of each other yes answer. The training teacher labels pairs with
this same prompt and contract, and the decision model learns one yes/no
question per label. `score_match_quality_with_llm` uses it by default
(`llm_match_quality.prompt_file`) and returns `llm_match_quality_score` (0–6, or
-1 when the answer fails the contract) plus one boolean `llm_<label>` column per
label (None when unparsed). Debug mode adds the answer, reasoning and parse
status.

## Decision-model scores

`score_match_quality` asks the decision model
([`matchminer_ai.matching.decision_model`](#decision-model); the `decision_model`
configuration section, base `LiquidAI/d1-2B` by default) the five match questions
about one state: the patient's key section and the cohort's key criteria. Each
question's instructions are its section of `match_quality.labels.system.txt`
(without the teacher's evidence-status bookkeeping), except `reasonableness`,
which opens: *Is this patient potentially eligible for this cohort after review
of the patient history and these key criteria? If the patient is potentially
eligible, answer yes; if the patient is clearly ineligible, answer no.* It
returns each P(yes) as `match_quality_<label>` and

```text
match_quality_score = P(reasonable) × (1 + P(cancer_type) + P(burden)
                      + P(treatment_history) + 2·P(biomarker)) / 6
```

the expected match score as a fraction of 6. `match_quality_pass` is
`match_quality_score >= match_quality.score_cutoff` (0.2 by default; tuned for
the earlier checker, so retune it for a fine-tuned decision model).

`interpret_match_quality` attributes one match question's decision (`head`,
default `"reasonableness"`): gradient times input embedding for the question's
yes-minus-no logit at its answer slot, read in the same packed sequence as the
scores. Token offsets refer to the `cohort_key_criteria` and
`patient_key_summary` sources. Attributions are local model sensitivities for
debugging and human review, not clinical rationales.

Scores and labels are ranking and screening signals, not eligibility
determinations. The LLM check sends patient key text to the configured LLM
endpoint; use only an endpoint authorized for that data.

## Decision model

::: matchminer_ai.matching.decision_model
    options:
      members:
        - match_state
        - match_questions
        - boilerplate_state
        - boilerplate_question
        - criteria_state
        - criterion_question
        - run_decisions

::: matchminer_ai.matching.rerank
    options:
      members:
        - score_match_quality
        - interpret_match_quality

::: matchminer_ai.matching.llm_checks
    options:
      members:
        - score_match_quality_with_llm

::: matchminer_ai.matching.match_labels
    options:
      members:
        - match_score
        - parse_match_labels
        - build_match_label_messages
        - match_labels_contract
