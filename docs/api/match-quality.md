# Match Quality

`score_match_quality` asks the decision model five match questions about each candidate pair, and
`score_match_quality_with_llm` asks an LLM the same five. Both read **key text only**: the patient summary's key
section and the cohort's key criteria. Boilerplate is the [exclusion-criteria](exclusion-criteria.md) check's.

Each input row is one candidate pair with:

| Column | Content |
| --- | --- |
| `patient_id` | Patient identifier |
| `cohort_trial_id` (or historical `space_trial_id`) | Cohort identifier |
| `cancer_history_summary` | The patient's key section; else a complete `patient_summary`, whose text before `Other relevant history:` is taken |
| `cohort_key_criteria` | The cohort's key criteria; else `cohort_eligibility_criteria` when the boilerplate step did not run; else historical `clinical_space_summary` |

Merge the summaries and cohorts onto the candidates from `generate_candidate_matches` to get these columns (see the
[quickstart](../getting-started/quickstart.md#4-score-the-candidates)).

## Match labels

One prompt, `match_quality.labels.system.txt` (`matchminer_ai.matching.match_labels`, schema
`trial-match-labels-v1`), asks five yes/no questions about one patient's key section and one cohort's key criteria.
The training pipeline's teacher labels pairs with it, the decision model learns one question per label, and
`score_match_quality_with_llm` uses it.

| Label | Yes means | Points |
| --- | --- | --- |
| `reasonableness` | No clear evidence that the patient fails any of the key criteria. Missing information is not a mismatch. | 1 (gate) |
| `cancer_type` | The cohort specifically names a cancer type or histology matching the patient's active cancer. | 1 |
| `burden` | A specific stage or disease-burden requirement is documented as satisfied. | 1 |
| `treatment_history` | A specific prior-treatment requirement is documented as satisfied. | 1 |
| `biomarker` | A specific required biomarker condition is documented as satisfied. | 2 |

The match score (`match_score`) is 0-6: 0 unless the answer to `reasonableness` is yes, otherwise 1 plus the points
of each other yes answer. In the LLM's answer, each label carries an `evidence_status` (`supported`, `contradicted`,
`insufficient_information` or `not_applicable`) and a `reason`.

## Decision-model scores

`score_match_quality` reads one state per pair, the patient's key section and the cohort's key criteria, and asks the
five questions together. Each question is its label's section of `match_quality.labels.system.txt`, without the
teacher's evidence-status bookkeeping. The `reasonableness` question is reworded for a yes/no answer: *Is this
patient potentially eligible for this cohort after review of the patient history and these key criteria? If the
patient is potentially eligible, answer yes; if the patient is clearly ineligible, answer no.*

| Output column | Content |
| --- | --- |
| `patient_id`, and `cohort_trial_id` or `space_trial_id` | The pair, with the input's identifier column |
| `match_quality_score` | The expected match score as a fraction of 6, for ranking (formula below) |
| `match_quality_pass` | `match_quality_reasonableness >= match_quality.reasonableness_cutoff` (0.5): the model's own potentially-eligible answer |
| `match_quality_reasonableness`, `match_quality_cancer_type`, `match_quality_burden`, `match_quality_treatment_history`, `match_quality_biomarker` | Each question's P(yes) |

```text
match_quality_score = P(reasonable) × (1 + P(cancer_type) + P(burden)
                      + P(treatment_history) + 2·P(biomarker)) / 6
```

By default (`filter_low_quality=True`) only passing pairs are returned; `filter_low_quality=False` returns every
pair. With `return_metadata=True`, `model_metadata["score_match_quality"]` records the model, its revision or
checkpoint fingerprint, a digest of the question texts, and request and token counts.

`interpret_match_quality` explains one question's answer (`head`, default `"reasonableness"`) for specific pairs. It
computes gradient times input embedding for the question's yes-minus-no logit at its answer slot, in the same packed
sequence as the scores. It returns the score columns plus `attribution_method`, `attribution_target`
(`<head>_yes` or `<head>_no`, by P(yes) ≥ 0.5), `token_attributions` (the `top_k` tokens per source, default 30),
`attribution_coverage` and `attribution_caveat`. Token offsets refer to the `cohort_key_criteria` and
`patient_key_summary` sources. Attributions are local model sensitivities for debugging and human review, not
clinical rationales.

## LLM scores

`score_match_quality_with_llm` sends one request per pair with the `llm_match_quality` section's model (by default
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` with thinking on and 32,768 output tokens) and prompt
(`llm_match_quality.prompt_file`, `match_quality.labels.system.txt`). It returns every pair, unfiltered:

| Output column | Content |
| --- | --- |
| `llm_match_quality_score` | The match score, 0-6, or -1 when the answer fails the label contract |
| `llm_reasonableness`, `llm_cancer_type`, `llm_burden`, `llm_treatment_history`, `llm_biomarker` | True for a yes answer; None when unparsed |
| `llm_match_quality_answer_text`, `llm_match_quality_reasoning_text`, `llm_match_quality_parse_status` | With `debug_mode` only; the status is `parsed` or `parse_failed: <error>` |

Scores and labels are ranking and screening signals, not eligibility determinations. The LLM check sends patient key
text to the configured LLM endpoint; use only an endpoint authorized for that data.

## Decision model

The decision model (`matchminer_ai.matching.decision_model`) answers yes/no questions about a state in one forward
pass, in-process with Transformers. It serves the match questions, the boilerplate question of
[`exclusion_criteria_check`](exclusion-criteria.md) and the per-line question of
[`check_eligibility_criteria`](criteria-check.md). The `decision_model` configuration section selects it:

| Key | Default | Meaning |
| --- | --- | --- |
| `model_name` | `LiquidAI/d1-2B` | Hub model, or a fine-tuned checkpoint directory from the training pipeline |
| `revision` | a pinned commit | Required for a Hub model, because the model's own code runs (`trust_remote_code`) |
| `backend` | `transformers` | The only backend |
| `device`, `dtype` | `cuda`, `bfloat16` | Where and in what precision it runs |
| `max_length` | `10000` | The fine-tuning length; longer requests still run, without truncation, and are counted in the metadata's `over_max_length` |
| `set_size` | `null` | Questions read together after one state; `null` uses the model's trained size |
| `token_budget`, `batch_size` | `65536`, `256` | Padded tokens per forward pass; requests per call |
| `compile` | `false` | Compile the model |

The section is read from the configuration's raw mapping:

```python
config.raw["decision_model"]["model_name"] = "/path/to/decision-model"   # a fine-tuned checkpoint
```

The preset's default is the base model without fine-tuning. Treat its scores as provisional until a fine-tuned
checkpoint has been evaluated for your setting. The decision model runs locally, so patient text reaches only that
model. See [how matching works](../user-guide/patient-centric-matching.md#the-decision-models-questions) for the
questions of all three checks.

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
