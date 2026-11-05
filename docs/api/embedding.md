# Embedding

Retrieval uses TrialSpace, a single-vector Sentence Transformer: each output row
holds the input identifier and one normalized `embedding` vector, and
`generate_candidate_matches` ranks by cosine similarity. The default model is
`ksg-dfci/TrialSpace-0526` with `embedding.max_seq_length: 4096`.

Retrieval reads **key text** when it is there. Trial processing is kept in separate steps that
callers may run or skip, so the cohort text depends on which ran: after `remove_boilerplate_criteria`
it is the key criteria; without that step, the complete criteria, boilerplate included. TrialSpace
models trained by `matchminer-ai-training` learn from merged cohorts' key criteria
(`merge_equivalent_cohorts`, then `remove_boilerplate_criteria`) and patients' key sections, so
inputs prepared the same way match their training:

| Side | Identifier | Text embedded (first column present) |
| --- | --- | --- |
| Patient | `patient_id` | `cancer_history_summary` (the summary's key section); else the key section of a complete `patient_summary`, the text before `Other relevant history:` |
| Cohort | `cohort_trial_id` (or historical `space_trial_id`) | `cohort_key_criteria` from `remove_boilerplate_criteria`; else `cohort_eligibility_criteria` when the boilerplate step did not run; else historical `clinical_space_summary` |

A TrialSpace model trained by `matchminer-ai-training` saves one prompt per
side, named `patient_summary` (patients) and `this_space` (cohorts), and the
runtime encodes each side with its own prompt. `embedding.trialspace_contract`
holds these names (`PATIENT_PROMPT`, `COHORT_PROMPT`) and the retrieval
`INSTRUCTION` and `DIRECTIONS` shared with training. A model without side
prompts, such as TrialSpace-0526, encodes both sides with its saved `query`
instruction. Prompts are applied by name and prepended exactly, as in training
and its evaluation scripts, and token counts (`count_embedding_tokens`) count
that same text. Releases before November 2026 prepended the literal word
"query" instead of 0526's instruction, so embeddings they stored are not
comparable with current ones; regenerate them. `embed_texts(texts,
entity_type="patient" | "trial")` selects a side's prompt for plain strings;
without `entity_type` the shared query prompt is used.

TrialSpace-0526 was trained on historical `cancer_history_summary` and
`clinical_space_summary` text. It accepts cohort key criteria but has not been
evaluated on them; pair it with hierarchical trial extraction for the evaluated
workflow, as described in the README.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import generate_candidate_matches

config = load_default_preset()
# config.embedding["model_path"] = "/path/to/trained/trialspace"
patients = embed_for_matching(patient_summaries, entity_type="patient", config=config)
cohorts = embed_for_matching(trial_cohorts, entity_type="trial", config=config)
patient_matches = generate_candidate_matches(patients, cohorts, k=20)
trial_matches = generate_candidate_matches(cohorts, patients, k=40)
```

Cosine similarity is a ranking score, not a probability or a determination of
eligibility. Do not mix embeddings from different checkpoint generations; when
changing the model, regenerate both sides. Use an immutable checkpoint
path/revision, and restart the process when replacing a cached model at the
same path.

::: matchminer_ai.embedding.embed
    options:
      members:
        - embed_for_matching
        - embed_texts
