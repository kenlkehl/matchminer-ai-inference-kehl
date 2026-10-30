# Embedding

The default embedder is the single-vector `ksg-dfci/TrialSpace-0526` Sentence
Transformer (`backend: sequence`, 2,500-token maximum). Each output row holds the
historical identifier and one normalized `embedding` vector, and
`generate_candidate_matches` ranks by cosine similarity. It was trained on
historical `cancer_history_summary` and `clinical_space_summary` text, so pair it
with hierarchical trial extraction as described in the README.

## ColBERT token embeddings (opt-in)

Select ColBERT explicitly:

```python
config.embedding = dict(backend="colbert", model_path="lightonai/GTE-ModernColBERT-v1",
                        device="cuda")
```

`lightonai/GTE-ModernColBERT-v1` is the base checkpoint; set `model_path` to a
trained TrialSpace ColBERT checkpoint directory when available. The runtime and
training package share the native encoder, trained Q/D markers, punctuation
mask, 128-dimensional projection, retrieval instruction, and mean-MaxSim scorer.

Patient input uses `patient_id` and `patient_summary`. Historical split summaries
are combined with an `Other relevant history:` section. Cohort input uses
`cohort_trial_id` and complete `cohort_eligibility_criteria`, as returned by
`summarize_trials`. This field already includes common criteria. Historical
`space_trial_id` / `clinical_space_summary` inputs remain accepted.

Each output row contains `embedding` (document token matrix), `query_embedding`
(direction-specific instruction plus full input, encoded with the query marker),
and `embedding_contract` (JSON provenance). Keep all three columns when storing
or passing embeddings. Parquet round trips are supported. Both context limits
default to 8192 tokens, including markers/instructions; oversized text raises an
error rather than being truncated. Encoding is batched (default 16 texts), and
retrieval defaults to `pair_batch_size=512`, `retrieval_query_batch_size=64`,
`retrieval_document_batch_size=256`, and `device_cache_gb=8`.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import generate_candidate_matches

config = load_default_preset()
config.embedding = dict(backend="colbert", model_path="lightonai/GTE-ModernColBERT-v1",
                        device="cuda")
# config.embedding["model_path"] = "/path/to/trained/trialspace_round2"
patients = embed_for_matching(patient_summaries, entity_type="patient", config=config)
cohorts = embed_for_matching(trial_cohorts, entity_type="trial", config=config)
patient_matches = generate_candidate_matches(patients, cohorts, k=20, config=config)
trial_matches = generate_candidate_matches(cohorts, patients, k=40, config=config)
```

The matcher validates model weights, text contract, and context settings before
comparing tokens. Switching direction uses the other side's query embeddings;
it does not transpose patient-centric scores. Mean-MaxSim is a ranking score,
not a probability or a determination of eligibility.

Do not mix token embeddings, pooled vectors, or different checkpoint generations.
Use an immutable checkpoint path/revision; restart the process when replacing a
cached model at the same path.

::: matchminer_ai.embedding.embed
    options:
      members:
        - embed_for_matching
        - embed_texts
