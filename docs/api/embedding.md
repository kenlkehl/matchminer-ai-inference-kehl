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

## Large encoding jobs: several GPUs or vLLM embedding servers

`open_encoder(EncoderSettings(...))` encodes many texts at once and returns float32, L2-normalized NumPy
vectors in input order. It is what the training repository mines with; `embed_for_matching` keeps its own
configuration and defaults.

- One device (`devices=("cuda:0",)`): the model's own `encode`.
- Several devices (`devices=("cuda:0", "cuda:1", ...)`): one spawned worker process per device. The
  coordinator takes the model's own length order and deals out the batches a one-device run would form, so
  every text is encoded in the same batch and the vectors equal the one-device vectors on the same GPU type.
- `backend="openai"`: OpenAI-compatible `/v1/embeddings` servers (`remote={"server_urls": [...]}` or a
  `server_urls_file` registry). vLLM serves EmbeddingGemma 2 natively from version 0.32
  (`build_embedding_server_command` gives `vllm serve <model> --runner pooling ...`; one server per GPU). The
  client prepends the prompt, truncates and tokenizes exactly as sentence-transformers does, sends token ids
  to the least-loaded server, and normalizes. vLLM's kernels and bf16 weights give vectors close to (about
  0.999 cosine), not equal to, the local backends'.

`encoder_identity(settings)` names everything that changes the vectors (backend, model and revision, prompts,
length, precision, batching) and no paths or devices; store it with the vectors and never mix vectors with
different identities. TrialSpace's prompt profile is `trialspace_contract.model_prompts("trialspace")`.

```python
from matchminer_ai.embedding import EncoderSettings, open_encoder
from matchminer_ai.embedding.trialspace_contract import model_prompts

settings = EncoderSettings("google/embeddinggemma-2", revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
                           devices=tuple(f"cuda:{i}" for i in range(8)), batch_size=32, max_seq_length=8192,
                           dtype="float32", autocast="bfloat16", prompts=model_prompts("trialspace"),
                           include_prompt_in_pooling=True, text_only=True)
with open_encoder(settings) as encoder:
    patients = encoder.encode(patient_key_texts, "patient_summary")
```

Exact search over such vectors in query batches, on one or more devices, is
`matchminer_ai.matching.retrieval.score_batches` (every batch against every document, so the scores do not
depend on the number of devices), with `stable_top_k` for index-order ties.

::: matchminer_ai.embedding.encoder
    options:
      members:
        - EncoderSettings
        - open_encoder
        - encoder_identity

::: matchminer_ai.matching.retrieval
