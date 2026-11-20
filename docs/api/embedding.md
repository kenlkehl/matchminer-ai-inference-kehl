# Embedding

`embed_for_matching` encodes patient summaries and cohorts with TrialSpace, a single-vector Sentence Transformer,
into one vector space. `generate_candidate_matches` then ranks pairs by cosine similarity (see
[candidate matching](candidate-matching.md)). Cosine similarity is a ranking score, not a probability or a
determination of eligibility.

## Inputs and outputs

Retrieval reads **key text only**, never boilerplate:

| `entity_type` | Identifier | Text embedded (first column present) |
| --- | --- | --- |
| `"patient"` | `patient_id` | `cancer_history_summary`, the summary's key section; else the key section of a complete `patient_summary` (the text before `Other relevant history:`) |
| `"trial"` | `cohort_trial_id` (or historical `space_trial_id`) | `cohort_key_criteria`; else `cohort_eligibility_criteria`, when `remove_boilerplate_criteria` did not run; else historical `clinical_space_summary` |

Cohorts from `process_trials` have `cohort_key_criteria`. TrialSpace models from the
[training pipeline](../training/index.md) learn from merged cohorts' key criteria and patients' key sections, so
inputs prepared the same way match their training.

The result has the identifier column and `embedding`, one L2-normalized float32 vector per row (a list of floats),
in input order. With `return_metadata=True` it also returns a metadata dict with `package`, `config_snapshot` and
`model_metadata["embed_for_matching"]` (the model's `model_name` and `model_sha`).

```python
from matchminer_ai import load_preset
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import generate_candidate_matches

config = load_preset("default")
config.embedding["model_path"] = "/path/to/trialspace"
patient_vectors = embed_for_matching(patients, entity_type="patient", config=config)
cohort_vectors = embed_for_matching(cohorts, entity_type="trial", config=config)
cohorts_for_each_patient = generate_candidate_matches(patient_vectors, cohort_vectors, k=20)
patients_for_each_cohort = generate_candidate_matches(cohort_vectors, patient_vectors, k=50)
```

`embed_texts(texts, config=config, entity_type="patient" | "trial")` embeds plain strings and returns the vectors
and the model metadata. Without `entity_type` it uses the model's shared `query` prompt.

## Model and settings

The `embedding` configuration section selects the model and how it encodes. The preset leaves `model_path` unset:
set it to your TrialSpace checkpoint (`google/embeddinggemma-2` fine-tuned by the training pipeline), or name it as
`paths.trialspace_model` in the [site config](../reference/site-config.md). With neither, embedding raises
`ValueError`; a local path that is missing or unreadable raises `FileNotFoundError`. Embedding never falls back to
another model.

| Key | Default | Meaning |
| --- | --- | --- |
| `model_path` | `null` (the site config's `paths.trialspace_model`) | TrialSpace directory or Hugging Face model ID |
| `device` | `cuda` | One device |
| `dtype`, `autocast` | `float32`, `bfloat16` | Weight precision; autocast precision, applied on CUDA only |
| `prompt_profile` | `trialspace` | Per-side prompts (see below) |
| `default_prompt_name` | `SentenceSimilarity` | The model's default prompt |
| `include_prompt_in_pooling` | `true` | Mean pooling over the prompt and the text |
| `text_only` | `true` | Load EmbeddingGemma 2 without its vision and audio towers |
| `batch_size` | `32` | Texts per batch |
| `max_seq_length` | `4096` | Truncation length in tokens (required) |
| `prompt_file` | `embedding.txt` | The `query` prompt for a model that saved none and has no prompt profile |

These keys map onto [`EncoderSettings`](#large-encoding-jobs), so `embed_for_matching` encodes exactly as the
training pipeline's mining does, on one device. The [configuration reference](../reference/configuration.md#embedding)
describes each key. Autocast runs on CUDA only, so CPU vectors are float32 throughout and differ slightly from CUDA
vectors. The encoder identity records which was used.

## Prompts

A TrialSpace model from the training pipeline saves one prompt per side: `patient_summary` for patients and
`this_space` for cohorts. Each side is encoded with its own prompt. `prompt_profile: trialspace` applies
`embedding.trialspace_contract.model_prompts("trialspace")`, which gives the same prompts: the prefix
`task: sentence similarity | query: `, the retrieval `INSTRUCTION`, then `Patient history:` or
`Cohort eligibility criteria:`. `trialspace_contract` also holds the prompt names (`PATIENT_PROMPT`,
`COHORT_PROMPT`) and the retrieval `DIRECTIONS` shared with training.

Prompts are prepended exactly, as in training. `embedding.inference.count_embedding_tokens` counts the same text.

A model without side prompts and without a `prompt_profile` encodes both sides with its saved `query` prompt, or with
`prompt_file` when it saved none.

## Other TrialSpace models

The published `ksg-dfci/TrialSpace-0526` (based on Qwen3-Embedding-0.6B) was trained on a different text format,
`cancer_history_summary` and `clinical_space_summary`. To use it, set `model_path` to it, remove the EmbeddingGemma 2
settings (`dtype`, `autocast`, `prompt_profile`, `default_prompt_name`, `include_prompt_in_pooling`, `text_only`),
and set `max_seq_length` to the model's limit.

## Never mix vectors

Vectors from different models or encoding settings are not comparable. When you change the model, embed both sides
again. Use an immutable checkpoint path or Hub revision, and restart the process after replacing a model at the
same path, because loaded models are cached per process.

`model_metadata` names the model by its Hub SHA, or a local directory by a digest of its files' sizes and times.
Stored vectors that the package reuses carry a fuller encoder identity and are never reused when it differs: the
guideline embedding cache of `retrieve_guideline_considerations` and `PatientEmbeddings` from
`embed_guideline_patients` (see [guideline retrieval](guideline-retrieval.md)). That identity covers a digest of
the loaded weights for a local directory (with its training stage receipt), the encoding settings, tokenizer,
prompts and library versions.

::: matchminer_ai.embedding.embed
    options:
      members:
        - embed_for_matching
        - embed_texts

## Large encoding jobs

`open_encoder(EncoderSettings(...))` encodes many texts at once and returns float32, L2-normalized NumPy vectors in
input order. The training pipeline's mining uses it, and `embed_for_matching` loads and encodes with the same
functions on one device.

- **One device** (`devices=("cuda:0",)`): the model's own `encode`.
- **Several devices** (`devices=("cuda:0", "cuda:1", ...)`): one spawned worker process per device. The coordinator
  takes the model's own length order and deals out the batches a one-device run would form, so every text is encoded
  in the same batch and the vectors equal the one-device vectors on the same GPU type.
- **vLLM servers** (`backend="openai"`): OpenAI-compatible `/v1/embeddings` servers, given as
  `remote={"server_urls": [...]}` or a `server_urls_file` registry. vLLM 0.32 and later serve EmbeddingGemma 2
  natively; `matchminer_ai.llm.vllm_server.build_embedding_server_command` builds the
  `vllm serve <model> --runner pooling ...` command (one server per GPU). The client prepends the prompt, truncates
  and tokenizes exactly as sentence-transformers does, sends token IDs to the least-loaded server, and normalizes.
  vLLM's kernels and bf16 weights give vectors close to (about 0.999 cosine), not equal to, the local backends'.

`encoder_identity(settings)` names everything that changes the vectors (backend, model and revision, prompts, length,
precision, batching) and no paths or devices. Store it with the vectors and never mix vectors with different
identities.

```python
from matchminer_ai.embedding import EncoderSettings, open_encoder
from matchminer_ai.embedding.trialspace_contract import COHORT_PROMPT, PATIENT_PROMPT, model_prompts

settings = EncoderSettings(
    "google/embeddinggemma-2", revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
    devices=tuple(f"cuda:{i}" for i in range(8)), batch_size=32, max_seq_length=8192,
    dtype="float32", autocast="bfloat16", prompts=model_prompts("trialspace"),
    default_prompt_name="SentenceSimilarity", include_prompt_in_pooling=True, text_only=True,
)
with open_encoder(settings) as encoder:
    patient_matrix = encoder.encode(patient_key_texts, PATIENT_PROMPT)
    cohort_matrix = encoder.encode(cohort_key_texts, COHORT_PROMPT)
```

`matchminer_ai.matching.retrieval.score_batches` is exact search over such vectors in query batches, on one or more
devices. Every batch is scored against every document, so the scores do not depend on the number of devices.
`stable_top_k` breaks ties by document index.

::: matchminer_ai.embedding.encoder
    options:
      members:
        - EncoderSettings
        - open_encoder
        - encoder_identity

::: matchminer_ai.matching.retrieval
