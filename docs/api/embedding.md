# Embedding

Retrieval uses TrialSpace, a single-vector Sentence Transformer: each output row
holds the input identifier and one normalized `embedding` vector, and
`generate_candidate_matches` ranks by cosine similarity.

**The default model is, for now and on purpose, a local unpublished directory**
(user, 2026-11-12): the n=5000 training run's round-2 TrialSpace,
`google/embeddinggemma-2` fine-tuned for one epoch after round 2 by the training
repository's `finetune_embedder.py` (stage receipt `d7ef7270…`), at
`/ksg/kehl_mm_data/mmai/v23/data/no_phi/synthetic_dataset_n5000_screened_20261106/trialspace/models/trialspace_round2`.
Runs where it is missing or unreadable, such as off camus or its NFS mount,
fail with `FileNotFoundError` rather than mix vectors from different models:
nothing falls back to `ksg-dfci/TrialSpace-0526` or any Hub model. Loading it
needs sentence-transformers 6.1 or later (the version that saved it).

`embed_for_matching` loads and encodes with the functions training's mining uses
(`embedding.encoder`; see below), so the default encodes exactly as training
encoded this model:

| Setting | Default | Training (`trialspace/mining.py` `encoder_settings`) |
| --- | --- | --- |
| Weights / compute | `dtype: float32`, `autocast: bfloat16` (CUDA only) | float32 weights, bf16 autocast |
| Prompts | `prompt_profile: trialspace`: patients `patient_summary`, cohorts `this_space` | `model_prompts("trialspace")`, the same per-side prompts |
| Pooling | `include_prompt_in_pooling: true` (mean pooling over prompt and text) | the same |
| Model | `text_only: true` (no vision or audio towers) | the same |
| Batches | `batch_size: 32` | `--encode-batch-size 32` |
| Length | `max_seq_length: 4096` (the fit's cap) | the fit used 4,096; round-3 mining encoded at 8,192 (one synthetic cohort in 15,263 is longer than 4,096 tokens) |
| Output | L2-normalized float32 | the same |

On fabricated texts, the two paths give identical vectors on CPU and on CUDA.
CPU vectors are float32 throughout (autocast is CUDA only), so they differ
slightly from CUDA vectors (patient-cohort similarities by up to about 0.0015
on those texts); the encoder identity records which was used, so they are not
mixed either.

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
prompts and no `prompt_profile`, such as the historical TrialSpace-0526, encodes
both sides with its saved `query` instruction. Prompts are applied by name and prepended exactly, as in training
and its evaluation scripts, and token counts (`count_embedding_tokens`) count
that same text. Releases before November 2026 prepended the literal word
"query" instead of 0526's instruction, so embeddings they stored are not
comparable with current ones; regenerate them. `embed_texts(texts,
entity_type="patient" | "trial")` selects a side's prompt for plain strings;
without `entity_type` the shared query prompt is used.

The historical `ksg-dfci/TrialSpace-0526` (Qwen3-Embedding-0.6B) was trained on
`cancer_history_summary` and `clinical_space_summary` text. To use it, set
`model_path` and remove the EmbeddingGemma 2 settings (`dtype`, `autocast`,
`prompt_profile`, `default_prompt_name`, `include_prompt_in_pooling`,
`text_only`), and never compare its vectors with the default model's.

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
same path. Stored vectors are checked against the encoder that made them: a
local directory is named by a digest of the weights actually loaded (and its
training stage receipt), with the encoding settings, tokenizer, prompts and
library versions.

::: matchminer_ai.embedding.embed
    options:
      members:
        - embed_for_matching
        - embed_texts

## Large encoding jobs: several GPUs or vLLM embedding servers

`open_encoder(EncoderSettings(...))` encodes many texts at once and returns float32, L2-normalized NumPy
vectors in input order. It is what the training repository mines with; `embed_for_matching` loads and encodes
with the same functions on one device, its `embedding` configuration section giving the settings.

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
