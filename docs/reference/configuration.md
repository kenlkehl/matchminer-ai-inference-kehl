# Configuration Reference

Most package behavior is controlled through configuration. Configuration tells
`matchminer-ai` which models to use, whether LLM calls should run locally or
through a remote endpoint, where to cache model metadata, and how individual
workflow steps should be run.

Most users can start with the built-in default settings. You usually only need
to change configuration when you want to change models, switch between local and
remote inference, adjust runtime settings, enable debug output, or point the
package at a different endpoint.

Use `load_default_preset()` when you want to start from the package defaults and
change a few values in Python:

```python
from matchminer_ai import load_default_preset

config = load_default_preset()
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://localhost:8000/v1"]
```

## Custom Config Files

The optional guideline extractor uses `config.guideline` with the same task-level
`remote.model_name`, `remote.request_params`, and `remote.extra_body` structure.
It requires shared remote mode and writes to an explicit external output directory.
See [guideline extraction](../user-guide/guideline-extraction.md) for context,
output reserve, reasoning, concurrency, and source-provenance settings. Existing
custom YAML files without a `guideline` section continue to load; copy that section
from the default preset before using this optional API.

For package installs, treat the built-in preset files as read-only package data.
For larger or reusable changes, copy the default preset values into a YAML file
in your project, edit them, and load that file by path:

```python
from matchminer_ai import load_config

config = load_config("my_config.yaml")
```

## Root Keys

### `debug_mode`

Boolean flag used by summarization postprocessing. When true, selected
intermediate columns are retained in output tables.

### `model_metadata_cache_dir`

Directory path used by model metadata helpers to cache Hugging Face model
metadata JSON files.

## Returned Metadata

Functions with `return_metadata=True` return the package version, resolved
configuration snapshot, and model provenance collected by the package. Model
metadata is keyed by the public operation that used the model:

```json
{
  "model_metadata": {
    "summarize_patients": {
      "model_name": "organization/model",
      "model_sha": "..."
    }
  }
}
```

For HTTP inference, the entry describes the requested `remote.model_name`
looked up on Hugging Face. When that name is not a Hub model ID, `model_sha` is
`openai-compatible-endpoint` and `metadata_error` records the lookup failure.
Where the server reports per-response provenance, it is listed under
`responses`. Neither verifies the weights the endpoint actually serves.

Payloads written before 0.6 used role keys such as `embedding_model` and
`match_quality_checker`. Readers of saved embedding metadata still accept
`embedding_model`.

## `remote`

Global transport settings used when `remote.enabled` is true and LLM tasks send
OpenAI-compatible chat completion requests to external endpoints. The default
tested setup is a vLLM server with a compatible reasoning parser.
Task-specific request payload settings live under each task's `remote` block.
The remote backend reads the API key from the `OPENAI_API_KEY` environment
variable. API keys are not stored in preset files.

Google Agent Platform (Gemini and MaaS) is supported through its OpenAI-compatible Chat
Completions route. Set `remote.provider` to `google_agent_platform`; the package
then uses Google Application Default Credentials (ADC), refreshes the OAuth
access token for each request, converts a leading system message to the
user-first form required by managed open models, and omits vLLM-only
`extra_body` values. Install-time dependencies include `google-auth`; credential
files and tokens are never stored in package configuration.

### `remote.enabled`

Selects the remote LLM backend when true.

### `remote.provider`

Authentication and message-compatibility profile. `openai` is the default and
reads `OPENAI_API_KEY`. `google_agent_platform` uses ADC and ignores
`OPENAI_API_KEY`.

For example, Gemma 4 26B A4B IT MaaS in a project named `profile-notes` uses:

```python
config.remote.update(
    {
        "enabled": True,
        "provider": "google_agent_platform",
        "google_project_id": "profile-notes",
        "server_urls": [
            "https://aiplatform.googleapis.com/v1/projects/profile-notes/"
            "locations/global/endpoints/openapi"
        ],
    }
)
for task in (
    config.trial,
    config.patient,
    config.llm_match_quality,
    config.llm_exclusion_criteria,
    config.llm_good_option,
):
    task["remote"]["model_name"] = "google/gemma-4-26b-a4b-it-maas"
```

Set up ADC through the runtime environment, an attached service account, or
`gcloud auth application-default login`. The selected endpoint must still be
authorized for the sensitivity of any clinical text sent to it.

`check_openai_endpoint()` uses a minimal patient-free chat completion for this
provider because the MaaS OpenAI route does not document `/models`.

Gemini 3.8 Flash uses the same global endpoint and ADC profile with model
`google/gemini-3.8-flash`. System messages are retained for Gemini. Its supported
reasoning efforts are `low`, `medium`, and `high`; the provider maps an `xhigh`
request to `high`. Explicit sampling fields and vLLM extensions are omitted,
because [Gemini 3.8 manages sampling internally](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/guides/gemini-3-8-flash).

For patient summarization and `review_patient_workup`, configure the patient task
after setting the Google remote profile above:

```python
config.patient.update(
    sampling_profile="none", reasoning_effort="high",
    context_window=1048576, tokenizer_mode="bytes",
)
config.patient["local"]["engine"]["max_model_len"] = 1048576
config.patient["local"]["chat_template_kwargs"] = {}
config.patient["remote"].update(
    model_name="google/gemini-3.8-flash",
    tokenizer_name="Qwen/Qwen3-0.6B",
    request_params={"max_tokens": 65536, "reasoning_effort": "high"},
    extra_body={},
)
```

The local tokenizer supplies chunk boundaries only; byte accounting conservatively
bounds the hosted-model prompt instead of claiming exact Gemini token counts.
Workup review skips vLLM discovery/tokenization routes, refreshes ADC for structured
requests, and retains its normal JSON schema and exact patient-note quote validation.
These settings do not change a separately configured guideline catalog run.

### `remote.google_project_id`

Optional Google Cloud quota project supplied when `remote.provider` is
`google_agent_platform`. The endpoint URL still determines the project that
receives the inference request.

### `remote.server_urls`

List of OpenAI-compatible base URLs. Values are passed to the OpenAI client as
`base_url`. For the default Nemotron 3.5 Super configuration, the server should
be a vLLM chat endpoint launched with the `nemotron_v3` reasoning parser, the
`qwen3_coder` tool-call parser and `--trust-remote-code`; the package
`start_vllm_servers()` helper adds these flags from the selected LLM task's
`reasoning_parser`, `tool_call_parser` and local `engine` settings.

Model names and request parameters are configured per LLM task under that
task's backend block. For example, `trial.local.model_name` is the model loaded
by local vLLM, `trial.remote.model_name` is the model string sent to the
endpoint, `trial.remote.request_params` contains top-level chat completion
request fields, and `trial.remote.extra_body` contains fields sent through
request `extra_body`.

Remote mode expects the endpoint to return final answer text in
`message.content`. The default vLLM/Nemotron setup can expose reasoning separately
via vLLM reasoning parser support. Other OpenAI-compatible endpoints may work
only if they return final answer text in `message.content`; endpoints that
include reasoning text in `message.content` are not currently supported.

### `remote.max_concurrent_requests`

Maximum number of concurrent requests per remote server; default 32. Guideline
clients in one Python process share this cap across diseases for the same endpoint.
Independent processes have separate caps, so use the collection runner rather
than launching multiple independent 32-request processes.

### `remote.request_timeout`

Request timeout in seconds.

### `remote.max_retries`

Maximum retry attempts for a failed remote request.
The package disables the OpenAI SDK's separate internal retry loop so this
setting is the single retry bound.

### `remote.batch_size`

Number of prompts processed per remote-server batch.

### `remote.retry_backoff_base`

Base value, in seconds, for exponential retry backoff.

## Model-aware sampling and reasoning

The default preset sets `sampling_profile: auto` and `reasoning_effort: xhigh`
on every LLM stage: `trial`, `patient`, `llm_match_quality`,
`llm_exclusion_criteria`, and `guideline`. The selected model determines defaults
for both in-process vLLM and OpenAI-compatible vLLM requests. Quantized model IDs
containing the same family name are recognized. Guideline discovery applies the
profile after `/v1/models` identifies the actual served model.

| Model / mode | Temperature | Top-p | Top-k | Other sampling defaults |
| --- | ---: | ---: | ---: | --- |
| Nemotron 3.5 Super (default) | 1.0 | 0.95 | Disabled (-1) | min-p 0, presence penalty 0, repetition penalty 1 |
| Gemma 4 | 1.0 | 0.95 | 64 | Engine defaults for penalties |
| Qwen 3.8, thinking | 1.0 | 0.95 | 20 | min-p 0, presence penalty 0, repetition penalty 1 |
| Qwen 3.8, non-thinking | 0.7 | 0.8 | 20 | min-p 0, presence penalty 1.5, repetition penalty 1 |
| Reflection Beam 1.0 | 0.7 | 0.99 | Disabled (-1) | min-p 0, presence penalty 0, repetition penalty 1 |

Nemotron's temperature and top-p follow NVIDIA's
[model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4)
(checked 2026-10-15); its other sampling controls are neutral. The others
follow the vendor recommendations checked on 2026-09-24:
[Google Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4),
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), and
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
Beam's temperature, top-p, and repetition penalty follow its
[generation configuration](https://huggingface.co/reflection/Beam-1.0-501B-A23B/blob/main/generation_config.json),
verified on 2026-10-16; its other sampling controls are neutral. Official Beam
1.0 IDs, quantized variants, and the short aliases `beam` and `reflection-beam`
are recognized. Local server startup selects `beam` for both reasoning and tool
parsing and requires a [Beam-compatible vLLM build](https://developers.reflection.ai/serving-vllm).
Beam always reasons. The workspace defaults `chat_template_kwargs.reasoning_effort`
to `medium` rather than the upstream template's `max`, without inferring an
`enable_thinking` switch. Explicit template or API effort overrides and output
budgets are retained. When both API and template effort are supplied, they must agree.

Qwen defaults to thinking enabled, preserved thinking, and `xhigh` effort.
Supported effort values are `xhigh`, `medium`, and `low`. Remote requests send
`reasoning_effort` at the API level and in `chat_template_kwargs`; local prompt
rendering and endpoint token counting use the same template setting. Gemma 4
supports thinking enabled/disabled, not graded reasoning effort: its profile
sets `enable_thinking: true` and does not invent an `xhigh` control.

Explicit `local.generation`, `local.chat_template_kwargs`,
`remote.request_params`, and `remote.extra_body` values override profile defaults.
To turn thinking off, set the applicable `chat_template_kwargs.enable_thinking`
to false. The Qwen profile then selects the non-thinking sampling defaults.
Set a stage's `sampling_profile` to `none` to use entirely explicit settings;
`gemma4`, `qwen3.8`, and `reflection-beam` (or `beam`) also allow explicit family
selection for endpoint aliases.
Unknown model families receive no automatic vendor settings. Existing custom
configs without `sampling_profile` keep their explicit behavior, with one
exception: a section without `sampling_profile` served by Nemotron 3.5 Super
(any `...nemotron-3.5-super-vl...` model name, profile `replaces_task_sampling`;
user instruction 2026-11-05) sends NVIDIA's sampling (temperature 1.0, top_p
0.95, top_k -1, min_p 0, repetition_penalty 1.0) in place of its fixed values,
on the local and remote paths, keeping an explicit thinking switch. Name a
profile (explicit keys then win) or set `sampling_profile: none` to keep a
section's own sampling for Nemotron.

This intentionally replaces the earlier preset's greedy patient/checker LLM
sampling and mismatched Gemma top-k. Matching model versions, output budgets,
and clinical prompts are unchanged. Existing generated artifacts retain their
original settings; changed extraction settings require a fresh run directory.

```python
config = load_default_preset()
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://your-server:8001/v1"]
config.remote["max_concurrent_requests"] = 32
config.trial["remote"]["model_name"] = "Qwen/Qwen3.8-Flash-Next"
# For patient summarization, update both model_name and tokenizer_name.
# Optional explicit override:
config.trial["remote"]["request_params"]["reasoning_effort"] = "medium"
```

Model-specific reasoning text: the literal `Reasoning: high` system header is
added only for GPT-OSS model identities. Gemma, Nemotron, Qwen, and other families
use their configured chat-template/API thinking controls; task instructions are
preserved without the GPT-OSS header. Unknown model identities receive no header.

## `trial`

Task configuration for trial summarization.

### `trial.local`

Local in-process vLLM runtime settings:

- `model_name`: model identifier loaded by local vLLM and used for local
  tokenizer/chat-template rendering and model metadata lookup.
- `engine`: keyword arguments passed to `vllm.LLM(...)`; the package passes
  `model=trial.local.model_name` separately. Every default engine serves
  Nemotron 3.5 Super as the production servers do: `trust_remote_code`,
  `language_model_only`, `enable_prefix_caching`, `mamba_backend: flashinfer`,
  `mamba_ssm_cache_dtype: float16`, `enable_mamba_cache_stochastic_rounding`,
  `mamba_cache_philox_rounds: 5` and `mamba_cache_mode: align`, with
  `tensor_parallel_size: 1` (the NVFP4 build needs a Blackwell GPU). The vLLM
  server helpers pass these keys as the same `vllm serve` flags.
- `generation`: keyword arguments passed to `vllm.SamplingParams(...)`.
- `chat_template_kwargs`: keyword arguments passed to tokenizer chat-template
  rendering.

Additional `engine` and `generation` keys may be included if they are valid vLLM
keyword arguments. vLLM validates those keys when the engine/request is created.

### `trial.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts` for the legacy
`one_shot` method (`primer`, `question`).

### `trial.cohorts`

Settings of the default `cohorts` method, the three-step workflow shared with
`protocol_cohort_extraction`: `max_attempts` (per request, 1-10, default 3),
`structured_outputs` (default true; each remote request is constrained by its
step's JSON schema), `chunk_size` (default 500) and `chunks_in_flight`
(default 2, at most 16), and optional `prompt_files` mapping `screen`, `list`
and `criteria` to other packaged prompt files. A remote backend keeps
`chunk_size * chunks_in_flight` requests in flight, sending each on its own as
another returns, so one slow answer holds up only itself; a local backend gets
batches of `chunk_size` requests of one step, `chunks_in_flight` at once. The
merge and boilerplate steps (`merge_equivalent_cohorts`,
`remove_boilerplate_criteria`) and `screen_trials` use the same settings.

### `trial.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including the default
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` to `nemotron_v3` and
`google/gemma-4-31B-it` to `gemma4`. Set this explicitly when using a
model not covered by the package mapping, or use `none` to disable reasoning
parsing for a non-reasoning model. This setting applies to local vLLM execution
and vLLM server launch helpers; non-vLLM remote endpoints may ignore it or
return no separate reasoning field.

### `trial.remote`

Task-specific remote chat completion request settings:

- `model_name`: model name sent in OpenAI-compatible chat completion requests
  and used for remote endpoint metadata.
- `request_params`: top-level chat completion request fields sent as-is,
  including the output-token budget. Use `max_tokens` for vLLM and many
  compatible endpoints, or `max_completion_tokens` for endpoints that require
  it.
- `extra_body`: provider-specific fields sent as request `extra_body` when
  non-empty.

The package interprets `model_name` and fills missing values from the selected
sampling profile. Explicit values inside `request_params` and
`extra_body` otherwise pass through; the Qwen profile validates its supported
reasoning levels. The OpenAI client or remote endpoint validates other API fields.

### `trial.boilerplate_marker`

Line marker used by trial postprocessing to identify the boilerplate exclusion
section heading.

## `patient`

Task configuration for patient summarization.

### `patient.chunk_size`

Maximum character count used when splitting patient notes into serial summary
chunks.

### `patient.chunk_overlap`

Character overlap between adjacent patient-note chunks.

### `patient.prompt_margin_tokens`

Token margin reserved when truncating patient chunks before prompt rendering.

### `patient.local`

Local in-process vLLM runtime settings. See `trial.local`.
The patient context (`engine.max_model_len`, 200,000 tokens) also bounds remote
requests: each one's output budget (`max_tokens`, 50,000) is reduced to what fits after its prompt.

### `patient.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `patient.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including the default
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` to `nemotron_v3` and
`google/gemma-4-31B-it` to `gemma4`. Set this explicitly when using a
model not covered by the package mapping, or use `none` to disable reasoning
parsing for a non-reasoning model. Non-vLLM remote endpoints may return no
separate reasoning field.

### `patient.remote`

Task-specific remote chat completion request settings. See `trial.remote`.
Patient summarization also supports `tokenizer_name`, which is the local
tokenizer used for patient chunk truncation and prompt sizing before sending
requests to the remote endpoint. For self-hosted vLLM this is usually the same
as `patient.remote.model_name`.

### `patient.text_token_threshold`

Maximum token count used by local truncation before patient summarization.

## `raw_patient_note_qa`

Configuration for focused question answering over one patient's raw notes. Its
`local`, `remote`, and `reasoning_parser` fields select the same in-process vLLM
or OpenAI-compatible backend used by other LLM tasks.

- `embedding_model_name`: Hugging Face SentenceTransformer identifier or local
  path. The function argument of the same name overrides this setting. The
  default is `Qwen/Qwen3-Embedding-0.6B`, independently of TrialSpace.
- `embedding_device` and `embedding_batch_size`: embedding runtime settings.
  The default device is `cuda` for both focused Q&A and one-time full-screen
  index construction.
- `chunk_size` and `chunk_overlap`: raw-note chunk sizes in tokens from the
  embedding model's own tokenizer. The effective chunk size is capped so the
  document prefix and special tokens fit the model's sequence limit.
- `document_prefix` and `query_prefix`: optional model-specific text prepended
  before document and query embedding. The default query prefix supplies
  Qwen3-Embedding with a patient-note passage-retrieval instruction.
- `initial_top_k`: chunk count retrieved for the original question.
- `tool_top_k`: chunk count returned by each agent-requested retrieval.
- `min_similarity`: minimum cosine similarity for returned chunks. The default
  `-1.0` does not filter cosine results.
- `max_agent_steps`: maximum parent-agent actions before forced finalization.
- `response_retry_limit`: retry limit for malformed JSON responses.

Thinking is on and shares the output budget: 16,000 tokens by default, in a
50,000-token local context.

The agent can request additional semantic retrieval or a separately grounded
answer to a narrower related question. Both tools operate only on the in-memory
raw-note index. Retrieved excerpts are sent to the configured LLM backend; no
web-search tool is used. Final citations are checked against retrieved chunks
and must contain exact source substrings. With DataFrame input, each note is
chunked independently, and retrieved chunks and final evidence receive one
code-derived `note_date`. String input returns `note_date: null` because it has
no structured note-to-date provenance.

## `full_patient_screen`

Configuration for screening complete trial eligibility criteria through the
raw-note QA workflow. Its `local`, `remote`, and `reasoning_parser` fields select
the same LLM backend pattern used by other tasks.

- `max_workers`: maximum CPU worker processes for independent criterion
  questions. The runtime also caps this by question count and available CPUs.
- `process_start_method`: multiprocessing context. The default `spawn` avoids
  inheriting initialized model or HTTP-client state.
- `max_questions`: safety bound on the number of independently assessable
  criteria returned by decomposition. Exceeding it fails rather than silently
  dropping protocol criteria.
- `synthesis_evidence_limit_per_question`: maximum validated evidence records
  copied from each raw-note answer into the final synthesis prompt. Complete
  raw-note responses remain in the returned JSON.
- `response_retry_limit`: malformed JSON and schema-validation retry bound for
  criterion decomposition and final synthesis.

The default decomposition prompt sends code-assigned source IDs with exact
criterion text. The model returns those IDs, and the package reattaches the
source text after validation instead of trusting the model to copy it. The
default remote request also uses JSON-object response formatting and disables
the thinking template for this bounded schema-generation task.

Process parallelism requires `remote.enabled: true`. The package embeds all
raw-note chunks once on `raw_patient_note_qa.embedding_device`, transfers the
normalized index to CPU, and gives the worker processes access to that prepared
index. The parent-owned embedding model serves the much smaller original and
agent-generated query vectors; cosine retrieval runs inside each worker on CPU.
The workers therefore neither re-embed the note record nor load their own copy
of the embedding model. Use `max_workers=1` with local in-process vLLM. This
workflow performs no web search, but retrieved patient excerpts and aggregated
answers reach the configured LLM backend.

## `trial_space_criteria_extraction`

Configuration for extracting the complete inclusion and exclusion criteria
relevant to one trial space from a UTF-8 OCR text file. Its `local`, `remote`,
and `reasoning_parser` fields select the same LLM backend pattern as other
tasks.

- `max_document_characters`: hard input bound. Oversized documents fail rather
  than being silently truncated and losing criteria.
- `max_criteria_per_type`: maximum number of inclusion criteria and maximum
  number of exclusion criteria accepted from the model.
- `response_retry_limit`: retry bound for malformed JSON, invalid schemas,
  duplicate criteria, and criteria that are not grounded in the OCR text.

The default extraction task keeps model thinking enabled and allows up to
50,000 completion tokens. The default remote request uses JSON-object response
formatting. The entire OCR document and trial space reach the configured LLM
backend; no web search is used. Every accepted criterion must match a source
excerpt after conservative whitespace, ordinal-suffix, repeated page-edge, and
referenced trailing-footnote normalization.

## `protocol_cohort_extraction`

Configuration for `extract_protocol_cohorts`, which screens full protocol
documents (PDF, OCR text or Markdown) and extracts their cohort eligibility criteria. Its
`local`, `remote`, `sampling_profile`, `reasoning_effort` and
`reasoning_parser` fields follow the other LLM task sections.

- `max_document_characters`: hard input bound (default 800,000 characters).
  Oversized documents fail rather than being truncated.
- `max_attempts`: validation attempts per request (1-10, default 3); only
  failed requests are retried, and a cohort that fails every attempt fails its
  protocol.
- `chunk_size` (default 500) and `chunks_in_flight` (default 2, at most 16): a
  remote backend keeps `chunk_size * chunks_in_flight` requests in flight, each
  sent as another returns; a local backend gets batches of `chunk_size`
  requests of one step, `chunks_in_flight` at once. With `output_dir`, each
  protocol is saved as soon as it is decided.
- `structured_outputs`: when true and the remote backend is enabled, each
  request carries a strict JSON-schema `response_format` for its step.
- `prompt_files` (optional): maps `screen`, `list` and `criteria` to other
  packaged prompt files; by default protocols use the same three prompts as
  ctgov listings (`trial.cohorts.screen.user.txt`,
  `trial.cohorts.list.user.txt`, `trial.cohorts.criteria.user.txt`).

The default model context is 262,144 tokens with up to 50,000 completion tokens,
so that full industry protocols (often 50,000-160,000 tokens) fit with room for
reasoning.

## `synthetic_protocol_generation`

Configuration for `generate_synthetic_protocols`. Its `local`, `remote`,
`sampling_profile`, `reasoning_effort` and `reasoning_parser` fields follow the
other LLM task sections.

- `chunk_size` (default 500) and `chunks_in_flight` (default 2, at most 16): a
  remote backend keeps `chunk_size * chunks_in_flight` requests in flight, each
  sent as another returns, so one slow protocol holds up only itself; a local
  backend gets batches of `chunk_size`, `chunks_in_flight` at once. Outputs are
  written when their request returns.
- `max_attempts`: attempts per trial for empty or truncated outputs (1-10,
  default 3).
- `layouts_file`: optional override of `synthetic_protocol_layouts.json`.

The default model context is 262,144 tokens with up to 100,000 completion
tokens, so long protocols and reasoning traces fit.

## `embedding`

Configuration for TrialSpace embedding. Retrieval embeds key text only: patient
key sections (`cancer_history_summary`) and cohort key criteria
(`cohort_key_criteria`); see the [embedding API](../api/embedding.md).

### `embedding.model_path`

Sentence-transformer model path passed to `SentenceTransformer(...)`.

### `embedding.device`

Device string passed to `SentenceTransformer(...)`.

On CUDA, embedding first uses PyTorch's default scaled-dot-product-attention
dispatcher. If cuDNN reports that it cannot build a valid attention execution
plan, the package retries with the CUDA flash, memory-efficient, or math
attention backends and remembers that fallback for the same model and device.
Other runtime errors are not retried.

### `embedding.prompt_file`

Prompt filename loaded from `matchminer_ai.prompts` and used as the `query`
prompt only for a model that saved none; a model's saved prompts are never
replaced. The file is prepended exactly as written (only its final newline is
removed). The default, `embedding.txt`, is byte-identical to TrialSpace-0526's
saved `query` prompt, including its trailing space. A TrialSpace model that
saves prompts named `patient_summary` and `this_space` encodes each side with
its own prompt.

### `embedding.max_seq_length`

Runtime truncation cutoff for embedding inputs (default 4096). `SentenceTransformer`
uses this value during `encode()`, so inputs longer than this limit are
truncated before embedding generation. QC reports use the same value, counting
tokens with each side's prompt, when flagging key text that exceeds the
embedding input limit.

This value should match the `model_path` configured alongside it — it is
specific to the embedding model in use, not a general-purpose default.
Changing `embedding.model_path` to a different model without reviewing
`max_seq_length` against that model's own native truncation cutoff (e.g.
its `sentence_bert_config.json`) can silently truncate inputs at the wrong
point, producing different embeddings than the model would otherwise
produce for longer inputs.

## `decision_model`

The decision model that answers the match, boilerplate and per-criterion yes/no
questions (`matchminer_ai.matching.decision_model`; see the
[match-quality](../api/match-quality.md#decision-model),
[exclusion-criteria](../api/exclusion-criteria.md) and
[criteria-check](../api/criteria-check.md) APIs). It runs in-process with
Transformers and the checkpoint's own System One code (`trust_remote_code`).

### `decision_model.model_name`

A Hugging Face model id or a local checkpoint directory (default
`LiquidAI/d1-2B`). A checkpoint fine-tuned by the training repository's
`train_decision_model.py` is a local directory holding the weights, the d1 code
and `decision_contract.json`, whose fingerprint is reported as `model_sha`.

### `decision_model.revision`

The commit of a Hub model (default `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`,
d1-2B's main on 2026-11-05). Required for every Hub model, because its code
runs; ignored for a local directory.

### `decision_model.backend`

`transformers` (the only backend).

### `decision_model.device` / `decision_model.dtype`

Where the model runs (default `cuda`) and its weight type (default `bfloat16`).

### `decision_model.max_length`

The fine-tuning request length (default 10000 tokens). Longer requests still
run, without truncation, and are counted in the metadata's `over_max_length`.

### `decision_model.set_size`

Questions read together after one state (null, the default, uses the model's
trained size, 16). Larger is faster and departs from training.

### `decision_model.token_budget` / `decision_model.batch_size`

Padded tokens per forward pass (default 65536) and requests per
`system_one_batch` call (default 256).

### `decision_model.compile`

`torch.compile` the model before use (default false).

## `match_quality`

### `match_quality.score_cutoff`

Minimum `match_quality_score`, the expected match score fraction of the five
match questions, required for `match_quality_pass == true` (default 0.2).

## `exclusion_criteria`

### `exclusion_criteria.pass_cutoff`

Minimum P(yes) of the boilerplate question (yes = potentially eligible) for
`exclusion_criteria_pass == true` (default 0.5).

## `criteria_check`

### `criteria_check.fail_cutoff`

A criteria line whose P(yes) (yes = the patient does not clearly fail the line)
is below this is flagged `criterion_clearly_fails` (default 0.5).

## `llm_match_quality`

Configuration for the LLM-based match-quality checker.

### `llm_match_quality.local`

Local in-process vLLM runtime settings. See `trial.local`. The default context
is 100,000 tokens with 32,768 completion tokens, reasoning included (the training
teacher's label budget for the same prompt).

### `llm_match_quality.prompt_file`

System prompt filename loaded from `matchminer_ai.prompts`. The default,
`match_quality.labels.system.txt`, is the training teacher's five-label prompt;
a replacement must return the same JSON contract.

### `llm_match_quality.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including the default
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` to `nemotron_v3` and
`google/gemma-4-31B-it` to `gemma4`. Non-vLLM remote endpoints may
return no separate reasoning field.

### `llm_match_quality.remote`

Task-specific remote chat completion request settings. See `trial.remote`.

## `llm_exclusion_criteria`

Configuration for the LLM-based exclusion-criteria checker.

### `llm_exclusion_criteria.local`

Local in-process vLLM runtime settings. See `trial.local`. The default context
is 100,000 tokens with 32,768 completion tokens, reasoning included (the training
teacher's boilerplate-label budget for the same prompt).

### `llm_exclusion_criteria.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`. The default,
`llm_exclusion_criteria.user.txt`, asks whether the patient clearly fails any
of the cohort's boilerplate criteria.

### `llm_exclusion_criteria.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including the default
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` to `nemotron_v3` and
`google/gemma-4-31B-it` to `gemma4`. Non-vLLM remote endpoints may
return no separate reasoning field.

### `llm_exclusion_criteria.remote`

Task-specific remote chat completion request settings. See `trial.remote`.

## `llm_good_option`

Configuration for experimental-drug selection during patient-free research and
for the patient-specific four-point-per-drug LLM scorer. Its `local` and
`remote` blocks follow the same structure as `trial.local` and `trial.remote`.
The default request enables model thinking, permits up to 100,000 completion
tokens within a 262,144-token local context, and uses a repetition penalty of
1.1. Each scoring message describes exactly one patient; the backend may batch
many independent messages in one run. `score_good_options_with_llm` can
selectively retry code-validation failures with `max_parse_attempts`; each
follow-up includes the prior invalid answer, exact parser error, and finish
reason. Set `reasoning_off_fallback=True` to add one final attempt with thinking
disabled after the ordinary attempts are exhausted.

## `good_option_catalog`

Configuration for patient-free trial drug extraction, evidence synthesis, and
clean task-specific projections. `synthesis_evidence_max_tokens` bounds the raw
evidence packed for one drug, and it bounds passage text only: the JSON
scaffolding around the passages adds roughly 6.5%, so the real prompt costs
about 1.08x this value. It defaults to 190,000, which pairs with the 262,144
`max_model_len` of the local synthesis engine and leaves roughly 50,000 tokens
for a reasoning trace and the final JSON. Evidence beyond the budget is
discarded round-robin across sources before the model sees it, so a lower value
silently drops passages rather than failing; when serving a smaller context,
reduce it to about `(context - completion budget) / 1.08`. Synthesis reads the ledger in
`synthesis_chunk_max_tokens` chunks (default 60,000; classes use
`class_evidence_max_tokens`). Each chunk's call sees the facts so far and returns
only new or revised items, which code merges in: an existing fact is removed
only when a validated item names it in `replaces`. The stage-specific
`synthesis_llm` override defaults to a 100,000-token completion budget so
reasoning-enabled models can think over a chunk and still finish the final JSON.
Larger chunks are not better: on 40 multi-chunk drugs with Gemma 4 31B,
100,000-token chunks gave about 25% fewer facts and cited passages than 60,000,
because the model condenses more of each chunk. A chunk plus passage metadata,
the existing facts (up to about 25,000 tokens for the broadest classes), and this
budget must fit together in the served context; lower both chunk limits when
serving less than 262,144 tokens, since the endpoint rejects a request whose
prompt plus `max_tokens` exceeds its context. A model that exhausts
the budget mid-trace returns an empty answer with `finish_reason=length`, which
is retried rather than silently accepted, and if every ordinary attempt ends
that way the final attempt disables thinking.
`screening_max_attempts` and `synthesis_max_attempts` control structured-output retries.
`screening_checkpoint_batch_size` and `class_checkpoint_batch_size` bound the
screening and classify LLM batches so successfully returned batches can be
checkpointed throughout long catalog builds. Synthesis is not batched: each
drug or class is its own chain of chunk calls, checkpointed as soon as that
chain finishes, with up to `remote.max_concurrent_requests` chains in flight. Screening and synthesis LLM overrides inherit from
`llm_good_option`. With a remote backend, `structured_outputs` (default `true`)
decodes the screening and class answers under their JSON schemas
(`response_format` `json_schema`); set it to `false` for a server without that
support, and the answers are parsed and validated from free text as before.

Evidence is also retrieved on two axes beside the drug's own name, and each has
its own budget so that neither can evict the other.

The `classify` stage gives an NCIt-matched drug every NCIt class above its
concept, below the fixed umbrella concepts, without an LLM call; those are not
capped. `class_max_per_drug` (default 3) caps the classes the LLM may add on
top, or name for a drug NCIt leaves unclassified; an added NCIt class's own
ancestors do not count against it. `class_candidate_limit` (default 25) bounds
the NCIt candidate classes, found by local search of class names and synonyms
in the drug's name, definition, and mechanism passages, that the LLM may choose
from by code. `class_assignment_evidence_max_tokens` (40,000) bounds the
mechanism passages that stage reads and searches, and `class_llm` gives it its
own completion budget.
`class_evidence_max_tokens` (60,000) bounds each chunk of the corpus that class
*synthesis* reads. Class synthesis is a separate call from agent synthesis rather than a
larger pooled one: 165 drugs already use more than half the agent budget and the
largest uses 91% of it, so pooling would let a flood of same-class passages evict
the agent's own data. `class_option_summary_max_tokens` bounds each class's
stored projection. The scoring prompt re-renders classes and drugs from their
structured facts instead, under `good_option_prompt`.

Retrieval settings for both axes live on `ResearchSettings` rather than in this
block: `class_facets`, `class_sources`, `class_research_rounds`, and the
`max_class_*` record caps for the class axis, and `indication_terms_per_drug`
(default 25), `indication_facets`, `indication_sources`, and the
`max_indication_*` caps for the indication axis. Indication terms come from the
`conditionsModule.conditions` of the trials a drug appears in, falling back to
`derivedSection.conditionBrowseModule.meshes`, so the catalog stays patient-free
and cacheable. Both axes call only the query-driven sources; the ctgov, label,
and curation adapters search by exact agent name and return nothing for a class
name.

`indication_facets` defaults to `efficacy_by_tumor`, `biomarker_prevalence`, and
`biomarker_directed_efficacy` for both drug–disease and class–disease retrieval.

PubMed requests read an optional `NCBI_API_KEY` from the environment, never from
configuration snapshots. Without it, the NCBI API works at its lower public rate limit.

## `good_option_prompt`

Patient-time packing of catalog drug and class evidence into the
`llm_good_option` prompt. It lives outside `llm_good_option` and
`good_option_catalog` because those sections fingerprint catalog checkpoints.

- `context_tokens` (default `null`): the teacher's context window. `null` uses
  `llm_good_option.local.engine.max_model_len` (262,144); set it when a remote
  endpoint serves a different length.
- `chars_per_token` (3.5): conversion rate for the budget. Conservative for
  Gemma 4, which averages about 4.4 characters per token on catalog text, and
  for Nemotron 3.5 Super, whose tokenizer averages the same as Gemma 4's (4.6)
  on ctgov eligibility criteria.
- `safety_tokens` (4,096): margin kept beyond the completion allowance, which
  is the active backend's `llm_good_option` `max_tokens`.
- `max_drug_section_tokens`, `max_class_section_tokens` (20,000 each): upper
  bound on any one drug or class, however much room remains; `null` removes it.

See [Evidence packing](../api/good-options.md#evidence-packing) for how the
budget is shared and when evidence is condensed.

## `good_option_check`

Settings for `check_good_options`, the on-demand patient-trial check. They are
applied to a copy of the config at call time and stay outside `llm_good_option`
so catalog-build fingerprints are unaffected.

- `max_output_tokens` (50,000): completion cap for the scoring request,
  replacing the active backend's `llm_good_option` `max_tokens` (100,000) so
  more of the context carries evidence. `null` keeps the `llm_good_option`
  value.
- `discover_context_tokens` (`true`): when `good_option_prompt.context_tokens`
  is unset and remote inference is enabled with an OpenAI-compatible provider,
  size evidence packing to the smallest `max_model_len` the servers report at
  `/v1/models`.
- `discovery_timeout_seconds` (10.0): timeout for each lookup. A failed lookup
  falls back to the configured context and is reported in the metadata.
- `max_parse_attempts` (3): total attempts for answers that fail validation.
- `reasoning_off_fallback` (`true`): add one final attempt with thinking
  disabled.

## `good_option_checker`

Deprecated. Configuration for the retired GoodOptionChecker classifier, kept so
existing configs load. GoodOption scoring uses the LLM teacher
(`llm_good_option`); the classifier APIs emit `DeprecationWarning`.
