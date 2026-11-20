# Configuration

Configuration selects the models each step uses, whether LLM calls run in-process or against an OpenAI-compatible
endpoint, and each step's limits, retries and cutoffs. The package ships one preset, `default`
(`matchminer_ai/presets/default.yaml`). This page describes its sections. Values that belong to one installation
(endpoints, cloud projects, paths) are not here; they come from the [site config](site-config.md). How requests reach
LLM servers (model profiles, the endpoint pool, traces, pacing, cancellation) is described in
[LLM runtime](llm-runtime.md).

## Loading and changing a configuration

Start from the preset and change a few values in Python:

```python
from matchminer_ai import load_default_preset

config = load_default_preset()
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://localhost:8000/v1"]
config.embedding["model_path"] = "/path/to/your/trialspace"
config.raw["decision_model"]["device"] = "cpu"
```

For larger or reusable changes, copy `default.yaml` into your project, edit it and load it by path. Treat the
installed preset file as read-only package data.

```python
from matchminer_ai import load_config, load_preset

config = load_config("my_config.yaml")
config = load_preset("default")  # the same as load_default_preset()
```

- **Sections with attributes:** `remote`, `embedding`, `trial`, `patient` and the other LLM sections are attributes of
  the returned `MMAIConfig` (`config.trial`, `config.llm_good_option`, ...).
- **Sections read from `config.raw`:** `decision_model`, `match_quality`, `exclusion_criteria`, `criteria_check`,
  `llm_criteria_check`, `good_option_prompt`, `good_option_check` and `good_option_checker`. Change them through
  `config.raw[...]`.
- **Required keys:** a YAML file must contain `debug_mode`, `model_metadata_cache_dir`, `trial`, `patient` and
  `embedding`. Other sections may be left out; a function that needs a missing section raises an error naming it.
  Sections the package does not read are kept in `config.raw` and ignored.

## Root keys

| Key | Default | Meaning |
| --- | --- | --- |
| `version` | `0` | Preset format version; not interpreted |
| `debug_mode` | `false` | Keep intermediate columns and raw responses in outputs (for example `trial_input_text`, per-attempt summaries, LLM answer and reasoning text) |
| `model_metadata_cache_dir` | `.mmai_cache/model_metadata` | Cache of Hugging Face model metadata, relative to the working directory; `null` uses `~/.cache/matchminer_ai/model_metadata`. Local model directories are identified from their files and never cached |

## Returned metadata

Functions called with `return_metadata=True` return the package version, a snapshot of the configuration, and the
provenance of each model used, keyed by the public function that used it:

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

For remote inference the entry describes `remote.model_name` as looked up on Hugging Face. When that name is not a
Hub ID, `model_sha` is `openai-compatible-endpoint` and `metadata_error` records why the lookup failed. Where the
server reports per-response provenance, it is listed under `responses`. Neither verifies the weights the endpoint
actually serves. Readers of saved embedding metadata also accept the older key `embedding_model`.

## `remote`

Shared transport settings. With `remote.enabled`, every LLM section sends OpenAI-compatible chat-completion requests
to these endpoints instead of loading a model in-process. Model names and request fields stay per section (see
[LLM sections](#llm-sections)).

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Use remote endpoints for every LLM section |
| `provider` | `openai` | Authentication and message profile: `openai`, `google_agent_platform` or `databricks_anthropic` (below) |
| `server_urls` | `["http://localhost:8000/v1"]` | Endpoint base URLs, as a list or a comma-separated string. `http://` and `/v1` are added when missing |
| `server_urls_file` | unset | A JSON server registry re-read during a run ([LLM runtime](llm-runtime.md#remote-endpoint-pool)) |
| `max_concurrent_requests` | `32` | Requests in flight per server; the ceiling of adaptive concurrency |
| `adaptive_concurrency` | see below | Per-endpoint concurrency control |
| `request_timeout` | `600` | Seconds a request may take |
| `max_retries` | `6` | Attempts per request on the direct transport (below), and for structured requests such as guideline extraction |
| `batch_size` | `1000` | Prompts per server batch on the direct transport |
| `retry_backoff_base` | `1.0` | Base, in seconds, of the direct transport's exponential backoff |
| `recovery_timeout` | unset (86,400) | Seconds the endpoint pool keeps retrying a request after transient failures |
| `pool_max_workers` | unset (1,536) | Client threads of the endpoint pool |
| `trace_dir` | unset | Directory for complete response archives ([LLM runtime](llm-runtime.md#reasoning-trace-archives)) |
| `google_project_id` | unset | Google Cloud quota project for `google_agent_platform` |
| `databricks_profile` | unset | Databricks CLI profile for `databricks_anthropic` |
| `request_start_interval_seconds`, `capacity_retry_initial_seconds`, `capacity_retry_max_seconds` | `0`, `0`, `60` | Pacing of structured requests ([LLM runtime](llm-runtime.md#request-pacing)) |

The `openai` provider reads the API key from `OPENAI_API_KEY`; keys and credentials are never stored in configuration
files. The endpoint must return the final answer in `message.content`. A vLLM server with the model's reasoning
parser does this; endpoints that put reasoning text in `message.content` are not supported.

**Transport.** With `adaptive_concurrency.enabled` (the default) or a `server_urls_file`, requests go through the
shared endpoint pool. The pool retries transient failures (connection errors, timeouts, HTTP 408, 409, 429 and 5xx)
on whichever endpoint is free until `recovery_timeout` passes; other errors fail at once. With the pool off, and for
providers whose credentials refresh per request (`google_agent_platform`), the direct transport sends `batch_size`
prompts per server and retries each request up to `max_retries` times. The OpenAI SDK's own retries are disabled in
both. Limits are per process: separate processes have separate limits.

**`adaptive_concurrency`** starts each endpoint at `initial` requests and moves between `minimum` and
`max_concurrent_requests` (or a registry server's `upstream_concurrency`):

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Adapt limits; `false` keeps fixed limits |
| `initial` | `32` | Starting limit per endpoint |
| `minimum` | `1` | Lowest limit |
| `success_threshold` | `32` | Successful requests in a saturated window before the limit grows |
| `increase_step` | `32` | Growth: the larger of twice the limit and the limit plus this step |
| `adjustment_interval` | `10.0` | Seconds between adjustments |
| `backoff_factor` | `0.5` | Factor applied to the limit after a transport or capacity failure |
| `metrics_interval` | `0` | Seconds between vLLM `/metrics` polls for queue and KV-cache feedback; `0` uses transport outcomes only |

Model-output validation failures never reduce concurrency.

### Google Agent Platform

`provider: google_agent_platform` calls Google's OpenAI-compatible Chat Completions route with Application Default
Credentials (ADC), refreshing the access token for each request. Set up ADC through the runtime environment, an
attached service account or `gcloud auth application-default login`; the `google-auth` package is a dependency.
`check_openai_endpoint()` checks this provider with a minimal patient-free chat completion, because the route does not
document `/models`.

- **Gemma 4 MaaS** (`google/gemma-4-26b-a4b-it-maas`): a leading system message is folded into the first user
  message. Of `extra_body`, only `top_k` and `chat_template_kwargs.enable_thinking` are sent, and `reasoning_effort`
  is dropped.
- **Gemini** (`google/gemini-4-hydrogen`, `google/gemini-3.8-flash`): system messages are kept and `extra_body` is not
  sent. Gemini manages its own sampling, so set `sampling_profile: none` and send no sampling fields. For
  `gemini-3.8-*`, sampling fields are removed and the section's `reasoning_effort` is sent, with `xhigh` as `high`.

For example, Gemma 4 MaaS in a project named `my-gcp-project`:

```python
config.remote.update(
    enabled=True,
    provider="google_agent_platform",
    google_project_id="my-gcp-project",
    server_urls=[
        "https://aiplatform.googleapis.com/v1/projects/my-gcp-project/"
        "locations/global/endpoints/openapi"
    ],
)
for section in (
    config.trial,
    config.patient,
    config.llm_match_quality,
    config.llm_exclusion_criteria,
    config.llm_good_option,
):
    section["remote"]["model_name"] = "google/gemma-4-26b-a4b-it-maas"
```

Patient summarization and `review_patient_workup` against Gemini need a local tokenizer for chunk boundaries and
byte-based prompt budgets, because Gemini's tokenizer is not available:

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

Byte accounting bounds the hosted prompt conservatively; it does not claim Gemini token counts. The selected endpoint
must be authorized for the sensitivity of any clinical text sent to it.

### Claude through Databricks

`provider: databricks_anthropic` sends requests to Claude through a Databricks AI Gateway's Anthropic Messages route,
authenticated with a Databricks CLI profile (`databricks_profile`, else the site config's `databricks.profile`).
`server_urls` is the gateway URL, `https://<workspace>/ai-gateway/anthropic`, and a section's `model_name` is the
gateway's three-part endpoint name.

- **Where it works:** structured requests, such as the
  [SQL workup agent](../api/workup-sql.md#claude-through-a-databricks-ai-gateway), and the remote path that patient
  summarization, the cohort steps and the LLM checks use.
- **Requests:** each request streams through the Anthropic SDK, keeping only the token budget, the effort (the
  section's `reasoning_effort`) and a response format. Claude manages its own sampling and thinking, so sampling and
  `extra_body` fields are dropped.
- **Endpoint pool:** requests use the direct transport (`max_concurrent_requests` per gateway URL); the adaptive
  endpoint pool serves OpenAI-compatible servers only.
- **Tokenizer:** patient summarization still needs a Hugging Face tokenizer for chunking (`tokenizer_name`, for
  example the preset's Nemotron); Claude counts tokens somewhat differently.
- **Monitoring:** with `MMAI_GATEWAY_REQUEST_LOG` set to a file, every request appends one line of metadata (times,
  HTTP status, stop reason, served model, token counts; never content), for watching rate limits.

## LLM sections

These sections each configure one LLM task: `trial`, `patient`, `document_extraction`, `patient_extraction`,
`workup_query_compile`, `workup_sql_agent`, `protocol_cohort_extraction`, `synthetic_protocol_generation`,
`llm_criteria_check`, `llm_match_quality`, `llm_exclusion_criteria`, `llm_good_option` and `guideline`. Every one
except `guideline` (remote only, [below](#guideline)) defaults to `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`,
locally and remotely, with thinking on. In-process, the NVFP4 build needs a GPU with native FP4 (Blackwell); see
[LLM runtime](llm-runtime.md#nvidia-nemotron-35-super-vl-default). In the section tables, `max_tokens` stands for
both `local.generation.max_tokens` and `remote.request_params.max_tokens`.

| Key | Default | Meaning |
| --- | --- | --- |
| `local.model_name` | Nemotron 3.5 Super | Model loaded by in-process vLLM; also the tokenizer and chat template for local prompt rendering |
| `local.engine` | see below | Keyword arguments to `vllm.LLM(...)` (the model is passed separately) |
| `local.generation` | section's `max_tokens`, `skip_special_tokens: false` | Keyword arguments to `vllm.SamplingParams(...)` |
| `local.chat_template_kwargs` | `enable_thinking: true` | Chat-template arguments for rendering and token counting |
| `remote.model_name` | Nemotron 3.5 Super | Model name sent to the endpoint |
| `remote.request_params` | section's `max_tokens` | Top-level chat-completion fields, sent as given: the output budget (`max_tokens`, or `max_completion_tokens` for endpoints that require it), sampling fields, `response_format` |
| `remote.extra_body` | `skip_special_tokens: false`, `chat_template_kwargs.enable_thinking: true` | Fields sent as the request's `extra_body`, such as vLLM's `top_k`, `min_p`, `repetition_penalty` and `chat_template_kwargs` |
| `sampling_profile` | `auto` | Publisher sampling for the model ([below](#model-aware-sampling-and-reasoning)) |
| `reasoning_effort` | `xhigh` | Effort for models with graded reasoning ([below](#model-aware-sampling-and-reasoning)) |
| `reasoning_parser` | `auto` | vLLM reasoning parser for local runs and the server helpers. `auto` resolves known models (Nemotron to `nemotron_v3`, Gemma 4 to `gemma4`) and raises for an unknown one; name a parser, or `none` for a non-reasoning model |
| `tool_call_parser` | unset (`auto`) | vLLM tool-call parser that `start_vllm_servers()` passes (Nemotron: `qwen3_coder`); `none` disables it |

- **Engines:** every default `local.engine` (except `workup_sql_agent`'s, which never runs in-process) sets
  `max_model_len` (per section), `tensor_parallel_size: 1`,
  `gpu_memory_utilization: 0.9`, and the settings Nemotron is served with: `trust_remote_code`,
  `language_model_only`, `enable_prefix_caching`, `mamba_backend: flashinfer`, `mamba_ssm_cache_dtype: float16`,
  `enable_mamba_cache_stochastic_rounding`, `mamba_cache_philox_rounds: 5` and `mamba_cache_mode: align`.
  [`start_vllm_servers()`](../api/llm.md) passes the same settings as `vllm serve` flags. Other valid `vllm.LLM` and
  `SamplingParams` keys may be added; vLLM validates them.
- **Context in remote mode:** `local.engine.max_model_len` is also the context budget for remote requests in steps
  that size their own prompts (patient summarization, patient-level extraction, the SQL agent and GoodOption evidence
  packing). Those steps reduce each request's output budget to what fits after its prompt.
- **Output budget:** reasoning and the answer share `max_tokens`, so the defaults leave room for 10,000-15,000
  reasoning tokens.

### Model-aware sampling and reasoning

- **`sampling_profile: auto`** recognizes the model by name, quantized re-uploads included, and fills in its
  publisher's sampling settings and thinking switch, locally and remotely. Explicit sampling keys in the section
  (`local.generation`, `remote.request_params`, `remote.extra_body`, `chat_template_kwargs`) win. A profile can be
  named for a served alias the package cannot recognize: `nemotron-3.5-super-vl`, `gemma4`, `qwen3.5`, `qwen3.8`,
  `reflection-beam` or `beam`. `none` sends only the section's own settings. Unknown models get no vendor settings.
  The profiles and their values are listed in [LLM runtime](llm-runtime.md#model-profiles).
- **Sections without `sampling_profile`** send their own fixed sampling. The one exception is Nemotron 3.5 Super,
  whose profile replaces a section's fixed sampling (keeping an explicit thinking switch). In the preset this applies
  to `llm_good_option`: its greedy settings are not used while the model is Nemotron. Set `sampling_profile: none`
  to keep a section's own values.
- **Thinking:** turn it off with `chat_template_kwargs.enable_thinking: false` (local and `remote.extra_body`).
  Qwen's profile then switches to its non-thinking sampling.
- **`reasoning_effort`:** used by Qwen 3.8 (`xhigh`, `medium` or `low`, sent as an API field and a chat-template
  argument) and by `gemini-3.8-*` on the Google provider. Nemotron, Gemma 4 and Qwen 3.5 have only a thinking switch.
  Beam's template effort defaults to `medium`. For other hosted models, put `reasoning_effort` in
  `remote.request_params`.
- **System header:** the `Reasoning: high` system line is added only for GPT-OSS models.

```python
config = load_default_preset()
config.remote.update(enabled=True, server_urls=["http://localhost:8001/v1"])
config.trial["remote"]["model_name"] = "Qwen/Qwen3.8-Flash-Next"
config.trial["remote"]["request_params"]["reasoning_effort"] = "medium"
# For patient summarization, set both patient.remote.model_name and tokenizer_name.
```

## `trial`

Trial cohort extraction from ClinicalTrials.gov listings (`trials.extract_cohorts`, `trials.process_trials`) and the
two optional steps `merge_equivalent_cohorts` and `remove_boilerplate_criteria`, plus `screen_trials`, when they run
with `section="trial"` (the default). See [Trials](../api/trials.md).

| Key | Default | Meaning |
| --- | --- | --- |
| `cohorts.max_attempts` | `3` | Attempts per request for an answer that fails validation (1-10) |
| `cohorts.structured_outputs` | `true` | Constrain each remote answer to its step's JSON schema |
| `cohorts.chunk_size` | `500` | See below |
| `cohorts.chunks_in_flight` | `2` | See below (1-16) |
| `cohorts.prompt_files` | unset | Maps `screen`, `list` and `criteria` to other packaged prompt files |
| `local.engine.max_model_len` | `100000` | Local context |
| `max_tokens` | `50000` | Output budget (`local.generation`, `remote.request_params`) |

A remote backend keeps `chunk_size * chunks_in_flight` requests in flight, sending each as another returns, so one
slow answer holds up only itself. A local backend gets batches of `chunk_size` requests of one step,
`chunks_in_flight` at once. The default prompts are `trial.cohorts.screen.user.txt`, `trial.cohorts.list.user.txt`
and `trial.cohorts.criteria.user.txt`, shared with full protocols.

## `patient`

Serial patient summarization (`patients.summarize_patients`). See [Patients](../api/patients.md).

| Key | Default | Meaning |
| --- | --- | --- |
| `chunk_size` | `50000` | Tokens per chunk of a patient's dated, deduplicated notes (the model's tokenizer, or `remote.tokenizer_name`) |
| `chunk_overlap` | `500` | Tokens shared by adjacent chunks |
| `prompt_margin_tokens` | `5000` | A chunk longer than `max_model_len` minus this margin (at least 1,024 tokens) is cut to its beginning and end |
| `scheduling` | `continuous` | How patients share the backend ([below](#patientscheduling)) |
| `prompt_build_workers` | unset (CPU count, at most 32) | Processes that render prompts |
| `summary_quality_max_attempts` | unset (3) | Attempts per chunk whose summary fails the [quality checks](../api/patients.md#summary-quality-checks) |
| `summary_quality_trace_dir` | unset | Local directory keeping every prompt, answer and reasoning trace, rejected attempts included |
| `prompt_files` | `patient.serial.user.primer.txt`, `patient.serial.user.question.txt` | Serial-summary prompt |
| `tokenizer_mode` | unset | `bytes`: budget prompts by UTF-8 bytes, for hosted models whose tokenizer is unavailable |
| `context_window` | unset | Context for `review_patient_workup`; unset uses the endpoint's reported length |
| `text_token_threshold` | `95000` | Read only by the `LocalBackend.truncate_texts` helper; summarization does not use it |
| `remote.tokenizer_name` | Nemotron 3.5 Super | Local tokenizer for chunking and prompt sizing in remote mode; usually the served model |
| `local.engine.max_model_len` | `200000` | Context, locally and for remote requests |
| `max_tokens` | `50000` | Output budget, reduced per request to what fits after its prompt |

### `patient.scheduling`

Each patient's chunks are always summarized in order, each prompt carrying the summary accepted for the previous
chunk.

- **`continuous`:** with a remote backend, each patient's next chunk is sent as soon as the previous one is final,
  after its quality retries. Requests up to the endpoints' capacity are in flight: `remote.max_concurrent_requests`
  per listed server, or with a registry the sum of its servers' `upstream_concurrency`, re-read during the run.
  Waiting patients start longest remaining chain first. A local engine, which batches prompts itself, runs in rounds.
- **`rounds`:** every patient's first chunk is sent as one batch, then every second chunk, and so on; each round waits
  for its slowest patient.

Both give the same prompts, quality checks, retries and results, and either can resume the other's `checkpoint_dir`
([Resuming patient summarization](../api/patients.md#resuming-patient-summarization)). Under an endpoint pool,
`rounds` reports its first round's per-request endpoints in `model_metadata["responses"]`; `continuous` records where
each request ran only in the debug attempt traces.

## `document_extraction`

`patients.extract_document_records`: one structured record per note
([Structured Clinical Extraction](../api/extraction.md)).

| Key | Default | Meaning |
| --- | --- | --- |
| `max_document_characters` | `200000` | Longer notes fail rather than being truncated |
| `max_attempts` | `3` | Attempts per note for an answer that fails validation, including a quote not found in the note (1-10) |
| `reasoning_off_fallback` | `true` | One more attempt with thinking off. After the last attempt, items without a quote found in the note are dropped and the record is `partial`; any other problem fails the note |
| `structured_outputs` | `true` | Constrain each remote answer to the record's strict JSON schema |
| `batch_size`, `batches_in_flight` | `256`, `2` | See below (`batches_in_flight` 1-16) |
| `extensions` | `[]` | Disease-specific item types and guidance, by name; the package provides `thoracic` |
| `ncit_agent_codes` | `false` | Add each treatment agent's NCIt concept from the bundled snapshot |
| `ncit_resource` | unset (`good_option_catalog.ncit_resource`) | The NCIt snapshot to use |
| `local.engine.max_model_len` | `131072` | Local context |
| `max_tokens` | `50000` | Output budget, reasoning included |

A local backend gets batches of `batch_size`, `batches_in_flight` at once. A remote backend keeps
`remote.max_concurrent_requests` requests in flight per listed server, each sent as another returns, and never fewer
than `batch_size * batches_in_flight`. With a registry (`remote.server_urls_file`) the window is
`batch_size * batches_in_flight` and the registry's ceilings limit each server.

A long completion can take longer than `remote.request_timeout` (600 seconds) on a busy server. Raise it (for example
to 7,200 seconds), since a timed-out request is retried from the start.

## `patient_extraction`

`patients.extract_patient_records`: one record per patient from all of their notes. It takes the
`document_extraction` keys except `max_document_characters`, plus:

| Key | Default | Meaning |
| --- | --- | --- |
| `chunk_tokens` | `40000` | Notes are packed in date order into chunks of at most this many tokens (at least 256) |
| `chunk_overlap_tokens` | `500` | Overlap when a longer note is split at token boundaries (less than half of `chunk_tokens`) |
| `context_tokens` | `null` | Context a chunk's prompt, including the record so far, plus the output budget must fit; `null` uses `local.engine.max_model_len`. A patient whose next prompt does not fit fails rather than being truncated |
| `batch_size` | `64` | As in `document_extraction`; each patient's chunks still run one after another |
| `local.engine.max_model_len` | `200000` | Context |
| `max_tokens` | `60000` | Output budget |

Tokens are counted with `remote.tokenizer_name`, or the model's tokenizer.

## `workup_query_compile`

`patients.compile_workup_queries`: one request per distinct guideline workup requirement (never per patient), whose
answer is a declarative query validated in code ([Extraction-Based Workup Checks](../api/workup-queries.md)).

| Key | Default | Meaning |
| --- | --- | --- |
| `max_attempts` | `3` | Attempts for an answer that is not valid JSON or fails validation, each naming the problems |
| `reasoning_off_fallback` | `true` | One more attempt with thinking off |
| `structured_outputs` | `true` | Constrain remote answers to the query's JSON schema |
| `batch_size`, `batches_in_flight` | `16`, `1` | Request window, as in `document_extraction` |
| `local.engine.max_model_len` | `131072` | Context |
| `max_tokens` | `40000` | Output budget |

The query vocabulary follows the `extensions` of `document_extraction` and `patient_extraction`.

## `workup_sql_agent`

`patients.compile_workup_sql`: for each distinct workup requirement, an LLM agent explores an extraction database with
read-only SQL in an isolated worker and accepts one cohort-level query
([SQL-Agent Workup Checks](../api/workup-sql.md)). It needs an endpoint (`remote.enabled`; a local vLLM server works),
not an in-process engine. Exploration results are patient-derived extracted values (no IDs, note text or quotes) and
reach that endpoint, so use only an endpoint authorized for the records.

| Key | Default | Meaning |
| --- | --- | --- |
| `max_attempts` | `3` | Attempts per turn for an invalid action |
| `structured_outputs` | `true` | JSON-schema responses; `false` gives the schema in the prompt only |
| `tokenizer_mode` | `bytes` | Conservative local prompt budgets; `endpoint` counts with the server's `/tokenize` |
| `context_window` | `null` | `null` uses `local.engine.max_model_len`, then the endpoint's `/v1/models` |
| `request_timeout` | `600` | Seconds a turn's request may go without data before it is retried, instead of `remote.request_timeout`. HTTP 429 and 503 are waited out (from 10 seconds, at most 60) within `max_attempts` |
| `max_concurrent_requests` | unset | Request slots; at least `limits.max_concurrent_items` |
| `reasoning_effort` | `high` | |
| `remote.server_urls`, `remote.provider`, `remote.google_project_id`, `remote.databricks_profile` | unset | A connection for the agent alone, overriding `config.remote` |
| `local.engine.max_model_len` | `131072` | Context |
| `max_tokens` | `32000` | Output budget per turn |

`limits`:

| Key | Default | Meaning |
| --- | --- | --- |
| `max_actions`, `max_submits` | `12`, `3` | Actions per requirement, and submissions among them |
| `submit_by` | `6` | Action by which the agent is asked to submit a first candidate |
| `max_review_queries` | `3` | Checking queries after a valid submission; then code removes the query action, and the agent accepts or revises |
| `max_rows`, `max_cell_chars`, `max_result_chars` | `30`, `160`, `6000` | What the agent sees of a result |
| `max_memory_chars`, `max_recent_steps`, `max_sql_chars` | `4000`, `8`, `16000` | Agent memory, recent steps in the prompt, query length |
| `max_search_terms`, `max_search_rows` | `8`, `50` | Words per search, and matching text values a search shows |
| `max_detail_chars`, `max_evidence_items` | `300`, `8` | Per-patient output |
| `sample_rows_per_status` | `3` | Example rows shown after a submission |
| `max_concurrent_items` | `8` | Requirements explored at once, sharing one SQL worker |
| `query_timeout_seconds`, `final_timeout_seconds` | `20`, `120` | Exploration and final-query timeouts |
| `worker_memory_mb`, `duckdb_memory_mb` | `4096`, `2048` | Worker and DuckDB memory |

## `protocol_cohort_extraction`

`trials.extract_protocol_cohorts`: cohort extraction from full protocol documents (PDF, OCR text or Markdown), and the
merge and boilerplate steps when they run with `section="protocol_cohort_extraction"`.

| Key | Default | Meaning |
| --- | --- | --- |
| `max_document_characters` | `800000` | Longer documents fail rather than being truncated |
| `max_attempts` | `3` | Attempts per request (1-10); a cohort that fails every attempt fails its protocol |
| `chunk_size`, `chunks_in_flight` | `500`, `2` | Request window, as in `trial.cohorts` (`chunks_in_flight` 1-16) |
| `structured_outputs` | `true` | Constrain each remote answer to its step's JSON schema |
| `prompt_files` | unset | Maps `screen`, `list` and `criteria` to other packaged prompts; by default the ctgov prompts |
| `local.engine.max_model_len` | `262144` | Context, so that full industry protocols (often 50,000-160,000 tokens) fit with room for reasoning |
| `max_tokens` | `50000` | Output budget |

## `synthetic_protocol_generation`

`trials.generate_synthetic_protocols`: synthetic protocols seeded by ctgov records, for training data.

| Key | Default | Meaning |
| --- | --- | --- |
| `chunk_size`, `chunks_in_flight` | `500`, `2` | Request window, as in `trial.cohorts` (`chunks_in_flight` 1-16); each output is written when its request returns |
| `max_attempts` | `3` | Attempts per trial for an empty or truncated output (1-10) |
| `layouts_file` | `synthetic_protocol_layouts.json` | Weighted eligibility layouts in `matchminer_ai.data` |
| `local.engine.max_model_len` | `262144` | Context |
| `max_tokens` | `100000` | Output budget |

## `embedding`

TrialSpace embedding (`embedding.embed_for_matching`). Retrieval embeds key text only: patients'
`cancer_history_summary` and cohorts' `cohort_key_criteria`. The keys map onto
`matchminer_ai.embedding.encoder.EncoderSettings`, so the preset encodes exactly as training's mining encodes. See
[Embedding](../api/embedding.md), which also covers other TrialSpace models.

| Key | Default | Absent | Meaning |
| --- | --- | --- | --- |
| `model_path` | `null` | required | TrialSpace directory or Hugging Face model ID; `null` uses the site config's `paths.trialspace_model` |
| `device` | `cuda` | `cpu` | One device |
| `dtype` | `float32` | model's own | Weight precision: `float32` or `bfloat16` |
| `autocast` | `bfloat16` | off | CUDA autocast precision |
| `prompt_profile` | `trialspace` | model's saved prompts | `trialspace` (per-side prompts: `patient_summary` for patients, `this_space` for cohorts) or `sentence_similarity` |
| `default_prompt_name` | `SentenceSimilarity` | model's own | Prompt for calls that name none |
| `include_prompt_in_pooling` | `true` | model's own | Mean pooling over the prompt tokens as well as the text |
| `text_only` | `true` | `false` | Load EmbeddingGemma 2 without its vision and audio towers |
| `batch_size` | `32` | `12` | Texts per batch |
| `prompt_file` | `embedding.txt` | required | `query` prompt for a model that saved none and has no `prompt_profile`, prepended exactly (only its final newline removed) |
| `max_seq_length` | `4096` | required | Truncation length in tokens, also used by QC to flag key text over the limit |

- **`model_path`:** your TrialSpace checkpoint (`google/embeddinggemma-2` fine-tuned by `mmai-train`). The preset
  leaves it unset; then the [site config](site-config.md)'s `paths.trialspace_model` is used, and with neither,
  embedding raises `ValueError`. A path that starts with `/` or `.`, or that exists, must be a readable directory, or
  embedding raises `FileNotFoundError`. Nothing falls back to another model. The checkpoint
  was saved by sentence-transformers 6.1 and needs 6.1 or later. The newest public model, `ksg-dfci/TrialSpace-0526`,
  is a different model family with a different text format.
- **Model-specific keys:** `dtype`, `autocast`, `prompt_profile`, `default_prompt_name`, `include_prompt_in_pooling`,
  `text_only` and `max_seq_length` describe the EmbeddingGemma 2 checkpoint. Review them when changing `model_path`;
  a wrong `max_seq_length` silently truncates at the wrong point.
- **Precision:** autocast applies on CUDA only, so CPU vectors (float32 throughout) are not interchangeable with CUDA
  vectors. The encoder identity stored with vectors records which was used. Never compare vectors made with different
  models or settings.
- **Attention fallback:** on CUDA, if cuDNN cannot build an attention execution plan, encoding is retried with the
  flash, memory-efficient or math attention backends, and that fallback is kept for the same model and device.

## `decision_model`

The decision model answers the match, boilerplate and per-criterion yes/no questions in one forward pass, in-process
with Transformers and the checkpoint's own code (`trust_remote_code`). See
[Match Quality](../api/match-quality.md#decision-model), [Exclusion Criteria](../api/exclusion-criteria.md) and
[Criteria Check](../api/criteria-check.md). Read from `config.raw["decision_model"]`.

| Key | Default | Meaning |
| --- | --- | --- |
| `model_name` | `LiquidAI/d1-2B` | Hugging Face model ID, or a local checkpoint directory from `mmai-train` (`matchminer_ai.training.decision_model`) holding the weights, the d1 code and `decision_contract.json`, whose fingerprint is reported as `model_sha` |
| `revision` | `652f920e544ea37f81e4b0ce89b81d4ed0da90b1` | Commit of a Hub model; required for every Hub model because its code runs, ignored for a local directory |
| `backend` | `transformers` | The only backend |
| `device` | `cuda` | Where the model runs |
| `dtype` | `bfloat16` | Weight type |
| `max_length` | `10000` | The fine-tuning request length in tokens; longer requests still run untruncated and are counted in the metadata's `over_max_length` |
| `set_size` | `null` | Questions read together after one state; `null` uses the model's trained size (16). Larger is faster and departs from training |
| `token_budget` | `65536` | Padded tokens per forward pass |
| `batch_size` | `256` | Requests per `system_one_batch` call |
| `compile` | `false` | `torch.compile` the model before use |

## Decision-model cutoffs

Each cutoff turns a decision-model probability into a pass flag for screening. The probabilities and scores rank pairs
for human review; they are not eligibility decisions.

| Key | Default | Meaning |
| --- | --- | --- |
| `match_quality.reasonableness_cutoff` | `0.5` | `match_quality_pass` when P(reasonable) (`match_quality_reasonableness`) is at least this; `score_match_quality(filter_low_quality=True)` keeps only passing pairs. `match_quality_score` ranks pairs and is not a pass criterion. A `score_cutoff` key from older configs is ignored |
| `exclusion_criteria.pass_cutoff` | `0.5` | `exclusion_criteria_pass` when P(yes) of the boilerplate question (yes means potentially eligible) is at least this |
| `criteria_check.fail_cutoff` | `0.5` | A criteria line whose P(yes) (yes means the patient does not clearly fail the line) is below this is flagged `criterion_clearly_fails` |

## `llm_criteria_check`

`matching.check_eligibility_criteria_with_llm`: the LLM version of the per-criterion check, one request per criteria
line with the training teacher's per-criterion prompt ([LLM check](../api/criteria-check.md#llm-check)). Read from
`config.raw["llm_criteria_check"]`.

| Key | Default | Meaning |
| --- | --- | --- |
| `structured_outputs` | `true` | Constrain each remote answer to the criterion label schema |
| `max_attempts` | `3` | Attempts for a line whose answer fails validation, each retry naming the error |
| `chunk_size`, `chunks_in_flight` | `64`, `1` | Request window, as in `trial.cohorts` |
| `local.engine.max_model_len` | `100000` | Context |
| `max_tokens` | `32768` | Output budget, reasoning included (the training teacher's) |

## `llm_match_quality` and `llm_exclusion_criteria`

`matching.score_match_quality_with_llm` and `matching.exclusion_criteria_check_with_llm`: the LLM versions of the
match-quality and boilerplate checks. Both use the training teacher's prompts and output budget.

| Key | Default | Meaning |
| --- | --- | --- |
| `llm_match_quality.prompt_file` | `match_quality.labels.system.txt` | System prompt with the teacher's five match labels; a replacement must return the same JSON contract |
| `llm_exclusion_criteria.prompt_file` | `llm_exclusion_criteria.user.txt` | Asks whether the patient clearly fails any of the cohort's boilerplate criteria |
| `local.engine.max_model_len` | `100000` | Context |
| `max_tokens` | `32768` | Output budget, reasoning included |

## `llm_good_option`

The LLM behind GoodOption: drug screening, classification and evidence synthesis during the patient-free catalog
build, and patient scoring (`score_good_options_with_llm`, `check_good_options`), where each request describes one
patient. See [Good Option Research and Scoring](../api/good-options.md).

| Key | Default | Meaning |
| --- | --- | --- |
| `local.engine.max_model_len` | `262144` | Context |
| `max_tokens` | `100000` | Output budget |
| sampling fields | temperature 0, `top_k: 1`, `top_p: 1.0`, `min_p: 0`, `repetition_penalty: 1.1` | Fixed greedy sampling (no `sampling_profile`), replaced by Nemotron's profile while the model is Nemotron ([above](#model-aware-sampling-and-reasoning)) |

`score_good_options_with_llm` retries only answers that fail validation (`max_parse_attempts`), each retry carrying the
invalid answer, the parser error and the finish reason; `reasoning_off_fallback=True` adds one final attempt with
thinking off. A catalog checkpoint is fingerprinted with this section, so changing it requires a new checkpoint
directory.

## `good_option_catalog`

The patient-free catalog build: trial drug screening, NCIt classes, evidence research and synthesis. The stages are
explained in [How the Good Options pipeline works](../user-guide/good-options-pipeline.md).

| Key | Default | Meaning |
| --- | --- | --- |
| `ncit_resource`, `ncit_version` | `Thesaurus_26.07d.FLAT.zip`, `26.07d` | Bundled NCIt snapshot (CC BY 4.0) the drug classes come from, and the version recorded with them |
| `screening_max_attempts`, `class_assignment_max_attempts`, `synthesis_max_attempts` | `3` each | Attempts for answers that fail validation |
| `structured_outputs` | `true` | Decode remote screening and class answers under their JSON schemas; `false` for a server without that support |
| `screening_checkpoint_batch_size`, `class_checkpoint_batch_size` | `64` each | Trials screened and drugs classified at once (below) |
| `class_max_per_drug` | `3` | Classes the LLM may add beyond the drug's NCIt classes, or name for a drug NCIt does not classify |
| `class_candidate_limit` | `25` | NCIt candidate classes, found by local search of the drug's text, that the LLM chooses from |
| `class_assignment_evidence_max_tokens` | `40000` | Mechanism passages the class step reads and searches |
| `synthesis_chunk_max_tokens` | `60000` | Passage text per synthesis call for a drug |
| `class_evidence_max_tokens` | `60000` | Passage text per synthesis call for a class |
| `good_option_summary_max_tokens`, `help_me_choose_summary_max_tokens`, `class_option_summary_max_tokens` | `6000`, `9000`, `4000` | Bounds of the stored text projections (`help_me_choose_summary` is a historical column name) |
| `synthesis_evidence_max_tokens` | `190000` | Not read by the build; kept for tooling that reports a whole-ledger budget |
| `screening_llm`, `class_llm`, `synthesis_llm` | `max_tokens` 50,000, 50,000, 100,000 | Overrides merged over `llm_good_option` for that stage |
| `host_requests_per_second` | `{}` | Per-host research pacing overrides (below) |
| `ncbi_api_key_file` | `null` | File holding an NCBI API key for PubMed (below) |

- **Batching:** with a remote backend, each trial or drug is its own request, validated and checkpointed as soon as it
  finishes, with up to the batch size in flight; the endpoint pool still caps each server. A local engine gets groups
  of these sizes. The build command's `--llm-batch-size` overrides both for one run.
- **Synthesis:** a drug's or class's evidence is split into as many chunks as it takes; each call sees the facts so far
  and returns only new or revised items, so nothing is discarded unread. A chunk, its passage metadata, the existing
  facts (up to about 25,000 tokens for the broadest classes) and the stage's `max_tokens` must fit in the served
  context: lower both chunk limits when serving less than 262,144 tokens. Larger chunks yield fewer facts, because the
  model condenses more of each. An answer cut off by the output budget is retried, and the final attempt disables
  thinking.
- **Research pacing:** requests are paced per host in one budget shared by every subject and stage, retries and
  redirects included. Defaults: `clinicaltrials.gov` 0.6 per second, `dailymed.nlm.nih.gov` 1.0,
  `eutils.ncbi.nlm.nih.gov` 2.0 (7.0 with an NCBI API key), about 30% under each service's per-IP limit. Other hosts
  are unpaced. `host_requests_per_second` overrides a host's rate (`0` leaves it unpaced), and the command line's
  `--host-requests-per-second HOST=RATE` overrides that. Limits are per IP, so two builds on one machine should split
  them.
- **NCBI API key:** `--ncbi-api-key-file` overrides `ncbi_api_key_file`, and either overrides the `NCBI_API_KEY`
  environment variable. The key is sent only in request bodies, never in a URL, log, error, checkpoint or manifest.
- **Checkpoints:** a catalog checkpoint is fingerprinted with `llm_good_option`, this section and the remote provider.
  A checkpoint directory with a different fingerprint is refused. `host_requests_per_second` and `ncbi_api_key_file`
  are left out of the fingerprint, as is `--llm-batch-size`, so a build resumes after they change.

Retrieval limits (sources, facets, record caps, indication terms) are `ResearchSettings` fields, not configuration
keys.

## `good_option_prompt`

How catalog evidence is packed into the patient-time `llm_good_option` prompt. It is kept apart from the fingerprinted
catalog sections. Read from `config.raw["good_option_prompt"]`. See
[Evidence packing](../api/good-options.md#evidence-packing).

| Key | Default | Meaning |
| --- | --- | --- |
| `context_tokens` | `null` | The model's context; `null` uses `llm_good_option.local.engine.max_model_len` (262,144). Set it when an endpoint serves another length |
| `chars_per_token` | `3.5` | Conversion rate for the budget; conservative for Nemotron 3.5 Super and Gemma 4, which average about 4.4-4.6 characters per token on catalog and criteria text |
| `safety_tokens` | `4096` | Margin beyond the output budget (`llm_good_option` `max_tokens`) |
| `max_drug_section_tokens`, `max_class_section_tokens` | `20000` each | Upper bound on any one drug or class; `null` removes it |

## `good_option_check`

`check_good_options`, the on-demand patient-trial check. Applied to a copy of the configuration at call time and kept
outside `llm_good_option`, so catalog fingerprints are unaffected. Read from `config.raw["good_option_check"]`.

| Key | Default | Meaning |
| --- | --- | --- |
| `max_output_tokens` | `50000` | Output budget for the scoring request in place of `llm_good_option`'s 100,000, so more of the context carries evidence; `null` keeps that value |
| `discover_context_tokens` | `true` | When `good_option_prompt.context_tokens` is unset and an OpenAI-compatible remote backend is enabled, size evidence to the smallest `max_model_len` the servers report at `/v1/models` |
| `discovery_timeout_seconds` | `10.0` | Timeout of each lookup; a failed lookup falls back to the configured context and is reported in the metadata |
| `max_parse_attempts` | `3` | Attempts in all for answers that fail validation |
| `reasoning_off_fallback` | `true` | One final attempt with thinking off |

## `good_option_checker`

Deprecated. Settings of the retired GoodOptionChecker classifier (`model_name: ""`, `device: cuda`,
`max_length: 8192`), read only by the deprecated classifier APIs, which emit `DeprecationWarning`. No checker
artifact is published; GoodOption scoring uses `llm_good_option`.

## `guideline`

Source-grounded NCCN guideline catalogs from a locally supplied library. Remote only, with an explicit output
directory. The keys (`packet_pages`, `context_window`, `safety_tokens`, `tokenizer_mode`, `response_format`, `stream`,
`api_key_env`, and `remote.model_name`, `null` to use the model the endpoint reports) are described in
[Configure the endpoint](../user-guide/guideline-extraction.md#configure-the-endpoint).
