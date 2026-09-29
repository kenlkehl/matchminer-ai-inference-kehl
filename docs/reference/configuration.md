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
`base_url`. For the default Gemma 4 configuration, the server should be a vLLM
chat endpoint launched with the `gemma4` reasoning parser; the package
`start_vllm_servers()` helper adds this flag from the selected LLM task's
`reasoning_parser`.

Model names and request parameters are configured per LLM task under that
task's backend block. For example, `trial.local.model_name` is the model loaded
by local vLLM, `trial.remote.model_name` is the model string sent to the
endpoint, `trial.remote.request_params` contains top-level chat completion
request fields, and `trial.remote.extra_body` contains fields sent through
request `extra_body`.

Remote mode expects the endpoint to return final answer text in
`message.content`. The default vLLM/Gemma setup can expose reasoning separately
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
| Gemma 4 | 1.0 | 0.95 | 64 | Engine defaults for penalties |
| Qwen 3.8, thinking | 1.0 | 0.95 | 20 | min-p 0, presence penalty 0, repetition penalty 1 |
| Qwen 3.8, non-thinking | 0.7 | 0.8 | 20 | min-p 0, presence penalty 1.5, repetition penalty 1 |

These follow the vendor recommendations checked on 2026-09-24:
[Google Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4),
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), and
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

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
`gemma4` and `qwen3.8` also allow explicit family selection for endpoint aliases.
Unknown model families receive no automatic vendor settings. Existing custom
configs without `sampling_profile` keep their explicit behavior.

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

## `trial`

Task configuration for trial summarization.

### `trial.local`

Local in-process vLLM runtime settings:

- `model_name`: model identifier loaded by local vLLM and used for local
  tokenizer/chat-template rendering and model metadata lookup.
- `engine`: keyword arguments passed to `vllm.LLM(...)`; the package passes
  `model=trial.local.model_name` separately.
- `generation`: keyword arguments passed to `vllm.SamplingParams(...)`.
- `chat_template_kwargs`: keyword arguments passed to tokenizer chat-template
  rendering.

Additional `engine` and `generation` keys may be included if they are valid vLLM
keyword arguments. vLLM validates those keys when the engine/request is created.

### `trial.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `trial.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Set this explicitly when using a
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

### `patient.prompt_files`

Prompt template filenames loaded from `matchminer_ai.prompts`.

### `patient.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Set this explicitly when using a
model not covered by the package mapping, or use `none` to disable reasoning
parsing for a non-reasoning model. Non-vLLM remote endpoints may return no
separate reasoning field.

### `patient.remote`

Task-specific remote chat completion request settings. See `trial.remote`.
Patient summarization also supports `tokenizer_name`, which is the local
tokenizer used for patient chunk truncation and prompt sizing before sending
requests to the remote endpoint. For self-hosted vLLM this is usually the same
as `patient.remote.model_name`.

### `patient.boilerplate_marker`

Line marker used by patient postprocessing to identify the boilerplate
conditions section heading.

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

## `patient_structuring`

Configuration for ontology-grounded JSON conversion of an existing patient
summary. The `local`, `remote`, and `reasoning_parser` fields follow the other
LLM task sections.

- `oncotree_resource` and `ncit_resource` select bundled ontology snapshots or
  explicit local file paths.
- `oncotree_max_depth` bounds hierarchical OncoTree descent.
- `ontology_retry_limit` bounds malformed JSON and invalid-index retries.
- `ncit_candidate_limit` bounds each locally searched candidate page.
- `ncit_max_agent_steps` bounds revised NCIt search attempts.
- `local.generation.max_tokens` and `remote.request_params.max_tokens` default
  to 30,000 completion tokens so reasoning-capable endpoints retain enough
  budget to emit the required JSON answer after their reasoning trace.

Complete ontology contents remain local. Only immediate OncoTree children,
bounded NCIt candidate labels, and selected NCIt definitions are sent to the
configured LLM backend.

`structure_patient_summaries` batches every dependency-ready prompt wave. In
remote mode, the root-level `remote.max_concurrent_requests`, `batch_size`, and
`server_urls` settings control request concurrency, chunking, and server
distribution. `batch_size` is a scheduler chunk size, not one multi-prompt HTTP
payload; an OpenAI-compatible chat-completion request is made for each prompt.

## `trial_space_structuring`

Configuration for ontology-grounded JSON conversion of an existing clinical
trial-space summary. The `local`, `remote`, and `reasoning_parser` fields follow
the other LLM task sections. Its ontology fields and limits have the same
meaning as `patient_structuring`:

- `oncotree_resource` and `ncit_resource` select bundled ontology snapshots or
  explicit local file paths.
- `oncotree_max_depth` bounds hierarchical OncoTree descent.
- `ontology_retry_limit` bounds malformed JSON and invalid-index retries.
- `ncit_candidate_limit` bounds each locally searched candidate page.
- `ncit_max_agent_steps` bounds revised NCIt search attempts.

The extraction prompt is trial-specific, but it reuses the bounded ontology
selection harness. Complete ontology contents remain local.

## `embedding`

Configuration for summary embedding.

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

Prompt filename loaded from `matchminer_ai.prompts` and used as the embedding
query prompt.

### `embedding.max_seq_length`

Runtime truncation cutoff for embedding inputs. `SentenceTransformer`
uses this value during `encode()`, so inputs longer than this limit are
truncated before embedding generation. QC reports use the same value when
flagging summaries that exceed the embedding input limit.

## `match_quality`

Configuration for the match-quality checker model.

### `match_quality.model_name`

Text-classification model identifier used by the checker pipeline and model
metadata lookup.

### `match_quality.device`

Device passed to the checker pipeline.

### `match_quality.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`.

### `match_quality.max_length`

Maximum token length passed to the text-classification checker pipeline.

### `match_quality.score_cutoff`

Minimum sigmoid-transformed checker score required for
`match_quality_pass == true`.

## `exclusion_criteria`

Configuration for the exclusion-criteria checker model.

### `exclusion_criteria.model_name`

Text-classification model identifier used by the checker pipeline and model
metadata lookup.

### `exclusion_criteria.device`

Device passed to the checker pipeline.

### `exclusion_criteria.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`.

### `exclusion_criteria.max_length`

Maximum token length passed to the text-classification checker pipeline.

## `llm_match_quality`

Configuration for the LLM-based match-quality checker.

### `llm_match_quality.local`

Local in-process vLLM runtime settings. See `trial.local`.

### `llm_match_quality.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`.

### `llm_match_quality.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Non-vLLM remote endpoints may
return no separate reasoning field.

### `llm_match_quality.remote`

Task-specific remote chat completion request settings. See `trial.remote`.

## `llm_exclusion_criteria`

Configuration for the LLM-based exclusion-criteria checker.

### `llm_exclusion_criteria.local`

Local in-process vLLM runtime settings. See `trial.local`.

### `llm_exclusion_criteria.prompt_file`

Prompt template filename loaded from `matchminer_ai.prompts`.

### `llm_exclusion_criteria.reasoning_parser`

vLLM reasoning parser name. The default `auto` resolves known model names,
including `google/gemma-4-31B-it` to `gemma4`. Non-vLLM remote endpoints may
return no separate reasoning field.

### `llm_exclusion_criteria.remote`

Task-specific remote chat completion request settings. See `trial.remote`.

## `help_me_choose`

LLM task configuration for the optional matched-trial comparison. Its
`local` and `remote` blocks follow the same structure as `trial.local` and
`trial.remote`. Drug-information retrieval is intentionally separate from this
LLM block and never accepts patient text.

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
`class_evidence_max_tokens`) and carries a running set of facts across them.
The stage-specific `synthesis_llm` override defaults to a 100,000-token
completion budget so reasoning-enabled models can think over a chunk, rewrite
the carried facts, and still finish the final JSON; a chunk, the carried facts,
and this budget must fit together in the served context. A model that exhausts
the budget mid-trace returns an empty answer with `finish_reason=length`, which
is retried rather than silently accepted, and if every ordinary attempt ends
that way the final attempt disables thinking.
`screening_max_attempts` and `synthesis_max_attempts` control structured-output retries.
`screening_checkpoint_batch_size` and `class_checkpoint_batch_size` bound the
screening and classify LLM batches so successfully returned batches can be
checkpointed throughout long catalog builds. Synthesis is not batched: each
drug or class is its own chain of chunk calls, checkpointed as soon as that
chain finishes, with up to `remote.max_concurrent_requests` chains in flight. Screening and synthesis LLM overrides inherit from
`llm_good_option`.

Evidence is also retrieved on two axes beside the drug's own name, and each has
its own budget so that neither can evict the other.

`class_max_per_drug` (default 3) caps how many pharmacologic classes the
`classify` stage may assign to one drug; a drug legitimately holds more than one,
since atezolizumab is both a PD-L1 inhibitor and an immune checkpoint inhibitor.
`class_assignment_evidence_max_tokens` (40,000) bounds the mechanism passages
that stage reads, and `class_llm` gives it its own completion budget.
`class_evidence_max_tokens` (60,000) bounds the corpus that class *synthesis*
reads. Class synthesis is a separate call from agent synthesis rather than a
larger pooled one: 165 drugs already use more than half the agent budget and the
largest uses 91% of it, so pooling would let a flood of same-class passages evict
the agent's own data. `class_option_summary_max_tokens` (1,600) bounds each
rendered class block in the scoring prompt, where one block is emitted per
distinct class across the trial's drugs.

Retrieval settings for both axes live on `ResearchSettings` rather than in this
block: `class_facets`, `class_sources`, `class_research_rounds`, and the
`max_class_*` record caps for the class axis, and `indication_terms_per_drug`
(default 4), `indication_facets`, `indication_sources`, and the
`max_indication_*` caps for the indication axis. Indication terms come from the
`conditionsModule.conditions` of the trials a drug appears in, falling back to
`derivedSection.conditionBrowseModule.meshes`, so the catalog stays patient-free
and cacheable. Both axes call only the query-driven sources; the registry, label,
and curation adapters search by exact agent name and return nothing for a class
name.

## `good_option_checker`

Configuration for the optional local GoodOptionChecker regression model. Its
input is the patient summary followed by one clean synthesized investigational-
drug summary; URLs, registry metadata, and raw evidence are excluded.
`model_name` is empty by default until a versioned trained artifact is
configured. `device` and `max_length` are passed to the text-classification
checker pipeline.

## `trial_space_contextualization`

Configuration for trial-only public-source retrieval and grounded synthesis.
Its `local` and `remote` blocks select the synthesis LLM. Additional fields
include:

- `sources`: default source adapter names (`nci_pdq`, `fda`, `civic`,
  `pubmed`, and `europe_pmc_open_guidelines`);
- `request_timeout`: public-source HTTP timeout in seconds;
- `max_concurrency`: maximum simultaneous source requests;
- `max_evidence_per_source`: per-space adapter result limit;
- `pubmed_retmax`: candidate count for each diagnostic, molecular-testing, and
  treatment-guidance PubMed query;
- `europe_pmc_languages`: allowed Europe PMC article language codes (English by
  default);
- `evidence_context_max_tokens`: model-token budget for raw evidence excerpts
  (the implementation enforces a minimum of 10,000; default 12,000);
- `diagnostic_context_min_tokens`: evidence budget reserved for diagnostic
  passages before therapeutic/general evidence is packed; and
- `evidence_item_max_tokens`: maximum contribution from one evidence record.

The packer uses `tokenizer_name` from the active local or remote task block. If
that tokenizer cannot be loaded, it records a warning and uses a lexical-token
fallback rather than reverting to character counts. Context rows report packed
token counts, dropped/truncated records, diagnostic coverage facets, and a
deterministic diagnostic-evidence sufficiency signal.

The Europe PMC adapter fetches structured full text only when core metadata
reports an allowlisted CC BY or CC0 license. It then applies disease relevance,
guideline/consensus, and diagnostic-section checks. Other open-access licenses
and records without explicit license metadata are rejected.

Optional source credentials and identification are read from environment
variables rather than configuration snapshots:

- `CIVIC_API_KEY`;
- `NCBI_EMAIL` and `NCBI_API_KEY`.

The NCBI API works without an API key at its lower public rate limit. Supplying
`NCBI_EMAIL` is recommended so requests identify the application operator.

## `patient_contextualization`

LLM task configuration for the separate per-patient, per-space review. This
stage does not call public-source adapters. Its `local` and `remote` blocks
follow the same structure as `trial.local` and `trial.remote`.
