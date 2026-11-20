# LLM runtime

How the package talks to LLMs: the sampling profiles it applies per model, the pool that spreads requests over
remote servers, optional reasoning archives, request pacing, cancellation, and a terminal tool for checking one
patient-cohort pair. The section-by-section configuration keys are in [Configuration](configuration.md).

## Model profiles

LLM sections with `sampling_profile: auto` recognize the model by name and apply its publisher's sampling settings.
Explicit sampling parameters in a request or section still override them, and `apply_model_profile` replaces a
previous model's sampling settings rather than merging them. A served alias the package cannot recognize can name a
profile explicitly.

### NVIDIA Nemotron 3.5 Super VL (default)

`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` is the default model of every LLM section of the default
preset (trial, patient, criteria extraction, protocol cohorts, synthetic protocols, and the LLM match, exclusion and
GoodOption checks), locally and remotely. Gemma 4 and the other profiles remain available by setting a section's
`model_name`.

- **Serving:** each local engine serves it in-process the way the production vLLM servers do: `trust_remote_code`,
  `language_model_only`, prefix caching, the FlashInfer Mamba backend, a float16 SSM cache with stochastic rounding
  (5 Philox rounds) and `mamba_cache_mode: align`. `start_vllm_servers()` passes the same settings as `vllm serve`
  flags, with `--reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`
  (`reasoning_parser` and `tool_call_parser` override them; `none` disables one).
- **Hardware:** the NVFP4 build needs a GPU with native FP4 (Blackwell). Another build, such as FP8, is selected by
  changing `model_name`. NVIDIA recommends vLLM 0.32.0 or its documented nightly.
- **Recognized names:** the NVFP4 and BF16 checkpoints and the served alias `nemotron-3.5-super-vl`
  (`sampling_profile: nemotron-3.5-super-vl` for other aliases).
- **Sampling:** temperature 1.0 and top-p 0.95 for text, from NVIDIA's
  [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) and generation config
  (checked 2026-10-15). The profile neutralizes inherited filters and penalties (`top_k=-1`, `min_p=0`,
  `presence_penalty=0`, `repetition_penalty=1`). These are text settings; the card recommends 0.6 for vision, and
  the profile adds no image or video input.
- **Reasoning:** controlled by `chat_template_kwargs.enable_thinking`; medium effort is
  `enable_thinking=true, medium_effort=true`. Reasoning and the answer share the output-token budget, so sections
  with long tasks leave room for 10-15k reasoning tokens: the LLM match and exclusion checks reserve the training
  teacher's 32,768 tokens.

### Qwen 3.5

Qwen 3.5 names, including `Qwen/Qwen3.5-4B`, get Alibaba's
[general-task sampling defaults](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices): temperature 0.7 and top-p
0.8 with `chat_template_kwargs.enable_thinking=false`, 1.0 and 0.95 with thinking on, and in both modes top-k 20,
min-p 0, presence penalty 1.5 and repetition penalty 1.0. Qwen 3.5 has a boolean thinking switch, not Qwen 3.8's
graded effort, so no effort is inferred. Explicit profile name: `qwen3.5`.

### Reflection Beam 1.0

Beam 1.0 is recognized in its BF16, FP8 and NVFP4 builds, quantized re-uploads, and the served names
`Beam-1.0-501B-A23B`, `beam` and `reflection-beam` (explicit profile: `reflection-beam` or `beam`).

- **Sampling:** temperature 0.7 and top-p 0.99 from its
  [published vLLM defaults](https://developers.reflection.ai/serving-vllm), and repetition penalty 1.0 from the
  checkpoint (checked 2026-10-16). Top-k, min-p and presence penalty are disabled so another model's settings are not
  inherited.
- **Serving:** both vLLM parsers are `beam`, which needs a vLLM build with Beam support. Local raw completions are
  split on Beam's own analysis/final delimiters, also on clients whose vLLM lacks the parser; unframed or tool-call
  output is never taken as a final answer.
- **Reasoning:** Beam always reasons. `chat_template_kwargs.reasoning_effort` defaults to `medium` instead of the
  upstream template's `max`; template or API overrides still apply.

## Remote endpoint pool

With `remote.enabled`, OpenAI-compatible requests go through one shared, adaptive endpoint pool.

- **Concurrency:** `remote.max_concurrent_requests` is a per-server ceiling. The `remote.adaptive_concurrency`
  settings start below it, admit more requests after successful saturated windows, and back off on transport or
  capacity failures; model-output validation failures do not reduce concurrency. `enabled: false` gives fixed
  limits. `metrics_interval` (seconds, default 0) adds vLLM queue, KV-cache and preemption feedback, falling back to
  transport outcomes when metrics are unavailable. `shared_pool(...).snapshot()` in
  `matchminer_ai.llm.endpoint_pool` reports limits and counters.
- **Endpoint choice:** a ready request goes to the available endpoint with the lowest fraction of its current limit
  in use, ties broken by least recent use. Idle endpoints reserve no queue positions, so a fast server refills while
  slower ones are busy. The training labeler uses the same rule.
- **Changing fleets:** `remote.server_urls_file` names a JSON registry that is replaced atomically and re-read during
  a run: `{"servers": [{"url": "http://host:8000/v1", "model": "served-alias", "model_root": "verified-checkpoint",
  "upstream_concurrency": 768}]}`. Whoever adds an endpoint must first verify the model it serves. `pool_max_workers`
  bounds the client workers. Cohort attempt metadata records the endpoint, the served alias and the registry's model
  root, so different quantizations stay traceable.
- **Scope:** concurrency is per process; separate processes have separate limits. Callable credentials and providers
  keep their own asynchronous transport.
- **Background work:** `llm.async_processing` provides bounded background work and batched, untruncated prompt
  tokenization; teacher labeling prepares prompts with four workers in batches of up to 32, with a separate
  checkpoint I/O pool. Remote transports do not reuse idle connections, and vLLM metrics polling probes up to 16
  endpoints at once so one failed server does not stall the others' feedback.

## Reasoning trace archives

Pooled requests can keep complete local archives with `remote.trace_dir` or `MMAI_LLM_TRACE_DIR`. Each response is
saved atomically before validation, token-limited and malformed attempts included: the exact messages, the reasoning
channel, the complete response, model and endpoint provenance, sampling settings and finish reason. Credentials and
HTTP headers are left out. Response metadata includes `trace_path`, and `MMAI_LLM_TRACE_STAGE` names the workflow.

Archives are opt-in because they contain the input text: write them only to a directory authorized for that data.
They do not add reasoning to normal answers or change any training target.

## Request pacing

Structured note and workup requests can space their starts without reducing worker concurrency:
`request_start_interval_seconds`, `capacity_retry_initial_seconds` and `capacity_retry_max_seconds` on
`NoteSearchLLMConfig` or the shared remote runtime (pacing is off by default).

- Clients share dispatch spacing and capacity cooldowns per endpoint and model within one process.
- HTTP 429 and 503 retries use randomized exponential backoff, and numeric or HTTP-date `Retry-After` delays are
  honored up to the ceiling.
- Capacity retries resend the original request and count toward the existing attempt and call limits.
- No patient text, credentials or provider error bodies enter pacing state. Accepted-request metadata records the
  dispatch wait.

## Cancellation

Applications can stop patient workflows cooperatively with `CancellationToken` and `cancellation_scope` from
`matchminer_ai.cancellation`: run the workflow inside the scope and call `token.cancel()` from the stop handler.

- **What stops:** worker thread pools inherit the scope. Capacity waits, retry delays and Python cells check it.
  Cancellation raises `InferenceCancelled`, a control-flow `BaseException`, rather than returning a finding or
  retrying as a model failure.
- **Cleanup:** isolated REPL processes close. Remote summarization cancels its pending tasks and closes its clients,
  while local model operations finish the current step.
- **What keeps running:** a blocking HTTP call stops being awaited, but its thread holds the concurrency slot until
  the response closes or times out. A request the server already accepted may finish there; its late answer is
  discarded.

Without a scope, behavior is unchanged.

## Terminal LLM trial check

`matchminer-ai-llm-trial-check` (or `python llm_trial_check.py` in a checkout) asks a running OpenAI-compatible
endpoint about one pair:

```shell
matchminer-ai-llm-trial-check http://localhost:8000/v1
```

It finds the endpoint's model through `/v1/models`, then reads a multiline patient summary (its key section) and the
cohort's key criteria; end each paste with a line containing only `.done`. It prints the complete reasoning, the five
match-label answers and the 0-6 score computed from them. Set `OPENAI_API_KEY` if the server needs a key. The summary
is sent to that endpoint, so use only an endpoint authorized for the data.
