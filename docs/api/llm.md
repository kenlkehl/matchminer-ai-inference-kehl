# LLM server helper

Helpers for running the LLM on local vLLM servers and for pointing a configuration at whatever model a server
serves. [Choosing an inference setup](../user-guide/local-vs-remote.md) explains when to use a server.

## Starting vLLM servers

`start_vllm_servers()` starts one `vllm serve` process for each URL in `config.remote["server_urls"]`. With the
default configuration that is one server, at `http://localhost:8000/v1`.

```python
from matchminer_ai import load_preset
from matchminer_ai.llm.vllm_server import start_vllm_servers

config = load_preset("default")
servers = start_vllm_servers(config=config)
config.remote["enabled"] = True
...
for server in servers:
    server.terminate()
```

- **Command:** the `task` section (`"patient"` by default, or `"trial"`, `"llm_match_quality"`,
  `"llm_exclusion_criteria"`) supplies the model (`local.model_name`), served under its `remote.model_name`. Its
  `local.engine` supplies the context length, tensor parallel size, memory fraction and the Nemotron engine settings
  (remote code, text only, prefix caching, Mamba cache), passed as `vllm serve` flags. The model's reasoning and
  tool-call parsers are added, and `local.chat_template_kwargs` become the server's default chat-template kwargs.
- **Other flags:** pass any other `vllm serve` flags with `extra_args`.
- **Readiness:** each call waits until the server answers `/v1/models` (`ready_timeout`, 600 seconds) and returns
  the `subprocess.Popen` handles.
- **GPUs:** each server inherits the calling process's environment, including `CUDA_VISIBLE_DEVICES`.

`build_vllm_server_commands()` returns the commands without starting them, `start_vllm_server()` starts the one for
a single `server_index`, and `check_openai_endpoint()` reports whether endpoints respond.

::: matchminer_ai.llm.vllm_server
    options:
      members:
        - start_vllm_servers
        - start_vllm_server
        - build_vllm_server_command
        - build_vllm_server_commands
        - wait_for_vllm_server
        - check_openai_endpoint

## Model sampling profiles

`configure_served_model()` enables remote inference, asks each server which model it serves, and applies that
model's registered profile to the named config sections:

```python
from matchminer_ai.llm.model_profiles import configure_served_model

model_name, profile = configure_served_model(
    config,
    ["http://localhost:8000/v1"],
    sections=["trial", "patient", "protocol_cohort_extraction"],
)
```

- **Model:** every server must report the same model. Each section's `remote.model_name` becomes that model, and its
  `remote.tokenizer_name` is removed, so the tokenizer is then loaded by the served name. Set it again if the server
  reports an alias.
- **Sampling:** a profile replaces the section's sampling parameters and chat-template kwargs with the publisher's
  recommendations, and sets a top-level section's reasoning and tool-call parsers. It may raise, never lower,
  `max_tokens` when the reasoning trace shares the completion budget. Output budgets above the floor stay with each
  task.
- **Unregistered models** keep the preset's sampling, with a warning.
- **Stage overrides:** dotted section names, such as `good_option_catalog.screening_llm`, are accepted.

Profiles match the model name case-insensitively, by substring, so quantized re-uploads get the base model's
profile. The first match wins, in this order:

| Profile | Matches | Sampling | Chat template | `max_tokens` floor |
| --- | --- | --- | --- | --- |
| `reflection-beam` | Reflection Beam 1.0 builds and the aliases `beam` and `reflection-beam` | temperature 0.7, top_p 0.99, top_k -1, min_p 0, presence 0, repetition 1.0 | `reasoning_effort: medium` | none |
| `nemotron-3.5-super-vl` (default model) | `nemotron-3.5-super-vl` | temperature 1.0, top_p 0.95, top_k -1, min_p 0, presence 0, repetition 1.0 | `enable_thinking: true` | none |
| `qwen3.8-flash-next-thinking` | `qwen3.8-flash-next` | temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0 | `enable_thinking: true` | 32,768 |
| `gemma-4-thinking` | `gemma-4`, `gemma4` | temperature 1.0, top_p 0.95, top_k 64, min_p 0, repetition 1.0 | `enable_thinking: true` | none |

The Nemotron profile also replaces, at request time, the fixed sampling of sections that set no `sampling_profile`.
The per-section `sampling_profile: auto` rules, which also cover Qwen 3.5, are described in
[LLM runtime](../reference/llm-runtime.md#model-profiles).

::: matchminer_ai.llm.model_profiles
    options:
      members:
        - configure_served_model
        - apply_model_profile
        - resolve_model_profile
        - discover_served_model
        - discover_served_context_tokens
