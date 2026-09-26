# LLM Server Helper

Use `start_vllm_servers()` to start one local vLLM server for each URL in
`config.remote.server_urls`. With the default configuration, this starts one
server.

::: matchminer_ai.llm.vllm_server
    options:
      members:
        - start_vllm_servers

## Model sampling profiles

`configure_served_model()` enables remote inference, asks each server which
model it serves, and applies that model's registered profile to the named
config sections. A profile replaces sampling parameters and chat-template
kwargs with the publisher's recommendations and may raise, never lower,
`max_tokens` when the reasoning trace shares the completion budget. Output
budgets above the floor stay with each task. Unregistered models keep the
preset's sampling and log a warning.

| Model match | Mode | Sampling | Floor |
| --- | --- | --- | --- |
| `*qwen3.8-flash-next*` | thinking (`enable_thinking: true`) | temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0, repetition 1.0 | 32,768 |

::: matchminer_ai.llm.model_profiles
    options:
      members:
        - configure_served_model
        - apply_model_profile
        - resolve_model_profile
        - discover_served_model
