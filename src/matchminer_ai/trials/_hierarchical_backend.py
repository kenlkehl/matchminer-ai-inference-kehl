"""Batched backend adapter for the hierarchical extraction contract."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path

from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.llm.prompt_rendering import (
    _get_chat_template_tokenizer,
    build_prompt_list,
)

from .hierarchy import (
    VERSION,
    FIELDS,
    Completion,
    Hierarchy,
    ReadyHierarchy,
    ResponseCache,
    load_prompts,
    parse_response,
    retry_messages,
    run_ready_hierarchies,
)

logger = logging.getLogger(__name__)


def _bounded_prompts(messages, runtime):
    """Reserve output within the configured context without truncating evidence."""
    prompts = build_prompt_list(messages, llm_config=runtime)
    if not runtime.get("max_model_len"):
        return prompts, runtime
    tokenizer = _get_chat_template_tokenizer(
        str(runtime.get("tokenizer_name") or runtime["model_name"])
    )
    template = dict(runtime.get("chat_template_kwargs", {}))
    if runtime.get("backend_mode") == "remote":
        template.update(
            runtime.get("remote", {})
            .get("extra_body", {})
            .get("chat_template_kwargs", {})
        )
    for prompt in prompts:
        rendered = (
            tokenizer.apply_chat_template(
                conversation=prompt.messages,
                add_generation_prompt=True,
                tokenize=False,
                **template,
            )
            if runtime.get("backend_mode") == "remote"
            else prompt.prompt_text
        )
        count = len(tokenizer.encode(rendered, add_special_tokens=False))
        available = int(runtime["max_model_len"]) - count - 256
        if available < 1:
            raise ValueError(
                f"Hierarchy prompt needs {count} tokens; configured context leaves no output budget. Source was not truncated."
            )
        prompt.max_tokens = min(prompt.max_tokens, available)
    # LocalBackend has one sampling budget per batch. Use the smallest safe one.
    if runtime.get("backend_mode") == "local" and prompts:
        runtime = deepcopy(runtime)
        runtime["sampling_params"]["max_tokens"] = min(p.max_tokens for p in prompts)
    return prompts, runtime


def retry_config(config, failures):
    """After two invalid answers, alternate thinking (Beam alternates effort)."""
    updated = deepcopy(config)
    if (failures // 2) % 2 == 0:
        return updated
    template = updated.setdefault("chat_template_kwargs", {})
    extra = updated.setdefault("remote", {}).setdefault("extra_body", {})
    remote_template = extra.setdefault("chat_template_kwargs", {})
    if (
        str(updated.get("reasoning_parser", "")).lower() == "beam"
        or "beam" in str(updated.get("model_name", "")).lower()
    ):
        template["reasoning_effort"] = "low"
        remote_template["reasoning_effort"] = "low"
        updated["reasoning_effort"] = "low"
        updated["remote"].setdefault("request_params", {})["reasoning_effort"] = "low"
    else:
        enabled = remote_template.get(
            "enable_thinking", template.get("enable_thinking", True)
        )
        template["enable_thinking"] = not enabled
        remote_template["enable_thinking"] = not enabled
    return updated


def summarize_hierarchically(
    trial_texts, backend, *, trial_config, model_metadata_cache_dir=None, settings=None
):
    settings = dict(settings or {})
    prompts = load_prompts(Path(__file__).resolve().parents[1] / "prompts")
    max_attempts = int(settings.get("max_attempts", 4))
    max_nodes = int(settings.get("max_nodes", 4096))
    max_calls = int(settings.get("max_calls", 20000))
    batch_size = int(settings.get("batch_size", 256))
    if (
        min(
            max_attempts,
            max_nodes,
            max_calls,
            batch_size,
            int(settings.get("active_trials", 32)),
        )
        < 1
    ):
        raise ValueError("Hierarchy limits must be positive")
    cache = ResponseCache(
        settings.get("checkpoint_dir"),
        {"version": VERSION, "prompts": prompts, "runtime": trial_config},
    )
    if trial_config.get("backend_mode") == "remote":
        from matchminer_ai.llm.remote_inference import _run_sync

        return _run_sync(
            lambda: _summarize_ready(
                trial_texts,
                backend,
                trial_config,
                prompts,
                cache,
                settings,
                model_metadata_cache_dir,
            )
        )
    states = [
        Hierarchy(i, text, prompts, max_nodes=max_nodes, max_calls=max_calls)
        for i, text in enumerate(trial_texts)
    ]
    attempts = [[] for _ in states]
    model_metadata = {"model_name": trial_config.get("model_name")}
    for stage in range(len(FIELDS) + 1):
        requests_by_trial = [state.requests() for state in states]
        requests = [request for group in requests_by_trial for request in group]
        if not requests:
            break
        logger.info(
            "Trial hierarchy stage %d/%d: %d parent requests",
            stage + 1, len(FIELDS) + 1, len(requests)
        )
        responses = {}
        for offset in range(0, len(requests), batch_size):
            chunk = requests[offset : offset + batch_size]
            messages = {}
            pending = []
            for request in chunk:
                identity = (request.trial_index, request.node_index)
                cached = cache.get(request)
                if cached is not None:
                    responses[identity] = cached
                else:
                    pending.append(request)
                    messages[identity] = request.messages
            for attempt in range(max_attempts):
                if not pending:
                    break
                runtime = retry_config(trial_config, attempt)
                prompt_list, runtime = _bounded_prompts(
                    [messages[(r.trial_index, r.node_index)] for r in pending], runtime
                )
                generation = backend.generate_llm_outputs(
                    prompt_list=prompt_list,
                    llm_config=runtime,
                    model_metadata_cache_dir=model_metadata_cache_dir,
                )
                model_metadata = generation.model_metadata
                next_pending = []
                for index, (request, final, reasoning, finish) in enumerate(
                    zip(
                        pending,
                        generation.final_outputs,
                        generation.reasoning_outputs,
                        generation.finish_reasons,
                        strict=True,
                    )
                ):
                    identity = (request.trial_index, request.node_index)
                    completion = Completion(
                        final,
                        reasoning,
                        finish,
                        generation.raw_outputs[index] if generation.raw_outputs else "",
                    )
                    record = {
                        "request_key": request.key,
                        "field": request.field_name,
                        "attempt": attempt + 1,
                        **vars(completion),
                    }
                    try:
                        parse_response(final, request.field_name, finish)
                    except ValueError as exc:
                        record["validation_error"] = str(exc)
                        # Transport retries already belong to the backend. Do not
                        # make error placeholders into model correction prompts.
                        if finish == "error" or attempt + 1 == max_attempts:
                            completion = Completion(
                                str(exc), reasoning, "error", completion.raw
                            )
                            responses[identity] = completion
                        else:
                            messages[identity] = retry_messages(
                                request, completion, str(exc)
                            )
                            next_pending.append(request)
                    else:
                        cache.put(request, completion)
                        responses[identity] = completion
                    attempts[request.trial_index].append(record)
                pending = next_pending
        for state, group in zip(states, requests_by_trial, strict=True):
            if group:
                state.accept(
                    group, [responses[(r.trial_index, r.node_index)] for r in group]
                )
    finals, reasons, finishes, raw = [], [], [], []
    for state, trial_attempts in zip(states, attempts, strict=True):
        trace = json.dumps(
            {
                "version": VERSION,
                "nodes": state.trace,
                "attempts": trial_attempts,
                "error": state.error,
            },
            ensure_ascii=False,
        )
        raw.append(trace)
        reasons.append(trace)
        finals.append("" if state.error else state.output())
        finishes.append("error" if state.error else "stop")
    return LLMGenerationResult(
        finals, {**model_metadata, "extraction_method": VERSION}, finishes, reasons, raw
    )


@asynccontextmanager
async def _request_session(backend, config, metadata_dir):
    if hasattr(backend, "open_request_session"):
        async with backend.open_request_session(config, metadata_dir) as session:

            async def generate(prompt, runtime):
                result = await session.generate(prompt, runtime)
                return Completion(
                    result.summary, result.reasoning, result.finish_reason
                )

            yield session.capacity, session.model_metadata, generate
    else:
        # Custom/test backends retain their existing public interface. Each
        # invocation contains one prompt; the outer scheduler owns the cap.
        metadata = {"model_name": config.get("model_name")}

        async def generate(prompt, runtime):
            result = await asyncio.to_thread(
                backend.generate_llm_outputs,
                prompt_list=[prompt],
                llm_config=runtime,
                model_metadata_cache_dir=metadata_dir,
            )
            metadata.update(result.model_metadata)
            return Completion(
                result.final_outputs[0],
                result.reasoning_outputs[0],
                result.finish_reasons[0],
                result.raw_outputs[0] if result.raw_outputs else "",
            )

        yield max(1, int(config.get("max_concurrent_requests", 16))), metadata, generate


async def _summarize_ready(
    texts, backend, config, prompts, cache, settings, metadata_dir
):
    states = [None] * len(texts)

    def admit_states():
        for i, text in enumerate(texts):
            states[i] = ReadyHierarchy(
                i,
                text,
                prompts,
                max_nodes=int(settings.get("max_nodes", 4096)),
                max_calls=int(settings.get("max_calls", 20000)),
            )
            yield states[i]

    attempts = [[] for _ in states]
    max_attempts = int(settings.get("max_attempts", 4))
    async with _request_session(backend, config, metadata_dir) as (
        capacity,
        metadata,
        generate,
    ):

        async def complete(request):
            cached = await asyncio.to_thread(cache.get, request)
            if cached is not None:
                return cached
            messages = request.messages
            for attempt in range(max_attempts):
                runtime = retry_config(config, attempt)
                prompt_list, runtime = _bounded_prompts([messages], runtime)
                completion = await generate(prompt_list[0], runtime)
                record = {
                    "request_key": request.key,
                    "field": request.field_name,
                    "attempt": attempt + 1,
                    **vars(completion),
                }
                attempts[request.trial_index].append(record)
                try:
                    parse_response(
                        completion.final, request.field_name, completion.finish_reason
                    )
                except ValueError as exc:
                    record["validation_error"] = str(exc)
                    if (
                        completion.finish_reason == "error"
                        or attempt + 1 == max_attempts
                    ):
                        return Completion(
                            str(exc), completion.reasoning, "error", completion.raw
                        )
                    messages = retry_messages(request, completion, str(exc))
                else:
                    await asyncio.to_thread(cache.put, request, completion)
                    return completion
            raise AssertionError("Positive retry limit must produce a completion")

        await run_ready_hierarchies(
            admit_states(),
            complete,
            concurrency=capacity,
            active_trials=max(capacity, int(settings.get("active_trials", 32))),
        )
    finals, traces, finishes = [], [], []
    for state, trial_attempts in zip(states, attempts, strict=True):
        traces.append(
            json.dumps(
                {
                    "version": VERSION,
                    "nodes": state.trace,
                    "attempts": trial_attempts,
                    "error": state.error,
                },
                ensure_ascii=False,
            )
        )
        finals.append("" if state.error else state.output())
        finishes.append("error" if state.error else "stop")
    return LLMGenerationResult(
        finals, {**metadata, "extraction_method": VERSION}, finishes, traces, traces
    )
