"""Retry failed serial summaries from the original notes and last valid prior."""
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import uuid

from matchminer_ai.llm.backends import LLMGenerationResult
from .summary_quality import check_patient_summary, summary_retry_codes


def _attempt_trace(prompt, *, attempt, answer, reasoning, raw, finish, metadata,
                   response_metadata, trace_dir):
    """Check one answer; return its attempt trace and retry codes.

    Both schedulers use this, so an answer is checked and traced the same way
    whether it came back in a batch or on its own.
    """
    codes = summary_retry_codes(answer, finish)
    trace = dict(attempt=attempt, answer=answer, reasoning=reasoning, raw_output=raw,
                 finish_reason=finish, quality=check_patient_summary(answer), retry_codes=codes,
                 messages=prompt.messages, prompt_text=prompt.prompt_text,
                 model_metadata={k: v for k, v in metadata.items() if k != 'responses'})
    if response_metadata is not None:
        trace['response_metadata'] = response_metadata
    if trace_dir:
        folder = Path(trace_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"summary_{uuid.uuid4().hex}.json"
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(trace))
        temporary.replace(path)
        trace['trace_path'] = str(path)
    return trace, codes


def generate_checked_summaries(backend, prompts, *, llm_config, metadata_cache_dir,
                               max_attempts=3, trace_dir=None):
    """Regenerate rejected rows; rows still rejected after max_attempts fail.

    A failed row keeps its last answer and reasoning with finish_reason "error",
    so serial summarization drops that patient. Its attempts keep every trace.
    """
    if max_attempts < 1:
        raise ValueError("summary_quality_max_attempts must be positive")
    pending = list(range(len(prompts)))
    accepted, rejected, attempts = {}, {}, {i: [] for i in pending}
    metadata = {}
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        # Backends index within their batch, not within the original patient table.
        batch = [replace(prompts[i], row_idx=j) for j, i in enumerate(pending)]
        generation = backend.generate_llm_outputs(prompt_list=batch, llm_config=llm_config,
                                                  model_metadata_cache_dir=metadata_cache_dir)
        # Remote chat backends expose parsed content/reasoning without a raw
        # completion string. Preserve that absence; do not invent a raw trace.
        raw_outputs = generation.raw_outputs or [None] * len(pending)
        fields = [generation.final_outputs, generation.reasoning_outputs,
                  raw_outputs, generation.finish_reasons]
        if any(len(values) != len(pending) for values in fields):
            raise ValueError("Patient summarization backend returned an incomplete batch")
        metadata = generation.model_metadata
        responses = metadata.get('responses', [])
        remaining = []
        for batch_index, (i, answer, reasoning, raw, finish) in enumerate(zip(pending, *fields, strict=True)):
            trace, codes = _attempt_trace(
                prompts[i], attempt=attempt, answer=answer, reasoning=reasoning, raw=raw,
                finish=finish, metadata=metadata, trace_dir=trace_dir,
                response_metadata=responses[batch_index] if len(responses) == len(pending) else None)
            attempts[i].append(trace)
            if codes:
                remaining.append(i)
                rejected[i] = (answer, reasoning, raw, "error")
            else:
                accepted[i] = (answer, reasoning, raw, finish)
        pending = remaining
    accepted.update((i, rejected[i]) for i in pending)
    values = [accepted[i] for i in range(len(prompts))]
    return LLMGenerationResult(final_outputs=[v[0] for v in values],
        reasoning_outputs=[v[1] for v in values],
        raw_outputs=[v[2] for v in values] if all(v[2] is not None for v in values) else [],
        finish_reasons=[v[3] for v in values], model_metadata=metadata), attempts


async def generate_checked_summary(session, prompt, *, llm_config, max_attempts=3, trace_dir=None):
    """Check and regenerate one prompt through a backend's request session.

    Returns what ``generate_checked_summaries`` returns for a one-prompt batch
    of a remote backend: the result with its one row, and that row's attempt
    traces. So a chunk summarized on its own is retried, traced and failed the
    same way. Each attempt is one ``session.generate`` request; the checks and
    trace files run in a worker thread, off the event loop.
    """
    if max_attempts < 1:
        raise ValueError("summary_quality_max_attempts must be positive")
    attempts = []
    single = replace(prompt, row_idx=0)
    for attempt in range(1, max_attempts + 1):
        result = await session.generate(single, llm_config)
        # A pooled session reports where each request ran, as a pooled batch does.
        response = (None if result.response_metadata is None
                    else {'row_idx': 0, **result.response_metadata})
        metadata = {**session.model_metadata, **({'responses': [response]} if response else {})}
        trace, codes = await asyncio.to_thread(
            _attempt_trace, prompt, attempt=attempt, answer=result.summary,
            reasoning=result.reasoning, raw=None, finish=result.finish_reason,
            metadata=metadata, response_metadata=response, trace_dir=trace_dir)
        attempts.append(trace)
        finish = result.finish_reason
        if not codes:
            break
        finish = "error"
    return LLMGenerationResult(final_outputs=[result.summary], reasoning_outputs=[result.reasoning],
        raw_outputs=[], finish_reasons=[finish], model_metadata=metadata), attempts
