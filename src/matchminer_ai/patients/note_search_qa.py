"""Question-focused raw-note QA through a small persistent Python REPL loop."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import copy
from importlib import resources
import json
import math
import re
import time
from typing import Callable

import pandas as pd

from matchminer_ai.llm.structured import (
    EndpointError,
    StructuredClient,
    StructuredConfig,
    resolve_structured_config,
)
from matchminer_ai.llm.model_profiles import resolve_model_profile

from ._note_repl import NoteREPL, NoteREPLError
from ._note_record import prepare_record, source_excerpts

NOTICE = (
    "Research documentation review of selected excerpts from supplied notes. "
    "Attached excerpts are automatically retained review context, not individually "
    "selected or verified supporting citations. "
    "Unknown or missing documentation does not mean an event did not occur. "
    "Search may miss evidence. This is not a guideline-concordance or eligibility "
    "determination; clinical interpretation and source completeness need human review."
)


@dataclass(frozen=True)
class NoteSearchLLMConfig(StructuredConfig):
    """Endpoint using shared vendor sampling, thinking on, and no trace replay.

    Optional sampling fields override vendor defaults. With max_tokens omitted,
    the existing model profile supplies its completion-budget floor, or 8192.
    search_reasoning_effort applies only to the first search-only turn where
    graded effort is supported; later turns retain the configured answer effort.
    """

    base_url: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    context_window: int = 65536
    max_concurrent_requests: int = 8
    timeout: float = 120
    attempts: int = 2
    tokenizer_mode: str = "bytes"
    response_format: str = "json_schema"
    sampling_profile: str = "auto"
    reasoning_effort: str = "xhigh"
    search_reasoning_effort: str | None = "low"


def _resolve_llm(config):
    params, extra = (
        copy.deepcopy(config.request_params),
        copy.deepcopy(config.extra_body),
    )
    for name in ("temperature", "top_p"):
        value = getattr(config, name)
        if value is not None:
            params.setdefault(name, value)
    if config.top_k is not None:
        extra.setdefault("top_k", config.top_k)
    template = extra.setdefault("chat_template_kwargs", {})
    if not isinstance(template, dict):
        raise ValueError("chat_template_kwargs must be a mapping.")
    if config.thinking not in {"on", "off", "default"}:
        raise ValueError("thinking must be on, off, or default.")
    template.setdefault("enable_thinking", config.thinking != "off")
    # Override shared Qwen chat-history preservation for this no-replay harness.
    template["preserve_thinking"] = False
    profile = resolve_model_profile(config.model)
    params["max_tokens"] = (
        config.max_tokens
        if config.max_tokens is not None
        else (profile.min_max_tokens if profile and profile.min_max_tokens else 8192)
    )
    resolved, _ = resolve_structured_config(
        {
            "server_urls": [config.base_url],
            "model_name": config.model,
            "provider": config.provider,
            "google_project_id": config.google_project_id,
            "context_window": config.context_window,
            "safety_tokens": config.safety_tokens,
            "max_concurrent_requests": config.max_concurrent_requests,
            "request_timeout": config.timeout,
            "max_retries": config.attempts,
            "response_format": config.response_format,
            "tokenizer_mode": config.tokenizer_mode,
            "stream": config.stream,
            "api_key_env": config.api_key_env,
            "sampling_profile": config.sampling_profile,
            "reasoning_effort": config.reasoning_effort,
            "remote": {"request_params": params, "extra_body": extra},
        },
        cache_dir=None,
    )
    return replace(resolved, max_prompt_chars=config.max_prompt_chars)


def _resolve_search_llm(config, resolved):
    """Lower effort only for the first, search-only request on supported models.

    The shared resolver determines whether graded effort is supported. Gemma's
    on/off thinking switch and unknown models receive no guessed effort knob.
    Subsequent turns can answer, so they keep the original assessment settings.
    """
    effort = config.search_reasoning_effort
    if effort is not None and (
        not isinstance(effort, str) or effort not in {"low", "medium", "high", "xhigh"}
    ):
        raise ValueError(
            "search_reasoning_effort must be None, low, medium, high, or xhigh."
        )
    if effort is None or "reasoning_effort" not in resolved.request_params:
        return resolved
    params, extra = (
        copy.deepcopy(config.request_params),
        copy.deepcopy(config.extra_body),
    )
    params["reasoning_effort"] = effort
    extra.pop("reasoning_effort", None)
    template = extra.get("chat_template_kwargs", {})
    if "reasoning_effort" in template:
        template["reasoning_effort"] = effort
    return _resolve_llm(
        replace(
            config,
            reasoning_effort=effort,
            request_params=params,
            extra_body=extra,
        )
    )


@dataclass(frozen=True)
class NoteSearchLimits:
    max_cells: int = 12
    max_output_chars: int = 12000
    max_memory_chars: int = 6000
    max_code_chars: int = 6000
    max_question_chars: int = 6000
    max_history_bytes: int = 32_000_000
    max_summary_chars: int = 100_000
    cell_timeout_seconds: float = 5.0
    worker_startup_timeout_seconds: float = 20.0
    worker_memory_mb: int = 512
    max_calls: int | None = 16
    max_scan_patterns: int = 128

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "max_calls":
                if value is not None and (type(value) is not int or value < 2):
                    raise ValueError("max_calls must be None or an integer >= 2.")
            elif name in {"cell_timeout_seconds", "worker_startup_timeout_seconds"}:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(
                        "cell_timeout_seconds must be finite and positive."
                    )
            elif type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.worker_memory_mb < 512:
            raise ValueError("worker_memory_mb must be at least 512 for pandas.")


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class _AnswerFormat:
    """Internal extension for a typed final answer in the same search loop."""

    schema: dict
    instructions: str
    validate: Callable
    prepare_evidence: Callable | None = None
    request_review: bool = False


def _schema(limits, answer_format=None):
    schema = _object(
        {
            "action": {"type": "string", "enum": ["final", "python"]},
            "code": {
                "type": "array",
                "maxItems": 200,
                "items": {
                    "type": "string",
                    "maxLength": limits.max_code_chars,
                },
            },
            "memory": {"type": "string", "maxLength": limits.max_memory_chars},
            "status": {"type": "string", "enum": ["answered", "unknown"]},
            "answer": (
                answer_format.schema
                if answer_format
                else {"type": "string", "maxLength": 6000}
            ),
            "limitations": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string", "maxLength": 1000},
            },
        }
    )

    if answer_format and answer_format.request_review:
        schema["properties"]["needs_review"] = {"type": "boolean"}
        schema["required"].append("needs_review")
    return schema


def _validate(
    value, history, limits, final_only, cells, answer_format=None, evidence=()
):
    expected = set(_schema(limits, answer_format)["properties"])
    # Accept older endpoint replies without the optional conflict flag.
    if not isinstance(value, dict) or set(value) not in (
        expected,
        expected - {"needs_review"},
    ):
        raise ValueError("Return all response fields.")
    if "needs_review" in value and type(value["needs_review"]) is not bool:
        raise ValueError("needs_review must be boolean.")
    text_fields = [("memory", limits.max_memory_chars)]
    if answer_format is None:
        text_fields.append(("answer", 6000))
    for name, maximum in text_fields:
        if not isinstance(value[name], str) or len(value[name]) > maximum:
            raise ValueError("Invalid response text length.")
    if (
        not isinstance(value["code"], list)
        or len(value["code"]) > 200
        or any(not isinstance(line, str) for line in value["code"])
        or len("\n".join(value["code"])) > limits.max_code_chars
    ):
        raise ValueError("Code must be a bounded array of Python source lines.")
    if value["action"] not in ("python", "final"):
        raise ValueError("Invalid action.")
    if value["status"] not in ("answered", "unknown"):
        raise ValueError("Invalid answer status.")
    if (
        not isinstance(value["limitations"], list)
        or len(value["limitations"]) > 12
        or any(not isinstance(s, str) or len(s) > 1000 for s in value["limitations"])
    ):
        raise ValueError("Invalid limitations.")
    if value["action"] == "python":
        if final_only or not "\n".join(value["code"]).strip():
            raise ValueError("A final answer is required, or Python code is empty.")
        # Only code and memory are consumed for this action. Unused answer
        # fields must not turn a valid search step into a failed question.
        return
    if answer_format is None and not value["answer"].strip():
        raise ValueError("Final answers require answer text.")
    if cells == 0:
        raise ValueError("Final answers require searching for exact evidence first.")
    if value["status"] == "answered" and not evidence:
        raise ValueError("Answered questions require source excerpts from search/read.")
    if value["status"] == "unknown" and not value["limitations"]:
        raise ValueError("Unknown answers require an explanation of limitations.")
    if answer_format is not None:
        answer_format.validate({**value, "evidence": evidence})


def _source_observation(observation, history, limits, answer_format=None, record=None):
    """Rebuild bounded original excerpts in the parent, never from worker text.

    Helpers register spans automatically, even when a cell does not print them.
    Explicitly supplying these excerpts on the next request makes provenance
    independent of Python repr, model quoting, and model-managed citation IDs.
    """
    candidates = []
    omitted = observation.pop("sources_truncated")
    for start, end in observation.pop("source_spans"):
        # The subprocess is untrusted. Validate again before slicing parent text.
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(history)
        ):
            raise NoteREPLError("Worker returned invalid source spans.")
        stop = min(end, start + 4000)
        omitted |= stop < end
        candidates.append({"start": start, "end": stop, "quote": history[start:stop]})
    if answer_format and answer_format.prepare_evidence:
        candidates = answer_format.prepare_evidence(candidates)
    elif record and record.structured:
        candidates = source_excerpts(candidates, record.spans)
    sources, display = [], []
    remaining = max(0, limits.max_output_chars - 100)
    for candidate in candidates:
        if not candidate["quote"].strip():
            continue
        visible = (
            dict(candidate)
            if answer_format and answer_format.request_review
            else {k: v for k, v in candidate.items() if k not in {"start", "end"}}
        )
        cost = len(json.dumps(visible, ensure_ascii=False)) + 2
        if cost > remaining:
            omitted = True
            continue
        sources.append(candidate)
        display.append(visible)
        remaining -= cost
    if len(observation["output"]) > remaining:
        observation["output"] = observation["output"][:remaining]
        observation["truncated"] = True
    observation["source_excerpts"] = display
    observation["sources_truncated"] = omitted
    return sources


def _remember_excerpts(retained, fresh):
    for item in fresh:
        if any(e["start"] <= item["start"] < item["end"] <= e["end"] for e in retained):
            continue
        retained[:] = [
            e
            for e in retained
            if not item["start"] <= e["start"] < e["end"] <= item["end"]
        ]
        retained.append(item)


class _MeasuredClient(StructuredClient):
    def __init__(self, config):
        super().__init__(config, None)
        self.requests = 0
        self.max_calls = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_reports = 0
        self.finish_reasons = []
        self.validation_errors = []
        self.request_metrics = []

    def complete(self, job, messages, schema, validator):
        def checked(value):
            try:
                return validator(value)
            except ValueError as exc:
                # This harness's validator emits fixed messages, never input text.
                self.validation_errors.append(str(exc))
                raise

        return super().complete(job, messages, schema, checked)

    def _http(self, endpoint, body=None, **kwargs):
        if endpoint == "/chat/completions":
            if self.max_calls is not None and self.requests >= self.max_calls:
                raise EndpointError("Question LLM call budget exhausted.")
            self.requests += 1
            metric = {"reasoning_effort": (body or {}).get("reasoning_effort")}
            started = time.monotonic()
        try:
            result = super()._http(endpoint, body, **kwargs)
        except EndpointError as exc:
            if endpoint == "/chat/completions":
                # Transport failures otherwise look like model validation errors.
                # Store only the fixed HTTP status, never server bodies/URLs,
                # authentication errors, or exception text that might echo input.
                metric["transport_error"] = True
                match = re.fullmatch(r"Endpoint HTTP (\d{3}) for /chat/completions", str(exc))
                if match:
                    metric["http_status"] = int(match[1])
            raise
        finally:
            if endpoint == "/chat/completions":
                metric["seconds"] = round(time.monotonic() - started, 4)
                self.request_metrics.append(metric)
        if endpoint == "/chat/completions":
            self.finish_reasons.append(
                result.get("choices", [{}])[0].get("finish_reason")
            )
            usage = result.get("usage") or {}
            if all(
                type(usage.get(k)) is int
                for k in ("prompt_tokens", "completion_tokens")
            ):
                self.usage_reports += 1
                self.prompt_tokens += usage["prompt_tokens"]
                self.completion_tokens += usage["completion_tokens"]
        return result


def _answer(
    history,
    question,
    index,
    llm,
    limits,
    system,
    answer_format=None,
    search_llm=None,
    record=None,
):
    started = time.monotonic()
    client = _MeasuredClient(llm)
    client.max_calls = limits.max_calls
    memory, observation = "", None
    evidence, pending_evidence = [], []
    sources_truncated = False
    cells, output_chars, truncated_cells = 0, 0, 0
    successful_cells, cell_errors = 0, []
    worker_startup_seconds, cell_seconds = 0.0, []
    history_lines = history.count("\n") + 1
    termination = "search_budget"
    result = {
        "question_index": index,
        "question": question,
        "status": "unknown",
        "answer": "Unknown: the search budget was exhausted.",
        "evidence": [],
        "limitations": ["Search budget exhausted before a validated answer."],
    }
    try:
        worker_started = time.monotonic()
        with NoteREPL(
            history,
            limits,
            notes=record.spans if record else None,
            patient_summary=record.patient_summary if record else None,
        ) as worker:
            worker_startup_seconds = time.monotonic() - worker_started
            for turn in range(limits.max_cells + 1):
                remaining = (
                    None
                    if limits.max_calls is None
                    else limits.max_calls - client.requests
                )
                if remaining is not None and remaining <= 0:
                    termination = "call_budget"
                    break
                final_only = turn == limits.max_cells or remaining == 1
                search_only = cells == 0 and not final_only
                turn_llm = (search_llm or llm) if search_only else llm
                client.config = turn_llm
                cells_remaining = limits.max_cells - cells
                if remaining is not None:
                    cells_remaining = min(cells_remaining, remaining - 1)
                    # Retries count too. Keep one request available to answer
                    # after searching; a final response can use the remainder.
                    client.config = replace(
                        turn_llm,
                        attempts=min(
                            llm.attempts, remaining if final_only else remaining - 1
                        ),
                    )
                payload = {
                    "question": question,
                    "history_chars": len(history),
                    "history_lines": history_lines,
                    "notes_metadata": record.describe() if record else None,
                    "patient_summary": record.patient_summary if record else None,
                    "memory": memory,
                    "last_cell_result": observation,
                    "cells_remaining": cells_remaining,
                    "calls_remaining": remaining,
                    "final_only": final_only,
                    "search_only": search_only,
                    "next_step": (
                        "Explore notes with pandas filters or regex; display relevant original excerpts."
                        if search_only
                        else "Answer when evidence is sufficient, or navigate additional notes "
                        "to resolve missing components, timing or conflicting updates. "
                        "Do not reread to extract quotes, "
                        "recalculate offsets, or assign source dates; code handles provenance."
                    ),
                    "max_memory_chars": limits.max_memory_chars,
                    "max_output_chars": limits.max_output_chars,
                }
                schema = _schema(limits, answer_format)
                if final_only:
                    schema["properties"]["action"]["enum"] = ["final"]
                    schema["properties"]["code"]["maxItems"] = 0
                elif search_only:
                    schema["properties"]["action"]["enum"] = ["python"]
                messages = [
                    {
                        "role": "system",
                        "content": system
                        + "\n\nResponse JSON schema:\n"
                        + json.dumps(schema),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ]
                if not client.fits(messages):
                    termination = "context_limit"
                    result["answer"] = (
                        "Unknown: search state exceeded the context budget."
                    )
                    result["limitations"] = [
                        "Reduce output/memory limits or increase the configured context window."
                    ]
                    break
                # Retain only excerpts included in an actual assessment request;
                # a last cell that cannot fit into context was never reviewed.
                _remember_excerpts(evidence, pending_evidence)
                try:
                    value = client.complete(
                        "patient-note-search",
                        messages,
                        schema,
                        lambda v: _validate(
                            v,
                            history,
                            limits,
                            final_only,
                            cells,
                            answer_format,
                            evidence,
                        ),
                    )
                except EndpointError:
                    # A failed early answer or follow-up may leave the reserved
                    # final request unused. Spend it on the evidence already
                    # gathered, within the same hard total attempt cap.
                    if (
                        not final_only
                        and successful_cells
                        and limits.max_calls is not None
                        and limits.max_calls - client.requests == 1
                    ):
                        continue
                    raise
                memory = value["memory"]
                if value["action"] == "final":
                    termination = (
                        "answered" if value["status"] == "answered" else "model_unknown"
                    )
                    if answer_format and answer_format.request_review:
                        result["needs_review"] = value.get("needs_review", False)
                    result.update(
                        {
                            k: value[k]
                            for k in (
                                "status",
                                "answer",
                                "limitations",
                            )
                        }
                    )
                    break
                code = "\n".join(value["code"])
                cell_started = time.monotonic()
                observation = worker.execute(code)
                cell_seconds.append(round(time.monotonic() - cell_started, 4))
                pending_evidence = _source_observation(
                    observation, history, limits, answer_format, record
                )
                sources_truncated |= observation["sources_truncated"]
                # Code helps recover from a failed cell or recall variable names;
                # provider reasoning and earlier assistant messages stay discarded.
                observation["code"] = code
                cells += 1
                if observation["error"] is None:
                    successful_cells += 1
                else:
                    cell_errors.append(observation["error"])
                output_chars += len(observation["output"])
                truncated_cells += int(observation["truncated"])
    except (EndpointError, NoteREPLError) as exc:
        termination = (
            "endpoint_failure" if isinstance(exc, EndpointError) else "worker_failure"
        )
        budget_exhausted = (
            isinstance(exc, EndpointError)
            and limits.max_calls is not None
            and client.requests >= limits.max_calls
        )
        if budget_exhausted:
            termination = "call_budget"
        result.update(
            status="error",
            answer=None,
            evidence=[],
            limitations=[
                "LLM call budget exhausted before a validated answer."
                if budget_exhausted
                else "Endpoint request/validation failed."
                if isinstance(exc, EndpointError)
                else "Isolated worker failed or exceeded its resource limits."
            ],
        )
    if cells and not successful_cells:
        termination = "cell_execution_failure"
        result.update(
            status="error",
            answer=None,
            evidence=[],
            limitations=[
                "Every Python cell failed; no successful record search supports an answer."
            ],
        )
    result["evidence"] = evidence
    if sources_truncated:
        result["limitations"].append(
            "Some retrieved source text was omitted by the excerpt budget."
        )
    for item in result["evidence"]:
        # Full-text histories carry no trustworthy structured date provenance.
        item.setdefault("note_date", None)
    complete_usage = client.requests == client.usage_reports
    result["metadata"] = {
        "cells": cells,
        "successful_cells": successful_cells,
        "cell_errors": cell_errors,
        "requests": client.requests,
        "request_metrics": client.request_metrics,
        "worker_startup_seconds": round(worker_startup_seconds, 4),
        "cell_seconds": cell_seconds,
        "max_calls": limits.max_calls,
        "max_cells": limits.max_cells,
        "structured_notes": bool(record and record.structured),
        "patient_summary_used": bool(record and record.patient_summary),
        "patient_summary_role": "navigation_context_only",
        "max_scan_patterns": limits.max_scan_patterns,
        "model": llm.model,
        "elapsed_seconds": round(time.monotonic() - started, 4),
        "prompt_tokens": client.prompt_tokens if complete_usage else None,
        "completion_tokens": client.completion_tokens if complete_usage else None,
        "usage_complete": complete_usage,
        "finish_reasons": client.finish_reasons,
        "validation_errors": client.validation_errors,
        "cell_output_chars": output_chars,
        "truncated_cells": truncated_cells,
        "scope": "searched_excerpts",
        "evidence_selection": "automatic_reviewed_excerpts",
        "source_excerpts_truncated": sources_truncated,
        "termination_reason": termination,
    }
    return result


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def answer_patient_question_batch(
    patients: list[dict],
    *,
    llm: NoteSearchLLMConfig,
    limits: NoteSearchLimits | None = None,
    max_parallel_patients: int = 4,
    max_parallel_questions: int = 4,
    max_active_questions: int = 8,
    progress_callback=None,
) -> dict:
    """Answer per-patient question lists with bounded two-level concurrency.

    Each input contains ``patient_id`` and ``questions`` plus ``history`` (text),
    ``notes`` (single-patient DataFrame), or both. An optional ``patient_summary``
    guides navigation but is not evidence. Structured notes take precedence when
    both inputs exist. Results preserve both input orders. Endpoint calls share
    ``llm.max_concurrent_requests`` across this process. No patient checkpoints,
    embeddings, web searches, recursive LLM calls, or application integration.
    Progress callbacks run on the caller thread and contain numeric indices only.
    """
    return _run_question_batch(
        patients,
        llm=llm,
        limits=limits,
        max_parallel_patients=max_parallel_patients,
        max_parallel_questions=max_parallel_questions,
        max_active_questions=max_active_questions,
        progress_callback=progress_callback,
    )


def _run_question_batch(
    patients,
    *,
    llm,
    limits=None,
    max_parallel_patients=4,
    max_parallel_questions=4,
    max_active_questions=8,
    progress_callback=None,
    answer_format=None,
    prepared_records=None,
):
    limits = limits or NoteSearchLimits()
    if not isinstance(limits, NoteSearchLimits) or not isinstance(
        llm, NoteSearchLLMConfig
    ):
        raise TypeError("Supply NoteSearchLimits and an explicit NoteSearchLLMConfig.")
    if not isinstance(llm.model, str) or not llm.model.strip():
        raise ValueError("Set the endpoint's model ID explicitly.")
    if not isinstance(llm.base_url, str) or not llm.base_url.strip():
        raise ValueError("Set the LLM endpoint URL explicitly.")
    for name, value in (
        ("max_parallel_patients", max_parallel_patients),
        ("max_parallel_questions", max_parallel_questions),
        ("max_active_questions", max_active_questions),
        ("max_concurrent_requests", llm.max_concurrent_requests),
        ("attempts", llm.attempts),
    ):
        _positive(name, value)
    if (
        isinstance(llm.timeout, bool)
        or not isinstance(llm.timeout, (int, float))
        or not math.isfinite(llm.timeout)
        or llm.timeout <= 0
    ):
        raise ValueError("Set a finite, positive endpoint timeout.")
    if llm.tokenizer_mode not in {"bytes", "endpoint"}:
        raise ValueError("Use bytes or endpoint token accounting.")
    # StructuredConfig is a low-level client configuration. Keep the REPL's
    # messages, schema, and single-response protocol owned by this harness.
    reserved = {
        "model",
        "messages",
        "stream",
        "stream_options",
        "response_format",
        "n",
        "tools",
        "tool_choice",
        "extra_headers",
        "extra_query",
        "extra_body",
        "api_key",
        "max_tokens",
        "max_completion_tokens",
    }
    if reserved.intersection(llm.request_params.keys() | llm.extra_body.keys()):
        raise ValueError(
            "Endpoint parameters cannot override the note-search protocol or output reserve."
        )
    _positive("context_window", llm.context_window)
    # Explicit model/context: no discovery request. Build both request profiles
    # once, preserving sampling and the original assessment effort.
    original_llm = llm
    llm = _resolve_llm(original_llm)
    search_llm = _resolve_search_llm(original_llm, llm)
    if not isinstance(patients, list) or not patients:
        raise ValueError("Supply at least one patient question set.")
    normalized, ids = [], set()
    for pi, patient in enumerate(patients):
        if (
            not isinstance(patient, dict)
            or not {"patient_id", "questions"} <= set(patient)
            or set(patient)
            - {"patient_id", "history", "notes", "questions", "patient_summary"}
        ):
            raise ValueError(
                "Each patient requires patient_id, questions, and notes or history."
            )
        identity, questions = patient["patient_id"], patient["questions"]
        if not isinstance(identity, str) or not identity.strip() or identity in ids:
            raise ValueError("Patient IDs must be unique, nonempty strings.")
        record = (
            prepared_records[pi]
            if prepared_records is not None
            else prepare_record(
                patient.get("history"),
                patient.get("notes"),
                patient.get("patient_summary"),
                max_bytes=limits.max_history_bytes,
                max_summary_chars=limits.max_summary_chars,
            )
        )
        if (
            not isinstance(questions, list)
            or not questions
            or any(
                not isinstance(q, str)
                or not q.strip()
                or len(q) > limits.max_question_chars
                for q in questions
            )
        ):
            raise ValueError(
                "Supply a nonempty list of questions within max_question_chars."
            )
        normalized.append(
            dict(
                patient_id=identity,
                history=record.history,
                questions=list(questions),
                record=record,
            )
        )
        ids.add(identity)
    # Fail closed before any endpoint sees patient questions if isolation is unavailable.
    with NoteREPL("", limits):
        pass
    system = (
        resources.files("matchminer_ai.prompts")
        .joinpath("patient.note_search.system.txt")
        .read_text(encoding="utf-8")
        .replace("{max_scan_patterns}", str(limits.max_scan_patterns))
    )
    if answer_format is not None:
        system += "\n\n" + answer_format.instructions
    results = [
        dict(patient_id=p["patient_id"], answers=[None] * len(p["questions"]))
        for p in normalized
    ]
    waiting = deque(range(len(normalized)))
    active, futures = {}, {}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_active_questions) as pool:
        while waiting or active:
            while waiting and len(active) < max_parallel_patients:
                active[waiting.popleft()] = {"next": 0, "running": 0, "done": 0}
            # Round-robin scheduling gives each active patient an opportunity per pass.
            while len(futures) < max_active_questions:
                submitted = False
                for pi, state in active.items():
                    patient = normalized[pi]
                    if (
                        len(futures) == max_active_questions
                        or state["running"] >= max_parallel_questions
                        or state["next"] == len(patient["questions"])
                    ):
                        continue
                    qi = state["next"]
                    futures[
                        pool.submit(
                            _answer,
                            patient["history"],
                            patient["questions"][qi],
                            qi,
                            llm,
                            limits,
                            system,
                            answer_format,
                            search_llm,
                            patient["record"],
                        )
                    ] = (pi, qi)
                    state["next"] += 1
                    state["running"] += 1
                    submitted = True
                if not submitted:
                    break
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                pi, qi = futures.pop(future)
                result = future.result()
                results[pi]["answers"][qi] = result
                state = active[pi]
                state["running"] -= 1
                state["done"] += 1
                if state["done"] == len(normalized[pi]["questions"]):
                    del active[pi]
                if progress_callback is not None:
                    progress_callback(
                        {
                            "patient_index": pi,
                            "question_index": qi,
                            "stage": "question_complete",
                            "status": result["status"],
                        }
                    )
    return {
        "patients": results,
        "notice": NOTICE,
        "metadata": {
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "max_parallel_patients": max_parallel_patients,
            "max_parallel_questions": max_parallel_questions,
            "max_active_questions": max_active_questions,
            "max_concurrent_requests": llm.max_concurrent_requests,
            "max_calls_per_question": limits.max_calls,
            "max_cells_per_question": limits.max_cells,
            "max_scan_patterns_per_call": limits.max_scan_patterns,
            "model": llm.model,
            "thinking": llm.thinking,
            "reasoning_effort": llm.request_params.get("reasoning_effort"),
            "search_reasoning_effort": search_llm.request_params.get(
                "reasoning_effort"
            ),
            "sampling": {
                "temperature": llm.temperature,
                "top_p": llm.top_p,
                "top_k": llm.top_k,
            },
            "preserve_thinking": False,
            "evidence_selection": "automatic_reviewed_excerpts",
            "reserved_output_tokens": llm.max_tokens,
        },
    }


def answer_patient_questions(
    history: str | pd.DataFrame | None = None,
    questions: list[str] | None = None,
    *,
    notes: pd.DataFrame | None = None,
    patient_summary: str | None = None,
    llm: NoteSearchLLMConfig,
    limits: NoteSearchLimits | None = None,
    max_parallel_questions: int = 4,
    progress_callback=None,
) -> dict:
    """Search text and/or original notes, optionally guided by a patient summary.

    DataFrame note_text is required; note_date and note_type are optional. The
    DataFrame is authoritative when both forms exist. Summary text is navigation
    context, never a substitute for original-note evidence.
    """
    result = answer_patient_question_batch(
        [
            {
                "patient_id": "patient",
                "history": history,
                "notes": notes,
                "patient_summary": patient_summary,
                "questions": questions,
            }
        ],
        llm=llm,
        limits=limits,
        max_parallel_patients=1,
        max_parallel_questions=max_parallel_questions,
        max_active_questions=max_parallel_questions,
        progress_callback=progress_callback,
    )
    return {
        "answers": result["patients"][0]["answers"],
        "notice": result["notice"],
        "metadata": result["metadata"],
    }
