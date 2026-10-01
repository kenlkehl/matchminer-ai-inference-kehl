"""Opt-in, concurrent compression of independent notes at a remote endpoint."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from importlib import resources
from typing import cast

import pandas as pd

from matchminer_ai.cancellation import (
    cancel_sleep,
    check_cancelled,
    submit_cancellable,
)
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, remote_enabled
from matchminer_ai.llm.structured import (
    EndpointError,
    StructuredClient,
    resolve_structured_config,
)

NoteCompressionProgress = Callable[[int, int], None]


class NoteCompressionError(RuntimeError):
    """A remote request failed to return a complete compressed note."""


class _CompressionClient(StructuredClient):
    def __init__(self, config, fallback_config=None):
        super().__init__(config, None)
        self._fallback_config = fallback_config

    def complete(self, job, messages, schema, validator):
        try:
            return super().complete(job, messages, schema, validator)
        except EndpointError as exc:
            if self._fallback_config is None or not _unsupported_thinking(exc):
                raise
            # Retry once at this same endpoint without the thinking switch,
            # including when its tokenizer rejected the switch before generation.
            self.config = self._fallback_config()
            self._fallback_config = None
            return super().complete(job, messages, schema, validator)

    def _complete_in_memory(self, body, messages, schema, validator):
        """Use the shared transport without JSON parsing or patient checkpoints."""
        feedback = None
        for attempt in range(self.config.attempts):
            check_cancelled()
            request = dict(body)
            request["messages"] = messages + (
                [{"role": "user", "content": feedback}] if feedback else []
            )
            if self.config.stream:
                request.update(stream=True, stream_options={"include_usage": True})
            if not self.fits(request["messages"], use_safety_margin=True):
                raise NoteCompressionError(
                    "Compression retry exceeds the context budget."
                )
            try:
                response = self._http("/chat/completions", request)
                choice = response["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Incomplete completion.")
                message = choice["message"]
                content = message["content"]
                if message.get("refusal") or not isinstance(content, str):
                    raise ValueError("Expected final note text.")
                content = content.strip()
                validator(content)
                return content
            except (ValueError, KeyError, TypeError, IndexError, EndpointError) as exc:
                if self._fallback_config is not None and _unsupported_thinking(exc):
                    raise
                if self._capacity_retry(exc, attempt):
                    continue
                # Fixed feedback and errors never echo provider or patient text.
                feedback = (
                    "Return the complete compressed note only. No explanations, "
                    "commentary, preamble, or reasoning."
                )
                if attempt + 1 < self.config.attempts:
                    cancel_sleep(min(2**attempt, 8))
        raise NoteCompressionError(
            "Note compression failed after bounded endpoint retries."
        ) from None


def _unsupported_thinking(exc: Exception) -> bool:
    """Recognize only request rejections concerning the thinking control."""
    if not isinstance(exc, EndpointError) or exc.http_status not in (400, 422):
        return False
    return exc.thinking_unsupported


def _validate_output(text: str) -> None:
    if not text:
        raise ValueError("Empty compressed note.")
    if re.search(r"</?(?:think|analysis|reasoning)>", text, re.IGNORECASE):
        raise ValueError("Reasoning must not appear in the compressed note.")
    if re.match(
        r"^(?:here(?:'s| is| are)\b|sure(?:[!,.:\s]|$)|explanation\s*:)",
        text,
        re.IGNORECASE,
    ):
        raise ValueError("Explanatory preamble must not appear in the compressed note.")


def compress_patient_notes(
    notes: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    thinking: str = "off",
    max_concurrent_requests: int | None = None,
    progress_callback: NoteCompressionProgress | None = None,
) -> pd.DataFrame:
    """Compress each note independently, preserving input rows and source text.

    The input requires one string-valued ``note_text`` column. The returned copy
    adds ``compressed_note_text`` without sorting, deduplicating, or modifying
    any original column or index. Multiple patients may share the input table;
    each request receives only that row's note text, with whitespace runs
    normalized to one ASCII space. Blank strings return an empty compression
    without an LLM call. Null and non-string note values are rejected.

    Uses the configured remote patient LLM, including its model, sampling,
    context budget, timeout, retries, authentication, pacing and concurrency.
    ``thinking`` defaults to ``"off"`` where supported, overriding inherited
    patient reasoning settings for this call only. Adapters that cannot disable
    reasoning omit the switch and retain their configured reasoning settings.
    An endpoint that explicitly rejects the switch is retried once without it.
    Set ``thinking="on"`` to opt in. The returned DataFrame's
    ``attrs["note_compression"]`` records the requested and dispatched thinking
    modes (``"mixed"`` if workers used different modes).
    ``max_concurrent_requests`` optionally overrides the positive worker/HTTP
    limit for this call. ``progress_callback(completed, total)`` runs in the
    calling thread, initially and as notes complete. Completion order does not
    change output order. Cancellation scopes propagate to worker requests.

    Only final note text is returned; reasoning and provider responses are not
    retained. No clinical prompts or outputs are logged or written to disk.
    Oversized notes fail rather than being truncated or split. The prompt asks
    for lossless compression; semantic losslessness is not automatically
    verified. Retain original notes for review and exact quotations, and use
    only an endpoint authorized for the sensitivity of the input.
    """
    check_cancelled()
    config = config or load_default_preset()
    if not isinstance(config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    if not remote_enabled(config):
        raise ValueError("Note compression requires a configured remote LLM endpoint.")
    if thinking not in ("off", "on"):
        raise ValueError("thinking must be 'off' or 'on'.")
    if not isinstance(notes, pd.DataFrame):
        raise TypeError("notes must be a DataFrame with a note_text column.")
    if not notes.columns.is_unique or "note_text" not in notes:
        raise ValueError("Supply unique columns including note_text.")
    if "compressed_note_text" in notes:
        raise ValueError("Input already contains a compressed_note_text column.")
    texts = notes["note_text"].tolist()
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every note_text value must be a string.")
    normalized = [re.sub(r"\s+", " ", text).strip() for text in texts]
    limit = (
        config.remote.get("max_concurrent_requests", 32)
        if max_concurrent_requests is None
        else max_concurrent_requests
    )
    if type(limit) is not int or limit < 1:
        raise ValueError("max_concurrent_requests must be a positive integer.")
    total = len(normalized)
    progress = progress_callback or (lambda completed, total: None)
    progress(0, total)
    outputs = [""] * total
    thinking_modes = set()
    positions = [i for i, text in enumerate(normalized) if text]
    completed = total - len(positions)
    if completed:
        progress(completed, total)
    if positions:
        patient = copy.deepcopy(config.patient)
        template = (
            patient["remote"]
            .setdefault("extra_body", {})
            .setdefault("chat_template_kwargs", {})
        )
        template["enable_thinking"] = thinking == "on"
        if thinking == "off":
            # Inherited effort settings must not contradict the thinking switch.
            template.pop("reasoning_effort", None)
            patient["remote"]["request_params"].pop("reasoning_effort", None)
            patient["remote"]["extra_body"].pop("reasoning_effort", None)
        runtime = build_llm_runtime_config("patient", patient, config=config)
        runtime["response_format"] = "none"
        runtime["max_concurrent_requests"] = limit
        client_config, _ = resolve_structured_config(runtime, cache_dir=None)

        def default_config():
            # Resolve inherited sampling/effort at the already-discovered model,
            # so unsupported switches do not erase the endpoint's normal settings.
            fallback_runtime = build_llm_runtime_config(
                "patient", copy.deepcopy(config.patient), config=config
            )
            fallback_runtime.update(
                model_name=client_config.model,
                context_window=client_config.context_window,
                response_format="none",
                max_concurrent_requests=limit,
            )
            fallback, _ = resolve_structured_config(fallback_runtime, cache_dir=None)
            extra = copy.deepcopy(fallback.extra_body)
            if "chat_template_kwargs" in extra:
                extra["chat_template_kwargs"].pop("enable_thinking", None)
                if not extra["chat_template_kwargs"]:
                    extra.pop("chat_template_kwargs")
            return replace(fallback, thinking="default", extra_body=extra)

        fallback_config = default_config if thinking == "off" else None
        if thinking == "off" and client_config.thinking != "off":
            client_config, fallback_config = default_config(), None
        system = (
            resources.files("matchminer_ai.prompts")
            .joinpath("patient.note_compression.system.txt")
            .read_text(encoding="utf-8")
            .strip()
        )

        def compress(position: int) -> tuple[str, str]:
            check_cancelled()
            client = _CompressionClient(client_config, fallback_config)
            output = cast(
                str,
                client.complete(
                    f"note {position + 1}",
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": normalized[position]},
                    ],
                    {},
                    _validate_output,
                ),
            )
            return output, client.config.thinking

        with ThreadPoolExecutor(max_workers=min(limit, len(positions))) as pool:
            futures = {submit_cancellable(pool, compress, i): i for i in positions}
            try:
                for future in as_completed(futures):
                    check_cancelled()
                    output, dispatched_thinking = future.result()
                    outputs[futures[future]] = output
                    thinking_modes.add(dispatched_thinking)
                    completed += 1
                    progress(completed, total)
            finally:
                for future in futures:
                    future.cancel()
    check_cancelled()
    result = notes.copy()
    result["compressed_note_text"] = outputs
    result.attrs["note_compression"] = {
        "thinking_requested": thinking,
        "thinking": (
            next(iter(thinking_modes))
            if len(thinking_modes) == 1
            else "mixed"
            if thinking_modes
            else "not_requested"
        ),
    }
    return result


def compress_patient_note(
    note: str, *, config: MMAIConfig | None = None, thinking: str = "off"
) -> str:
    """Return only compressed text, requesting reasoning off where supported."""
    result = compress_patient_notes(
        pd.DataFrame({"note_text": [note]}), config=config, thinking=thinking
    )
    return cast(str, result.iloc[0]["compressed_note_text"])


__all__ = [
    "NoteCompressionError",
    "NoteCompressionProgress",
    "compress_patient_note",
    "compress_patient_notes",
]
