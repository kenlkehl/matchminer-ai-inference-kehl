"""Agentic, embedding-retrieved question answering over raw patient notes."""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from multiprocessing.connection import Client, Listener
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np
import pandas as pd

from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.backends import (
    build_llm_runtime_config,
    get_llm_backend,
    get_model_metadata,
)
from matchminer_ai.llm.prompt_rendering import build_prompt_list

RawPatientNoteQAProgress = Callable[[str, int, int, str], None]


class RawPatientNoteQuestionError(ValueError):
    """Raised when the raw-note question-answering workflow cannot finish safely."""


@dataclass(frozen=True)
class _NoteSourceSpan:
    note_date: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _PreparedPatientNotes:
    raw_text: str
    input_type: str
    source_spans: tuple[_NoteSourceSpan, ...]


@dataclass(frozen=True)
class _NoteChunk:
    chunk_id: str
    text: str
    token_start: int
    token_end: int
    note_date: str | None = None


@dataclass(frozen=True)
class _RetrievedChunk:
    chunk: _NoteChunk
    query: str
    score: float

    def prompt_record(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "similarity": round(self.score, 6),
            "note_date": self.chunk.note_date,
            "text": self.chunk.text,
        }


@dataclass(frozen=True)
class _PreparedRawNoteIndex:
    """CPU-resident note chunks and vectors reusable across QA agents."""

    chunks: tuple[_NoteChunk, ...]
    embeddings: np.ndarray
    input_type: str
    source_note_count: int
    embedding_model_name: str
    embedding_device: str
    embedding_batch_size: int
    query_prefix: str
    min_similarity: float
    configured_chunk_size: int
    effective_chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class _QueryEmbeddingEndpoint:
    """Authenticated loopback endpoint for the parent-owned embedding model."""

    address: tuple[str, int]
    authkey: bytes


def _load_prompt_text(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _emit_progress(
    callback: RawPatientNoteQAProgress | None,
    stage: str,
    completed: int,
    total: int,
    detail: str,
) -> None:
    if callback is not None:
        callback(stage, completed, total, detail)


def _format_note_date(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _parse_note_date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="raise")
    if not isinstance(parsed, pd.Timestamp):
        parsed = pd.Timestamp(parsed)
    return parsed


def _utc_sort_date(value: pd.Timestamp) -> pd.Timestamp:
    if value.tzinfo is None:
        return value.tz_localize("UTC")
    return value.tz_convert("UTC")


def _prepare_raw_text(
    patient_notes: str | pd.DataFrame,
    *,
    text_column: str,
    date_column: str,
) -> _PreparedPatientNotes:
    """Validate input and retain note-level source spans for dated DataFrames."""
    if isinstance(patient_notes, str):
        raw_text = patient_notes.strip()
        if not raw_text:
            raise ValueError("patient_notes must be a non-empty string or DataFrame.")
        return _PreparedPatientNotes(raw_text, "string", ())

    if not isinstance(patient_notes, pd.DataFrame):
        raise TypeError("patient_notes must be a string or pandas DataFrame.")
    if text_column == date_column:
        raise ValueError("text_column and date_column must name different columns.")
    missing = [
        column
        for column in (text_column, date_column)
        if column not in patient_notes.columns
    ]
    if missing:
        raise ValueError(
            "patient note DataFrame is missing required columns: " + ", ".join(missing)
        )

    if "patient_id" in patient_notes.columns:
        patient_ids = (
            patient_notes["patient_id"].dropna().astype(str).str.strip().unique()
        )
        patient_ids = [patient_id for patient_id in patient_ids if patient_id]
        if len(patient_ids) > 1:
            raise ValueError(
                "patient note DataFrame must represent exactly one patient; "
                "multiple patient_id values were found."
            )

    normalized = patient_notes[[date_column, text_column]].copy()
    normalized = normalized[normalized[text_column].notna()].copy()
    normalized[text_column] = normalized[text_column].astype(str)
    normalized = normalized[normalized[text_column].str.strip().ne("")].copy()
    if normalized.empty:
        raise ValueError("patient note DataFrame contains no non-empty note text.")
    if normalized[date_column].isna().any():
        raise ValueError("every non-empty patient note must have a date.")
    try:
        normalized["_parsed_note_date"] = normalized[date_column].map(_parse_note_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("patient note dates must be parseable as datetimes.") from exc
    normalized["_sort_note_date"] = normalized["_parsed_note_date"].map(_utc_sort_date)
    normalized = normalized.sort_values("_sort_note_date", kind="mergesort")

    blocks: list[str] = []
    source_spans: list[_NoteSourceSpan] = []
    cursor = 0
    for _, row in normalized.iterrows():
        if blocks:
            cursor += 2  # The two newlines inserted by the join below.
        note_date = _format_note_date(row["_parsed_note_date"])
        block = (
            f"=== Clinical Note dated {note_date} ===\n{str(row[text_column]).strip()}"
        )
        blocks.append(block)
        source_spans.append(
            _NoteSourceSpan(
                note_date=note_date,
                char_start=cursor,
                char_end=cursor + len(block),
            )
        )
        cursor += len(block)
    return _PreparedPatientNotes(
        raw_text="\n\n".join(blocks),
        input_type="dataframe",
        source_spans=tuple(source_spans),
    )


@lru_cache(maxsize=2)
def _get_embedding_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _embedding_tokenizer(model: Any, model_name: str) -> Any:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _extract_input_ids(encoded: Any) -> list[int]:
    input_ids = (
        encoded.get("input_ids") if isinstance(encoded, dict) else encoded.input_ids
    )
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _prefix_token_count(tokenizer: Any, prefix: str) -> int:
    if not prefix:
        return 0
    return len(_extract_input_ids(tokenizer(prefix, add_special_tokens=False)))


def _effective_chunk_size(
    configured_size: int,
    *,
    chunk_overlap: int,
    model: Any,
    tokenizer: Any,
    document_prefix: str,
) -> int:
    if configured_size < 1:
        raise ValueError("raw_patient_note_qa.chunk_size must be positive.")
    if chunk_overlap < 0:
        raise ValueError("raw_patient_note_qa.chunk_overlap cannot be negative.")

    model_limit = getattr(model, "max_seq_length", None)
    if model_limit is None:
        model_limit = getattr(tokenizer, "model_max_length", None)
    try:
        parsed_limit = int(model_limit)
    except (TypeError, ValueError, OverflowError):
        parsed_limit = 0
    if parsed_limit <= 0 or parsed_limit > 1_000_000:
        effective = configured_size
    else:
        try:
            special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        except (AttributeError, TypeError, ValueError):
            special_tokens = 2
        available = (
            parsed_limit
            - special_tokens
            - _prefix_token_count(tokenizer, document_prefix)
        )
        if available < 1:
            raise ValueError(
                "The embedding model has no token capacity left after the "
                "configured document prefix."
            )
        effective = min(configured_size, available)
    if chunk_overlap >= effective:
        raise ValueError(
            "raw_patient_note_qa.chunk_overlap must be less than the effective "
            "embedding chunk size."
        )
    return effective


def _tokenize_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], Any]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(text, add_special_tokens=False)
        return _extract_input_ids(encoded), None
    offsets = (
        encoded.get("offset_mapping")
        if isinstance(encoded, dict)
        else getattr(encoded, "offset_mapping", None)
    )
    return _extract_input_ids(encoded), offsets


def _chunk_raw_text(
    raw_text: str,
    tokenizer: Any,
    *,
    chunk_size: int,
    chunk_overlap: int,
    source_spans: tuple[_NoteSourceSpan, ...] = (),
) -> list[_NoteChunk]:
    chunk_sources = (
        [
            (
                raw_text[source_span.char_start : source_span.char_end],
                source_span.note_date,
            )
            for source_span in source_spans
        ]
        if source_spans
        else [(raw_text, None)]
    )
    chunks: list[_NoteChunk] = []
    stride = chunk_size - chunk_overlap
    for source_text, note_date in chunk_sources:
        token_ids, offsets = _tokenize_with_offsets(tokenizer, source_text)
        if not token_ids:
            continue
        if offsets and isinstance(offsets[0], list):
            offsets = offsets[0]
        has_offsets = offsets is not None and len(offsets) == len(token_ids)

        for start in range(0, len(token_ids), stride):
            end = min(start + chunk_size, len(token_ids))
            if has_offsets:
                char_start = int(offsets[start][0])
                char_end = int(offsets[end - 1][1])
                chunk_text = source_text[char_start:char_end]
            else:
                chunk_text = tokenizer.decode(
                    token_ids[start:end],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            chunk_text = str(chunk_text).strip()
            if chunk_text:
                chunks.append(
                    _NoteChunk(
                        chunk_id=f"chunk_{len(chunks):04d}",
                        text=chunk_text,
                        token_start=start,
                        token_end=end,
                        note_date=note_date,
                    )
                )
            if end >= len(token_ids):
                break
    if not chunks:
        raise ValueError("patient_notes could not be split into non-empty chunks.")
    return chunks


def _encode(model: Any, texts: list[str], *, batch_size: int) -> np.ndarray:
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    values = np.asarray(encoded, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != len(texts):
        raise RawPatientNoteQuestionError(
            "The embedding model returned an unexpected array shape."
        )
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _request_query_embeddings(
    endpoint: _QueryEmbeddingEndpoint,
    texts: list[str],
) -> np.ndarray:
    """Ask the parent process to encode queries with its CUDA model."""
    connection = Client(endpoint.address, authkey=endpoint.authkey)
    try:
        connection.send({"texts": list(texts)})
        status, payload = connection.recv()
    finally:
        connection.close()
    if status != "ok":
        raise RawPatientNoteQuestionError(
            f"The shared query embedding service failed: {payload}"
        )
    values = np.asarray(payload, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(texts):
        raise RawPatientNoteQuestionError(
            "The shared query embedding service returned an unexpected array shape."
        )
    return values


class _QueryEmbeddingServer:
    """Serve one parent-owned embedding model to spawned QA workers."""

    def __init__(self, model: Any, *, batch_size: int) -> None:
        self._model = model
        self._batch_size = max(1, int(batch_size))
        self._authkey = secrets.token_bytes(32)
        self._listener = Listener(
            ("127.0.0.1", 0),
            backlog=128,
            authkey=self._authkey,
        )
        address = self._listener.address
        if not isinstance(address, tuple) or len(address) != 2:
            self._listener.close()
            raise RuntimeError("Could not create the query embedding loopback service.")
        self.endpoint = _QueryEmbeddingEndpoint(
            address=(str(address[0]), int(address[1])),
            authkey=self._authkey,
        )
        self._thread = Thread(
            target=self._serve,
            name="mmai-query-embedding-server",
            daemon=True,
        )

    def __enter__(self) -> _QueryEmbeddingEndpoint:
        self._thread.start()
        return self.endpoint

    def __exit__(self, *_args: Any) -> None:
        try:
            connection = Client(self.endpoint.address, authkey=self.endpoint.authkey)
            try:
                connection.send({"stop": True})
                connection.recv()
            finally:
                connection.close()
        finally:
            self._thread.join(timeout=10)
            self._listener.close()

    def _serve(self) -> None:
        while True:
            connection = self._listener.accept()
            try:
                request_payload = connection.recv()
                if not isinstance(request_payload, dict):
                    raise TypeError("query embedding requests must be mappings.")
                if bool(request_payload.get("stop", False)):
                    connection.send(("ok", None))
                    return
                texts = request_payload.get("texts")
                if not isinstance(texts, list) or not all(
                    isinstance(text, str) for text in texts
                ):
                    raise TypeError("query embedding texts must be a list of strings.")
                values = _encode(
                    self._model,
                    texts,
                    batch_size=self._batch_size,
                )
                connection.send(("ok", values))
            except Exception as exc:  # noqa: BLE001 - report service errors to worker
                try:
                    connection.send(("error", f"{type(exc).__name__}: {exc}"))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            finally:
                connection.close()


@dataclass
class _RawNoteIndex:
    chunks: list[_NoteChunk]
    embeddings: np.ndarray
    query_encoder: Callable[[list[str]], np.ndarray]
    query_prefix: str
    min_similarity: float

    def search(self, query: str, *, top_k: int) -> list[_RetrievedChunk]:
        query = str(query).strip()
        if not query:
            raise ValueError("retrieval query must be non-empty.")
        query_vector = self.query_encoder([f"{self.query_prefix}{query}"])[0]
        scores = self.embeddings @ query_vector
        order = np.argsort(-scores, kind="stable")[: max(1, int(top_k))]
        return [
            _RetrievedChunk(
                chunk=self.chunks[int(index)],
                query=query,
                score=float(scores[int(index)]),
            )
            for index in order
            if float(scores[int(index)]) >= self.min_similarity
        ]


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RawPatientNoteQuestionError("The LLM response did not contain a JSON object.")


def _task_runtime_config(
    qa_config: dict[str, Any],
    *,
    config: MMAIConfig,
) -> dict[str, Any]:
    llm_only_config = {
        key: deepcopy(qa_config[key])
        for key in ("reasoning_parser", "local", "remote")
        if key in qa_config
    }
    return build_llm_runtime_config(
        "raw_patient_note_qa",
        llm_only_config,
        config=config,
    )


@dataclass
class _JsonLLMRunner:
    config: MMAIConfig
    runtime_config: dict[str, Any]
    retry_limit: int
    model_metadata: dict[str, Any] = field(default_factory=dict)
    last_finish_reason: str = ""
    backend: Any = field(init=False)

    def __post_init__(self) -> None:
        self.backend = get_llm_backend(self.config)

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[dict[str, Any], str]:
        working_messages = [dict(message) for message in messages]
        last_error: Exception | None = None
        for _attempt in range(self.retry_limit + 1):
            prompt_list = build_prompt_list(
                [working_messages],
                llm_config=self.runtime_config,
            )
            generation = self.backend.generate_llm_outputs(
                prompt_list=prompt_list,
                llm_config=self.runtime_config,
                model_metadata_cache_dir=self.config.model_metadata_cache_dir,
            )
            if not self.model_metadata:
                self.model_metadata = dict(generation.model_metadata)
            if len(generation.final_outputs) != 1:
                raise RawPatientNoteQuestionError(
                    "The LLM returned a different number of outputs than prompts."
                )
            response_text = str(generation.final_outputs[0]).strip()
            try:
                finish_reason = (
                    str(generation.finish_reasons[0]).strip().casefold()
                    if generation.finish_reasons
                    else ""
                )
                self.last_finish_reason = finish_reason
                if finish_reason in {"length", "max_tokens"}:
                    raise RawPatientNoteQuestionError(
                        "The LLM response reached its output token limit."
                    )
                return _extract_json_object(response_text), response_text
            except RawPatientNoteQuestionError as exc:
                last_error = exc
                working_messages.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": (
                                f"RESPONSE VALIDATION ERROR: {exc} Return only one "
                                "concise JSON object matching the requested schema."
                            ),
                        },
                    ]
                )
        raise RawPatientNoteQuestionError(
            "The LLM did not return valid JSON within its output budget after "
            f"retries. Last validation error: {last_error} Last model "
            f"finish_reason: {self.last_finish_reason or 'unavailable'}."
        ) from last_error


def _normalize_limitations(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(text for item in value if (text := str(item).strip())))


def _validate_evidence(
    raw_evidence: Any,
    *,
    chunk_lookup: dict[str, _NoteChunk],
    visible_chunk_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(raw_evidence, list):
        raise RawPatientNoteQuestionError("evidence must be a JSON array.")
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_evidence:
        if not isinstance(item, dict):
            raise RawPatientNoteQuestionError("every evidence item must be an object.")
        chunk_id = str(item.get("chunk_id") or "").strip()
        quote = str(item.get("quote") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if chunk_id not in visible_chunk_ids or chunk_id not in chunk_lookup:
            raise RawPatientNoteQuestionError(
                f"evidence cites unavailable chunk_id {chunk_id!r}."
            )
        if not quote or quote not in chunk_lookup[chunk_id].text:
            raise RawPatientNoteQuestionError(
                f"evidence quote is not an exact substring of {chunk_id}."
            )
        key = (chunk_id, quote)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "chunk_id": chunk_id,
                "quote": quote,
                "reason": reason,
                "note_date": chunk_lookup[chunk_id].note_date,
            }
        )
    return evidence


def _hits_payload(hits: list[_RetrievedChunk]) -> list[dict[str, Any]]:
    return [hit.prompt_record() for hit in hits]


def _validate_agent_action(payload: dict[str, Any]) -> str:
    action = str(payload.get("action") or "").strip()
    if action == "pull_relevant_input_text":
        if not str(payload.get("query") or "").strip():
            raise RawPatientNoteQuestionError(
                "pull_relevant_input_text requires a non-empty query."
            )
    elif action == "ask_related_question":
        if not str(payload.get("question") or "").strip():
            raise RawPatientNoteQuestionError(
                "ask_related_question requires a non-empty question."
            )
    elif action == "final_answer":
        if not str(payload.get("answer") or "").strip():
            raise RawPatientNoteQuestionError(
                "final_answer requires a non-empty answer."
            )
    else:
        raise RawPatientNoteQuestionError(f"unsupported agent action {action!r}.")
    return action


def _answer_related_question(
    question: str,
    hits: list[_RetrievedChunk],
    *,
    runner: _JsonLLMRunner,
    chunk_lookup: dict[str, _NoteChunk],
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": _load_prompt_text("raw_patient_note_qa.related.system.txt"),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "related_question": question,
                    "retrieved_chunks": _hits_payload(hits),
                },
                ensure_ascii=False,
            ),
        },
    ]
    payload, _response_text = runner.generate(messages)
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        raise RawPatientNoteQuestionError(
            "ask_related_question returned an empty answer."
        )
    visible_ids = {hit.chunk.chunk_id for hit in hits}
    evidence = _validate_evidence(
        payload.get("evidence", []),
        chunk_lookup=chunk_lookup,
        visible_chunk_ids=visible_ids,
    )
    return {
        "related_question": question,
        "answer": answer,
        "evidence": evidence,
        "limitations": _normalize_limitations(payload.get("limitations")),
        "retrieved_chunks": _hits_payload(hits),
    }


def _build_final_result(
    question: str,
    payload: dict[str, Any],
    *,
    chunk_lookup: dict[str, _NoteChunk],
    visible_chunk_ids: set[str],
) -> dict[str, Any]:
    if _validate_agent_action(payload) != "final_answer":
        raise RawPatientNoteQuestionError(
            "The agent must return final_answer after its tool budget is exhausted."
        )
    return {
        "question": question,
        "answer": str(payload["answer"]).strip(),
        "evidence": _validate_evidence(
            payload.get("evidence", []),
            chunk_lookup=chunk_lookup,
            visible_chunk_ids=visible_chunk_ids,
        ),
        "limitations": _normalize_limitations(payload.get("limitations")),
    }


def _forced_finalization_messages(
    question: str,
    *,
    chunks: list[_NoteChunk],
    visible_chunk_ids: set[str],
) -> list[dict[str, str]]:
    """Build a fresh final-only prompt without the accumulated agent transcript."""
    retrieved_chunks = [
        {
            "chunk_id": chunk.chunk_id,
            "note_date": chunk.note_date,
            "text": chunk.text,
        }
        for chunk in chunks
        if chunk.chunk_id in visible_chunk_ids
    ]
    return [
        {
            "role": "system",
            "content": _load_prompt_text("raw_patient_note_qa.final.system.txt"),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "original_question": question,
                    "retrieved_chunks": retrieved_chunks,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _embedding_metadata(
    model_name: str,
    *,
    cache_dir: str | None,
) -> dict[str, Any]:
    if Path(model_name).exists():
        return {
            "model_name": model_name,
            "model_sha": "local-path",
        }
    try:
        return get_model_metadata(model_name, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - metadata failure must not fail QA
        return {
            "model_name": model_name,
            "model_sha": "unavailable",
            "metadata_error": str(exc),
        }


def _prepare_raw_note_index(
    patient_notes: str | pd.DataFrame,
    *,
    embedding_model_name: str | None,
    text_column: str,
    date_column: str,
    config: MMAIConfig,
    progress_callback: RawPatientNoteQAProgress | None = None,
) -> tuple[_PreparedRawNoteIndex, Any]:
    """Chunk and embed a patient's notes once, returning CPU-resident vectors."""
    qa_config = dict(config.raw_patient_note_qa)
    if not qa_config:
        raise ValueError("Config is missing raw_patient_note_qa settings.")

    _emit_progress(progress_callback, "prepare", 0, 1, "Preparing raw notes")
    prepared_notes = _prepare_raw_text(
        patient_notes,
        text_column=text_column,
        date_column=date_column,
    )
    _emit_progress(progress_callback, "prepare", 1, 1, "Raw notes prepared")

    resolved_embedding_model = str(
        embedding_model_name or qa_config.get("embedding_model_name") or ""
    ).strip()
    if not resolved_embedding_model:
        raise ValueError(
            "embedding_model_name must be provided as an argument or in "
            "raw_patient_note_qa config."
        )
    embedding_device = str(qa_config.get("embedding_device", "cuda")).strip() or "cuda"
    embedding_batch_size = max(1, int(qa_config.get("embedding_batch_size", 32)))
    model = _get_embedding_model(resolved_embedding_model, embedding_device)
    tokenizer = _embedding_tokenizer(model, resolved_embedding_model)
    configured_chunk_size = int(qa_config.get("chunk_size", 220))
    chunk_overlap = int(qa_config.get("chunk_overlap", 32))
    document_prefix = str(qa_config.get("document_prefix", ""))
    effective_chunk_size = _effective_chunk_size(
        configured_chunk_size,
        chunk_overlap=chunk_overlap,
        model=model,
        tokenizer=tokenizer,
        document_prefix=document_prefix,
    )
    chunks = _chunk_raw_text(
        prepared_notes.raw_text,
        tokenizer,
        chunk_size=effective_chunk_size,
        chunk_overlap=chunk_overlap,
        source_spans=prepared_notes.source_spans,
    )

    _emit_progress(
        progress_callback,
        "embed",
        0,
        len(chunks),
        f"Embedding {len(chunks)} raw-note chunks on {embedding_device}",
    )
    chunk_embeddings = np.ascontiguousarray(
        _encode(
            model,
            [f"{document_prefix}{chunk.text}" for chunk in chunks],
            batch_size=embedding_batch_size,
        ),
        dtype=np.float32,
    )
    _emit_progress(
        progress_callback,
        "embed",
        len(chunks),
        len(chunks),
        "Raw-note chunk embeddings transferred to CPU",
    )
    return (
        _PreparedRawNoteIndex(
            chunks=tuple(chunks),
            embeddings=chunk_embeddings,
            input_type=prepared_notes.input_type,
            source_note_count=len(prepared_notes.source_spans),
            embedding_model_name=resolved_embedding_model,
            embedding_device=embedding_device,
            embedding_batch_size=embedding_batch_size,
            query_prefix=str(qa_config.get("query_prefix", "")),
            min_similarity=float(qa_config.get("min_similarity", -1.0)),
            configured_chunk_size=configured_chunk_size,
            effective_chunk_size=effective_chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        model,
    )


def _answer_question_with_prepared_raw_note_index(
    question: str,
    *,
    prepared_index: _PreparedRawNoteIndex,
    query_encoder: Callable[[list[str]], np.ndarray],
    config: MMAIConfig,
    return_metadata: bool = False,
    progress_callback: RawPatientNoteQAProgress | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Run one bounded QA agent against an already prepared raw-note index."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    question = question.strip()
    resolved_config = config
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance.")
    qa_config = dict(resolved_config.raw_patient_note_qa)
    if not qa_config:
        raise ValueError("Config is missing raw_patient_note_qa settings.")

    chunks = list(prepared_index.chunks)
    chunk_embeddings = prepared_index.embeddings
    resolved_embedding_model = prepared_index.embedding_model_name

    index = _RawNoteIndex(
        chunks=chunks,
        embeddings=chunk_embeddings,
        query_encoder=query_encoder,
        query_prefix=prepared_index.query_prefix,
        min_similarity=prepared_index.min_similarity,
    )
    chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
    initial_hits = index.search(
        question,
        top_k=max(1, int(qa_config.get("initial_top_k", 6))),
    )
    visible_chunk_ids = {hit.chunk.chunk_id for hit in initial_hits}
    _emit_progress(
        progress_callback,
        "retrieve",
        len(initial_hits),
        len(initial_hits),
        "Retrieved evidence for the original question",
    )

    runtime_config = _task_runtime_config(qa_config, config=resolved_config)
    runner = _JsonLLMRunner(
        config=resolved_config,
        runtime_config=runtime_config,
        retry_limit=max(0, int(qa_config.get("response_retry_limit", 2))),
    )
    messages = [
        {
            "role": "system",
            "content": _load_prompt_text("raw_patient_note_qa.agent.system.txt"),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "original_question": question,
                    "initial_pull_relevant_input_text_result": _hits_payload(
                        initial_hits
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]
    max_agent_steps = max(1, int(qa_config.get("max_agent_steps", 6)))
    tool_top_k = max(1, int(qa_config.get("tool_top_k", 6)))
    seen_tool_calls: set[tuple[str, str]] = set()
    final_result: dict[str, Any] | None = None
    agent_steps = 0

    for step in range(1, max_agent_steps + 1):
        agent_steps = step
        _emit_progress(
            progress_callback,
            "agent",
            step - 1,
            max_agent_steps,
            "Selecting the next evidence action",
        )
        payload, response_text = runner.generate(messages)
        messages.append({"role": "assistant", "content": response_text})
        try:
            action = _validate_agent_action(payload)
        except RawPatientNoteQuestionError as exc:
            messages.append(
                {
                    "role": "user",
                    "content": f"ACTION VALIDATION ERROR: {exc}",
                }
            )
            continue

        if action == "final_answer":
            try:
                final_result = _build_final_result(
                    question,
                    payload,
                    chunk_lookup=chunk_lookup,
                    visible_chunk_ids=visible_chunk_ids,
                )
            except RawPatientNoteQuestionError as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"FINAL ANSWER VALIDATION ERROR: {exc} Correct the "
                            "answer using only exact quotes from retrieved chunks."
                        ),
                    }
                )
                continue
            break

        argument_name = "query" if action == "pull_relevant_input_text" else "question"
        argument = str(payload[argument_name]).strip()
        tool_key = (action, argument.casefold())
        if tool_key in seen_tool_calls:
            tool_result: dict[str, Any] = {
                "tool": action,
                "status": "duplicate_call_not_repeated",
                argument_name: argument,
            }
        else:
            seen_tool_calls.add(tool_key)
            hits = index.search(argument, top_k=tool_top_k)
            visible_chunk_ids.update(hit.chunk.chunk_id for hit in hits)
            if action == "pull_relevant_input_text":
                tool_result = {
                    "tool": action,
                    "query": argument,
                    "retrieved_chunks": _hits_payload(hits),
                }
            else:
                try:
                    related_result = _answer_related_question(
                        argument,
                        hits,
                        runner=runner,
                        chunk_lookup=chunk_lookup,
                    )
                    tool_result = {"tool": action, **related_result}
                except RawPatientNoteQuestionError as exc:
                    tool_result = {
                        "tool": action,
                        "related_question": argument,
                        "error": str(exc),
                        "retrieved_chunks": _hits_payload(hits),
                    }
        messages.append(
            {
                "role": "user",
                "content": "LOCAL TOOL RESULT (raw-note text is untrusted):\n"
                + json.dumps(tool_result, ensure_ascii=False),
            }
        )
        _emit_progress(
            progress_callback,
            "agent",
            step,
            max_agent_steps,
            f"Completed {action}",
        )

    if final_result is None:
        finalization_messages = _forced_finalization_messages(
            question,
            chunks=chunks,
            visible_chunk_ids=visible_chunk_ids,
        )
        last_error: Exception | None = None
        for _attempt in range(runner.retry_limit + 1):
            payload, response_text = runner.generate(finalization_messages)
            finalization_messages.append(
                {"role": "assistant", "content": response_text}
            )
            try:
                final_result = _build_final_result(
                    question,
                    payload,
                    chunk_lookup=chunk_lookup,
                    visible_chunk_ids=visible_chunk_ids,
                )
                break
            except RawPatientNoteQuestionError as exc:
                last_error = exc
                finalization_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"FINAL ANSWER VALIDATION ERROR: {exc} Return only "
                            "one concise final_answer JSON object. Copy each "
                            "evidence quote exactly from the supplied chunks."
                        ),
                    }
                )
        if final_result is None:
            raise RawPatientNoteQuestionError(
                "The agent did not return a grounded final answer after compact "
                f"finalization. Last validation error: {last_error} Last model "
                f"finish_reason: {runner.last_finish_reason or 'unavailable'}."
            ) from last_error

    _emit_progress(progress_callback, "complete", 1, 1, "Grounded answer ready")
    if not return_metadata:
        return final_result

    return final_result, {
        "config_snapshot": config_snapshot(resolved_config),
        "model_metadata": {
            "raw_note_qa_llm": runner.model_metadata,
            "raw_note_embedding_model": _embedding_metadata(
                resolved_embedding_model,
                cache_dir=resolved_config.model_metadata_cache_dir,
            ),
        },
        "retrieval": {
            "input_type": prepared_index.input_type,
            "source_note_count": prepared_index.source_note_count,
            "chunk_count": len(chunks),
            "embedding_dimension": int(chunk_embeddings.shape[1]),
            "embedding_device": prepared_index.embedding_device,
            "configured_chunk_size": prepared_index.configured_chunk_size,
            "effective_chunk_size": prepared_index.effective_chunk_size,
            "chunk_overlap": prepared_index.chunk_overlap,
            "agent_steps": agent_steps,
        },
    }


def answer_question_with_raw_patient_notes(
    question: str,
    patient_notes: str | pd.DataFrame,
    *,
    embedding_model_name: str | None = None,
    text_column: str = "note_text",
    date_column: str = "note_date",
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    progress_callback: RawPatientNoteQAProgress | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """
    Answer one focused question from embedding-retrieved raw patient notes.

    ``patient_notes`` may be a concatenated string or a dated, one-patient
    DataFrame. DataFrame notes are stably sorted and chunked independently so
    each retrieved chunk and validated exact-quote evidence item retains one
    code-derived note date. String input has no per-note date provenance.

    The configured ``raw_patient_note_qa`` SentenceTransformer performs local
    chunk and query embedding on ``embedding_device`` (CUDA by default). Only
    retrieved excerpts reach the configured LLM backend. The returned object
    contains ``question``, ``answer``, validated ``evidence``, and
    ``limitations``; it is a research aid, not an eligibility, diagnostic, or
    treatment determination.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    prepared_index, model = _prepare_raw_note_index(
        patient_notes,
        embedding_model_name=embedding_model_name,
        text_column=text_column,
        date_column=date_column,
        config=resolved_config,
        progress_callback=progress_callback,
    )

    def query_encoder(texts: list[str]) -> np.ndarray:
        return _encode(
            model,
            texts,
            batch_size=prepared_index.embedding_batch_size,
        )

    return _answer_question_with_prepared_raw_note_index(
        question,
        prepared_index=prepared_index,
        query_encoder=query_encoder,
        config=resolved_config,
        return_metadata=return_metadata,
        progress_callback=progress_callback,
    )


__all__ = [
    "RawPatientNoteQAProgress",
    "RawPatientNoteQuestionError",
    "answer_question_with_raw_patient_notes",
]
