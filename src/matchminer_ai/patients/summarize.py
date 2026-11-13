"""Patient summarization logic."""

from __future__ import annotations

import asyncio
import heapq
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd
from transformers import AutoTokenizer

from matchminer_ai.llm.backends import (
    build_llm_runtime_config,
    get_llm_backend,
)
from matchminer_ai._metadata import package_metadata
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.prompt_rendering import Prompt

from .checkpoints import (
    ChunkJournal,
    load_chunk_journal,
    load_prepared_chunks,
    load_round_checkpoints,
    prepare_checkpoint_dir,
    save_prepared_chunks,
)
from .postprocess import postprocess_patient_summaries
from .prepare import prepare_patient_notes
from .prompt_builder import (
    PromptWorkItem,
    build_prompt_worker,
    prep_prompt_pool,
    shutdown_prompt_pool,
)


logger = logging.getLogger(__name__)

_SCHEDULING_MODES = ("continuous", "rounds")
# The continuous scheduler rereads the endpoints' capacity at least this often,
# so a server added to a registry gets work without waiting for an answer.
_SCHEDULER_TICK_SECONDS = 5.0
_PROGRESS_LOG_SECONDS = 300.0


def validate_existing_summaries(
    existing_summaries: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and normalize existing patient summary state."""
    required_columns = ["patient_id", "patient_summary"]
    missing = [
        column
        for column in required_columns
        if column not in existing_summaries.columns
    ]
    if missing:
        raise ValueError(
            "existing summaries input must include columns "
            "'patient_id' and 'patient_summary'. Missing: "
            f"{', '.join(missing)}"
        )

    normalized = existing_summaries.copy()
    normalized["patient_id"] = normalized["patient_id"].astype(str)
    normalized["patient_summary"] = normalized["patient_summary"].where(
        normalized["patient_summary"].notna(),
        None,
    )
    return normalized.drop_duplicates(subset=["patient_id"], keep="last")


def _build_existing_summary_lookup(
    existing_summaries: pd.DataFrame | None,
) -> dict[str, str | None]:
    if existing_summaries is None:
        return {}
    normalized = validate_existing_summaries(existing_summaries)
    return cast(
        dict[str, str | None],
        normalized.set_index("patient_id")["patient_summary"].to_dict(),
    )


def _patient_chunks(prepared_chunks: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Each patient's prepared chunks, in chunk order."""
    chunks: dict[str, list[dict[str, Any]]] = {}
    ordered = prepared_chunks.sort_values(
        ["patient_id", "chunk_index"], kind="mergesort"
    )
    for row in ordered.to_dict("records"):
        chunks.setdefault(str(row["patient_id"]), []).append(row)
    return chunks


def _scheduling_mode(patient_config: dict[str, Any]) -> str:
    mode = str(patient_config.get("scheduling", "continuous"))
    if mode not in _SCHEDULING_MODES:
        raise ValueError(
            "patient.scheduling must be one of "
            f"{', '.join(_SCHEDULING_MODES)}; got {mode!r}."
        )
    return mode


def _sends_requests_one_by_one(backend: Any) -> bool:
    """Whether the backend can send each prompt on its own (remote backends).

    A local engine batches its prompts itself and has no request session. The
    class is checked, not the instance, so a mock does not qualify by accident.
    """
    return callable(getattr(type(backend), "open_request_session", None))


@dataclass
class _SerialState:
    """Each patient's place in their chain of chunks, plus the run's tallies.

    ``summaries`` holds each patient's latest accepted summary: a supplied
    existing summary until the first chunk's, then each chunk's in turn. Both
    schedulers record results through :meth:`record`, and resuming adopts
    checkpointed results through :meth:`restore`, so the chain, the quality
    counts and the debug attempts advance the same way under either.
    """

    chunks: dict[str, list[dict[str, Any]]]
    summaries: dict[str, str | None]
    debug: bool
    next_chunk: dict[str, int] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)
    failed: set[str] = field(default_factory=set)
    attempts: dict[str, list] = field(default_factory=dict)
    quality: dict[str, int] = field(
        default_factory=lambda: {"attempts": 0, "rejected_attempts": 0}
    )

    def remaining(self, patient_id: str) -> int:
        """Chunks this patient still needs; none once their summary failed."""
        if patient_id in self.failed:
            return 0
        return len(self.chunks[patient_id]) - self.next_chunk.get(patient_id, 0)

    def counts(self) -> dict[str, int]:
        """Patients and chunks done, failed and to go; counts only, no IDs."""
        done = sum(self.next_chunk.values()) + len(self.failed)
        to_go = sum(self.remaining(patient_id) for patient_id in self.chunks)
        total = sum(len(chunks) for chunks in self.chunks.values())
        return {
            "patients": len(self.chunks),
            "patients_done": sum(not self.remaining(p) for p in self.chunks),
            "patients_failed": len(self.failed),
            "chunks": total,
            "chunks_done": done,
            "chunks_to_go": to_go,
            "chunks_skipped": total - done - to_go,
        }

    def restore(self, chunk_index: int, row: dict[str, Any]) -> bool:
        """Adopt a checkpointed chunk result if it continues its patient's chain.

        A result for another chunk, or one whose prior summary is not the
        chain's current summary (it came from a different chain, for example
        one an older run rebuilt), is not adopted. A first chunk's prior is not
        compared, so supplied existing summaries may change between attempts,
        as they always could.
        """
        patient_id = str(row["patient_id"])
        if (
            patient_id not in self.chunks
            or patient_id in self.failed
            or chunk_index != self.next_chunk.get(patient_id, 0)
            or chunk_index >= len(self.chunks[patient_id])
        ):
            return False
        if chunk_index and row["prior_summary"] != self.summaries.get(patient_id):
            return False
        self._advance(
            patient_id,
            chunk_index,
            row["patient_answer_text"],
            row["patient_reasoning_text"],
            str(row["finish_reason"]),
        )
        return True

    def record(
        self,
        patient_id: str,
        chunk_index: int,
        summary: Any,
        reasoning: Any,
        finish_reason: Any,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply one chunk's checked result; return its checkpoint row."""
        self.quality["attempts"] += len(attempts)
        self.quality["rejected_attempts"] += sum(
            bool(attempt["retry_codes"]) for attempt in attempts
        )
        if self.debug:
            self.attempts.setdefault(patient_id, []).extend(attempts)
        # Persist the final summary, not the reasoning trace, so it becomes the
        # prior summary for the patient's next chunk.
        row = {
            "patient_id": patient_id,
            "chunk_index": chunk_index,
            "prior_summary": self.summaries.get(patient_id),
            "patient_reasoning_text": str(reasoning),
            "patient_answer_text": str(summary),
            "finish_reason": str(finish_reason),
        }
        self._advance(
            patient_id,
            chunk_index,
            row["patient_answer_text"],
            row["patient_reasoning_text"],
            row["finish_reason"],
        )
        return row

    def _advance(self, patient_id, chunk_index, summary, reasoning, finish_reason):
        if finish_reason == "error":
            # Every attempt failed: this patient's later chunks are skipped.
            self.failed.add(patient_id)
            return
        self.summaries[patient_id] = summary
        self.reasoning[patient_id] = reasoning
        self.next_chunk[patient_id] = chunk_index + 1


def _restore_checkpoints(state: _SerialState, checkpoint_dir: str | Path) -> bool:
    """Rebuild each patient's chain from the checkpoint directory.

    Round files from runs before the journal come first, in round order, then
    the journal in the order it was written; each result is adopted only where
    it continues its patient's chain, so old rounds, a journal and a mix of
    both resume alike. Returns whether any saved result was found.
    """
    rounds = load_round_checkpoints(checkpoint_dir)
    journal = load_chunk_journal(checkpoint_dir)
    saved = [
        (round_idx, row)
        for round_idx, frame in sorted(rounds.items())
        for row in frame.to_dict("records")
    ] + [(row["chunk_index"], row) for row in journal]
    adopted = sum(state.restore(chunk_index, row) for chunk_index, row in saved)
    if saved:
        logger.info(
            "Restored %d saved chunk result(s) (%d patient(s) failed earlier) "
            "from %d round file(s) and %d journal line(s); %d did not continue "
            "a patient's chain and were not used.",
            adopted,
            len(state.failed),
            len(rounds),
            len(journal),
            len(saved) - adopted,
        )
    return bool(saved)


def _build_prompt_list(
    round_df: pd.DataFrame,
    *,
    current_summaries: dict[str, str | None],
    prompt_pool: Any,
    n_prompt_workers: int,
) -> tuple[list[Prompt], list[str]]:
    work_items: list[PromptWorkItem] = []
    round_patient_ids: list[str] = []
    for row_idx, (_, row) in enumerate(round_df.iterrows()):
        patient_id = str(row["patient_id"])
        round_patient_ids.append(patient_id)
        work_items.append(
            PromptWorkItem(
                row_idx=row_idx,
                prior_summary_text=current_summaries.get(patient_id),
                first_date=str(row["first_date"]),
                last_date=str(row["last_date"]),
                chunk_text=str(row["chunk_text"]),
            )
        )

    chunksize = max(1, len(work_items) // (n_prompt_workers * 4)) if work_items else 1
    return (
        list(prompt_pool.map(build_prompt_worker, work_items, chunksize=chunksize)),
        round_patient_ids,
    )


def _summarize_in_rounds(
    state: _SerialState,
    *,
    backend: Any,
    llm_config: dict[str, Any],
    prompt_pool: Any,
    n_prompt_workers: int,
    journal: ChunkJournal | None,
    metadata_cache_dir: str | None,
    max_attempts: int,
    trace_dir: str | None,
) -> dict[str, Any]:
    """Summarize every patient's next chunk together, one round at a time.

    Round N holds the Nth chunk of every patient who still needs it, sent as one
    batch, so each round waits for its slowest patient. This is the scheduler
    for a local engine, which batches prompts itself.
    """
    from .summary_retry import generate_checked_summaries

    model_metadata: dict[str, Any] = {}
    n_rounds = max((len(chunks) for chunks in state.chunks.values()), default=0)
    for round_idx in range(n_rounds):
        round_patient_ids = sorted(
            patient_id
            for patient_id in state.chunks
            if state.remaining(patient_id)
            and state.next_chunk.get(patient_id, 0) == round_idx
        )
        if not round_patient_ids:
            continue
        round_df = pd.DataFrame(
            [state.chunks[patient_id][round_idx] for patient_id in round_patient_ids]
        )
        prompt_list, round_patient_ids = _build_prompt_list(
            round_df,
            current_summaries=state.summaries,
            prompt_pool=prompt_pool,
            n_prompt_workers=n_prompt_workers,
        )
        generation, attempts = generate_checked_summaries(
            backend,
            prompt_list,
            llm_config=llm_config,
            metadata_cache_dir=metadata_cache_dir,
            max_attempts=max_attempts,
            trace_dir=trace_dir,
        )
        if not model_metadata:
            model_metadata = generation.model_metadata
        rows = [
            state.record(patient_id, round_idx, summary, reasoning, finish, attempts[i])
            for i, (patient_id, summary, reasoning, finish) in enumerate(
                zip(
                    round_patient_ids,
                    generation.final_outputs,
                    generation.reasoning_outputs,
                    generation.finish_reasons,
                    strict=False,
                )
            )
        ]
        failed_in_round_count = sum(row["finish_reason"] == "error" for row in rows)
        if failed_in_round_count:
            logger.warning(
                "Filtering %d patient(s) whose summary failed every attempt "
                "in round %d.",
                failed_in_round_count,
                round_idx + 1,
            )
        # A round becomes resumable only after inference finishes for the batch.
        if journal is not None:
            journal.append(rows)
    return model_metadata


def _request_capacity(session: Any) -> int:
    capacity = getattr(session, "request_capacity", None)
    value = capacity() if callable(capacity) else getattr(session, "capacity", 1)
    return max(1, int(value))


def _log_progress(state: _SerialState, in_flight: int, limit: int) -> None:
    counts = state.counts()
    logger.info(
        "Patient summarization: %d of %d patient(s) finished (%d failed), %d in "
        "progress; %d of %d chunk(s) done, %d to go, %d skipped after a "
        "failure; %d chunk(s) in flight of %d allowed.",
        counts["patients_done"],
        counts["patients"],
        counts["patients_failed"],
        in_flight,
        counts["chunks_done"],
        counts["chunks"],
        counts["chunks_to_go"],
        counts["chunks_skipped"],
        in_flight,
        limit,
    )


async def _summarize_continuously(
    state: _SerialState,
    *,
    backend: Any,
    llm_config: dict[str, Any],
    prompt_pool: Any,
    n_prompt_workers: int,
    journal: ChunkJournal | None,
    metadata_cache_dir: str | None,
    max_attempts: int,
    trace_dir: str | None,
) -> dict[str, Any]:
    """Advance each patient's chain on its own, keeping the endpoints full.

    As soon as a chunk's checked summary is final it is journaled and the
    patient's next prompt is built (in the prompt pool, off the event loop) and
    sent, with up to the session's request capacity in flight; there is no
    round barrier. Waiting patients start longest remaining chain first: with
    similar request times, the longest chain is the run's critical path.

    An exception in one chain cancels none of the others: no new chunk starts,
    the chunks already in flight finish and are journaled, and then the first
    exception is raised, as a round's batch call would raise it.
    """
    from .summary_retry import generate_checked_summary

    loop = asyncio.get_running_loop()
    ready = [
        (-state.remaining(patient_id), patient_id)
        for patient_id in state.chunks
        if state.remaining(patient_id)
    ]
    heapq.heapify(ready)
    builders = ThreadPoolExecutor(
        max_workers=n_prompt_workers, thread_name_prefix="mmai-summary-prompts"
    )
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmai-summary-journal")
    running: set[asyncio.Task] = set()
    failures: list[BaseException] = []

    async def advance(session: Any, patient_id: str) -> str:
        chunk_index = state.next_chunk.get(patient_id, 0)
        chunk = state.chunks[patient_id][chunk_index]
        item = PromptWorkItem(
            row_idx=0,
            prior_summary_text=state.summaries.get(patient_id),
            first_date=str(chunk["first_date"]),
            last_date=str(chunk["last_date"]),
            chunk_text=str(chunk["chunk_text"]),
        )
        (prompt,) = await loop.run_in_executor(
            builders, prompt_pool.map, build_prompt_worker, [item]
        )
        generation, attempts = await generate_checked_summary(
            session,
            prompt,
            llm_config=llm_config,
            max_attempts=max_attempts,
            trace_dir=trace_dir,
        )
        row = state.record(
            patient_id,
            chunk_index,
            generation.final_outputs[0],
            generation.reasoning_outputs[0],
            generation.finish_reasons[0],
            attempts,
        )
        if journal is not None:
            await loop.run_in_executor(writer, journal.append, [row])
        return patient_id

    try:
        async with backend.open_request_session(llm_config, metadata_cache_dir) as session:
            limit = _request_capacity(session)
            counts = state.counts()
            logger.info(
                "Summarizing %d patient(s) continuously: %d chunk(s) to go "
                "(%d done earlier), up to %d at once, longest remaining chain "
                "first.",
                counts["patients"] - counts["patients_done"],
                counts["chunks_to_go"],
                counts["chunks_done"],
                limit,
            )
            logged = time.monotonic()
            try:
                while True:
                    limit = _request_capacity(session)
                    while ready and not failures and len(running) < limit:
                        _, patient_id = heapq.heappop(ready)
                        running.add(asyncio.create_task(advance(session, patient_id)))
                    if not running:
                        break
                    done, running = await asyncio.wait(
                        running,
                        timeout=_SCHEDULER_TICK_SECONDS,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task.exception() is not None:
                            failures.append(task.exception())
                            continue
                        patient_id = task.result()
                        if state.remaining(patient_id):
                            heapq.heappush(
                                ready, (-state.remaining(patient_id), patient_id)
                            )
                    if time.monotonic() - logged >= _PROGRESS_LOG_SECONDS:
                        _log_progress(state, len(running), limit)
                        logged = time.monotonic()
            finally:
                # Cancellation or an interrupt stops every chain at once.
                for task in running:
                    task.cancel()
                await asyncio.gather(*running, return_exceptions=True)
            _log_progress(state, 0, limit)
            if failures:
                logger.warning(
                    "Patient summarization stopped after %d chunk(s) raised; the "
                    "chunks in flight finished and were saved, and the %d "
                    "patient(s) still waiting continue when the run resumes.",
                    len(failures),
                    len(ready),
                )
                raise failures[0]
            return dict(session.model_metadata)
    finally:
        builders.shutdown(wait=False, cancel_futures=True)
        # Finish a journal write in progress before the caller closes the file.
        writer.shutdown(wait=True)


def summarize_patient_notes(
    notes: pd.DataFrame,
    config: MMAIConfig | None = None,
    *,
    existing_summaries: pd.DataFrame | None = None,
    checkpoint_dir: str | Path | None = None,
    return_qc: bool = False,
) -> (
    tuple[pd.DataFrame, dict[str, Any]]
    | tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]
):
    """
    Summarize longitudinal patient notes using serial chunk-based updates.

    Each patient's notes are split into chunks, summarized in order: every
    chunk's prompt carries the summary accepted for the patient's previous
    chunk. ``patient.scheduling`` chooses how patients share the backend.
    ``continuous`` (the default) sends each patient's next chunk as soon as
    their previous one is final, with the backend's request capacity in
    flight. A local engine, which batches prompts itself, and ``rounds`` send
    every patient's Nth chunk together, round by round. Both give the same
    prompts, retries and results.

    Parameters
    ----------
    notes : pd.DataFrame
        Note-level input. One row per note.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        note_text : str
            Full note text.
        note_date : str or datetime
            Date of the note.
    existing_summaries : pd.DataFrame, optional
        Optional patient-level prior summaries used as the starting state for
        serial updates. Patients with no usable new notes retain their existing
        summary, subject to the usual summary postprocessing and filtering.

        Expected columns
        ----------------
        patient_id : str
            Unique patient identifier.
        patient_summary : str
            Existing full patient summary text to update.
    checkpoint_dir : str or pathlib.Path, optional
        Directory used to save progress during patient summarization: each
        chunk result is saved as soon as it is final. Reusing the same
        directory on retry continues each patient from their last saved chunk,
        including from round files written before the chunk journal. The
        caller is responsible for supplying a directory specific to this
        input and run.
    return_qc : bool, optional
        When True, also return a QC report DataFrame for this summarization step.
    """
    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")

    patient_config = dict(resolved_config.patient)
    scheduling = _scheduling_mode(patient_config)
    runtime_patient_config = build_llm_runtime_config(
        "patient",
        patient_config,
        config=resolved_config,
    )

    prepared_chunks = None
    if checkpoint_dir is not None:
        prepare_checkpoint_dir(checkpoint_dir)
        prepared_chunks = load_prepared_chunks(checkpoint_dir)

    if prepared_chunks is None:
        # Tokenization and chunking are only repeated when no reusable checkpoint exists.
        tokenizer_name = runtime_patient_config.get(
            "tokenizer_name",
            runtime_patient_config["model_name"],
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        prepared_chunks = prepare_patient_notes(
            notes,
            tokenizer,
            chunk_size=int(patient_config["chunk_size"]),
            chunk_overlap=int(patient_config["chunk_overlap"]),
        )
        if checkpoint_dir is not None:
            save_prepared_chunks(checkpoint_dir, prepared_chunks)

    existing_summary_lookup = _build_existing_summary_lookup(existing_summaries)
    # If the caller provided an existing summary, it is the prior summary for the
    # patient's first chunk; each accepted chunk summary then replaces it.
    state = _SerialState(
        chunks=_patient_chunks(prepared_chunks),
        summaries=dict(existing_summary_lookup),
        debug=bool(resolved_config.debug_mode),
    )
    resumed = (
        _restore_checkpoints(state, checkpoint_dir)
        if checkpoint_dir is not None
        else False
    )

    backend = get_llm_backend(resolved_config)
    continuous = scheduling == "continuous" and _sends_requests_one_by_one(backend)
    if scheduling == "continuous" and not continuous:
        logger.info(
            "Summarizing patients in rounds: this backend batches its own prompts."
        )
    pending = [patient_id for patient_id in state.chunks if state.remaining(patient_id)]
    n_rounds = max((len(chunks) for chunks in state.chunks.values()), default=0)
    if resumed and not pending:
        logger.info(
            "All %d patient summarization round(s) are already complete (%d "
            "chunk(s)); using saved results.",
            n_rounds,
            state.counts()["chunks"],
        )
    elif resumed and not continuous:
        logger.info(
            "Resuming patient summarization at round %d of %d.",
            min(state.next_chunk.get(patient_id, 0) for patient_id in pending) + 1,
            n_rounds,
        )

    model_metadata: dict[str, Any] = {}
    prompt_pool = None
    journal = None
    n_prompt_workers = max(
        1,
        int(patient_config.get("prompt_build_workers", min(os.cpu_count() or 4, 32))),
    )
    schedule = dict(
        backend=backend,
        llm_config=runtime_patient_config,
        n_prompt_workers=n_prompt_workers,
        metadata_cache_dir=resolved_config.model_metadata_cache_dir,
        max_attempts=int(patient_config.get("summary_quality_max_attempts", 3)),
        trace_dir=patient_config.get("summary_quality_trace_dir"),
    )
    try:
        if pending:
            prompt_pool = prep_prompt_pool(
                patient_config=runtime_patient_config,
                n_workers=n_prompt_workers,
            )
            if checkpoint_dir is not None:
                journal = ChunkJournal(checkpoint_dir)
            if continuous:
                from matchminer_ai.llm.remote_inference import _run_sync

                model_metadata = _run_sync(
                    lambda: _summarize_continuously(
                        state, prompt_pool=prompt_pool, journal=journal, **schedule
                    )
                )
            else:
                model_metadata = _summarize_in_rounds(
                    state, prompt_pool=prompt_pool, journal=journal, **schedule
                )
    finally:
        if prompt_pool is not None:
            shutdown_prompt_pool(prompt_pool)
        if journal is not None:
            journal.close()

    current_summaries = state.summaries
    failed_patient_ids = state.failed
    # Collapse chunk-level work to one final row for each summarized patient.
    final_rows = prepared_chunks[["patient_id"]].drop_duplicates().copy()
    total_patient_count = int(final_rows["patient_id"].nunique())
    # Retain supplied summaries when there is no new content to process.
    final_rows = pd.concat(
        [final_rows, pd.DataFrame({"patient_id": list(existing_summary_lookup)})],
        ignore_index=True,
    ).drop_duplicates(subset=["patient_id"])
    final_rows = final_rows.loc[
        ~final_rows["patient_id"].astype(str).isin(failed_patient_ids)
    ].copy()
    final_rows["patient_answer_text"] = final_rows["patient_id"].map(current_summaries)
    if resolved_config.debug_mode:
        # These columns preserve final-chunk debug traces without feeding them
        # back into the serial patient summary state.
        final_rows["patient_reasoning_text"] = final_rows["patient_id"].map(
            state.reasoning
        )
        final_rows["patient_summary_attempts"] = final_rows["patient_id"].map(
            state.attempts
        )
    final_rows = final_rows.dropna(subset=["patient_answer_text"]).copy()

    patient_count_before_cleaning = int(final_rows["patient_id"].nunique())
    final_rows, removed_patient_ids = postprocess_patient_summaries(
        final_rows,
        resolved_config,
    )

    metadata = {
        "package": package_metadata(),
        "config_snapshot": config_snapshot(resolved_config),
        "model_metadata": model_metadata,
        "summary_quality": state.quality,
    }

    if return_qc:
        from matchminer_ai._qc.common import build_qc_artifact
        from matchminer_ai._qc.patients import patient_summary_qc_report

        noninformative_summary_qc_artifact = build_qc_artifact(
            metric="patients_dropped_noninformative_summary",
            ids=sorted(removed_patient_ids),
            denominator=patient_count_before_cleaning,
        )
        failed_inference_qc_artifact = build_qc_artifact(
            metric="patients_failed_inference",
            ids=sorted(failed_patient_ids),
            denominator=total_patient_count,
        )
        qc_report = patient_summary_qc_report(
            final_rows,
            noninformative_summary_qc_artifact=(noninformative_summary_qc_artifact),
            failed_inference_qc_artifact=failed_inference_qc_artifact,
            config=resolved_config,
        )
        return final_rows, metadata, qc_report
    return final_rows, metadata


__all__ = [
    "summarize_patient_notes",
    "validate_existing_summaries",
]
