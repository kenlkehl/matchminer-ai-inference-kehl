"""A queue of independent LLM requests that keeps a steady number in flight.

The trial stages (screening, cohort extraction, merging, boilerplate classification,
synthetic protocol generation) send many independent requests and handle each answer on its
own. Sending them in fixed batches and waiting for each batch to return lets one slow answer
hold up the rest of its batch, so most of a batch's slots sit idle while its slowest requests
finish. :class:`RequestQueue` instead sends each request on its own through the backend's
request session (``open_request_session``, which remote backends have) and starts the next
waiting request as soon as any returns, keeping ``chunk_size * chunks_in_flight`` requests in
flight; the endpoint pool still limits each server. Backends without a request session (local
vLLM, which batches its own requests, and test doubles) get batches of ``chunk_size``, up to
``chunks_in_flight`` at once.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Hashable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time
from typing import Any

from .prompt_rendering import build_prompt_list

# (item, answer, reasoning, finish_reason, response metadata or None)
Record = Callable[[Any, str, str, str, "dict[str, Any] | None"], None]


class RequestQueue:
    """Requests grouped by LLM runtime, sent so that a steady number stay in flight.

    ``runtimes`` maps each group to its runtime config; waiting requests of an earlier group are
    sent first. :meth:`add` queues an item with its chat messages. :meth:`run` sends them all and
    calls ``record(item, answer, reasoning, finish_reason, response_metadata)`` as each answer
    returns, one answer at a time (never from two threads at once). Items that ``record`` adds (retries, a trial's next step) go
    ahead of the items still waiting in their group, so a started trial finishes early.
    """

    def __init__(self, backend: Any, runtimes: dict[Hashable, dict[str, Any]], *, chunk_size: int,
                 chunks_in_flight: int, model_metadata_cache_dir: str | None = None):
        self.backend, self.runtimes = backend, runtimes
        self.chunk_size, self.chunks_in_flight = chunk_size, chunks_in_flight
        self.cache_dir = model_metadata_cache_dir
        self.waiting: dict[Hashable, deque] = {group: deque() for group in runtimes}
        self.added: dict[Hashable, list] | None = None  # items added by record, while it runs
        self.model_metadata: dict[str, Any] = {}

    def add(self, item: Any, messages: list[dict[str, str]], group: Hashable = None) -> None:
        if self.added is None:
            self.waiting[group].append((item, messages))
        else:
            self.added.setdefault(group, []).append((item, messages))

    def __len__(self) -> int:
        return sum(map(len, self.waiting.values()))

    def _deliver(self, record: Record, answers) -> None:
        """Record answers (item, answer, reasoning, finish_reason, provenance) that returned together.

        The items they add go ahead of their groups' waiting items, in the order they were added.
        """
        self.added = {}
        try:
            for answer in answers:
                record(*answer)
        finally:
            added, self.added = self.added, None
            for group, entries in added.items():
                self.waiting[group].extendleft(reversed(entries))

    def _next(self) -> tuple[Hashable, deque] | None:
        """The first group with waiting requests, and its queue (a group may be None)."""
        return next(((group, queue) for group, queue in self.waiting.items() if queue), None)

    def run(self, record: Record, progress: Callable[[], None] | None = None,
            progress_seconds: float = 10.0) -> dict[str, Any]:
        """Send every queued request; return the model metadata.

        ``progress()`` runs after each batch, or at most every ``progress_seconds`` while
        requests are sent one by one, and once at the end.
        """
        if not len(self):
            return self.model_metadata
        if hasattr(self.backend, "open_request_session"):
            from .remote_inference import _run_sync
            _run_sync(lambda: self._stream(record, progress, progress_seconds))
        else:
            self._batches(record, progress)
        if progress is not None:
            progress()
        return self.model_metadata

    async def _stream(self, record, progress, progress_seconds):
        window = self.chunk_size * self.chunks_in_flight
        first = next(iter(self.runtimes.values()))
        # The session's threads hold the requests waiting for an endpoint, so it needs one per request.
        session_config = {**first, "pool_max_workers": max(window, int(first.get("pool_max_workers", 0)))}
        async with self.backend.open_request_session(session_config, self.cache_dir) as session:
            self.model_metadata = dict(session.model_metadata)
            finished: asyncio.Queue = asyncio.Queue()
            in_flight, reported = 0, time.monotonic()
            while in_flight or len(self):
                while in_flight < window and (picked := self._next()) is not None:
                    group, queue = picked
                    item, messages = queue.popleft()
                    runtime = self.runtimes[group]
                    prompt = build_prompt_list([messages], llm_config=runtime)[0]
                    task = asyncio.ensure_future(session.generate(prompt, runtime))
                    task.add_done_callback(lambda task, item=item: finished.put_nowait((item, task)))
                    in_flight += 1
                done = [await finished.get()]
                while not finished.empty():
                    done.append(finished.get_nowait())
                in_flight -= len(done)
                self._deliver(record, ((item, result.summary, result.reasoning, result.finish_reason,
                                        result.response_metadata)
                                       for item, result in ((item, task.result()) for item, task in done)))
                if progress is not None and time.monotonic() - reported >= progress_seconds:
                    progress()
                    reported = time.monotonic()

    def _batches(self, record, progress):
        with ThreadPoolExecutor(max_workers=self.chunks_in_flight) as executor:
            running = {}
            while running or len(self):
                while len(running) < self.chunks_in_flight and (picked := self._next()) is not None:
                    group, queue = picked
                    chunk = [queue.popleft() for _ in range(min(self.chunk_size, len(queue)))]
                    runtime = self.runtimes[group]
                    running[executor.submit(
                        self.backend.generate_llm_outputs,
                        prompt_list=build_prompt_list([messages for _, messages in chunk], llm_config=runtime),
                        llm_config=runtime, model_metadata_cache_dir=self.cache_dir)] = chunk
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    chunk = running.pop(future)
                    generation = future.result()
                    self.model_metadata = {k: v for k, v in generation.model_metadata.items() if k != "responses"}
                    provenance = {item["row_idx"]: item for item in generation.model_metadata.get("responses", [])}
                    self._deliver(record, ((item, answer, reasoning, finish_reason, provenance.get(position))
                                           for position, ((item, _), answer, reasoning, finish_reason) in enumerate(zip(
                                               chunk, generation.final_outputs, generation.reasoning_outputs,
                                               generation.finish_reasons, strict=True))))
                if progress is not None:
                    progress()


__all__ = ["RequestQueue"]
