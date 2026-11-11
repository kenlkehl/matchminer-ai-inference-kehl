"""
Async dispatcher for vLLM inference against a dynamic, fault-tolerant
pool of OpenAI-compatible servers (vLLM in OpenAI server mode).

Moved from the training repository's remote_vllm_pool.py (2026-11-06) so that
training's teacher labeling, note generation and summaries share the inference
package's transport (matchminer_ai.labeling builds on it). Behaviour, flags and
request bodies are unchanged.

Designed for the GCP-orchestrated training pipeline where:
- the set of healthy servers changes over time (spot instances come and go);
- per-server concurrency must back off when servers get overloaded;
- every work item must eventually succeed ("retry until done"); a hard
  ceiling of max_attempts protects against true poison-pill items.

An orchestrator (for example a GCP fleet controller) writes the canonical list of
healthy server URLs to a JSON file; DynamicServerRegistry re-reads that
file periodically. A fixed snapshot of URLs (--server_urls) is also
supported for ad-hoc runs.

Callers provide work items, a work coroutine, and a shard_writer; this
module owns the queue, semaphores, retries, and dispatch.

Servers JSON file schema (written atomically by the orchestrator):
{
  "updated_at": "2026-05-16T12:00:00Z",
  "ready": true,
  "servers": [
    {"url": "http://10.0.0.5:8000/v1", "instance": "worker-a", "gpus": [0],
     "kind": "remote"},
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, List, Optional, Sequence, Tuple

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from .scheduler_control import Controller, now, parse_metrics, apply_tuning


def _read_json_file(path):
    with open(path) as stream:
        return json.load(stream)


class OutputValidationError(ValueError):
    """A completed model response failed validation, not server capacity.

    Retried under the usual per-item attempt budget without changing the
    concurrency limit. Callers should use this for invalid output schemas;
    json.JSONDecodeError is recognized directly as well.
    """


# -------------------------
# CLI helpers
# -------------------------

def add_remote_cli_args(parser: argparse.ArgumentParser) -> None:
    """Attach the standard remote-mode CLI flags to a script's argparse."""
    g = parser.add_argument_group("Remote vLLM pool (GCP orchestrator)")
    g.add_argument(
        "--server_urls", type=str, default=None,
        help="Comma-separated URLs of existing vLLM servers "
             "(e.g. 'http://10.0.0.5:8000/v1,http://10.0.0.6:8000/v1'). "
             "Static snapshot; for a dynamic set use --server_urls_file.",
    )
    g.add_argument(
        "--server_urls_file", type=str, default=None,
        help="Path to a JSON file maintained by gcp_vllm_orchestrator.py "
             "containing the current list of healthy server URLs. The file "
             "is re-read every --server_urls_refresh seconds.",
    )
    g.add_argument(
        "--server_urls_refresh", type=float, default=15.0,
        help="Seconds between re-reads of --server_urls_file.",
    )
    g.add_argument(
        "--max_concurrent_per_server", type=int, default=100,
        help="Per-server concurrency ceiling (adaptive; output-validation errors do not cause backoff).",
    )
    g.add_argument(
        "--concurrency-control-file", default=None,
        help="Optional live JSON file with max_concurrent_per_server and an optional "
             "per_server mapping of endpoint URLs to limits; values must be positive "
             "and no greater than the CLI ceiling. Reloaded with --server_urls_refresh.",
    )
    g.add_argument(
        "--concurrency_success_threshold", type=int, default=3,
        help="Successful requests needed before increasing a server's adaptive "
             "concurrency limit. Below half of max_concurrent_per_server the "
             "limit doubles per streak (slow start); above it grows by "
             "--concurrency_increase_step.",
    )
    g.add_argument(
        "--scheduler-adaptive-concurrency", action=argparse.BooleanOptionalAction, default=False,
        help="Adjust each endpoint's concurrency using vLLM queue/cache/preemption and throughput metrics, in addition to request-error backoff.",
    )
    g.add_argument(
        "--initial-concurrent-per-server", type=int, default=128,
        help="Starting concurrency for scheduler adaptation, bounded by the per-server ceiling.",
    )
    g.add_argument(
        "--scheduler-metrics-file", default=None,
        help="Optional live JSON snapshot of scheduler metrics and adaptive concurrency limits.",
    )
    g.add_argument(
        "--concurrency_increase_step", type=int, default=2,
        help="Adaptive concurrency slots to add after each clean success "
             "streak once the limit is past slow-start (>= max_limit/2).",
    )
    g.add_argument(
        "--concurrency_backoff_factor", type=float, default=0.5,
        help="Multiplier applied to a server's adaptive concurrency limit after "
             "a request error other than output validation.",
    )
    g.add_argument(
        "--results_per_shard", "--results-per-shard", type=int, default=200,
        help="Number of completed items per shard written by the pool.",
    )
    g.add_argument(
        "--request_timeout", type=float, default=600.0,
        help="Per-request timeout in seconds (passed to work_fn).",
    )
    g.add_argument(
        "--api_key_env", "--api-key-env", type=str, default="OPENAI_API_KEY",
        help="Environment variable containing the endpoint API key. If it is "
             "unset, use 'not-needed' for ordinary local vLLM servers.",
    )
    g.add_argument(
        "--max_attempts", type=int, default=200,
        help="Per-item max retries before recording an ERROR placeholder. "
             "Set very high (default 200) for 'retry until done' behavior; "
             "the ceiling exists only as a poison-pill guard.",
    )


def parse_static_urls(server_urls: Optional[str]) -> List[str]:
    if not server_urls:
        return []
    return [u.strip() for u in server_urls.split(",") if u.strip()]


# -------------------------
# Adaptive concurrency
# -------------------------

class AdaptiveSemaphore:
    """
    A condition-variable semaphore whose capacity can shrink (on failure)
    and grow back (on sustained success). Slow-start + AIMD:
      - failure: multiplicative decrease by `backoff_factor` (min 1).
      - success: after `success_threshold` consecutive successes since the
        last decrease, grow the limit. While `limit < max_limit/2` the limit
        doubles (slow start); once at or above that threshold it grows
        additively by `increase_step` up to max_limit.
    """

    def __init__(
        self,
        max_limit: int,
        *,
        success_threshold: int = 3,
        increase_step: int = 2,
        backoff_factor: float = 0.5,
    ):
        self.max_limit = max(1, int(max_limit))
        self.limit = self.max_limit
        self.in_flight = 0
        self._cond = asyncio.Condition()
        self._success_streak = 0
        self._success_threshold = max(1, int(success_threshold))
        self._increase_step = max(1, int(increase_step))
        self._backoff_factor = min(0.99, max(0.01, float(backoff_factor)))
        self._slow_start_ceiling = max(2, self.max_limit // 2)

    async def acquire(self):
        async with self._cond:
            while self.in_flight >= self.limit:
                await self._cond.wait()
            self.in_flight += 1

    async def try_acquire(self):
        """Reserve a ready request without blocking dispatch to other servers."""
        async with self._cond:
            if self.in_flight >= self.limit:
                return False
            self.in_flight += 1
            return True

    async def set_max_limit(self, maximum, *, reset=False):
        """Resize without cancelling active requests or undoing error backoff."""
        async with self._cond:
            maximum = max(1, int(maximum))
            self.limit = maximum if reset or self.limit == self.max_limit else min(self.limit, maximum)
            self.max_limit = maximum
            self._slow_start_ceiling = max(2, maximum // 2)
            self._cond.notify_all()

    async def release(self, *, success: Optional[bool]):
        """Release a slot; None leaves the limit and success streak unchanged."""
        async with self._cond:
            self.in_flight = max(0, self.in_flight - 1)
            if success:
                self._success_streak += 1
                if (self._success_streak >= self._success_threshold
                        and self.limit < self.max_limit):
                    if self.limit < self._slow_start_ceiling:
                        new_limit = min(self.max_limit, self.limit * 2)
                    else:
                        new_limit = min(self.max_limit,
                                        self.limit + self._increase_step)
                    self.limit = new_limit
                    self._success_streak = 0
            elif success is False:
                old = self.limit
                self.limit = max(1, int(self.limit * self._backoff_factor))
                self._success_streak = 0
                if self.limit != old:
                    # capacity shrank; some waiters may need to re-check
                    pass
            self._cond.notify_all()

    def stats(self) -> dict:
        return {
            "limit": self.limit,
            "max_limit": self.max_limit,
            "in_flight": self.in_flight,
            "streak": self._success_streak,
        }


# -------------------------
# Server registry
# -------------------------

@dataclass
class ServerEntry:
    url: str
    client: AsyncOpenAI
    semaphore: AdaptiveSemaphore
    instance: str = ""
    retired: bool = False
    # rolling stats for status logs
    ok_count: int = 0
    err_count: int = 0
    first_seen_at: float = field(default_factory=time.monotonic)
    output_validation_err_count: int = 0
    error_types: dict[str, int] = field(default_factory=dict)
    last_error: Optional[dict] = None


class DynamicServerRegistry:
    """
    Tracks the current set of vLLM servers. Two modes:
      - static: built from a list of URLs (server_urls); never changes.
      - dynamic: watches a JSON file written by gcp_vllm_orchestrator.py
        and reconciles every `refresh_seconds`.

    Adds new servers when they appear; marks vanished servers `retired`
    (in-flight calls drain, no new dispatches sent). The reconciler is
    started by `start()` and stopped by `stop()`.
    """

    def __init__(
        self,
        *,
        max_concurrent_per_server: int,
        request_timeout: float,
        concurrency_success_threshold: int = 3,
        concurrency_increase_step: int = 2,
        concurrency_backoff_factor: float = 0.5,
        servers_file: Optional[str] = None,
        static_urls: Optional[List[str]] = None,
        refresh_seconds: float = 15.0,
        api_key: str = "not-needed",
        concurrency_file: Optional[str] = None,
        scheduler_adaptive_concurrency: bool = False,
        initial_concurrent_per_server: int = 128,
        scheduler_metrics_file: Optional[str] = None,
    ):
        if not servers_file and not static_urls:
            raise ValueError("Either servers_file or static_urls must be set.")
        self._servers_file = servers_file
        self._static_urls = list(static_urls) if static_urls else []
        self._refresh_seconds = max(1.0, float(refresh_seconds))
        self._max_concurrent = max_concurrent_per_server
        self._effective_max_concurrent = max_concurrent_per_server
        self._per_server_max_concurrent: dict[str, int] = {}
        self._concurrency_file = concurrency_file
        self._concurrency_success_threshold = concurrency_success_threshold
        self._concurrency_increase_step = concurrency_increase_step
        self._concurrency_backoff_factor = concurrency_backoff_factor
        self._request_timeout = request_timeout
        self._api_key = api_key or "not-needed"
        self._scheduler_enabled = scheduler_adaptive_concurrency
        self._scheduler_initial = max(1, initial_concurrent_per_server)
        self._scheduler_metrics_file = scheduler_metrics_file
        self._scheduler_controllers: dict[str, Controller] = {}
        self._scheduler_ceilings: dict[str, int] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._pending_tuning = None
        self._applied_tuning_revision = None

        self._entries: dict[str, ServerEntry] = {}
        self._lock = asyncio.Lock()
        self._change_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._reconcile_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        # Idempotent: safe to call multiple times.
        if self._reconcile_task is not None or self._scheduler_task is not None:
            return
        # Always reconcile once synchronously so callers see initial set.
        await self._reconcile_once()
        if self._servers_file or self._concurrency_file:
            self._reconcile_task = asyncio.create_task(self._reconcile_loop())
        if self._scheduler_enabled:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        self._stop_event.set()
        for task in (self._reconcile_task, self._scheduler_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Close all clients
        async with self._lock:
            for entry in self._entries.values():
                try:
                    await entry.client.close()
                except Exception:
                    pass

    async def wait_for_change(self, timeout: float) -> bool:
        """Wait until the active server set changes, or timeout."""
        self._change_event.clear()
        try:
            await asyncio.wait_for(self._change_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def active_entries(self) -> List[ServerEntry]:
        async with self._lock:
            return [e for e in self._entries.values() if not e.retired]

    def endpoint_stats(self) -> List[dict]:
        """Every endpoint's admission and outcome counters, retired ones included (label client processes)."""
        return [dict(url=entry.url, retired=entry.retired, **entry.semaphore.stats(),
                     successful_requests=entry.ok_count, request_errors=entry.err_count,
                     validation_errors=entry.output_validation_err_count, error_types=dict(entry.error_types),
                     last_error=entry.last_error) for entry in self._entries.values()]

    def request_capacity(self) -> int:
        """Maximum active slots; read on the registry's owning event loop."""
        return sum(
            entry.semaphore.max_limit for entry in self._entries.values()
            if not entry.retired
        ) or self._effective_max_concurrent

    async def _read_servers_file(self) -> Optional[List[dict]]:
        try:
            data = await asyncio.to_thread(_read_json_file, self._servers_file)
            return list(data.get("servers", []))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            # Mid-write; try again next tick
            return None

    def _make_client(self, url: str) -> AsyncOpenAI:
        # Long generation requests and local checkpoint/tokenization work can
        # leave idle connections beyond a vLLM proxy's keep-alive timeout. A
        # fresh connection avoids reset races being mistaken for GPU overload;
        # active requests still share the same bounded asynchronous pool.
        client = DefaultAsyncHttpxClient(
            limits=httpx.Limits(max_connections=max(1000, self._max_concurrent),
                               max_keepalive_connections=0),
            timeout=self._request_timeout + 60,
        )
        return AsyncOpenAI(
            base_url=url,
            api_key=self._api_key,
            timeout=self._request_timeout + 60,
            max_retries=0,  # we own retry policy
            http_client=client,
        )

    async def _reconcile_once(self) -> None:
        if self._concurrency_file:
            try:
                control = await asyncio.to_thread(_read_json_file, self._concurrency_file)
                maximum = control["max_concurrent_per_server"]
                if type(maximum) is not int or not 1 <= maximum <= self._max_concurrent:
                    raise ValueError("live concurrency must be an integer within the CLI ceiling")
                per_server = control.get("per_server", {})
                if not isinstance(per_server, dict) or any(
                    not isinstance(url, str) or not url
                    or type(limit) is not int or not 1 <= limit <= self._max_concurrent
                    for url, limit in per_server.items()
                ):
                    raise ValueError("per_server must map endpoint URLs to integers within the CLI ceiling")
                self._effective_max_concurrent = maximum
                self._per_server_max_concurrent = per_server
                self._pending_tuning = control.get('tuning')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"[remote_vllm_pool] retaining concurrency {self._effective_max_concurrent}: {exc}")
        if self._servers_file:
            servers = await self._read_servers_file()
            if servers is None:
                # File doesn't exist yet; treat as empty set, retry next tick.
                desired = {}
            else:
                desired = {
                    s["url"]: s
                    for s in servers
                    if isinstance(s, dict) and s.get("url")
                }
        else:
            desired = {url: {"url": url, "instance": ""} for url in self._static_urls}

        changed = False
        async with self._lock:
            # Add new
            for url, info in desired.items():
                maximum = self._per_server_max_concurrent.get(url, self._effective_max_concurrent)
                if self._scheduler_enabled:
                    if self._scheduler_ceilings.get(url) != maximum:
                        old = self._scheduler_controllers.get(url)
                        initial = old.cap if old is not None else self._scheduler_initial
                        self._scheduler_controllers[url] = Controller(min(initial, maximum), maximum)
                        self._scheduler_ceilings[url] = maximum
                    maximum = self._scheduler_controllers[url].cap
                if url in self._entries:
                    semaphore = self._entries[url].semaphore
                    if semaphore.max_limit != maximum:
                        await semaphore.set_max_limit(maximum)
                        print(f"[remote_vllm_pool] concurrency ceiling for {url}: {semaphore.max_limit}")
                        changed = True
                    if self._entries[url].retired:
                        # Server came back; un-retire and reset stats
                        self._entries[url].retired = False
                        self._entries[url].first_seen_at = time.monotonic()
                        if self._scheduler_enabled:
                            self._scheduler_controllers[url].reset_observations()
                        changed = True
                    continue
                client = self._make_client(url)
                self._entries[url] = ServerEntry(
                    url=url,
                    client=client,
                    semaphore=AdaptiveSemaphore(
                        maximum,
                        success_threshold=self._concurrency_success_threshold,
                        increase_step=self._concurrency_increase_step,
                        backoff_factor=self._concurrency_backoff_factor,
                    ),
                    instance=info.get("instance", ""),
                )
                changed = True

            # Retire vanished
            for url, entry in self._entries.items():
                if url not in desired and not entry.retired:
                    entry.retired = True
                    changed = True

            if self._scheduler_enabled and self._pending_tuning:
                try:
                    event = apply_tuning(self._scheduler_controllers, self._scheduler_ceilings,
                                         self._pending_tuning, self._applied_tuning_revision)
                    if event:
                        for change in event['changes']:
                            await self._entries[change['endpoint']].semaphore.set_max_limit(change['new_cap'], reset=True)
                        self._applied_tuning_revision = event['revision']
                        print('[concurrency_tuning] ' + json.dumps(event), flush=True)
                        changed = True
                except (ValueError, KeyError, TypeError) as error:
                    print(f'[concurrency_tuning] retaining prior targets: {error}', flush=True)

        if changed:
            self._change_event.set()

    async def _sample_scheduler_once(self, client):
        """Tune endpoints independently; unavailable metrics retain error-based adaptation."""
        async def sample(entry):
            controller = self._scheduler_controllers[entry.url]
            result = {"url": entry.url}
            try:
                response = await client.get(entry.url.rstrip('/').removesuffix('/v1') + '/metrics')
                response.raise_for_status()
                metrics = parse_metrics(response.text)
                # A live registry update may retire or reconfigure this endpoint during I/O.
                if entry.retired or self._scheduler_controllers.get(entry.url) is not controller:
                    return None
                event = controller.observe(metrics)
                await entry.semaphore.set_max_limit(controller.cap)
                if event is not None:
                    print('[adaptive_concurrency] ' + json.dumps({"url": entry.url, **event}), flush=True)
                    self._change_event.set()
                result['metrics'] = metrics
            except Exception as exc:
                controller.reset_observations()
                result['metrics_error'] = str(exc) or type(exc).__name__
                result['metrics_error_type'] = type(exc).__name__
            result.update(entry.semaphore.stats(), configured_ceiling=self._scheduler_ceilings[entry.url],
                          scheduler_ceiling=controller.upper, successful_requests=entry.ok_count,
                          request_errors=entry.err_count, validation_errors=entry.output_validation_err_count,
                          error_types=dict(entry.error_types), last_error=entry.last_error)
            return result

        entries = await self.active_entries()
        results = await asyncio.gather(*(sample(entry) for entry in entries))
        snapshot = dict(updated_at=now(), scheduler_adaptive_concurrency=True,
                        servers=[result for result in results if result is not None])
        if self._scheduler_metrics_file:
            def write_snapshot():
                path = Path(self._scheduler_metrics_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + '.tmp')
                temporary.write_text(json.dumps(snapshot, indent=2) + '\n')
                temporary.replace(path)
            await asyncio.to_thread(write_snapshot)
        return snapshot

    async def _scheduler_loop(self):
        async with httpx.AsyncClient(timeout=15, trust_env=False,
                                     headers={'Authorization': 'Bearer ' + self._api_key}) as client:
            while not self._stop_event.is_set():
                try:
                    await self._sample_scheduler_once(client)
                except Exception as exc:
                    print(f'[adaptive_concurrency] metrics update failed: {exc}', flush=True)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._refresh_seconds)
                except asyncio.TimeoutError:
                    continue

    async def _reconcile_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._refresh_seconds,
                    )
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    await self._reconcile_once()
                except Exception as e:
                    print(f"[remote_vllm_pool] reconcile error: {e}")
        except asyncio.CancelledError:
            raise


# -------------------------
# Result buffering / sharding
# -------------------------

WorkFn = Callable[[AsyncOpenAI, Any], Awaitable[Any]]
ShardWriter = Callable[[List[Tuple[Any, Any]], int], None]


class ResultBuffer:
    """Accumulates completed results and flushes them via shard_writer."""

    def __init__(
        self,
        *,
        results_per_shard: int,
        shard_writer: ShardWriter,
        starting_shard_idx: int = 0,
        requeue_fn: Optional[Callable[[List[Tuple[Any, Any]]], Awaitable[None]]] = None,
        write_retry_attempts: int = 5,
        write_retry_initial_sleep: float = 1.0,
    ):
        self._buf: List[Tuple[Any, Any]] = []
        self._lock = asyncio.Lock()
        self._results_per_shard = max(1, int(results_per_shard))
        self._writer = shard_writer
        self._next_idx = int(starting_shard_idx)
        self.total_written = 0
        # Unique item IDs that have been successfully added (i.e. work_fn
        # returned a result or a poison-pill placeholder was recorded). Used
        # by run_pool to identify items that never finished so it can write
        # placeholders for them rather than wait forever.
        self.completed_ids: set = set()
        # Called with the list of (item_id, result) tuples whose shard write
        # permanently failed; the callback should put them back on the work
        # queue. Without this, failed-shard items would be silently lost
        # (they're already in completed_ids by the time _call_writer runs).
        self._requeue_fn = requeue_fn
        self._write_retry_attempts = max(1, int(write_retry_attempts))
        self._write_retry_initial_sleep = max(0.1, float(write_retry_initial_sleep))

    @property
    def buf_size(self) -> int:
        """Items added but not yet flushed to disk."""
        return len(self._buf)

    async def add(self, item_id: Any, result: Any) -> None:
        flush_payload: Optional[List[Tuple[Any, Any]]] = None
        flush_idx = 0
        async with self._lock:
            self._buf.append((item_id, result))
            self.completed_ids.add(item_id)
            if len(self._buf) >= self._results_per_shard:
                flush_payload = self._buf
                self._buf = []
                flush_idx = self._next_idx
                self._next_idx += 1
        if flush_payload is not None:
            await self._call_writer(flush_payload, flush_idx)

    async def flush(self) -> None:
        async with self._lock:
            if not self._buf:
                return
            flush_payload = self._buf
            self._buf = []
            flush_idx = self._next_idx
            self._next_idx += 1
        await self._call_writer(flush_payload, flush_idx)

    async def _call_writer(self, payload: List[Tuple[Any, Any]], idx: int) -> None:
        # Run sync writer in a thread so disk I/O doesn't block the loop.
        # Retry transient failures (e.g. EMFILE under FD pressure) with
        # exponential backoff. If still failing, hand the payload back to
        # the requeue callback so the items can be re-inferenced; otherwise
        # those rows would be silently lost (they're already in
        # completed_ids by the time we get here).
        last_err: Optional[BaseException] = None
        for attempt in range(self._write_retry_attempts):
            try:
                await asyncio.to_thread(self._writer, payload, idx)
                self.total_written += len(payload)
                return
            except Exception as e:
                last_err = e
                if attempt + 1 < self._write_retry_attempts:
                    sleep_s = self._write_retry_initial_sleep * (2 ** attempt)
                    print(
                        f"[remote_vllm_pool] shard writer failed for "
                        f"shard_idx={idx} (attempt {attempt + 1}/"
                        f"{self._write_retry_attempts}): {e!r}; "
                        f"retrying in {sleep_s:.1f}s"
                    )
                    await asyncio.sleep(sleep_s)
                    continue

        ids = [iid for iid, _ in payload]
        async with self._lock:
            for iid in ids:
                self.completed_ids.discard(iid)
        if self._requeue_fn is not None:
            print(
                f"[remote_vllm_pool] CRITICAL: shard writer failed for "
                f"shard_idx={idx} after {self._write_retry_attempts} attempts "
                f"({last_err!r}); requeuing {len(payload)} item(s) for retry."
            )
            try:
                await self._requeue_fn(payload)
                return
            except Exception as e:
                print(
                    f"[remote_vllm_pool] requeue callback failed for "
                    f"shard_idx={idx}: {e!r}; the {len(payload)} item(s) "
                    f"are now lost."
                )
                raise
        else:
            print(
                f"[remote_vllm_pool] CRITICAL: shard writer failed for "
                f"shard_idx={idx} after {self._write_retry_attempts} attempts "
                f"and no requeue_fn was set; dropping {len(payload)} item(s): "
                f"{last_err!r}"
            )
            raise last_err  # type: ignore[misc]


# -------------------------
# Dispatch core
# -------------------------

async def _dispatch_one(
    entry: ServerEntry,
    item_id: Any,
    payload: Any,
    attempt: int,
    *,
    work_fn: WorkFn,
    work_queue: asyncio.Queue,
    results: ResultBuffer,
    max_attempts: int,
    error_placeholder: Callable[[Any, Exception], Any],
) -> None:
    success: Optional[bool] = False
    try:
        result = await work_fn(entry.client, payload)
        await results.add(item_id, result)
        entry.ok_count += 1
        success = True
    except Exception as e:
        entry.err_count += 1
        kind = type(e).__name__
        entry.error_types[kind] = entry.error_types.get(kind, 0) + 1
        # Error messages and response bodies can contain clinical inputs. Keep
        # only transport/type metadata in shared operational telemetry.
        entry.last_error = dict(at=now(), type=kind, http_status=getattr(e, 'status_code', None),
                               cause_type=type(e.__cause__).__name__ if e.__cause__ else None)
        if isinstance(e, (OutputValidationError, json.JSONDecodeError)):
            # Content errors require another generation, not fewer requests.
            # They are neutral feedback, not successful work or server overload.
            success = None
            entry.output_validation_err_count += 1
        if attempt + 1 >= max_attempts:
            # Poison pill — record an error placeholder so the run can complete.
            print(f"[remote_vllm_pool] item {item_id!r} exhausted "
                  f"{max_attempts} attempts ({e!r}); writing error placeholder.")
            await results.add(item_id, error_placeholder(payload, e))
            # Keep the failure classification, including neutral feedback for
            # exhausted output-validation retries, but don't requeue.
        else:
            # Requeue with a tiny stagger so the queue doesn't hot-spin
            # when every server is failing.
            await asyncio.sleep(min(0.1 * (2 ** min(attempt, 6)), 5.0))
            await work_queue.put((item_id, payload, attempt + 1))
    finally:
        await entry.semaphore.release(success=success)


async def _registry_supervisor(
    registry: DynamicServerRegistry,
    *,
    work_queue: asyncio.Queue,
    work_fn: WorkFn,
    results: ResultBuffer,
    max_attempts: int,
    error_placeholder: Callable[[Any, Exception], Any],
    stop_event: asyncio.Event,
) -> None:
    """Assign ready work to the endpoint with the most proportional headroom.

    Idle endpoints hold no queue items or semaphore slots. Fast servers therefore
    receive replacement work before slower, more occupied servers, even when the
    producer supplies only a few requests at a time.
    """
    from .dispatch import least_loaded_endpoint

    running = set()
    wake = asyncio.Event()
    last_used, announced = {}, set()
    sequence, dispatched = 0, 0
    pending = None

    async def watch_registry():
        while not stop_event.is_set():
            await registry.wait_for_change(timeout=2.0)
            wake.set()

    def finished(task):
        running.discard(task)
        wake.set()

    watcher = asyncio.create_task(watch_registry())
    try:
        while not stop_event.is_set():
            if pending is None:
                try:
                    pending = work_queue.get_nowait()
                except asyncio.QueueEmpty:
                    try:
                        pending = await asyncio.wait_for(work_queue.get(), timeout=.5)
                    except asyncio.TimeoutError:
                        continue
            wake.clear()
            active = [entry for entry in await registry.active_entries() if not entry.retired]
            for entry in active:
                if entry.url not in announced:
                    print(f"[remote_vllm_pool] + server {entry.url} ({entry.instance})")
                    announced.add(entry.url)
            selected = least_loaded_endpoint(
                (entry.url, entry.semaphore.in_flight, entry.semaphore.limit, last_used.get(entry.url, -1))
                for entry in active)
            if selected is None:
                try:
                    await asyncio.wait_for(wake.wait(), timeout=.5)
                except asyncio.TimeoutError:
                    pass
                continue
            entry = next(entry for entry in active if entry.url == selected)
            if not await entry.semaphore.try_acquire():
                continue  # A metrics update changed admission while choosing.
            if entry.retired or stop_event.is_set():
                await entry.semaphore.release(success=None)
                continue
            item_id, payload, attempt = pending
            pending = None
            last_used[entry.url] = sequence
            sequence += 1
            task = asyncio.create_task(_dispatch_one(
                entry, item_id, payload, attempt, work_fn=work_fn,
                work_queue=work_queue, results=results, max_attempts=max_attempts,
                error_placeholder=error_placeholder))
            running.add(task)
            task.add_done_callback(finished)
            dispatched += 1
            if dispatched % 16 == 0:
                await asyncio.sleep(0)
    except BaseException:
        for task in list(running):
            task.cancel()
        raise
    finally:
        if pending is not None:
            work_queue.put_nowait(pending)
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        if running:
            await asyncio.gather(*list(running), return_exceptions=True)


async def _status_logger(
    registry: DynamicServerRegistry,
    work_queue: asyncio.Queue,
    results: ResultBuffer,
    *,
    total_items: int,
    interval: float,
    stop_event: asyncio.Event,
) -> None:
    start = time.monotonic()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass
        try:
            entries = await registry.active_entries()
            pending = work_queue.qsize()
            done = results.total_written
            buf = results.buf_size
            in_flight_sum = sum(e.semaphore.in_flight for e in entries)
            completed = len(results.completed_ids)
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            per_server = ", ".join(
                f"{e.instance or e.url.split('//')[-1].split('/')[0]}"
                f"[{e.semaphore.in_flight}/{e.semaphore.limit}"
                f" ok={e.ok_count} err={e.err_count}"
                f" output_err={e.output_validation_err_count}]"
                for e in entries
            ) or "(no active servers)"
            # "missing" = items not in any tracked state. If non-zero with
            # an empty queue, the pool can never make progress without help.
            missing = max(0, total_items - done - pending - in_flight_sum - buf)
            print(
                f"[pool] done={done}/{total_items} pending={pending} "
                f"in_flight={in_flight_sum} buf={buf} missing={missing} "
                f"completed={completed} rate={rate:.1f}/s "
                f"servers: {per_server}"
            )
        except Exception as e:
            print(f"[pool] status logger error: {e}")


# -------------------------
# Public entry point
# -------------------------

def _default_error_placeholder(payload: Any, err: Exception) -> Any:
    return f"ERROR: {err!r}"


class RequestSession:
    """Persistent adaptive pool for a dynamically expanding dependency graph.

    Uses the same dispatch, retry and per-server semaphores as run_pool, but
    delivers each result immediately instead of waiting for a shard or batch.
    The caller persists validated nodes before releasing their children.
    """

    def __init__(self, registry, work_fn, *, max_attempts=4):
        self.registry, self.work_fn = registry, work_fn
        self.max_attempts = max_attempts
        self.queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.futures = {}
        self.dispatch_tasks = set()
        self.submitted = self.completed = 0

    async def __aenter__(self):
        await self.registry.start()
        self.supervisor = asyncio.create_task(_registry_supervisor(
            self.registry, work_queue=self.queue, work_fn=self._tracked_work,
            results=self, max_attempts=self.max_attempts,
            error_placeholder=_default_error_placeholder, stop_event=self.stop_event,
        ))
        self.logger = asyncio.create_task(self._log_progress())
        return self

    async def _tracked_work(self, client, payload):
        task = asyncio.current_task()
        self.dispatch_tasks.add(task)
        task.add_done_callback(self.dispatch_tasks.discard)
        return await self.work_fn(client, payload)

    async def submit(self, payload):
        item_id = self.submitted
        self.submitted += 1
        future = asyncio.get_running_loop().create_future()
        self.futures[item_id] = future
        self.queue.put_nowait((item_id, payload, 0))
        return await future

    async def add(self, item_id, result):
        future = self.futures.pop(item_id, None)
        if future is not None:
            self.completed += 1
            if not future.done():
                future.set_result(result)

    async def _log_progress(self):
        while not self.stop_event.is_set():
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                entries = await self.registry.active_entries()
                slots = sum(e.semaphore.in_flight for e in entries)
                print(f"[pool] graph completed={self.completed}/{self.submitted} "
                      f"queued={self.queue.qsize()} in_flight={slots}", flush=True)

    async def __aexit__(self, *exc):
        self.stop_event.set()
        if exc[0] is not None:
            tasks = list(self.dispatch_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.supervisor.cancel()
        try:
            await asyncio.wait_for(self.supervisor, timeout=30)
        except asyncio.CancelledError:
            self.supervisor.cancel()
            await asyncio.gather(self.supervisor, return_exceptions=True)
            if exc[0] is None:
                raise
        except asyncio.TimeoutError:
            self.supervisor.cancel()
            await asyncio.gather(self.supervisor, return_exceptions=True)
        finally:
            self.logger.cancel()
            await asyncio.gather(self.logger, return_exceptions=True)
            for future in self.futures.values():
                future.cancel()
            await self.registry.stop()


async def run_pool(
    work_items: Iterable[Tuple[Any, Any]],
    work_fn: WorkFn,
    registry: DynamicServerRegistry,
    *,
    shard_writer: ShardWriter,
    results_per_shard: int = 200,
    starting_shard_idx: int = 0,
    max_attempts: int = 200,
    status_interval: float = 30.0,
    error_placeholder: Callable[[Any, Exception], Any] = _default_error_placeholder,
    min_servers_warn_secs: float = 300.0,
    stale_giveup_secs: float = 1800.0,
) -> int:
    """
    Drive `work_fn` over `work_items` using a fault-tolerant, dynamic pool
    of vLLM servers from `registry`. Persists results in batches of
    `results_per_shard` via `shard_writer(payload, shard_idx)`.

    work_items: iterable of (item_id, payload). item_id is any hashable;
    payload is opaque to this module and passed straight to work_fn.

    work_fn(client, payload) -> result (any). Raise on transient failure.

    Returns the total number of items written (== len(work_items) modulo
    poison-pill placeholders).
    """
    await registry.start()

    work_queue: asyncio.Queue = asyncio.Queue()
    payload_by_id: dict = {}
    total = 0
    for item_id, payload in work_items:
        work_queue.put_nowait((item_id, payload, 0))
        payload_by_id[item_id] = payload
        total += 1
    if total == 0:
        print("[pool] no work items to process.")
        return 0
    all_ids = set(payload_by_id.keys())

    async def _requeue_failed_shard(payload: List[Tuple[Any, Any]]) -> None:
        # Put items back on the work_queue with attempt=0 so a permanent
        # shard-write failure (e.g. exhausted FD-pressure retries) doesn't
        # silently lose them. Items have already been removed from
        # completed_ids by _call_writer.
        for item_id, _result in payload:
            payload_for_item = payload_by_id.get(item_id)
            if payload_for_item is None:
                # Should not happen: items in a shard payload were always
                # enqueued in this run.
                print(f"[pool] cannot requeue unknown item_id {item_id!r}")
                continue
            await work_queue.put((item_id, payload_for_item, 0))

    results = ResultBuffer(
        results_per_shard=results_per_shard,
        shard_writer=shard_writer,
        starting_shard_idx=starting_shard_idx,
        requeue_fn=_requeue_failed_shard,
    )

    stop_event = asyncio.Event()
    supervisor_task = asyncio.create_task(
        _registry_supervisor(
            registry,
            work_queue=work_queue,
            work_fn=work_fn,
            results=results,
            max_attempts=max_attempts,
            error_placeholder=error_placeholder,
            stop_event=stop_event,
        )
    )
    status_task = asyncio.create_task(
        _status_logger(
            registry, work_queue, results,
            total_items=total,
            interval=status_interval,
            stop_event=stop_event,
        )
    )

    # Wait until everything has been written. We check periodically.
    # `last_progress` measures the last time results.total_written advanced.
    # If it stays put for `stale_giveup_secs` AND the queue is empty, we
    # force-flush any buffered shard and (if needed) write error placeholders
    # for items the pool never recorded — otherwise the loop would hang
    # forever on a tiny number of lost items.
    # We exit when every item has been processed (success or poison-pill
    # placeholder) — i.e. len(completed_ids) >= total. This works regardless
    # of whether shards have been flushed yet, which matters for callers
    # like summarize_patients.py that use a giant results_per_shard and
    # rely on the final flush() in `finally` to write the whole round.
    last_progress = time.monotonic()
    last_warned_at = last_progress
    last_completed = 0
    gave_up = False
    try:
        while len(results.completed_ids) < total and not gave_up:
            await asyncio.sleep(2.0)
            current_completed = len(results.completed_ids)
            if current_completed != last_completed:
                last_completed = current_completed
                last_progress = time.monotonic()
                last_warned_at = last_progress
                continue
            now = time.monotonic()
            stale = now - last_progress
            if stale > 0 and now - last_warned_at >= min_servers_warn_secs:
                active = await registry.active_entries()
                pending = work_queue.qsize()
                in_flight_sum = sum(e.semaphore.in_flight for e in active)
                completed = len(results.completed_ids)
                print(
                    f"[pool] no progress for {stale:.0f}s; "
                    f"{len(active)} active server(s), "
                    f"{pending} items pending in queue, "
                    f"{in_flight_sum} slots in flight, "
                    f"{completed}/{total} items completed."
                )
                last_warned_at = now
            if stale >= stale_giveup_secs and work_queue.qsize() == 0:
                # Flush whatever is in the buffer — these are real completions
                # that just haven't reached the per-shard threshold.
                print(f"[pool] giving up after {stale:.0f}s without progress; "
                      f"flushing buffered shard ({results.buf_size} items).")
                try:
                    await results.flush()
                except Exception as e:
                    print(f"[pool] final flush failed: {e!r}")
                # Any item that never reached results.add is truly lost
                # (e.g. swallowed shard-writer exception earlier in the run).
                # Write error placeholders so the pipeline can advance.
                missing = all_ids - results.completed_ids
                if missing:
                    sample = list(missing)[:5]
                    print(
                        f"[pool] writing error placeholders for "
                        f"{len(missing)} items the pool never recorded "
                        f"(sample ids: {sample!r})."
                    )
                    for item_id in missing:
                        payload = payload_by_id.get(item_id)
                        ph = error_placeholder(
                            payload,
                            RuntimeError(
                                f"pool gave up after {stale:.0f}s without progress"
                            ),
                        )
                        try:
                            await results.add(item_id, ph)
                        except Exception as e:
                            print(f"[pool] failed to record placeholder for "
                                  f"{item_id!r}: {e!r}")
                    try:
                        await results.flush()
                    except Exception as e:
                        print(f"[pool] flush after placeholders failed: {e!r}")
                gave_up = True
    finally:
        stop_event.set()
        await results.flush()
        try:
            await asyncio.wait_for(supervisor_task, timeout=30.0)
        except asyncio.TimeoutError:
            supervisor_task.cancel()
        try:
            await asyncio.wait_for(status_task, timeout=5.0)
        except asyncio.TimeoutError:
            status_task.cancel()

    print(f"[pool] complete: wrote {results.total_written}/{total} items.")
    return results.total_written


# -------------------------
# Standard completion work_fn
# -------------------------

@dataclass
class CompletionSampling:
    """Sampling parameters captured by make_completion_work_fn closures."""
    model: str
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    presence_penalty: float = 0.0
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    request_timeout: float = 600.0


def make_completion_work_fn(
    sampling: CompletionSampling,
    parser_name: str,
    tokenizer,
) -> WorkFn:
    """Returns a work_fn that POSTs a fully-rendered prompt to the OpenAI
    /v1/completions endpoint and parses the reasoning split via
    matchminer_ai.llm.reasoning.parse_reasoning_output.

    Payload schema (dict):
      - 'prompt': str | list[str] required; a list batches independent prompts
      - 'max_tokens': int      required
      - 'temperature': float   optional override
      - 'top_p': float         optional override
      - 'request_timeout': float optional per-call timeout override
      - 'include_completion_metadata': bool optional; return finish metadata
      - 'enable_thinking': bool optional; match the rendered template when parsing
      - 'include_raw_text': bool optional; include raw text in completion metadata

    A scalar prompt returns one (reasoning, answer) tuple, or a triple including
    metadata when requested. A prompt list returns a list of those results in
    prompt order. Prompt arrays are transport batching only: the rendered prompt
    strings remain semantically independent.
    """
    from .reasoning import parse_reasoning_output  # local import keeps module import-light

    async def work_fn(client: AsyncOpenAI, payload: dict):
        prompt = payload["prompt"]
        prompt_is_batch = isinstance(prompt, Sequence) and not isinstance(
            prompt,
            (str, bytes),
        )
        if prompt_is_batch:
            request_prompt = list(prompt)
            if not request_prompt or not all(
                isinstance(item, str) and item for item in request_prompt
            ):
                raise ValueError("Batched completion prompts must be non-empty strings.")
        elif isinstance(prompt, str) and prompt:
            request_prompt = prompt
        else:
            raise ValueError("Completion prompt must be a non-empty string or list.")
        max_tokens = int(payload["max_tokens"])
        temperature = float(payload.get("temperature", sampling.temperature))
        top_p = float(payload.get("top_p", sampling.top_p))
        request_timeout = float(
            payload.get("request_timeout", sampling.request_timeout)
        )
        if request_timeout <= 0:
            raise ValueError("Completion request_timeout must be positive.")
        extra = {
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
            "repetition_penalty": sampling.repetition_penalty,
            "skip_special_tokens": False,
        }
        if payload.get("json_schema"):
            # vLLM constrains the answer after the reasoning (the server's reasoning parser).
            extra["structured_outputs"] = {"json": payload["json_schema"]}

        response = await asyncio.wait_for(
            client.completions.create(
                model=sampling.model,
                prompt=request_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=sampling.presence_penalty,
                extra_body=extra,
                timeout=request_timeout,
            ),
            timeout=request_timeout,
        )

        def parse_choice(choice: Any) -> tuple[Any, ...]:
            raw_text = choice.text or ""
            reasoning, answer = parse_reasoning_output(
                raw_text,
                parser_name=parser_name,
                tokenizer=tokenizer,
                **({"enable_thinking": payload["enable_thinking"]}
                   if "enable_thinking" in payload else {}),
            )
            if payload.get("include_completion_metadata"):
                return (
                    reasoning,
                    answer,
                    {
                        "finish_reason": str(
                            getattr(choice, "finish_reason", "") or ""
                        ),
                        "raw_text_char_count": len(raw_text),
                        **({"raw_text": raw_text} if payload.get("include_raw_text") else {}),
                    },
                )
            return (reasoning, answer)

        choices = list(response.choices)
        if not prompt_is_batch:
            if not choices:
                raise RuntimeError("Completion endpoint returned no choices.")
            return await asyncio.to_thread(parse_choice, choices[0])

        choices_by_index: dict[int, Any] = {}
        for position, choice in enumerate(choices):
            raw_index = getattr(choice, "index", position)
            if (
                not isinstance(raw_index, int)
                or isinstance(raw_index, bool)
                or raw_index < 0
                or raw_index >= len(request_prompt)
                or raw_index in choices_by_index
            ):
                raise RuntimeError(
                    "Completion endpoint returned invalid batched choice indices."
                )
            choices_by_index[raw_index] = choice
        missing = sorted(set(range(len(request_prompt))) - set(choices_by_index))
        if missing:
            raise RuntimeError(
                "Completion endpoint omitted batched prompt choices at indices "
                f"{missing}."
            )
        def parse_batch():
            return [
                parse_choice(choices_by_index[index])
                for index in range(len(request_prompt))
            ]

        return await asyncio.to_thread(parse_batch)

    return work_fn


# -------------------------
# Convenience constructor
# -------------------------

def make_chat_completion_work_fn(sampling: CompletionSampling) -> WorkFn:
    """Use the server's chat/reasoning parser, without a matching local vLLM build.

    Accept the same sampling and metadata contract as raw completions, with
    ``messages`` and optional ``chat_template_kwargs`` in the work payload.
    Raw token text is unavailable on this endpoint; reasoning is saved separately.
    """
    async def work_fn(client: AsyncOpenAI, payload: dict):
        timeout = float(payload.get("request_timeout", sampling.request_timeout))
        extra = dict(top_k=sampling.top_k, min_p=sampling.min_p,
                     repetition_penalty=sampling.repetition_penalty)
        if payload.get("chat_template_kwargs"):
            extra["chat_template_kwargs"] = payload["chat_template_kwargs"]
        if payload.get("json_schema"):
            extra["structured_outputs"] = {"json": payload["json_schema"]}
        response = await asyncio.wait_for(client.chat.completions.create(
            model=sampling.model, messages=payload["messages"],
            max_tokens=int(payload["max_tokens"]),
            temperature=float(payload.get("temperature", sampling.temperature)),
            top_p=float(payload.get("top_p", sampling.top_p)),
            presence_penalty=sampling.presence_penalty,
            extra_body=extra, timeout=timeout,
        ), timeout=timeout)
        if not response.choices:
            raise RuntimeError("Chat endpoint returned no choices.")
        choice = response.choices[0]
        message = choice.message
        reasoning = (getattr(message, "reasoning", None)
                     or getattr(message, "reasoning_content", None) or "").strip()
        answer = (message.content or "").strip()
        if payload.get("include_completion_metadata"):
            return reasoning, answer, {
                "finish_reason": str(choice.finish_reason or ""),
                "raw_text": None,
                "transport": "chat_completions",
                **({"response_json": response.model_dump()} if payload.get("include_raw_text") else {}),
            }
        return reasoning, answer

    return work_fn


def build_registry_from_args(args) -> DynamicServerRegistry:
    """Build a DynamicServerRegistry from the standard remote CLI flags."""
    if not getattr(args, "server_urls", None) and not getattr(args, "server_urls_file", None):
        raise ValueError(
            "build_registry_from_args called without --server_urls or "
            "--server_urls_file; the script should not have entered remote mode."
        )
    if getattr(args, "server_urls", None) and getattr(args, "server_urls_file", None):
        raise ValueError("Specify only one of --server_urls / --server_urls_file.")
    static_urls = parse_static_urls(getattr(args, "server_urls", None))
    api_key_env = str(getattr(args, "api_key_env", "OPENAI_API_KEY") or "")
    api_key = os.environ.get(api_key_env, "not-needed") if api_key_env else "not-needed"
    return DynamicServerRegistry(
        max_concurrent_per_server=int(getattr(args, "max_concurrent_per_server", 25)),
        concurrency_success_threshold=int(getattr(args, "concurrency_success_threshold", 3)),
        concurrency_increase_step=int(getattr(args, "concurrency_increase_step", 2)),
        concurrency_backoff_factor=float(getattr(args, "concurrency_backoff_factor", 0.5)),
        request_timeout=float(getattr(args, "request_timeout", 600.0)),
        servers_file=getattr(args, "server_urls_file", None) or None,
        static_urls=static_urls or None,
        refresh_seconds=float(getattr(args, "server_urls_refresh", 15.0)),
        api_key=api_key,
        concurrency_file=getattr(args, "concurrency_control_file", None) or None,
        scheduler_adaptive_concurrency=bool(getattr(args, "scheduler_adaptive_concurrency", False)),
        initial_concurrent_per_server=int(getattr(args, "initial_concurrent_per_server", 128)),
        scheduler_metrics_file=getattr(args, "scheduler_metrics_file", None) or None,
    )
