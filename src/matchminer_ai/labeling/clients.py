"""Several client processes for one labeling stage (``--label-client-processes N``).

One labeler process is bound by its own client, not by the teacher servers: HTTP responses with several KB of
reasoning, their reasoning parse, validation, storage and the dispatch of the next request all share one event
loop under the GIL (about 35 pairs/s with 0.4-5 s of loop lag and servers able to take 3-4 times more,
2026-11-10). With N > 1 the labeling process prepares as before (manifest, reuse import, score journal), then
splits the pending pairs by question key, so that duplicate questions stay in one client, and starts N clients.
Every client writes the same response store (SQLite WAL on local disk) and appends to the same score journal
under a file lock; neither they nor any identity record N, so a run resumes with any N.

The labeling process stays the only adaptive scheduler. It reads the control and servers files and vLLM metrics
as before, and gives each running client an equal share of every server's limit in the client's own control
file (re-read every second), so the clients together never exceed it; a finished client's share goes to the
others. It aggregates the clients' status into status.json and concurrency_status.json, stops all of them and
fails if one fails, and exports and publishes once, after all of them finish.
"""
import asyncio
from contextlib import asynccontextmanager
import copy
import json
import os
from pathlib import Path
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from .files import atomic_json, require_local_disk, write_jsonl

BOOTSTRAP = ('import json,sys;sys.path[:]=json.loads(sys.argv[1]);'
             'from matchminer_ai.labeling.clients import client_main;client_main(sys.argv[2])')
SERVER_STATS = ('in_flight', 'successful_requests', 'request_errors', 'validation_errors')


def shard(key, count):
    return int(key[:15], 16) % count


def shares(limit, count, offset=0):
    """``limit`` split over ``count`` clients; the remainder rotates with ``offset`` (one per server)."""
    base, extra = divmod(limit, count)
    return [base + ((index - offset) % count < extra) for index in range(count)]


def _add(values):
    """Sum the clients' counters, key by key."""
    dicts = [value for value in values if isinstance(value, dict)]
    if dicts:
        return {key: _add([value[key] for value in dicts if key in value])
                for key in dict.fromkeys(key for value in dicts for key in value)}
    numbers = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(numbers) if numbers else (values[0] if values else None)


def spawn(spec, fds):
    """The client process; the labeling lock's descriptor is inherited, so no labeler starts while it lives."""
    return subprocess.Popen([sys.executable, '-u', '-c', BOOTSTRAP, json.dumps(sys.path), str(spec)], pass_fds=fds)


def client_main(path):
    try:  # Linux kills the client when the labeling process dies, so no orphan keeps writing
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass
    spec = pickle.loads(Path(path).read_bytes())
    if os.getppid() != spec['parent'] or Path(__file__).resolve() != Path(spec['module']):
        sys.exit('label client: its labeling process exited, or this is another matchminer_ai')

    def orphaned():
        while os.getppid() == spec['parent']:
            time.sleep(1)
        os._exit(1)
    threading.Thread(target=orphaned, daemon=True).start()
    run_client(spec)


def run_client(spec):
    from .runner import _label
    asyncio.run(_label(spec['args'], Path(spec['output']), client=spec))


class Clients:
    def __init__(self, args, output, count):
        self.args, self.output, self.count = args, Path(output), count
        require_local_disk(self.output)  # one SQLite store and journal written by several processes
        directory = getattr(args, 'label_client_dir', None)
        self.root = Path(directory) if directory else Path(tempfile.mkdtemp(prefix='mmai-label-clients-'))
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.processes, self.results = [None] * count, [None] * count
        self.statuses, self.written = [None] * count, {}
        self.completed, self.failures, self.done = [], {}, False
        self.refreshing = asyncio.Lock()

    def coordinator_args(self):
        """The labeling process's registry: its scheduler snapshot is merged with the clients' counters."""
        args = copy.copy(self.args)
        args.scheduler_metrics_file = str(self.root / 'scheduler.json')
        return args

    def _client_args(self, index):
        args, directory = copy.copy(self.args), self.root / str(index)
        workers = self.args.pending_requests
        automatic = getattr(self.args, 'max_auto_pending_requests', None)
        args.candidates = str(directory / 'candidates.jsonl')
        args.server_urls, args.server_urls_file = None, str(directory / 'control.json')
        args.concurrency_control_file, args.server_urls_refresh = str(directory / 'control.json'), 1.0
        args.scheduler_adaptive_concurrency, args.scheduler_metrics_file = False, None
        args.pending_requests = max(1, shares(workers, self.count)[index]) if workers else 0
        if automatic is not None:
            args.max_auto_pending_requests = max(1, shares(automatic, self.count)[index])
        args.label_client_processes = 1
        return args

    def _prepare(self, pairs, task, signature):
        from .runner import label_key
        parts = [[] for _ in range(self.count)]
        for pair in pairs:
            parts[shard(label_key({**pair, 'task': task}), self.count)].append(pair)
        for index, part in enumerate(parts):
            directory = self.root / str(index)
            write_jsonl(directory / 'candidates.jsonl', part)
            spec = dict(args=self._client_args(index), output=str(self.output), directory=str(directory),
                        index=index, count=self.count, signature=signature, parent=os.getpid(),
                        module=str(Path(__file__).resolve()))
            (directory / 'spec.pickle').write_bytes(pickle.dumps(spec))
        return [len(part) for part in parts]

    def _controls(self, entries, live):
        controls = {index: dict(ready=True, servers=[], max_concurrent_per_server=1, per_server={}) for index in live}
        for offset, entry in enumerate(sorted(entries, key=lambda entry: entry.url)):
            for index, share in zip(live, shares(entry.semaphore.max_limit, len(live), offset)):
                if share:  # a server without a share is absent: that client retires it
                    controls[index]['servers'].append(dict(url=entry.url, instance=entry.instance))
                    controls[index]['per_server'][entry.url] = share
        return controls

    async def _share(self, registry):
        live = [index for index, result in enumerate(self.results) if result is None]
        if not live:
            return
        for index, control in self._controls(await registry.active_entries(), live).items():
            if control != self.written.get(index):
                await asyncio.to_thread(atomic_json, self.root / str(index) / 'control.json', control)
                self.written[index] = control

    def _read(self, name, index):
        path = self.root / str(index) / name
        try:
            return json.loads(path.read_text()) if path.exists() else None
        except ValueError:
            return None

    def _refresh(self):
        self.statuses = [self._read('status.json', index) or prior for index, prior in enumerate(self.statuses)]
        snapshot = self.root / 'scheduler.json'
        target = getattr(self.args, 'scheduler_metrics_file', None)
        if not target or not snapshot.exists():
            return
        value = json.loads(snapshot.read_text())
        servers = {}
        for status in filter(None, self.statuses):
            for server in status.get('servers', []):
                servers.setdefault(server['url'], []).append(server)
        for server in value.get('servers', []):  # the labeling process's limits and metrics, the clients' counts
            mine = servers.get(server['url'], [])
            server.update({key: sum(item[key] for item in mine) for key in SERVER_STATS},
                          limit=sum(item['limit'] for item in mine if not item['retired']),
                          streak=max((item['streak'] for item in mine), default=0),
                          error_types=_add([item['error_types'] for item in mine]) if mine else {},
                          last_error=max((item['last_error'] for item in mine if item['last_error']),
                                         key=lambda error: error['at'], default=None),
                          client_limits=[0 if item['retired'] else item['limit'] for item in mine],
                          client_in_flight=[item['in_flight'] for item in mine])
        value['label_client_processes'] = self.count
        atomic_json(target, value)

    async def _refreshed(self):
        try:
            async with self.refreshing:
                await asyncio.to_thread(self._refresh)
        except Exception as error:  # telemetry never stops labeling
            print('[labels] client status delayed: ' + repr(error), flush=True)

    def merge(self, own):
        """The labeling process's status with its clients' added (supervisors read one file). Their counts are
        added only until the labeling process has taken over their results."""
        value = dict(own, label_client_processes=self.count, client_processes=[
            dict(index=index, pid=getattr(process, 'pid', None), exit_code=getattr(process, 'returncode', None),
                 state=(status or {}).get('state'), completed=(status or {}).get('completed', 0),
                 event_loop_lag=(status or {}).get('event_loop_lag'))
            for index, (process, status) in enumerate(zip(self.processes, self.statuses))])
        statuses = [status for status in self.statuses if status]
        if not statuses:
            return value
        value['attempts_started'] += sum(status['attempts_started'] for status in statuses)
        value['first_request_at'] = min(filter(None, [value['first_request_at'],
                                                      *(status['first_request_at'] for status in statuses)]), default=None)
        value['client'] = _add([status['client'] for status in statuses])
        value['event_loop_lag'] = {key: max(status['event_loop_lag'][key] for status in statuses)
                                   for key in ('latest_seconds', 'max_seconds')}
        limits = [status for status in statuses if 'pair_worker_limit' in status]
        if limits:
            value.update(pair_worker_limit=sum(status['pair_worker_limit'] for status in limits),
                         pair_worker_mode=limits[0]['pair_worker_mode'])
        if self.done:
            return value
        value['completed'] += sum(status['completed'] for status in statuses)
        value['failures'] = dict(value['failures'], **{k: v for s in statuses for k, v in s['failures'].items()})
        if 'scored_pairs' in value:
            value['scored_pairs'] += sum(status['scored_pairs'] for status in statuses)
            value['score_distribution'] = _add([value['score_distribution'],
                                                *(status['score_distribution'] for status in statuses)])
        return value

    @asynccontextmanager
    async def serving(self, registry):
        """The registry, with the merged status refreshed from its start (as the one process's scheduler writes it)."""
        async def refresh():
            while True:
                await self._refreshed()
                await asyncio.sleep(5)
        await registry.start()
        refresher = asyncio.create_task(refresh())
        try:
            yield self
        finally:
            refresher.cancel()
            await asyncio.gather(refresher, return_exceptions=True)
            await self.stop()
            await registry.stop()

    async def run(self, registry, pairs, task, signature):
        """Every client's share of ``pairs`` to completion; any client's failure stops the others and raises."""
        sizes = await asyncio.to_thread(self._prepare, pairs, task, signature)
        print(f'[labels] {self.count} client processes, pending pairs by client: {sizes}', flush=True)
        self.results = [None if size else dict(completed=[], failures={}) for size in sizes]
        await self._share(registry)
        fds = tuple(fd for fd in [getattr(self.args, 'label_lock_fd', None)] if fd is not None)
        self.processes = [spawn(self.root / str(index) / 'spec.pickle', fds) if size else None
                          for index, size in enumerate(sizes)]
        while any(result is None for result in self.results):
            for index, process in enumerate(self.processes):
                if self.results[index] is None and process.poll() is not None:
                    result = await asyncio.to_thread(self._read, 'result.json', index) if process.returncode == 0 else None
                    if result is None:
                        raise RuntimeError(f'Label client {index} of {self.count} exited with code {process.returncode}; '
                                           f'the others are stopped. Resume this command with any '
                                           f'--label-client-processes. Client files: {self.root}')
                    self.results[index] = result
            await self._share(registry)
            await asyncio.sleep(1)
        await self._refreshed()
        for result in self.results:
            self.completed.extend(result['completed'])
            self.failures.update(result['failures'])

    async def stop(self):
        running = [process for process in self.processes if process is not None and process.poll() is None]
        for process in running:
            process.terminate()
        deadline = time.monotonic() + 30
        while any(process.poll() is None for process in running) and time.monotonic() < deadline:
            await asyncio.sleep(.2)
        for process in running:
            if process.poll() is None:
                process.kill()
            await asyncio.to_thread(process.wait)

    def finish(self):
        """After the labeling process took over the results: only its own counters count from now on."""
        self.done = True
        shutil.rmtree(self.root, ignore_errors=True)
