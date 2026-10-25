"""Thread-safe OpenAI endpoint selection across batches and live worker changes."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time

from .adaptive_concurrency import AdaptiveConcurrency, vllm_load

MAX_POOL_WORKERS = 1536


def retryable(error):
    from openai import APIConnectionError, APITimeoutError
    status = getattr(error, 'status_code', None)
    return (isinstance(error, (APIConnectionError, APITimeoutError, TimeoutError, ConnectionError))
            or status in (408, 409, 429) or (status is not None and status >= 500))


class EndpointPool:
    """Keep transport availability separate from model/prompt identity.

    Registry entries may specify upstream_concurrency. A failed endpoint cools
    down, allowing healthy workers to take its requests. Registry updates admit
    returning Spot workers without restarting the caller. Only transient errors
    consume the recovery window; invalid requests fail immediately.
    """
    def __init__(self, urls=(), *, registry=None, per_server=96, timeout=7200,
                 recovery_timeout=86400, api_key='EMPTY', refresh_seconds=1,
                 client_factory=None, adaptive=None, trace_dir=None):
        from openai import OpenAI
        if per_server < 1 or timeout <= 0 or recovery_timeout <= 0:
            raise ValueError('Endpoint limits must be positive')
        self.urls = list(dict.fromkeys(url.rstrip('/') for url in urls))
        self.registry = Path(registry).resolve() if registry else None
        self.per_server, self.timeout = per_server, timeout
        self.recovery_timeout, self.api_key = recovery_timeout, api_key
        self.refresh_seconds = refresh_seconds
        self.client_factory = client_factory or OpenAI
        self.trace_dir = trace_dir or os.environ.get('MMAI_LLM_TRACE_DIR')
        self.adaptive = dict(adaptive or {})
        self.adaptive_enabled = self.adaptive.pop('enabled', False)
        self.metrics_interval = float(self.adaptive.pop('metrics_interval', 0))
        if self.metrics_interval < 0:
            raise ValueError('metrics_interval cannot be negative')
        self.waiters = 0
        self.condition = threading.Condition()
        self.states, self.entries, self.models, self.model_roots = {}, {}, {}, {}
        self.last_refresh, self.sequence = float('-inf'), 0
        self._refresh()
        if self.adaptive_enabled and self.metrics_interval:
            threading.Thread(target=self._monitor, daemon=True,
                             name='mmai-endpoint-load').start()

    def _refresh(self):
        now = time.monotonic()
        if now - self.last_refresh < self.refresh_seconds:
            return
        if self.registry:
            try:
                data = json.loads(self.registry.read_text())
            except (OSError, ValueError):
                # Atomic replacement or a temporarily unavailable mount: retry
                # without sending work to a stale/unverified endpoint list.
                self.entries = {}
                return
            rows = (data.get('servers', data.get('server_urls', [])) if data.get('ready', True) else []) if isinstance(data, dict) else data
        else:
            rows = [{'url': url} for url in self.urls]
        entries, models, roots = {}, {}, {}
        for row in rows:
            row = {'url': row} if isinstance(row, str) else row
            url = row['url'].rstrip('/')
            limit = int(row.get('upstream_concurrency', self.per_server))
            if not url.startswith(('http://', 'https://')) or not 1 <= limit <= 1024:
                raise ValueError('Invalid endpoint URL or concurrency in registry')
            entries[url] = limit
            if row.get('model'):
                models[url] = row['model']
            if row.get('model_root'):
                roots[url] = row['model_root']
            if url not in self.states:
                self.states[url] = {'active': 0, 'failures': 0, 'until': 0,
                                    'last': -1, 'client': None, 'completed': 0,
                                    'errors': 0, 'load': {}, 'metrics_error': None}
                if self.adaptive_enabled:
                    self.states[url]['controller'] = AdaptiveConcurrency(limit, **self.adaptive)
            elif self.adaptive_enabled:
                self.states[url]['controller'].resize(limit)
        self.entries, self.models, self.model_roots, self.last_refresh = entries, models, roots, now

    def _limit(self, url):
        return (self.states[url]['controller'].limit if self.adaptive_enabled
                else self.entries.get(url, self.per_server))

    def snapshot(self):
        """Operational counters only; contains no prompts, responses or API keys."""
        with self.condition:
            self._refresh()
            return {'adaptive': self.adaptive_enabled, 'waiting_clients': self.waiters,
                    'servers': {url: {
                        'active': state['active'], 'limit': self._limit(url),
                        'maximum': self.entries[url], 'completed': state['completed'],
                        'transport_errors': state['errors'], 'load': dict(state['load']),
                        'metrics_error': state['metrics_error'],
                        'reason': state['controller'].reason if self.adaptive_enabled else 'fixed',
                    } for url, state in self.states.items() if url in self.entries}}

    def observe_load(self, url, load):
        with self.condition:
            state = self.states[url]
            previous = state['load']
            state['load'], state['metrics_error'] = load, None
            if self.adaptive_enabled:
                control = state['controller']
                kv = load.get('kv_cache_usage', 0)
                waiting, running = load.get('waiting', 0), load.get('running', 0)
                preempted = ('preemptions' in previous
                             and load.get('preemptions', 0) > previous['preemptions'])
                control.growth_blocked = kv >= 0.80 or waiting > 0
                # Let already-admitted work drain before reducing again.
                # Lowering a client cap cannot evict running server sequences.
                pressure = kv >= 0.92 or preempted or waiting > max(8, running * 0.1)
                control.observe(pressure=pressure and state['active'] <= control.limit)
            self.condition.notify_all()

    def _monitor(self):
        import httpx
        # Missing metrics never prevents inference; transport outcomes remain
        # the fallback control signal. Probe only configured endpoints.
        with (httpx.Client(timeout=5, trust_env=False) as client,
              ThreadPoolExecutor(max_workers=16, thread_name_prefix='mmai-load') as executor):
            while True:
                with self.condition:
                    self._refresh()
                    urls = list(self.entries)
                # A slow/preempted peer must not serialize every healthy
                # endpoint's feedback behind its metrics timeout.
                list(executor.map(lambda url: self._sample_load(client, url), urls))
                time.sleep(self.metrics_interval)

    def _sample_load(self, client, url):
        try:
            response = client.get(url.removesuffix('/v1') + '/metrics')
            response.raise_for_status()
            self.observe_load(url, vllm_load(response.text))
        except Exception as exc:
            with self.condition:
                self.states[url]['metrics_error'] = type(exc).__name__
                self.states[url]['controller'].growth_blocked = False

    def acquire(self, deadline):
        from matchminer_ai.cancellation import check_cancelled
        with self.condition:
            self.waiters += 1
            try:
                return self._acquire(deadline, check_cancelled)
            finally:
                self.waiters -= 1

    def _acquire(self, deadline, check_cancelled):
        while True:
            check_cancelled()
            self._refresh()
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError('Teacher pool recovery deadline exceeded')
            available = [url for url, limit in self.entries.items()
                         if self.states[url]['active'] < self._limit(url) and self.states[url]['until'] <= now]
            if available:
                url = min(available, key=lambda u: (self.states[u]['active'] / self._limit(u), self.states[u]['last']))
                state = self.states[url]
                if state['client'] is None:
                    import httpx
                    state['client'] = self.client_factory(base_url=url, api_key=self.api_key,
                        timeout=self.timeout, max_retries=0,
                        http_client=httpx.Client(trust_env=False, timeout=self.timeout,
                            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=0)))
                state['active'] += 1
                state['last'] = self.sequence
                self.sequence += 1
                return url, state['client']
            self.condition.wait(timeout=min(max(self.refresh_seconds, 0.01), deadline - now))

    def release(self, url, *, failed=False, successful=True):
        with self.condition:
            state = self.states[url]
            old_limit = self._limit(url)
            saturated = self.waiters > 0 and state['active'] >= self._limit(url)
            state['active'] -= 1
            if failed:
                state['errors'] += 1
                state['failures'] += 1
                state['until'] = time.monotonic() + min(60, 2 ** min(state['failures'] - 1, 6))
            elif successful:
                state['completed'] += 1
                state['failures'], state['until'] = 0, 0
            if self.adaptive_enabled:
                state['controller'].observe(successful=successful and not failed,
                                            pressure=failed, saturated=saturated)
            if self._limit(url) > old_limit:
                self.condition.notify_all()
            else:
                self.condition.notify()

    def request(self, **request):
        return self.request_with_metadata(**request)[0]

    def request_with_metadata(self, **request):
        deadline = time.monotonic() + self.recovery_timeout
        while True:
            url, client = self.acquire(deadline)
            failed, successful = False, False
            try:
                with self.condition:
                    model = self.models.get(url, request.get('model'))
                    provenance = {'endpoint': url, 'request_model': model,
                                  'model_root': self.model_roots.get(url)}
                response = client.chat.completions.create(**{**request, 'model': model})
                if self.trace_dir:
                    from .traces import write_response_trace
                    provenance['trace_path'] = write_response_trace(self.trace_dir, response,
                        request={**request, 'model': model}, provenance=dict(provenance))
                successful = True
                return response, provenance
            except Exception as error:
                failed = retryable(error)
                if not failed:
                    raise
            finally:
                self.release(url, failed=failed, successful=successful)

    def map(self, requests, *, max_workers=MAX_POOL_WORKERS, with_metadata=False):
        requests = list(requests)
        call = self.request_with_metadata if with_metadata else self.request
        if len(requests) == 1:
            return [call(**requests[0])]
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(requests)))) as executor:
            return list(executor.map(lambda request: call(**request), requests))


_POOLS, _LOCK = {}, threading.Lock()


def shared_pool(config, urls, api_key):
    settings = dict(registry=config.get('server_urls_file'),
                    per_server=int(config.get('concurrency_per_server', config.get('max_concurrent_requests', 96))),
                    timeout=float(config.get('request_timeout', 7200)),
                    recovery_timeout=float(config.get('recovery_timeout', 86400)), api_key=api_key,
                    adaptive=config.get('adaptive_concurrency'),
                    trace_dir=config.get('trace_dir') or os.environ.get('MMAI_LLM_TRACE_DIR'))
    # Registry contents deliberately do not enter this key.
    key = (str(settings['registry']), tuple(urls) if not settings['registry'] else (),
           settings['per_server'], settings['timeout'], settings['recovery_timeout'], api_key,
           config.get('model_name'), json.dumps(settings['adaptive'], sort_keys=True),
           str(settings['trace_dir']))
    with _LOCK:
        if key not in _POOLS:
            _POOLS[key] = EndpointPool(urls, **settings)
        return _POOLS[key]


def generate_pooled_outputs(prompts, config, urls, api_key, response_metadata=None):
    from .remote_auth import prepare_messages_for_provider
    from .remote_inference import build_remote_request_config, request_params_for_prompt
    params, extra = build_remote_request_config(config)
    requests = []
    for prompt in prompts:
        messages = prompt.messages or [{'role': 'user', 'content': prompt.prompt_text}]
        request = dict(model=config['model_name'], messages=prepare_messages_for_provider(messages, config),
                       **request_params_for_prompt(params, prompt))
        if extra:
            request['extra_body'] = extra
        requests.append(request)
    responses = shared_pool(config, urls, api_key).map(
        requests, max_workers=int(config.get("pool_max_workers", MAX_POOL_WORKERS)),
        with_metadata=True)
    texts, reasonings, finishes = [''] * len(prompts), [''] * len(prompts), [''] * len(prompts)
    for prompt, (response, provenance) in zip(prompts, responses, strict=True):
        if response_metadata is not None:
            response_metadata.append({'row_idx': prompt.row_idx, **provenance})
        choice = response.choices[0]
        texts[prompt.row_idx] = choice.message.content or ''
        reasonings[prompt.row_idx] = (getattr(choice.message, 'reasoning', None)
                                     or getattr(choice.message, 'reasoning_content', None) or '')
        finishes[prompt.row_idx] = choice.finish_reason or 'stop'
    return texts, reasonings, finishes
