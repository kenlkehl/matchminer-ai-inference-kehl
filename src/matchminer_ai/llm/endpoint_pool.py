"""Thread-safe OpenAI endpoint selection across batches and live worker changes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

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
                 client_factory=None):
        from openai import OpenAI
        if per_server < 1 or timeout <= 0 or recovery_timeout <= 0:
            raise ValueError('Endpoint limits must be positive')
        self.urls = list(dict.fromkeys(url.rstrip('/') for url in urls))
        self.registry = Path(registry).resolve() if registry else None
        self.per_server, self.timeout = per_server, timeout
        self.recovery_timeout, self.api_key = recovery_timeout, api_key
        self.refresh_seconds = refresh_seconds
        self.client_factory = client_factory or OpenAI
        self.condition = threading.Condition()
        self.states, self.entries = {}, {}
        self.last_refresh, self.sequence = float('-inf'), 0
        self._refresh()

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
        entries = {}
        for row in rows:
            row = {'url': row} if isinstance(row, str) else row
            url = row['url'].rstrip('/')
            limit = int(row.get('upstream_concurrency', self.per_server))
            if not url.startswith(('http://', 'https://')) or not 1 <= limit <= 1024:
                raise ValueError('Invalid endpoint URL or concurrency in registry')
            entries[url] = limit
            self.states.setdefault(url, {'active': 0, 'failures': 0, 'until': 0, 'last': -1, 'client': None})
        self.entries, self.last_refresh = entries, now

    def acquire(self, deadline):
        with self.condition:
            while True:
                self._refresh()
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError('Teacher pool recovery deadline exceeded')
                available = [url for url, limit in self.entries.items()
                             if self.states[url]['active'] < limit and self.states[url]['until'] <= now]
                if available:
                    url = min(available, key=lambda u: (self.states[u]['active'] / self.entries[u], self.states[u]['last']))
                    state = self.states[url]
                    if state['client'] is None:
                        state['client'] = self.client_factory(base_url=url, api_key=self.api_key,
                            timeout=self.timeout, max_retries=0)
                    state['active'] += 1
                    state['last'] = self.sequence
                    self.sequence += 1
                    return url, state['client']
                self.condition.wait(timeout=min(max(self.refresh_seconds, 0.01), deadline - now))

    def release(self, url, *, failed=False):
        with self.condition:
            state = self.states[url]
            state['active'] -= 1
            if failed:
                state['failures'] += 1
                state['until'] = time.monotonic() + min(60, 2 ** min(state['failures'] - 1, 6))
            else:
                state['failures'], state['until'] = 0, 0
            self.condition.notify_all()

    def request(self, **request):
        deadline = time.monotonic() + self.recovery_timeout
        while True:
            url, client = self.acquire(deadline)
            failed = False
            try:
                return client.chat.completions.create(**request)
            except Exception as error:
                failed = retryable(error)
                if not failed:
                    raise
            finally:
                self.release(url, failed=failed)

    def map(self, requests, *, max_workers=MAX_POOL_WORKERS):
        requests = list(requests)
        if len(requests) == 1:
            return [self.request(**requests[0])]
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(requests)))) as executor:
            return list(executor.map(lambda request: self.request(**request), requests))


_POOLS, _LOCK = {}, threading.Lock()


def shared_pool(config, urls, api_key):
    settings = dict(registry=config.get('server_urls_file'),
                    per_server=int(config.get('concurrency_per_server', config.get('max_concurrent_requests', 96))),
                    timeout=float(config.get('request_timeout', 7200)),
                    recovery_timeout=float(config.get('recovery_timeout', 86400)), api_key=api_key)
    # Registry contents deliberately do not enter this key.
    key = (str(settings['registry']), tuple(urls) if not settings['registry'] else (),
           settings['per_server'], settings['timeout'], settings['recovery_timeout'], api_key)
    with _LOCK:
        if key not in _POOLS:
            _POOLS[key] = EndpointPool(urls, **settings)
        return _POOLS[key]


def generate_pooled_outputs(prompts, config, urls, api_key):
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
        requests, max_workers=int(config.get("pool_max_workers", MAX_POOL_WORKERS)))
    texts, reasonings, finishes = [''] * len(prompts), [''] * len(prompts), [''] * len(prompts)
    for prompt, response in zip(prompts, responses, strict=True):
        choice = response.choices[0]
        texts[prompt.row_idx] = choice.message.content or ''
        reasonings[prompt.row_idx] = (getattr(choice.message, 'reasoning', None)
                                     or getattr(choice.message, 'reasoning_content', None) or '')
        finishes[prompt.row_idx] = choice.finish_reason or 'stop'
    return texts, reasonings, finishes
