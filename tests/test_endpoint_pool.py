"""Transport regressions: persistent catalog balancing and live Spot recovery."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event, Lock
import time
from types import SimpleNamespace

import pytest

from matchminer_ai.llm.endpoint_pool import EndpointPool


def client_factory(respond):
    def factory(**settings):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **request: respond(settings['base_url'], request))))
    return factory


def registry(path, entries):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps({'ready': True, 'servers': [
        {'url': url, 'upstream_concurrency': cap} for url, cap in entries]}))
    temp.replace(path)


def test_singleton_catalog_calls_share_pool_across_batches(monkeypatch):
    import openai
    from matchminer_ai import load_default_preset
    from matchminer_ai.trials.drug_catalog import _run_llm_messages
    from matchminer_ai.llm.endpoint_pool import _POOLS
    _POOLS.clear()
    called = []
    def respond(url, request):
        called.append((url, request))
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content='Synthetic answer', reasoning='hidden'))])
    monkeypatch.setattr(openai, 'OpenAI', client_factory(respond))
    monkeypatch.setattr('matchminer_ai.llm.backends.get_model_metadata', lambda *a, **k: {})
    config = load_default_preset()
    urls = ['http://replica-a/v1', 'http://replica-b/v1', 'http://replica-c/v1']
    config.remote.update(enabled=True, server_urls=urls, max_concurrent_requests=2)
    for _ in range(6):
        outputs = _run_llm_messages([[{'role': 'user', 'content': 'fabricated evidence'}]],
                                   config=config, stage='synthesis')
        assert outputs[0].text == 'Synthetic answer'
        assert outputs[0].reasoning == 'hidden'
    assert [url for url, _ in called] == urls * 2


def test_registry_caps_and_removal_apply_without_restarting(tmp_path):
    path = tmp_path / 'servers.json'
    a, b = 'http://a/v1', 'http://b/v1'
    registry(path, [(a, 2), (b, 1)])
    lock, saturated, release = Lock(), Event(), Event()
    active, peaks = {a: 0, b: 0}, {a: 0, b: 0}
    def respond(url, request):
        with lock:
            active[url] += 1
            peaks[url] = max(peaks[url], active[url])
            if sum(active.values()) == 3:
                saturated.set()
        try:
            assert release.wait(3)
            return url
        finally:
            with lock:
                active[url] -= 1
    pool = EndpointPool(registry=path, refresh_seconds=0.01, recovery_timeout=3,
                        client_factory=client_factory(respond))
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(pool.request, model='teacher') for _ in range(8)]
        try:
            assert saturated.wait(3)
            registry(path, [(b, 1)])
            time.sleep(0.03)
        finally:
            release.set()
        result = [f.result(timeout=3) for f in futures]
    assert peaks == {a: 2, b: 1}
    assert result.count(a) == 2  # only requests already in flight before removal
    assert pool.request(model='teacher') == b
    registry(path, [(a, 2)])
    time.sleep(0.02)
    assert pool.request(model='teacher') == a


def test_transient_failure_reroutes_without_saving_a_failed_answer():
    calls = []
    def respond(url, request):
        calls.append(url)
        if url == 'http://spot/v1':
            raise ConnectionError('worker disappeared')
        return 'valid final answer'
    pool = EndpointPool(['http://spot/v1', 'http://local/v1'], recovery_timeout=2,
                        client_factory=client_factory(respond))
    assert pool.request(model='teacher') == 'valid final answer'
    assert calls == ['http://spot/v1', 'http://local/v1']
    assert all(s['active'] == 0 for s in pool.states.values())


def test_empty_registry_waits_for_recovered_worker(tmp_path):
    path = tmp_path / 'servers.json'
    registry(path, [])
    pool = EndpointPool(registry=path, refresh_seconds=0.01, recovery_timeout=3,
                        client_factory=client_factory(lambda url, request: url))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pool.request, model='teacher')
        assert not future.done()
        registry(path, [('http://recovered/v1', 1)])
        assert future.result(timeout=3) == 'http://recovered/v1'


def test_permanent_bad_request_fails_immediately():
    class InvalidRequest(Exception):
        status_code = 400
    calls = []
    def respond(url, request):
        calls.append(url)
        raise InvalidRequest('invalid request')
    pool = EndpointPool(['http://a/v1', 'http://b/v1'],
                        client_factory=client_factory(respond))
    with pytest.raises(InvalidRequest):
        pool.request(model='teacher')
    assert len(calls) == 1


def test_explicit_batch_limit_caps_total_requests():
    lock = Lock()
    active = peak = 0
    def respond(url, request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return url
    pool = EndpointPool(['http://a/v1', 'http://b/v1'], per_server=96,
                        client_factory=client_factory(respond))
    assert len(pool.map([{'model': 'teacher'}] * 8, max_workers=2)) == 8
    assert peak == 2


def test_recovery_window_expires_without_any_available_server(tmp_path):
    path = tmp_path / 'servers.json'
    registry(path, [])
    pool = EndpointPool(registry=path, recovery_timeout=0.02, refresh_seconds=0.01)
    with pytest.raises(TimeoutError, match='recovery deadline'):
        pool.request(model='teacher')
