from concurrent.futures import ThreadPoolExecutor
import json
from threading import Lock
import time
from types import SimpleNamespace

import pytest

from matchminer_ai.llm.adaptive_concurrency import AdaptiveConcurrency, vllm_load
from matchminer_ai.llm.endpoint_pool import EndpointPool


def test_ramp_backoff_recovery_and_cap_changes():
    control = AdaptiveConcurrency(16, initial=2, success_threshold=2,
                                  increase_step=2, adjustment_interval=10)
    for now in (0, 1):
        control.observe(successful=True, saturated=True, now=now)
    assert control.limit == 4
    control.observe(pressure=True, now=2)
    assert control.limit == 2  # capacity failure immediately reverses growth
    control.observe(pressure=True, now=3)
    assert control.limit == 2  # a concurrent error burst is one backoff
    for now in (12, 13):
        control.observe(successful=True, saturated=True, now=now)
    assert control.limit == 4
    control.resize(3)
    assert control.limit == 3
    control.observe(pressure=True, now=30)
    assert control.limit == 1


def test_no_growth_without_demand_or_under_memory_pressure():
    control = AdaptiveConcurrency(8, initial=2, success_threshold=1)
    assert control.observe(successful=True, saturated=False, now=0) == 2
    control.growth_blocked = True
    assert control.observe(successful=True, saturated=True, now=20) == 2
    control.growth_blocked = False
    assert control.observe(successful=True, saturated=True, now=40) == 8


@pytest.mark.parametrize('settings', [
    {'maximum': 0}, {'maximum': 8, 'initial': 0},
    {'maximum': 8, 'backoff_factor': 1}, {'maximum': 8, 'adjustment_interval': 0},
])
def test_invalid_settings_fail(settings):
    with pytest.raises(ValueError):
        AdaptiveConcurrency(**settings)


def test_metrics_aggregate_workers_and_ignore_nonfinite_values():
    assert vllm_load('''# HELP ignored
vllm:num_requests_running{engine="0"} 10
vllm:num_requests_running{engine="1"} 12
vllm:num_requests_waiting{engine="0"} 2
vllm:kv_cache_usage_perc{engine="0"} 0.5
vllm:kv_cache_usage_perc{engine="1"} 0.9
vllm:num_preemptions_total{engine="0"} 7
vllm:num_requests_waiting{engine="2"} NaN
''') == dict(running=22, waiting=2, kv_cache_usage=0.9, preemptions=7)


def test_live_pool_grows_without_exceeding_endpoint_limit():
    lock = Lock()
    active = peak = 0

    def create(**request):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return 'malformed model text'  # semantic/JSON failures are not overload

    pool = EndpointPool(['http://synthetic/v1'], per_server=8,
                        adaptive=dict(enabled=True, initial=2, increase_step=2,
                                      success_threshold=1, adjustment_interval=0.001),
                        client_factory=lambda **kw: SimpleNamespace(chat=SimpleNamespace(
                            completions=SimpleNamespace(create=create))))
    with ThreadPoolExecutor(max_workers=12) as executor:
        values = list(executor.map(lambda _: pool.request(model='fake'), range(40)))
    assert len(values) == 40 and 2 < peak <= 8
    state = pool.snapshot()['servers']['http://synthetic/v1']
    assert state['limit'] == 8 and state['transport_errors'] == 0
    assert state['completed'] == 40 and state['active'] == 0


def test_load_feedback_reduces_limit_and_blocks_growth():
    pool = EndpointPool(['http://synthetic/v1'], per_server=128,
                        adaptive=dict(enabled=True, initial=64))
    pool.observe_load('http://synthetic/v1', dict(running=64, waiting=20, kv_cache_usage=0.95))
    state = pool.snapshot()['servers']['http://synthetic/v1']
    assert state['limit'] == 32 and state['reason'] == 'capacity_backoff'
    pool.observe_load('http://synthetic/v1', dict(running=20, waiting=0, kv_cache_usage=0.5))
    assert not pool.states['http://synthetic/v1']['controller'].growth_blocked


def test_registry_addition_accepts_endpoint_specific_model_alias(tmp_path):
    path = tmp_path/'servers.json'
    path.write_text(json.dumps({'servers': []}))
    received = []
    pool = EndpointPool(registry=path, refresh_seconds=0.001,
                        adaptive=dict(enabled=True, initial=2),
                        client_factory=lambda **kw: SimpleNamespace(chat=SimpleNamespace(
                            completions=SimpleNamespace(create=lambda **req: received.append(req) or 'ok'))))
    path.write_text(json.dumps({'servers': [{'url':'http://ready/v1',
                                           'model':'different-served-alias', 'upstream_concurrency':8}]}))
    time.sleep(0.002)
    assert pool.request(model='original-alias') == 'ok'
    assert received[0]['model'] == 'different-served-alias'
    assert pool.snapshot()['servers']['http://ready/v1']['limit'] == 2


def test_public_cohort_retries_keep_each_endpoint_provenance(monkeypatch, tmp_path):
    import pandas as pd
    import openai
    from matchminer_ai.config import load_default_preset
    from matchminer_ai.trials import summarize_trials
    path = tmp_path/'registry.json'
    path.write_text(json.dumps({'servers': [
        {'url':f'http://synthetic-{n}/v1', 'model':f'alias-{n}',
         'model_root':f'quantization-{n}', 'upstream_concurrency':2} for n in (0, 1)
    ]}))
    calls = []
    def create(**request):
        calls.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(reasoning='synthetic trace', content=(
                'malformed output' if len(calls) == 1 else
                '```json\n{"cancer_treatment_trial":"yes","multiple_cohorts":"no"}\n```')))])
    monkeypatch.setattr(openai, 'OpenAI', lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr('matchminer_ai.llm.backends.get_endpoint_model_metadata', lambda *a, **k: {})
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=[], server_urls_file=str(path))
    frame = pd.DataFrame([dict(trial_id='synthetic', trial_title='Synthetic trial',
                               brief_summary='Cancer treatment', eligibility_criteria='Adult')])
    rows, meta = summarize_trials(frame, config=config, return_metadata=True)
    assert len(rows) == 1 and len(calls) == 2
    provenance = [item['response_metadata'] for item in meta['trial_attempts']['synthetic']]
    assert [item['model_root'] for item in provenance] == ['quantization-0', 'quantization-1']
    assert [item['request_model'] for item in provenance] == ['alias-0', 'alias-1']
    assert not meta['trial_errors']


def test_adaptive_persistent_session_uses_shared_pool(monkeypatch):
    import asyncio
    from matchminer_ai.llm.remote_inference import RemoteRequestSession
    calls = []
    def request(**kwargs):
        calls.append(kwargs)
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',
            message=SimpleNamespace(content='answer', reasoning='trace'))])
        return response, {'endpoint':'http://synthetic/v1', 'model_root':'checkpoint'}
    monkeypatch.setattr('matchminer_ai.llm.endpoint_pool.shared_pool',
                        lambda *a: SimpleNamespace(request_with_metadata=request))
    config = dict(server_urls=['http://synthetic/v1'], model_name='fake',
                  max_concurrent_requests=2, pool_max_workers=2,
                  adaptive_concurrency={'enabled':True},
                  remote={'request_params':{'max_tokens':10}, 'extra_body':{}})
    async def run():
        async with RemoteRequestSession(config, 'unused', {}) as session:
            result = await session.generate(SimpleNamespace(row_idx=7, messages=None,
                                            prompt_text='synthetic', max_tokens=10), config)
            assert result.row_idx == 7 and result.summary == 'answer'
            assert result.response_metadata['model_root'] == 'checkpoint'
    asyncio.run(run())
    assert len(calls) == 1
