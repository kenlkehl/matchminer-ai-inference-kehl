import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from matchminer_ai.llm.teacher_pool import AdaptiveSemaphore, OutputValidationError, ServerEntry, _dispatch_one


@pytest.mark.parametrize('error,neutral', [
    (OutputValidationError('invalid schema'), True),
    (json.JSONDecodeError('missing quote', '{bad}', 1), True),
    (TimeoutError('timeout'), False),
    (RateLimitError('overloaded', response=httpx.Response(429, request=httpx.Request('POST', 'http://unused')), body=None), False),
    (ValueError('unrelated work failure'), False),
])
def test_validation_retries_do_not_change_limit_even_when_exhausted(error, neutral):
    async def scenario():
        sem = AdaptiveSemaphore(16)
        sem.limit = 8
        sem._success_streak = 1
        entry = ServerEntry(url='http://unused', client=None, semaphore=sem)
        queue = asyncio.Queue()
        saved = []

        async def add(work_id, result):
            saved.append((work_id, result))

        async def fail(client, payload):
            raise error

        args = dict(work_fn=fail, work_queue=queue, results=SimpleNamespace(add=add),
                    max_attempts=2, error_placeholder=lambda payload, exc: f'ERROR: {type(exc).__name__}')
        await sem.acquire()
        await _dispatch_one(entry, 42, {'prompt': 'test'}, 0, **args)
        assert sem.in_flight == 0
        assert sem.limit == (8 if neutral else 4)
        assert sem._success_streak == (1 if neutral else 0)
        assert not saved  # Invalid output is retried, never a completed decision.
        work_id, payload, attempt = queue.get_nowait()
        assert (work_id, attempt) == (42, 1)
        await sem.acquire()
        await _dispatch_one(entry, work_id, payload, attempt, **args)
        assert queue.empty()
        assert saved == [(42, f'ERROR: {type(error).__name__}')]
        assert sem.limit == (8 if neutral else 2)
        assert sem.in_flight == 0
        assert entry.ok_count == 0 and entry.err_count == 2
        assert entry.output_validation_err_count == (2 if neutral else 0)

    asyncio.run(scenario())


def test_valid_retry_is_saved_and_success_can_grow_concurrency():
    async def scenario():
        sem = AdaptiveSemaphore(16, success_threshold=1)
        sem.limit = 8
        entry = ServerEntry(url='http://unused', client=None, semaphore=sem)
        queue, saved = asyncio.Queue(), []

        async def add(work_id, result):
            saved.append((work_id, result))

        async def work(client, payload):  # a malformed answer raises JSONDecodeError, a validation error
            return json.loads(payload['final'])

        args = dict(work_fn=work, work_queue=queue, results=SimpleNamespace(add=add),
                    max_attempts=3, error_placeholder=lambda *args: 'ERROR')
        await sem.acquire()
        await _dispatch_one(entry, 1, {'final': '{bad JSON'}, 0, **args)
        assert sem.limit == 8
        work_id, payload, attempt = queue.get_nowait()
        payload['final'] = '{"answer": "no", "reason": "Diagnostic-only imaging"}'
        await sem.acquire()
        await _dispatch_one(entry, work_id, payload, attempt, **args)
        assert sem.limit == 10
        assert saved == [(1, {'answer': 'no', 'reason': 'Diagnostic-only imaging'})]
        assert entry.ok_count == 1 and entry.output_validation_err_count == 1

    asyncio.run(scenario())

