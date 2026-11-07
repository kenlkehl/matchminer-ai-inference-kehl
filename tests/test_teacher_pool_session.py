import asyncio

import pytest

from matchminer_ai.llm.teacher_pool import (
    AdaptiveSemaphore,
    DynamicServerRegistry,
    OutputValidationError,
    RequestSession,
    ServerEntry,
)


def test_clients_do_not_reuse_idle_server_connections():
    """A vLLM endpoint's short keep-alive lifetime cannot poison later work."""
    async def scenario():
        connections = []
        handlers = set()

        async def handle(reader, writer):
            task = asyncio.current_task()
            handlers.add(task)
            connections.append(writer)
            try:
                while True:
                    await reader.readuntil(b'\r\n\r\n')
                    body = b'{"object":"list","data":[]}'
                    writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n'
                                 + f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
                    await writer.drain()
            except asyncio.IncompleteReadError:
                pass
            finally:
                writer.close()
                await writer.wait_closed()
                handlers.discard(task)

        server = await asyncio.start_server(handle, '127.0.0.1', 0)
        port = server.sockets[0].getsockname()[1]
        registry = DynamicServerRegistry(max_concurrent_per_server=768, request_timeout=60,
                                         static_urls=[f'http://127.0.0.1:{port}/v1'])
        client = registry._make_client(registry._static_urls[0])
        try:
            assert (await client.models.list()).data == []
            assert (await client.models.list()).data == []
            assert len(connections) == 2
        finally:
            await client.close()
            server.close()
            await server.wait_closed()
            await asyncio.gather(*list(handlers))

    asyncio.run(scenario())


class Registry:
    def __init__(self, limit=2):
        self.entry = ServerEntry("synthetic", None, AdaptiveSemaphore(limit))
        self.stopped = False

    async def start(self):
        pass

    async def active_entries(self):
        return [self.entry]

    async def wait_for_change(self, timeout):
        await asyncio.sleep(0.01)

    async def stop(self):
        self.stopped = True


def test_sparse_requests_fill_fast_endpoint_headroom_instead_of_round_robin():
    async def scenario():
        registry = Registry()
        fast = ServerEntry('fast', 'fast', AdaptiveSemaphore(8))
        slow = ServerEntry('slow', 'slow', AdaptiveSemaphore(2))
        await slow.semaphore.acquire()  # A separate long-running request.
        async def entries():
            return [slow, fast]
        registry.active_entries = entries
        calls = []
        async def work(client, payload):
            calls.append(client)
            await asyncio.sleep(.001)
            return payload
        async with RequestSession(registry, work) as session:
            # Allow idle consumers to form: the old dispatcher then alternated
            # these sparse arrivals even though the slow server was occupied.
            await asyncio.sleep(.02)
            for i in range(12):
                assert await session.submit(i) == i
                await asyncio.sleep(.002)
        assert calls == ['fast'] * 12
        assert fast.semaphore.in_flight == 0
        await slow.semaphore.release(success=None)
    asyncio.run(scenario())


def test_retired_endpoint_drains_while_replacement_receives_new_work():
    async def scenario():
        registry = Registry()
        old = ServerEntry('old', 'old', AdaptiveSemaphore(1))
        new = ServerEntry('new', 'new', AdaptiveSemaphore(2))
        current = [old]
        async def entries():
            return current[:]
        registry.active_entries = entries
        started, release = asyncio.Event(), asyncio.Event()
        async def work(client, payload):
            if client == 'old':
                started.set()
                await release.wait()
            return client
        async with RequestSession(registry, work) as session:
            first = asyncio.create_task(session.submit('first'))
            await started.wait()
            old.retired = True
            current[:] = [new]
            assert await asyncio.wait_for(session.submit('second'), 2) == 'new'
            release.set()
            assert await first == 'old'
        assert old.semaphore.in_flight == new.semaphore.in_flight == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_session_shares_capacity_retries_json_neutrally_and_accepts_new_work(limit):
    async def scenario():
        registry = Registry(limit)
        running = peak = 0
        attempts = {}

        async def work(client, payload):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            try:
                attempts[payload] = attempts.get(payload, 0) + 1
                await asyncio.sleep(0.01)
                if payload == "retry" and attempts[payload] == 1:
                    raise OutputValidationError("synthetic malformed JSON")
                return payload
            finally:
                running -= 1

        async with RequestSession(registry, work, max_attempts=2) as session:
            payloads = ["a", "b", "c", "d", "e", "retry"]
            assert await asyncio.gather(
                *(session.submit(x) for x in payloads)
            ) == payloads
            assert await session.submit("new child") == "new child"
            assert session.completed == len(payloads) + 1
        assert peak == limit and running == 0
        assert registry.entry.semaphore.limit == limit
        assert registry.entry.output_validation_err_count == 1
        assert registry.stopped

    asyncio.run(scenario())


def test_session_cancellation_releases_requests_and_clients():
    async def scenario():
        registry = Registry()
        running = 0
        started = asyncio.Event()

        async def work(client, payload):
            nonlocal running
            running += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                running -= 1

        async def run():
            async with RequestSession(registry, work) as session:
                await session.submit("blocked")

        task = asyncio.create_task(run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert running == 0 and registry.stopped
        assert registry.entry.semaphore.in_flight == 0

    asyncio.run(scenario())


def test_cancellation_during_normal_session_shutdown_is_not_swallowed():
    async def scenario():
        waiting, exiting = asyncio.Event(), asyncio.Event()

        class SlowRegistry(Registry):
            async def wait_for_change(self, timeout):
                waiting.set()
                await asyncio.Event().wait()

        registry = SlowRegistry()

        async def work(client, payload):
            return payload

        async def run():
            async with RequestSession(registry, work):
                await waiting.wait()
                exiting.set()

        task = asyncio.create_task(run())
        await exiting.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert registry.stopped

    asyncio.run(scenario())
