import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest

from matchminer_ai.llm import remote_inference as remote


@pytest.mark.parametrize("per_server", [1, 2, 5])
def test_persistent_session_obeys_each_server_cap_and_closes_clients(
    monkeypatch, per_server
):
    async def scenario():
        running, peak, calls, closed = Counter(), Counter(), Counter(), []

        class Client:
            def __init__(self, name):
                self.name = name

            async def close(self):
                closed.append(self.name)

        clients = [Client("one"), Client("two")]
        def connect(**kwargs):
            assert kwargs["max_connections"] == per_server
            return [(c, 0) for c in clients]

        monkeypatch.setattr(remote, "connect_to_remote_servers", connect)
        monkeypatch.setattr(
            remote,
            "build_remote_request_config",
            lambda config: ({"max_tokens": 10}, {}),
        )

        async def generate(client, **kwargs):
            name = client.name
            running[name] += 1
            peak[name] = max(peak[name], running[name])
            calls[name] += 1
            try:
                await asyncio.sleep(0.01)
                return remote.ModelResult(0, "reasoning", "answer")
            finally:
                running[name] -= 1

        monkeypatch.setattr(remote, "single_inference_request", generate)
        config = {
            "server_urls": ["http://one", "http://two"],
            "model_name": "fake",
            "max_concurrent_requests": per_server,
            "request_timeout": 10,
            "max_retries": 1,
        }
        prompt = SimpleNamespace(
            row_idx=0,
            messages=[{"role": "user", "content": "synthetic"}],
            max_tokens=10,
        )
        async with remote.RemoteRequestSession(config, "unused", {}) as session:
            assert session.capacity == 2 * per_server
            results = await asyncio.gather(
                *(session.generate(prompt, config) for _ in range(12))
            )
            assert len(results) == 12
        assert set(closed) == {"one", "two"}
        assert peak == {"one": per_server, "two": per_server}
        assert sum(calls.values()) == 12 and not any(running.values())

    asyncio.run(scenario())


def test_http_pool_allows_more_than_100_simultaneous_requests():
    """All requests must reach the server before any response is released."""
    async def scenario():
        capacity = 128
        arrived = 0
        all_arrived = asyncio.Event()

        async def handle(reader, writer):
            nonlocal arrived
            try:
                header = await reader.readuntil(b"\r\n\r\n")
                length = next(int(line.split(b":", 1)[1]) for line in header.split(b"\r\n")
                              if line.lower().startswith(b"content-length:"))
                await reader.readexactly(length)
                arrived += 1
                if arrived == capacity:
                    all_arrived.set()
                await asyncio.wait_for(all_arrived.wait(), timeout=10)
                body = b'{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop","index":0}]}'
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                             + str(len(body)).encode() + b"\r\n\r\n" + body)
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0, backlog=capacity)
        port = server.sockets[0].getsockname()[1]
        clients = remote.connect_to_remote_servers(
            [f"http://127.0.0.1:{port}/v1"], request_timeout=15,
            max_connections=capacity,
        )
        client = clients[0][0]
        try:
            async with server:
                results = await asyncio.wait_for(asyncio.gather(*(
                    client.chat.completions.create(
                        model="synthetic", messages=[{"role": "user", "content": "test"}],
                        max_tokens=2,
                    ) for _ in range(capacity)
                )), timeout=15)
                assert arrived == capacity
                assert all(result.choices[0].message.content == "ok" for result in results)
        finally:
            await client.close()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
