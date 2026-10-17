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
        monkeypatch.setattr(
            remote, "connect_to_remote_servers", lambda **kw: [(c, 0) for c in clients]
        )
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
