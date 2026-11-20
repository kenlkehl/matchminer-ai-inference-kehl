"""Loopback-only concurrency limiter for adding a teacher without restarting a labeler.

Reads a small JSON config: upstream, expected_root, served_model, concurrency.
Optional upstream_model maps the client model alias to the upstream model ID.
It preserves prompts and sampling settings and never logs request bodies.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time

from aiohttp import ClientConnectionError, ClientResponseError, ClientSession, ClientTimeout, TCPConnector, web


class Limiter:
    def __init__(self, limit):
        self.limit = limit
        self.active = 0
        self.waiting = 0
        self.completed = 0
        self.errors = 0
        self.condition = asyncio.Condition()

    async def resize(self, limit):
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 512:
            raise ValueError('Concurrency must be an integer between 1 and 512')
        async with self.condition:
            self.limit = limit
            self.condition.notify_all()

    @asynccontextmanager
    async def slot(self):
        async with self.condition:
            self.waiting += 1
            try:
                await self.condition.wait_for(lambda: self.active < self.limit)
                self.active += 1
            finally:
                self.waiting -= 1
        try:
            yield
        finally:
            async with self.condition:
                self.active -= 1
                self.condition.notify_all()


def make_app(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    upstream = config['upstream'].rstrip('/')
    upstream_model = config.get('upstream_model', config['served_model'])
    limit = Limiter(config['concurrency'])
    app = web.Application(client_max_size=8 * 1024 * 1024)

    async def resources(app):
        await limit.resize(config['concurrency'])
        async with ClientSession(timeout=ClientTimeout(total=float(config.get("request_timeout", 7260))),
                                 connector=TCPConnector(limit=600)) as client:
            app['client'] = client
            # Spot teachers may still be restoring their models when the proxy
            # restarts. Wait for transport recovery, then verify model identity.
            while True:
                try:
                    async with client.get(upstream + '/v1/models', timeout=ClientTimeout(total=10)) as response:
                        response.raise_for_status()
                        models = await response.json()
                    break
                except (ClientConnectionError, asyncio.TimeoutError):
                    await asyncio.sleep(5)
                except ClientResponseError as error:
                    if error.status < 500:
                        raise
                    await asyncio.sleep(5)
            matching = [m for m in models['data'] if m['id'] == upstream_model]
            if not matching or matching[0].get('root') != config['expected_root']:
                raise ValueError('Upstream model differs from the approved configuration')
            app['models'] = {**models, 'data': [{**m, 'id': config['served_model']} for m in matching]}

            async def reload_limit():
                while True:
                    await asyncio.sleep(1)
                    current = json.loads(config_path.read_text())
                    if any(current[k] != config[k] for k in ('upstream', 'expected_root', 'served_model')):
                        raise ValueError('Restart the proxy to change upstream/model')
                    if current.get('upstream_model', current['served_model']) != upstream_model:
                        raise ValueError('Restart the proxy to change upstream model ID')
                    if current.get('request_timeout', 7260) != config.get('request_timeout', 7260):
                        raise ValueError('Restart the proxy to change request timeout')
                    await limit.resize(current['concurrency'])

            watcher = asyncio.create_task(reload_limit())
            app['watcher'] = watcher
            try:
                yield
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)

    async def status(request):
        return web.json_response({'upstream': upstream, 'model_root': config['expected_root'],
                                  'served_model': config['served_model'], 'upstream_model': upstream_model,
                                  'request_timeout': float(config.get('request_timeout', 7260)),
                                  'limit': limit.limit, 'active': limit.active, 'queued': limit.waiting,
                                  'completed': limit.completed, 'errors': limit.errors,
                                  'config_watcher_healthy': not app['watcher'].done(),
                                  'epoch': time.time()})

    async def models(request):
        return web.json_response(app['models'])

    async def metrics(request):
        # The production pool needs the actual vLLM scheduler/cache metrics even
        # when requests use a model-alias proxy. Do not take a generation slot.
        async with app['client'].get(upstream + '/metrics', timeout=ClientTimeout(total=10)) as response:
            return web.Response(status=response.status, body=await response.read(),
                                content_type=response.content_type)

    async def forward(request):
        body = await request.json()
        if body.get('stream') or body.get('model') != config['served_model']:
            raise web.HTTPBadRequest(text='Requires non-streaming requests for the configured teacher')
        body = {**body, 'model': upstream_model}
        async with limit.slot():
            try:
                async with app['client'].post(upstream + request.path, json=body) as response:
                    content = await response.read()
                    if response.status >= 400:
                        limit.errors += 1
                    else:
                        limit.completed += 1
                    return web.Response(status=response.status, body=content,
                                        content_type=response.content_type)
            except (asyncio.CancelledError, Exception):
                limit.errors += 1
                raise

    app.cleanup_ctx.append(resources)
    app.router.add_get('/status', status)
    app.router.add_get('/v1/models', models)
    app.router.add_get('/metrics', metrics)
    app.router.add_post('/v1/completions', forward)
    app.router.add_post('/v1/chat/completions', forward)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int, default=18101)
    args = parser.parse_args()
    web.run_app(make_app(args.config), host='127.0.0.1', port=args.port, access_log=None, handler_cancellation=True)
