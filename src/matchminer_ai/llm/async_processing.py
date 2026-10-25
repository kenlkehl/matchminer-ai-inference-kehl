"""Shared bounded background preprocessing and durable I/O for LLM clients."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
import threading
import time

import hashlib
import json


def prompt_digest(messages):
    """Stable complete-message identity, shared with training checkpoints."""
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class BackgroundWork:
    """Keep filesystem/CPU jobs out of asyncio's DNS/default thread pool.

    A cancelled caller does not release capacity until its thread actually ends.
    Shutdown joins outstanding work, including atomic reasoning-checkpoint writes.
    """

    def __init__(self, workers=32, pending=64, name='label-storage'):
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name)
        self.slots = asyncio.Semaphore(pending)

    async def __aenter__(self):
        return self

    async def run(self, function, *args, **kwargs):
        await self.slots.acquire()
        try:
            future = asyncio.get_running_loop().run_in_executor(
                self.executor, copy_context().run, partial(function, *args, **kwargs))
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # In particular, a cancelled status writer must finish before its
            # replacement writes the same atomic temporary/checkpoint path.
            try:
                await asyncio.shield(future)
            except Exception:
                pass
            raise

    async def __aexit__(self, *exc):
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)


class PromptPreparer:
    """Batch independent prompts without changing their text or token budgets.

    Native tokenizer encoding is read-only and can run concurrently. Use a
    private backend with padding/truncation disabled, avoiding the mutable
    enable/disable operations in Hugging Face's high-level batch wrapper.
    The fallback supports custom/slow tokenizers under a lock.
    """

    def __init__(self, tokenizer, template_kwargs, *, context, max_tokens,
                 workers=4, batch_size=32):
        self.tokenizer = tokenizer
        self.template_kwargs = template_kwargs
        self.context, self.max_tokens = context, max_tokens
        self.workers, self.batch_size = workers, batch_size
        self.queue = asyncio.Queue(maxsize=workers * batch_size)
        self.background = BackgroundWork(workers, workers, 'label-preprocess')
        self.fallback_lock = threading.Lock()
        self.backend = None
        if getattr(tokenizer, 'backend_tokenizer', None) is not None:
            from tokenizers import Tokenizer
            self.backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
            self.backend.no_padding()
            self.backend.no_truncation()
        self.stats = dict(batches=0, requests=0, worker_seconds=0.0)

    def _prepare(self, messages):
        started = time.monotonic()
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False,
            add_generation_prompt=True, **self.template_kwargs) for m in messages]
        if self.backend is not None:
            lengths = [len(row) for row in self.backend.encode_batch(prompts, add_special_tokens=False)]
        else:
            with self.fallback_lock:
                lengths = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        prepared = [dict(prompt=prompt, tokens=length, prompt_sha256=prompt_digest(m))
                    for m, prompt, length in zip(messages, prompts, lengths)]
        return prepared, time.monotonic() - started

    async def _worker(self):
        while True:
            batch = [await self.queue.get()]
            try:
                # Yield once to coalesce simultaneous arrivals, without an artificial
                # timer delaying an isolated question or its dependent follow-up.
                await asyncio.sleep(0)
                while len(batch) < self.batch_size:
                    try:
                        batch.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                batch = [(m, f) for m, f in batch if not f.cancelled()]
                if not batch:
                    continue
                rows, elapsed = await self.background.run(self._prepare, [m for m, _ in batch])
                self.stats['batches'] += 1
                self.stats['requests'] += len(rows)
                self.stats['worker_seconds'] += elapsed
                for (_, future), row in zip(batch, rows):
                    if future.done():
                        continue
                    if row['tokens'] + self.max_tokens + 256 > self.context:
                        future.set_exception(ValueError(
                            'Full teacher input and output exceed context; truncation is disabled'))
                    else:
                        future.set_result(row)
            except asyncio.CancelledError:
                for _, future in batch:
                    future.cancel()
                raise
            except Exception as error:
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)

    async def prepare(self, messages):
        future = asyncio.get_running_loop().create_future()
        try:
            await self.queue.put((messages, future))
            return await future
        finally:
            if not future.done():
                future.cancel()

    async def __aenter__(self):
        self.tasks = [asyncio.create_task(self._worker()) for _ in range(self.workers)]
        return self

    async def __aexit__(self, *exc):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        while not self.queue.empty():
            _, future = self.queue.get_nowait()
            future.cancel()
        await self.background.__aexit__(*exc)
