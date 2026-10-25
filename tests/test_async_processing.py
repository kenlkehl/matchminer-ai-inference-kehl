import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import threading
import time

import pytest

from matchminer_ai.llm.async_processing import prompt_digest as digest
from matchminer_ai.llm.async_processing import BackgroundWork, PromptPreparer


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return '|'.join(m['content'] for m in messages)

    def encode(self, prompt, **kwargs):
        assert kwargs == dict(add_special_tokens=False)
        return list(prompt)


def test_batches_preserve_independent_prompts_hashes_and_full_context_check():
    async def scenario():
        messages = [[dict(role='user', content=text)] for text in ['short', 'x' * 101, 'é🙂']]
        async with PromptPreparer(Tokenizer(), {'enable_thinking': True},
                                  context=366, max_tokens=10, workers=1) as preparer:
            rows = await asyncio.gather(*(preparer.prepare(m) for m in messages), return_exceptions=True)
            assert rows[0] == dict(prompt='short', tokens=5, prompt_sha256=digest(messages[0]))
            assert isinstance(rows[1], ValueError) and 'truncation is disabled' in str(rows[1])
            assert rows[2] == dict(prompt='é🙂', tokens=2, prompt_sha256=digest(messages[2]))
            assert preparer.stats['batches'] == 1
    asyncio.run(scenario())


def test_native_batch_token_counts_match_untruncated_scalar_encoding():
    from tokenizers import Tokenizer as Native, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Native(models.WordLevel({'[UNK]': 0, 'a': 1, 'b': 2, 'é': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    messages = [[dict(role='user', content=text)] for text in ['a b é', 'a ' * 30, '🙂']]
    expected = [len(tokenizer.encode(m[0]['content'], add_special_tokens=False)) for m in messages]
    # The private backend must clear a saved tokenizer's mutable truncation/
    # padding state, and must not mutate the original tokenizer while running.
    backend = tokenizer.backend_tokenizer
    backend.enable_truncation(2)
    backend.enable_padding(length=40, pad_id=0)
    async def scenario():
        async with PromptPreparer(tokenizer, {}, context=1024, max_tokens=10) as preparer:
            rows = await asyncio.gather(*(preparer.prepare(m) for m in messages))
            assert [r['tokens'] for r in rows] == expected
        assert backend.truncation['max_length'] == 2
        assert backend.padding['length'] == 40
    asyncio.run(scenario())


def test_slow_preprocessing_does_not_block_network_event_loop():
    class SlowTokenizer(Tokenizer):
        def encode(self, *args, **kwargs):
            time.sleep(.04)
            return super().encode(*args, **kwargs)
    async def scenario():
        ticks = 0
        async def heartbeat():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(.005)
        async with PromptPreparer(SlowTokenizer(), {'enable_thinking': True},
                                  context=1024, max_tokens=10, workers=1) as preparer:
            pulse = asyncio.create_task(heartbeat())
            try:
                await asyncio.gather(*(preparer.prepare([dict(role='user', content='a')]) for _ in range(8)))
                assert ticks >= 20
            finally:
                pulse.cancel()
                await asyncio.gather(pulse, return_exceptions=True)
    asyncio.run(scenario())


def test_storage_saturation_leaves_default_network_executor_available_and_joins_cancelled_writes():
    async def scenario():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        entered, release = threading.Event(), threading.Event()
        durable = []
        def slow_write():
            entered.set()
            release.wait(3)
            durable.append(True)
        async with BackgroundWork(workers=1, pending=1) as storage:
            task = asyncio.create_task(storage.run(slow_write))
            try:
                while not entered.is_set():
                    await asyncio.sleep(.001)
                assert await asyncio.wait_for(asyncio.to_thread(lambda: 'DNS is free'), .5) == 'DNS is free'
                task.cancel()
                await asyncio.sleep(.01)
                assert not task.done()  # Do not abandon an in-progress atomic write.
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert durable == [True]
    asyncio.run(scenario())


def test_background_work_retains_caller_context():
    marker = ContextVar('synthetic_trace_stage', default='missing')
    async def scenario():
        marker.set('label-stage')
        async with BackgroundWork(workers=1, pending=1) as work:
            assert await work.run(marker.get) == 'label-stage'
    asyncio.run(scenario())


def test_preprocessing_errors_and_shutdown_do_not_leave_waiters():
    class BrokenTokenizer(Tokenizer):
        def encode(self, *args, **kwargs):
            raise ValueError('synthetic tokenizer error')
    async def scenario():
        async with PromptPreparer(BrokenTokenizer(), {'enable_thinking': True},
                                  context=1024, max_tokens=10, workers=1) as preparer:
            answers = await asyncio.gather(*(preparer.prepare([dict(role='user', content='x')])
                                            for _ in range(3)), return_exceptions=True)
            assert all(isinstance(answer, ValueError) for answer in answers)
        assert all(task.done() for task in preparer.tasks)
    asyncio.run(scenario())
