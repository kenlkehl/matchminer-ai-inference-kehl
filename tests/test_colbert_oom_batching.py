from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch

from matchminer_ai.embedding.colbert import ColBERT


def fake_encoder(monkeypatch, encode):
    monkeypatch.setattr(torch, 'autocast', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, 'device', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    model = SimpleNamespace(device=torch.device('cuda'), eval=lambda: None, encode_texts=encode)
    model.encode_batch = MethodType(ColBERT.encode_batch, model)
    return model


def test_oom_retries_smaller_complete_batches_and_preserves_roles_order_dtype(monkeypatch):
    calls = []
    def encode(texts, *, is_queries):
        calls.append((list(texts), is_queries))
        if len(texts) > 2:
            raise torch.cuda.OutOfMemoryError('synthetic batch limit')
        return [torch.full((len(text), 2), int(text[0]), dtype=torch.float32) for text in texts]
    model = fake_encoder(monkeypatch, encode)
    texts = [str(i) * (i + 1) for i in range(6)]
    result = model.encode_batch(texts, is_query=False)
    assert [r.shape[0] for r in result] == list(range(1, 7))
    assert [r[0, 0].item() for r in result] == list(range(6))
    assert all(r.dtype == torch.float16 and r.device.type == 'cpu' for r in result)
    assert all(roles == [False] * len(batch) for batch, roles in calls)


@pytest.mark.parametrize('error', [torch.cuda.OutOfMemoryError, ValueError])
def test_single_input_oom_and_other_errors_are_not_hidden(monkeypatch, error):
    def encode(texts, *, is_queries):
        raise error('synthetic error')
    model = fake_encoder(monkeypatch, encode)
    with pytest.raises(error, match='synthetic error'):
        model.encode_batch(['complete text'], is_query=True)
