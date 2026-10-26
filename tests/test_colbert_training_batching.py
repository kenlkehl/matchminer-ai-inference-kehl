import copy
from types import SimpleNamespace

import pytest
import torch

from matchminer_ai.embedding.colbert import ColBERT, maxsim, maxsim_batch


@pytest.mark.parametrize('tied', [False, True])
def test_batched_maxsim_preserves_scores_gradients_order_and_negative_padding(tied):
    torch.manual_seed(51)
    query = torch.randn(11, 7, requires_grad=True)
    documents = [torch.randn(n, 7, requires_grad=True) for n in (13, 3, 17, 6, 8)]
    if tied:
        query.data.fill_(-1)
        for document in documents:
            document.data.fill_(1)
    q2 = query.detach().clone().requires_grad_()
    d2 = [d.detach().clone().requires_grad_() for d in documents]
    reference = torch.stack([maxsim(query, d, query_block=3, document_block=4) for d in documents])
    actual = maxsim_batch(q2, d2, query_block=3, document_block=4, batch_size=2)
    torch.testing.assert_close(actual, reference)
    weights = torch.arange(1, len(documents) + 1)
    expected_grads = torch.autograd.grad((reference * weights).sum(), [query, *documents])
    actual_grads = torch.autograd.grad((actual * weights).sum(), [q2, *d2])
    for actual_grad, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected, rtol=2e-5, atol=2e-6)


class Tokenizer:
    padding_side = 'right'

    def __call__(self, texts, **kwargs):
        assert kwargs['truncation'] is False
        rows = [[1] + [4 if word == 'word' else 5 for word in text.split()] + [2] for text in texts]
        width = max(map(len, rows))
        return dict(input_ids=torch.tensor([r + [0] * (width - len(r)) for r in rows]),
                    attention_mask=torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]))


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.attention = torch.nn.MultiheadAttention(8, 2, dropout=0, batch_first=True)
        self.padded_tokens = 0

    def forward(self, input_ids, attention_mask):
        self.padded_tokens += input_ids.numel()
        x = self.embedding(input_ids)
        hidden, _ = self.attention(x, x, x, key_padding_mask=~attention_mask.bool(), need_weights=False)
        return SimpleNamespace(last_hidden_state=hidden)


def test_encoder_length_buckets_keep_complete_text_roles_order_and_gradients():
    torch.manual_seed(7)
    model = ColBERT.__new__(ColBERT)
    torch.nn.Module.__init__(model)
    model.tokenizer, model.encoder = Tokenizer(), Encoder()
    model.projection = torch.nn.Linear(8, 4, bias=False)
    model.prefixes, model.skiplist = {'query': 8, 'document': 9}, {5}
    model.query_length = model.document_length = 128
    model.length_bucketing = True
    reference = copy.deepcopy(model)
    reference.length_bucketing = False
    texts = ['word ' * 29, 'word .', 'word ' * 19, 'word . word']
    roles = [True, False, False, False]
    expected = reference.encode_texts(texts, is_queries=roles)
    actual = model.encode_texts(texts, is_queries=roles)
    assert model.encoder.padded_tokens < reference.encoder.padded_tokens
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
    sum(t[:, 0].sum() * (i + 1) for i, t in enumerate(actual)).backward()
    sum(t[:, 0].sum() * (i + 1) for i, t in enumerate(expected)).backward()
    for left, right in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=2e-5)
    model.query_length = 4
    with pytest.raises(ValueError, match='no truncation'):
        model.encode_texts(texts, is_queries=roles)


def test_batched_maxsim_rejects_empty_inputs():
    with pytest.raises(ValueError, match='empty'):
        maxsim_batch(torch.ones(1, 4), [torch.empty(0, 4)])
