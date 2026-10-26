"""Bounded CPU token cache; never cache learned embeddings or autograd graphs."""
from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedText:
    inputs: dict
    keep_indices: torch.Tensor

    @property
    def length(self):
        return len(self.inputs['input_ids'])


class TokenCache:
    def __init__(self, max_tokens=0):
        if max_tokens < 0:
            raise ValueError('Token cache capacity must be nonnegative')
        self.max_tokens = max_tokens
        self.rows = OrderedDict()
        self.tokens = self.hits = self.misses = 0

    def prepare(self, tokenizer, texts, roles, prefixes, skiplist):
        keys = list(zip(roles, texts, strict=True))
        selected, missing = {}, {}
        for key in keys:
            if key in selected or key in missing:
                continue
            if key in self.rows:
                selected[key] = self.rows[key]
                self.rows.move_to_end(key)
                self.hits += 1
            else:
                missing[key] = None
                self.misses += 1
        missing = list(missing)
        # Bound transient CPU padding even for a whole effective training batch.
        for start in range(0, len(missing), 128):
            batch = missing[start:start + 128]
            encoded = tokenizer([text for _, text in batch], return_tensors='pt',
                                padding=True, truncation=False)
            lengths = encoded['attention_mask'].sum(dim=1).tolist()
            for index, ((role, text), length) in enumerate(zip(batch, lengths, strict=True)):
                inputs = {}
                for name, values in encoded.items():
                    marker = prefixes['query' if role else 'document'] if name == 'input_ids' else int(name == 'attention_mask')
                    value = values[index, :length]
                    inputs[name] = torch.cat((value[:1], value.new_tensor([marker]), value[1:]))
                ids = inputs['input_ids']
                keep = torch.arange(len(ids))
                if not role and skiplist:
                    keep = keep[~torch.isin(ids, torch.tensor(sorted(skiplist), dtype=ids.dtype))]
                row = PreparedText(inputs, keep)
                key = role, text
                selected[key] = row
                if row.length <= self.max_tokens:
                    while self.rows and self.tokens + row.length > self.max_tokens:
                        _, old = self.rows.popitem(last=False)
                        self.tokens -= old.length
                    self.rows[key] = row
                    self.tokens += row.length
        return [selected[key] for key in keys]

    def info(self):
        return dict(tokens=self.tokens, capacity_tokens=self.max_tokens,
                    entries=len(self.rows), hits=self.hits, misses=self.misses)
