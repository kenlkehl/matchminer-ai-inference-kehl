"""Native ColBERT encoder shared by matching and TrialSpace training.

Uses the published 128-dimensional projection, Q/D markers and punctuation mask.
No dependency upgrades to the production generation environment are required.
"""
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .colbert_inputs import TokenCache

BASE_MODEL = 'lightonai/GTE-ModernColBERT-v1'


def maxsim(query, document, *, query_block=128, document_block=2048):
    """Mean MaxSim with bounded intermediates and autograd through every token.

Mean, rather than sum, reduces query-length scaling. It is a ranking score,
not a calibrated binary probability; training and retrieval use this same rule.
"""
    if not len(query) or not len(document):
        raise ValueError('Cannot score empty token embeddings')
    scores = []
    for q in query.split(query_block):
        best = None
        for d in document.split(document_block):
            values = (q @ d.T).float().max(dim=1).values
            best = values if best is None else torch.maximum(best, values)
        scores.append(best)
    return torch.cat(scores).mean()


def maxsim_batch(query, documents, *, query_block=128, document_block=2048, batch_size=32):
    """Differentiable mean-MaxSim over length-bucketed documents in parallel."""
    if not len(query) or not documents or any(not len(d) for d in documents):
        raise ValueError('Cannot score empty token embeddings')
    if min(query_block, document_block, batch_size) < 1:
        raise ValueError('MaxSim block and batch sizes must be positive')
    buckets = defaultdict(list)
    for index, document in enumerate(documents):
        buckets[(len(document) - 1).bit_length()].append(index)
    output = [None] * len(documents)
    for indices in buckets.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            padded = nn.utils.rnn.pad_sequence([documents[i] for i in selected], batch_first=True)
            lengths = torch.tensor([len(documents[i]) for i in selected], device=padded.device)
            chunks = []
            for q in query.split(query_block):
                best = None
                for dstart in range(0, padded.shape[1], document_block):
                    d = padded[:, dstart:dstart + document_block]
                    valid = torch.arange(dstart, dstart + d.shape[1], device=d.device)[None] < lengths[:, None]
                    similarities = torch.bmm(q.expand(len(selected), -1, -1), d.transpose(1, 2)).float()
                    values = similarities.masked_fill(~valid[:, None], -torch.inf).max(dim=-1).values
                    best = values if best is None else torch.maximum(best, values)
                chunks.append(best)
            scores = torch.cat(chunks, dim=1).mean(dim=1)
            for index, score in zip(selected, scores, strict=True):
                output[index] = score
    return torch.stack(output)


class ColBERT(nn.Module):
    def __init__(self, model=BASE_MODEL, *, revision=None, query_length=8192,
                 document_length=8192, cache_dir=None, checkpointing=False, length_bucketing=True):
        super().__init__()
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        folder = Path(model)
        if not folder.is_dir():
            folder = Path(snapshot_download(model, revision=revision, cache_dir=cache_dir,
                allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Dense/*.json', '1_Dense/*.safetensors']))
        self.folder = folder
        self.settings = json.loads((folder / 'config_sentence_transformers.json').read_text())
        projection = json.loads((folder / '1_Dense/config.json').read_text())
        if projection['activation_function'] != 'torch.nn.modules.linear.Identity':
            raise ValueError('Unsupported checkpoint projection')
        self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, use_fast=True)
        self.encoder = AutoModel.from_pretrained(folder, local_files_only=True,
            trust_remote_code=False, attn_implementation='sdpa')
        # Avoid implicit compilation inside ModernBERT, including long-context probes.
        self.encoder.config.reference_compile = False
        self.projection = nn.Linear(projection['in_features'], projection['out_features'], bias=projection['bias'])
        self.projection.load_state_dict({k.removeprefix('linear.'): v for k, v in
                                        load_file(folder / '1_Dense/model.safetensors').items()})
        self.query_length, self.document_length = query_length, document_length
        self.length_bucketing = length_bucketing
        capacity = self.encoder.config.max_position_embeddings
        if min(query_length, document_length) < 4 or max(query_length, document_length) > capacity:
            raise ValueError(f'Lengths must fit the checkpoint context ({capacity}); text is never silently truncated')
        self.tokenizer.model_max_length = max(query_length, document_length)
        if self.settings.get('do_query_expansion'):
            raise ValueError('This encoder requires a checkpoint without query expansion')
        self.prefixes = {}
        for role in ('query', 'document'):
            marker = self.settings[role + '_prefix']
            if marker not in self.tokenizer.get_vocab():
                raise ValueError(f'Missing trained {role} marker: {marker!r}')
            self.prefixes[role] = self.tokenizer.convert_tokens_to_ids(marker)
        self.skiplist = {self.tokenizer.convert_tokens_to_ids(s) for s in self.settings.get('skiplist_words', [])}
        if checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.source_revision = folder.name
        hasher = hashlib.sha256()
        for name in ('model.safetensors', '1_Dense/model.safetensors', 'config.json', 'config_sentence_transformers.json', 'tokenizer.json'):
            with (folder / name).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    hasher.update(block)
        self.artifact_sha256 = hasher.hexdigest()

    def fingerprint(self):
        return dict(artifact_sha256=self.artifact_sha256, query_length=self.query_length,
                    document_length=self.document_length, score='mean_maxsim')

    @property
    def device(self):
        return self.projection.weight.device

    def tokens(self, text, *, is_query):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Nonempty input text is required')
        inputs = self.tokenizer(text, return_tensors='pt', truncation=False)
        role = 'query' if is_query else 'document'
        for key, value in inputs.items():
            marker = self.prefixes[role] if key == 'input_ids' else 1 if key == 'attention_mask' else 0
            inputs[key] = torch.cat((value[:, :1], torch.full((1, 1), marker, dtype=value.dtype), value[:, 1:]), dim=1)
        length = inputs['input_ids'].shape[1]
        limit = self.query_length if is_query else self.document_length
        if length > limit:
            raise ValueError(f'{role} has {length} tokens, exceeding {limit}; full text retained, no truncation performed')
        return {key: value.to(self.device) for key, value in inputs.items()}

    def encode_text(self, text, *, is_query):
        return self.encode_texts([text], is_queries=[is_query])[0]

    def forward(self, query=None, documents=None, *, prepared_groups=None):
        """Score one query or several prepared groups through the DDP forward path."""
        if prepared_groups is None:
            prepared_groups = self.prepare_groups([query], [documents])
        elif query is not None or documents is not None:
            raise ValueError('Pass raw texts or prepared groups, not both')
        if not prepared_groups or any(len(group) < 2 for group in prepared_groups):
            raise ValueError('Each group requires a query and at least one document')
        embeddings = self.encode_prepared([row for group in prepared_groups for row in group])
        scores, offset = [], 0
        for group in prepared_groups:
            q, *docs = embeddings[offset:offset + len(group)]
            scores.append(maxsim_batch(q, docs))
            offset += len(group)
        return torch.cat(scores)

    def configure_token_cache(self, max_tokens):
        self._token_cache = TokenCache(max_tokens)

    def token_cache_info(self):
        return self._token_cache.info() if hasattr(self, '_token_cache') else {}

    def prepare_groups(self, queries, documents):
        if not queries or len(queries) != len(documents) or any(not docs for docs in documents):
            raise ValueError('Each query requires at least one document')
        texts, roles = [], []
        for query, docs in zip(queries, documents, strict=True):
            texts.extend([query, *docs])
            roles.extend([True] + [False] * len(docs))
        rows = self.prepare_texts(texts, is_queries=roles)
        groups, offset = [], 0
        for docs in documents:
            groups.append(rows[offset:offset + 1 + len(docs)])
            offset += 1 + len(docs)
        return groups

    def prepared_batch_tokens(self, groups):
        """Padded encoder tokens for packing; one oversized group stays complete."""
        buckets = defaultdict(list)
        for group in groups:
            for row in group:
                bucket = (row.length - 1).bit_length() if getattr(self, 'length_bucketing', True) else 0
                buckets[bucket].append(row.length)
        return sum(len(lengths) * max(lengths) for lengths in buckets.values())

    def execution_settings(self):
        return dict(encoder_length_bucketing=self.length_bucketing, scoring='batched_maxsim_v1',
                    inputs='cpu_prepared_integer_masks_v1')

    @torch.inference_mode()
    def encode(self, texts, *, is_query, storage_dtype=torch.float16, batch_size=16):
        if batch_size < 1:
            raise ValueError('Batch size must be positive')
        texts = list(texts)
        return [embedding for start in range(0, len(texts), batch_size)
                for embedding in self.encode_batch(texts[start:start + batch_size],
                    is_query=is_query, storage_dtype=storage_dtype)]

    def encode_texts(self, texts, *, is_queries):
        """Shared differentiable batch path for training and all inference roles."""
        return self.encode_prepared(self.prepare_texts(texts, is_queries=is_queries))

    def prepare_texts(self, texts, *, is_queries):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('A nonempty batch of nonempty texts is required')
        if len(is_queries) != len(texts) or any(type(role) is not bool for role in is_queries):
            raise ValueError('Each text requires an explicit query/document role')
        if self.tokenizer.padding_side != 'right':
            raise ValueError('The checkpoint batch contract requires right padding')
        if not hasattr(self, '_token_cache'):
            self.configure_token_cache(0)
        rows = self._token_cache.prepare(self.tokenizer, texts, is_queries, self.prefixes, self.skiplist)
        for row, query in zip(rows, is_queries, strict=True):
            limit = self.query_length if query else self.document_length
            if row.length > limit:
                raise ValueError(f'{"query" if query else "document"} has {row.length} tokens, exceeding {limit}; no truncation performed')
        return rows

    def encode_prepared(self, rows):
        buckets = defaultdict(list)
        for index, row in enumerate(rows):
            bucket = (row.length - 1).bit_length() if getattr(self, 'length_bucketing', True) else 0
            buckets[bucket].append(index)
        output = [None] * len(rows)
        for indices in buckets.values():
            batch = {key: nn.utils.rnn.pad_sequence([rows[i].inputs[key] for i in indices],
                        batch_first=True, padding_value=(getattr(self.tokenizer, 'pad_token_id', 0) or 0)
                        if key == 'input_ids' else 0).to(self.device) for key in rows[indices[0]].inputs}
            hidden = self.encoder(**batch).last_hidden_state
            embeddings = F.normalize(self.projection(hidden).float(), dim=-1)
            # CPU-derived integer indices avoid CUDA boolean compaction and its
            # device synchronization once per text. Transfer all indices together.
            sizes = [len(rows[i].keep_indices) for i in indices]
            keeps = torch.cat([rows[i].keep_indices for i in indices]).to(self.device).split(sizes)
            for index, embedding, keep in zip(indices, embeddings, keeps, strict=True):
                output[index] = embedding.index_select(0, keep)
        return output

    @torch.inference_mode()
    def encode_batch(self, texts, *, is_query, storage_dtype=torch.float16):
        self.eval()
        embeddings = None
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
                embeddings = self.encode_texts(texts, is_queries=[is_query] * len(texts))
            return [embedding.to(device='cpu', dtype=storage_dtype) for embedding in embeddings]
        except torch.cuda.OutOfMemoryError:
            if self.device.type != 'cuda' or len(texts) <= 1:
                raise
        # Leave the exception handler before retrying: its traceback can retain
        # the failed encoder's GPU tensors. This inference-only fallback keeps
        # every token and the original row order; training uses encode_texts.
        del embeddings
        logging.getLogger(__name__).warning(
            'ColBERT CUDA memory limit: retrying %d complete texts in smaller batches', len(texts))
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        middle = len(texts) // 2
        return (self.encode_batch(texts[:middle], is_query=is_query, storage_dtype=storage_dtype)
                + self.encode_batch(texts[middle:], is_query=is_query, storage_dtype=storage_dtype))

    def save(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        (path / '1_Dense').mkdir(exist_ok=True)
        save_file({'linear.' + k: v.detach().cpu().contiguous() for k, v in self.projection.state_dict().items()},
                  path / '1_Dense/model.safetensors')
        (path / '1_Dense/config.json').write_text((self.folder / '1_Dense/config.json').read_text())
        settings = {**self.settings, 'query_length': self.query_length, 'document_length': self.document_length}
        (path / 'config_sentence_transformers.json').write_text(json.dumps(settings, indent=2))
        (path / 'modules.json').write_text((self.folder / 'modules.json').read_text())
        (path / 'experiment_encoding.json').write_text(json.dumps(dict(
            score='mean_maxsim', query_length=self.query_length, document_length=self.document_length,
            source_revision=self.source_revision, silent_truncation=False), indent=2))

