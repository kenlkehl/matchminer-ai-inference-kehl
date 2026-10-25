"""Resident ragged token vectors and exact MaxSim across many query/document pairs."""
import torch


class PackedTokens:
    """One contiguous token matrix plus offsets; no corpus-wide padding."""
    def __init__(self, vectors, device):
        if not vectors or any(v.ndim != 2 or len(v) == 0 for v in vectors):
            raise ValueError('Nonempty token matrices are required')
        self.sizes = [len(v) for v in vectors]
        self.values = torch.cat(vectors).to(device)
        self.lengths = torch.tensor(self.sizes, device=device, dtype=torch.long)
        self.offsets = self.lengths.cumsum(0) - self.lengths

    def block(self, rows, start, width):
        positions = torch.arange(start, start + width, device=self.values.device)
        valid = positions[None] < self.lengths[rows, None]
        indices = self.offsets[rows, None] + positions[None]
        # Invalid positions are masked by the scorer, including positions that
        # land in the next document. Clamp only to make the gather memory-safe.
        values = self.values[indices.clamp_max(len(self.values) - 1)]
        return values, valid


@torch.inference_mode()
def score_pairs(queries, documents, query_indices, document_indices, *,
                pair_batch_size=512, query_block=128, document_block=2048):
    """Same mean-MaxSim as the reference, preserving pair order and padding masks.

    Bucket by both lengths to limit padding. Pair batching and the resident
    cache change execution only, not candidate selection or the scoring rule.
    """
    if len(query_indices) != len(document_indices):
        raise ValueError('Each query needs one paired document')
    if min(pair_batch_size, query_block, document_block) < 1:
        raise ValueError('Scoring batch and block sizes must be positive')
    if queries.values.device != documents.values.device or queries.values.dtype != documents.values.dtype:
        raise ValueError('Query and document vectors must share device and dtype')
    device = queries.values.device
    order = sorted(range(len(query_indices)), key=lambda i: (
        (queries.sizes[query_indices[i]] + 127) // 128,
        (documents.sizes[document_indices[i]] + 255) // 256))
    results = [None] * len(order)
    for begin in range(0, len(order), pair_batch_size):
        selected = order[begin:begin + pair_batch_size]
        qids = torch.tensor([query_indices[i] for i in selected], device=device)
        dids = torch.tensor([document_indices[i] for i in selected], device=device)
        nq = max(queries.sizes[query_indices[i]] for i in selected)
        nd = max(documents.sizes[document_indices[i]] for i in selected)
        scores = torch.zeros(len(selected), device=device, dtype=torch.float32)
        for qstart in range(0, nq, query_block):
            q, qvalid = queries.block(qids, qstart, min(query_block, nq - qstart))
            best = torch.full(qvalid.shape, -torch.inf, device=device)
            for dstart in range(0, nd, document_block):
                d, dvalid = documents.block(dids, dstart, min(document_block, nd - dstart))
                similarities = torch.bmm(q, d.transpose(1, 2)).float()
                similarities.masked_fill_(~dvalid[:, None, :], -torch.inf)
                best = torch.maximum(best, similarities.max(dim=-1).values)
            scores += best.masked_fill(~qvalid, 0).sum(dim=-1)
        scores /= queries.lengths[qids]
        for index, value in zip(selected, scores.cpu().tolist(), strict=True):
            results[index] = value
    return results
