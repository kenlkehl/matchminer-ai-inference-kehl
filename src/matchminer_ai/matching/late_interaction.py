"""Exact, bounded-memory mean-MaxSim retrieval with resident corpus tokens."""
import json
import math

import pandas as pd
import torch

from matchminer_ai.embedding.colbert_contract import INSTRUCTION
from matchminer_ai.embedding.packed_scoring import PackedTokens, score_pairs


def _contract(frame):
    if 'embedding_contract' not in frame:
        raise ValueError('ColBERT embeddings require embedding_contract; regenerate both sides')
    signatures = frame['embedding_contract'].unique().tolist()
    if len(signatures) != 1 or not isinstance(signatures[0], str):
        raise ValueError('Each ColBERT table must use one encoding contract')
    contract = json.loads(signatures[0])
    if (contract.get('version') != 'cohort-colbert-encoding-v1'
            or contract.get('score') != 'mean_maxsim'
            or not contract.get('artifact_sha256')
            or contract.get('query_instruction') != INSTRUCTION
            or contract.get('roles') != 'Q_instruction_full_text/D_full_text'
            or contract.get('storage_dtype') != 'float16'):
        raise ValueError('Unsupported ColBERT encoding contract; regenerate embeddings')
    return contract


def _vectors(values):
    import numpy as np
    result = []
    for value in values:
        # Parquet can round-trip a ragged matrix as an object array of rows.
        if not isinstance(value, torch.Tensor):
            value = np.asarray(value)
            if value.dtype == object:
                value = np.stack(value)
        tensor = torch.as_tensor(value, dtype=torch.float16, device='cpu')
        if tensor.ndim != 2 or min(tensor.shape) == 0 or not torch.isfinite(tensor).all():
            raise ValueError('ColBERT requires finite, nonempty token matrices')
        result.append(tensor)
    return result


def _corpus_chunks(vectors, budget):
    """Pack a fitting corpus once, or stream bounded subsets without truncation."""
    start, size = 0, 0
    for index, vector in enumerate(vectors):
        needed = vector.numel() * vector.element_size()
        if index > start and size + needed > budget:
            yield start, index
            start, size = index, 0
        size += needed
    if vectors:
        yield start, len(vectors)


def colbert_matches(query_df, corpus_df, *, query_id_col, corpus_id_col, k, settings):
    columns = [query_id_col, corpus_id_col, 'similarity_score', 'rank']
    if query_id_col == corpus_id_col or (query_id_col != 'patient_id' and corpus_id_col != 'patient_id'):
        raise ValueError('ColBERT matching requires one patient side and one trial/cohort side')
    for frame, column in ((query_df, 'query_embedding'), (corpus_df, 'embedding')):
        if column not in frame:
            raise ValueError(f'ColBERT requires {column}; embed both query/document roles')
    if query_df.empty or corpus_df.empty:
        return pd.DataFrame(columns=columns)
    if _contract(query_df) != _contract(corpus_df):
        raise ValueError('ColBERT model/prompt/context contracts differ; regenerate both sides')
    queries, documents = _vectors(query_df.query_embedding), _vectors(corpus_df.embedding)
    if len({vector.shape[1] for vector in queries + documents}) != 1:
        raise ValueError('ColBERT token dimensions differ')
    device = torch.device(settings.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    pair_batch = int(settings.get('pair_batch_size', 512))
    query_batch = int(settings.get('retrieval_query_batch_size', 64))
    document_batch = int(settings.get('retrieval_document_batch_size', 256))
    budget_gb = float(settings.get('device_cache_gb', 8))
    if min(pair_batch, query_batch, document_batch) < 1 or not math.isfinite(budget_gb) or budget_gb < 0:
        raise ValueError('Batch sizes must be positive and device_cache_gb nonnegative')
    budget = budget_gb * 1024**3
    if device.type == 'cuda':
        # Leave room for the encoder and bounded scoring intermediates.
        budget = min(budget, max(0, torch.cuda.mem_get_info(device)[0] - 8 * 1024**3))
    top_k = len(documents) if k is None else min(k, len(documents))
    best = [[] for _ in queries]
    chunks = 0
    for start, end in _corpus_chunks(documents, budget):
        packed_documents = PackedTokens(documents[start:end], device)
        chunks += 1
        for qstart in range(0, len(queries), query_batch):
            current = queries[qstart:qstart + query_batch]
            packed_queries = PackedTokens(current, device)
            for dstart in range(0, end - start, document_batch):
                dids = list(range(dstart, min(dstart + document_batch, end - start)))
                qids = [q for q in range(len(current)) for _ in dids]
                scores = score_pairs(packed_queries, packed_documents, qids, dids * len(current),
                                     pair_batch_size=pair_batch)
                for q in range(len(current)):
                    candidates = best[qstart + q] + list(zip(
                        scores[q * len(dids):(q + 1) * len(dids)],
                        [start + d for d in dids], strict=True))
                    # Deterministic corpus-order ties, including across chunks.
                    best[qstart + q] = sorted(candidates, key=lambda item: (-item[0], item[1]))[:top_k]
            del packed_queries
        del packed_documents
    query_ids = query_df[query_id_col].astype(str).tolist()
    corpus_ids = corpus_df[corpus_id_col].astype(str).tolist()
    rows = [{query_id_col: query_ids[q], corpus_id_col: corpus_ids[d],
             'similarity_score': score, 'rank': rank}
            for q, matches in enumerate(best) for rank, (score, d) in enumerate(matches, 1)]
    result = pd.DataFrame(rows, columns=columns)
    result.attrs['retrieval'] = dict(method='colbert', score='mean_maxsim', exhaustive=True,
        corpus_chunks=chunks, pair_batch_size=pair_batch, query_batch_size=query_batch,
        encoding_contract=_contract(query_df))
    return result
