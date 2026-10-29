import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import generate_candidate_matches
from matchminer_ai.config import load_default_preset
from matchminer_ai.embedding import late_interaction as encoding
from matchminer_ai.embedding.colbert import maxsim
from matchminer_ai.embedding.colbert_contract import INSTRUCTION, regression_query
from matchminer_ai.embedding.packed_scoring import PackedTokens, score_pairs


class Encoder:
    artifact_sha256 = 'test-weights'

    def __init__(self):
        self.calls = []

    def fingerprint(self):
        return dict(artifact_sha256=self.artifact_sha256, query_length=8192,
                    document_length=8192, score='mean_maxsim')

    def encode(self, texts, *, is_query, batch_size):
        self.calls.append((texts, is_query, batch_size))
        return [torch.eye(3).half()[:2 if is_query else 3] for _ in texts]


def table(identifier, ids, queries, documents, artifact='weights'):
    contract = dict(version='cohort-colbert-encoding-v1', artifact_sha256=artifact,
        query_instruction=INSTRUCTION, score='mean_maxsim', query_length=8192,
        document_length=8192, storage_dtype='float16', roles='Q_instruction_full_text/D_full_text')
    return pd.DataFrame({identifier: ids,
        'query_embedding': encoding.token_series(queries, range(len(ids))),
        'embedding': encoding.token_series(documents, range(len(ids))),
        'embedding_contract': json.dumps(contract, sort_keys=True)})


def test_public_embedding_uses_full_text_and_separate_roles(monkeypatch):
    model = Encoder()
    monkeypatch.setattr(encoding, '_model', lambda *args: model)
    cfg = load_default_preset()
    cfg.embedding = dict(backend='colbert', model_path='lightonai/GTE-ModernColBERT-v1')
    patient = pd.DataFrame([dict(patient_id='p', patient_summary='History.',
                                 patient_boilerplate_text='Kidney history.')])
    result, metadata = embed_for_matching(patient, entity_type='patient', config=cfg, return_metadata=True)
    expected = 'History.\n\nOther relevant history:\nKidney history.'
    assert model.calls == [([regression_query('patient_to_cohort', expected)], True, 16),
                           ([expected], False, 16)]
    assert np.asarray(result.embedding.iloc[0]).shape == (3, 3)
    assert np.asarray(result.query_embedding.iloc[0]).shape == (2, 3)
    assert json.loads(result.embedding_contract.iloc[0]) == metadata['model_metadata']['embedding_model']['encoding_contract']
    model.calls.clear()
    cohort = pd.DataFrame([dict(cohort_trial_id='t-1', cohort_eligibility_criteria='Specific.\n\nCommon.')])
    trial_result = embed_for_matching(cohort, entity_type='trial', config=cfg)
    assert model.calls[0][0] == [regression_query('cohort_to_patient', 'Specific.\n\nCommon.')]
    assert model.calls[1][0] == ['Specific.\n\nCommon.']
    assert trial_result.embedding_contract.iloc[0] == result.embedding_contract.iloc[0]


def test_packed_maxsim_is_exact_for_ragged_tokens_and_negative_padding():
    rng = torch.Generator().manual_seed(18)
    queries = [torch.nn.functional.normalize(torch.randn(n, 7, generator=rng), dim=-1) for n in (1, 3, 8)]
    documents = [torch.nn.functional.normalize(torch.randn(n, 7, generator=rng), dim=-1) for n in (2, 9, 4)]
    qids, dids = [2, 0, 1, 2, 0], [0, 2, 1, 0, 0]
    scores = score_pairs(PackedTokens(queries, 'cpu'), PackedTokens(documents, 'cpu'), qids, dids,
                         pair_batch_size=3, query_block=2, document_block=3)
    assert scores == pytest.approx([maxsim(queries[q], documents[d]).item() for q, d in zip(qids, dids)], abs=1e-6)


@pytest.mark.parametrize('budget', [0, 8])
@pytest.mark.parametrize('reverse', [False, True])
def test_public_retrieval_uses_query_role_in_both_directions_and_roundtrips_parquet(tmp_path, budget, reverse):
    p = table('patient_id', ['p1', 'p2'],
        [np.array([[1., 0.]]), np.array([[0., 1.], [1., 0.]])],
        [np.array([[-1., 0.], [0., 1.]]), np.array([[1., 0.]])])
    c = table('cohort_trial_id', ['c1', 'c2', 'c3'],
        [np.array([[0., -1.]]), np.array([[1., 0.]]), np.array([[-1., 0.]])],
        [np.array([[0., 1.], [1., 0.]]), np.array([[-1., 0.]]), np.array([[1., 0.]])])
    p.to_parquet(tmp_path / 'p.parquet')
    c.to_parquet(tmp_path / 'c.parquet')
    p, c = pd.read_parquet(tmp_path / 'p.parquet'), pd.read_parquet(tmp_path / 'c.parquet')
    query, corpus = (c, p) if reverse else (p, c)
    qid, did = ('cohort_trial_id', 'patient_id') if reverse else ('patient_id', 'cohort_trial_id')
    cfg = SimpleNamespace(embedding=dict(device='cpu', device_cache_gb=budget,
        retrieval_query_batch_size=2, retrieval_document_batch_size=2, pair_batch_size=3))
    for k in (1, 2, None, 20):
        result = generate_candidate_matches(query, corpus, k=k, config=cfg)
        for i, row in query.iterrows():
            q = torch.tensor(np.stack(row.query_embedding)).half()
            expected = [maxsim(q, torch.tensor(np.stack(d)).half()).item() for d in corpus.embedding]
            ordered = sorted(range(len(expected)), key=lambda j: (-expected[j], j))[:k]
            actual = result[result[qid] == row[qid]]
            assert actual[did].tolist() == corpus[did].iloc[ordered].tolist()
            assert actual.similarity_score.tolist() == pytest.approx([expected[j] for j in ordered], abs=1e-6)
        assert result.attrs['retrieval']['corpus_chunks'] == (len(corpus) if budget == 0 else 1)


def test_rejects_mixed_model_and_missing_role_or_provenance():
    q = table('patient_id', ['p'], [np.eye(2)], [np.eye(2)])
    d = table('cohort_trial_id', ['c'], [np.eye(2)], [np.eye(2)], artifact='other-weights')
    with pytest.raises(ValueError, match='contracts differ'):
        generate_candidate_matches(q, d)
    with pytest.raises(ValueError, match='query_embedding'):
        generate_candidate_matches(q.drop(columns='query_embedding'), d)
    with pytest.raises(ValueError, match='embedding_contract'):
        generate_candidate_matches(q.drop(columns='embedding_contract'), d)
    with pytest.raises(ValueError, match='Sequence embeddings must be vectors'):
        generate_candidate_matches(q[['patient_id', 'embedding']], d[['cohort_trial_id', 'embedding']])


def test_colbert_token_count_includes_direction_instruction_and_marker(monkeypatch):
    from matchminer_ai.embedding import inference
    seen = []
    def tokenize(texts, **kwargs):
        seen.extend(texts)
        return dict(input_ids=[[1, 2, 3] for _ in texts])
    monkeypatch.setattr(inference, '_get_embedding_tokenizer', lambda _: tokenize)
    cfg = dict(backend='colbert', model_path='model', query_direction='cohort_to_patient')
    assert inference.count_embedding_tokens(['Full criteria'], embedding_config=cfg) == [4]
    assert seen == [regression_query('cohort_to_patient', 'Full criteria')]


def test_retrieval_export_resolves_contract_once_per_table_without_prompt_io(monkeypatch):
    from pathlib import Path
    from matchminer_ai.matching import late_interaction as matching
    q = table('patient_id', ['p1', 'p2'], [np.eye(2)] * 2, [np.eye(2)] * 2)
    d = table('cohort_trial_id', ['c1', 'c2', 'c3'], [np.eye(2)] * 3, [np.eye(2)] * 3)
    calls = []
    original = matching._contract
    def once(frame):
        calls.append(id(frame))
        return original(frame)
    def no_prompt_reads(*args, **kwargs):
        raise AssertionError('Retrieval/export must not reread prompt files per pair')
    monkeypatch.setattr(matching, '_contract', once)
    monkeypatch.setattr(Path, 'read_text', no_prompt_reads)
    result = generate_candidate_matches(q, d, k=None, config=SimpleNamespace(embedding={'device': 'cpu'}))
    assert len(result) == 6
    assert calls == [id(q), id(d)]
    assert result.attrs['retrieval']['encoding_contract'] == original(q)
