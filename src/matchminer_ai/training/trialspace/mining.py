"""TrialSpace (dense, single-vector) mining: encode the corpus, rank both directions, add lexical and random candidates.

Patients' key summary sections and cohorts' key criteria are encoded once with EmbeddingGemma 2 (or a fitted
TrialSpace) through the inference package's encoder, on one or more GPUs (``--devices``) or over vLLM
embedding servers; scores are exact cosine similarities in query batches (``matchminer_ai.matching.retrieval``).
The mining policy is training's: each patient searches only the cohorts of its sampled trials, one version per
trial (TrialSampler); cohort queries use one version per trial and search every patient (query_cohorts); TF-IDF
and seeded random candidates come from the same search sets, and source-trial lineage pairs are added.
"""
import json
from pathlib import Path
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from matchminer_ai.embedding.encoder import EncoderSettings, encoder_identity, open_encoder
from matchminer_ai.embedding.trialspace_contract import DIRECTIONS, model_prompts
from matchminer_ai.labeling import score_contract
from matchminer_ai.matching.retrieval import score_batches, stable_top_k

from matchminer_ai.training.cohort_format import file_stamp
from .data import digest, load_corpus, read_jsonl, write_json, write_jsonl
from .retrieval import TRIAL_VERSION_POLICY, TrialSampler, has_trial_versions, merge_candidates, query_cohorts

# v3 (2026-11-06): the inference encoder and scorer; the identity names the encoder and the corpus receipt
# instead of source files and paths.
VERSION = 'trialspace-dense-mining-v3'
OUTPUTS = ('candidates.jsonl', 'rankings.jsonl', 'patient_to_cohort.parquet', 'cohort_to_patient.parquet',
           'patients.npy', 'cohorts.npy', 'complete.json')


def freeze(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Mining inputs/settings changed: {path}')
    write_json(path, value)


def finish(output, names, **metadata):
    write_json(Path(output) / 'complete.json', dict(**metadata,
        outputs={name: file_stamp(Path(output) / name) for name in names}))


def completed(output):
    path = Path(output) / 'complete.json'
    if not path.exists():
        return False
    for name, expected in json.loads(path.read_text())['outputs'].items():
        if file_stamp(Path(output) / name) != expected:
            raise ValueError('Completed mining output changed: ' + name)
    return True


def encoder_settings(args):
    """EmbeddingGemma 2 as TrialSpace mines with it: fp32 weights, bf16 autocast, the trialspace prompts."""
    model_identity = None
    model_path = Path(args.model)
    if model_path.is_dir():
        from .stage import read_receipt
        receipt = read_receipt(model_path)
        # A fitted model is named by its stage receipt; any other local directory by its files' stamps.
        model_identity = (receipt['receipt_id'] if receipt else
                          digest({str(p.relative_to(model_path)): file_stamp(p)
                                  for p in sorted(model_path.rglob('*')) if p.is_file()}))
    remote = {}
    if getattr(args, 'embedding_backend', 'sentence_transformers') == 'openai':
        remote = dict(server_urls=args.embedding_url or [], server_urls_file=args.embedding_servers_file,
                      model_name=args.embedding_served_model)
    return EncoderSettings(args.model, revision=args.revision, cache_dir=args.cache_dir,
                           devices=tuple(args.devices), batch_size=args.encode_batch_size, max_seq_length=args.max_tokens,
                           dtype='float32', autocast='bfloat16', prompts=model_prompts('trialspace'),
                           default_prompt_name='SentenceSimilarity', include_prompt_in_pooling=True, text_only=True,
                           model_identity=model_identity, backend=getattr(args, 'embedding_backend', 'sentence_transformers'),
                           remote=remote)


def encode_corpus(args, patients, cohorts):
    """``patients.npy`` and ``cohorts.npy`` in corpus order (reused when present), float32 and normalized."""
    output = Path(args.output)
    vectors, missing = {}, []
    sides = [('patients', patients, 'patient_summary'), ('cohorts', cohorts, 'this_space')]
    for name, rows, prompt in sides:
        if not (output / (name + '.npy')).exists():
            missing.append((name, rows, prompt))
    if missing:
        from matchminer_ai.training.trialspace.finetune import require_embeddinggemma2
        require_embeddinggemma2(args.model, revision=args.revision, cache_dir=args.cache_dir)
        with open_encoder(encoder_settings(args)) as encoder:
            for name, rows, prompt in missing:
                values = encoder.encode([row['text'] for row in rows], prompt)
                path = output / (name + '.npy')
                with path.with_suffix('.tmp').open('wb') as stream:
                    np.save(stream, values, allow_pickle=False)
                path.with_suffix('.tmp').replace(path)
    for name, rows, _ in sides:
        path = output / (name + '.npy')
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.ndim != 2 or len(values) != len(rows) or not np.isfinite(values).all():
            raise ValueError('Invalid saved dense embeddings: ' + str(path))
        if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-3):
            raise ValueError('Dense embeddings must be normalized')
        vectors[name] = values
    return vectors


def rank_dense(patients, cohorts, vectors, *, devices=('cpu',), score_batch_size=128,
               patient_k=20, cohort_k=40, lexical_k=5, random_k=5, seed=42, patient_trial_sample=0,
               trial_sample_round=1):
    """Cohort queries search every patient; patients search their sampled trials' cohorts.

    A trial with protocol and ctgov versions is searched in one version per
    patient and queries with one version, chosen per trial (TrialSampler, query_cohorts).
    Each mining round (trial_sample_round) draws a fresh trial sample per patient.
    """
    from matchminer_ai.training.synthetic.pool import require_full_pool
    require_full_pool(patients, cohorts)
    hits = []
    for direction in DIRECTIONS:
        qkind, dkind = (('patients', 'cohorts') if direction == 'patient_to_cohort'
                        else ('cohorts', 'patients'))
        queries, documents = (patients, cohorts) if qkind == 'patients' else (cohorts, patients)
        k = patient_k if qkind == 'patients' else cohort_k
        querying = {c['id'] for c in query_cohorts(cohorts, seed)} if qkind == 'cohorts' else None
        for split in ('train',):
            qi = [i for i, row in enumerate(queries) if row['split'] == split
                  and (querying is None or row['id'] in querying)]
            di = [i for i, row in enumerate(documents) if row['split'] == split]
            if not qi or not di:
                continue
            targets = [documents[i] for i in di]
            lexical = TfidfVectorizer(ngram_range=(1, 2), lowercase=True) if lexical_k else None
            matrix = lexical.fit_transform(row['text'] for row in targets) if lexical else None
            sampler = (TrialSampler(targets, patient_trial_sample, seed, trial_sample_round)
                       if qkind == 'patients' else None)
            everyone = list(range(len(targets)))
            for start, scores in score_batches(np.asarray(vectors[qkind][qi]), np.asarray(vectors[dkind][di]),
                                               batch_size=score_batch_size, devices=tuple(devices)):
                chosen = qi[start:start + score_batch_size]
                lex = ((lexical.transform([queries[i]['text'] for i in chosen]) @ matrix.T).toarray()
                       if lexical else None)
                for index, (i, values) in enumerate(zip(chosen, scores, strict=True)):
                    row = queries[i]
                    def add(j, source, rank=None, score=None):
                        p, c = (row, targets[j]) if qkind == 'patients' else (targets[j], row)
                        hits.append(dict(patient_id=p['id'], cohort_id=c['id'], direction=direction,
                                         source=source, task='gated_score', rank=rank, score=score))
                    allowed = sampler(row['id']) if sampler else everyone
                    for rank, j in enumerate(stable_top_k(values[allowed], k), 1):
                        add(allowed[j], 'embeddinggemma2', rank, float(values[allowed[j]]))
                    if lexical:
                        for rank, j in enumerate(stable_top_k(lex[index][allowed], lexical_k), 1):
                            add(allowed[j], 'tfidf_baseline', rank, float(lex[index, allowed[j]]))
                    rng = random.Random(f'{seed}:{direction}:{row["id"]}')
                    for j in rng.sample(allowed, min(random_k, len(allowed))):
                        add(j, 'random_exploration')
                print(f'[dense mine] {direction}/{split}: {start + len(chosen)}/{len(qi)}', flush=True)
    return hits


def mining_identity(args, corpus_receipt):
    """The stage identity: the corpus receipt, the encoder and the candidate policy (no paths or devices)."""
    identity = dict(version=VERSION, corpus=corpus_receipt, contract=score_contract(),
                    encoder=encoder_identity(encoder_settings(args)), score_batch_size=args.score_batch_size,
                    patient_k=args.patient_k, cohort_k=args.cohort_k, lexical_k=args.lexical_k,
                    random_k=args.random_k, seed=args.seed, patient_trial_sample=args.patient_trial_sample)
    if args.trial_sample_round != 1:  # Round 1 keeps its original draws.
        identity['trial_sample_round'] = args.trial_sample_round
    return identity


def _dense_mine(args, identity):
    from .corpus import export_candidates
    patients, cohorts = load_corpus(args.corpus)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(identity, patients=digest(patients), cohorts=digest(cohorts),
                     dimension=768, scope='exhaustive full synthetic corpus')
    if has_trial_versions(cohorts):
        signature['trial_versions'] = TRIAL_VERSION_POLICY
    from matchminer_ai.training.trialspace.retrieval_cache_import import validate_encoding_import
    validate_encoding_import(output, signature)
    freeze(output / 'manifest.json', signature)
    if completed(output):
        return
    vectors = encode_corpus(args, patients, cohorts)
    hits = rank_dense(patients, cohorts, vectors, devices=args.retrieval_devices or args.devices,
                      score_batch_size=args.score_batch_size, patient_k=args.patient_k,
                      cohort_k=args.cohort_k, lexical_k=args.lexical_k, random_k=args.random_k, seed=args.seed,
                      patient_trial_sample=args.patient_trial_sample, trial_sample_round=args.trial_sample_round)
    lineage = Path(args.corpus) / 'source_trial_pairs.jsonl'
    if lineage.exists():
        hits.extend(dict(patient_id=p['patient_id'], cohort_id=p['cohort_id'], direction='both',
                         source='source_trial_or_independent_proposal', task=None) for p in read_jsonl(lineage))
    pairs = merge_candidates(patients, cohorts, hits)
    write_jsonl(output / 'candidates.jsonl', pairs)
    write_jsonl(output / 'rankings.jsonl', [h for h in hits if h['source'] == 'embeddinggemma2'])
    export_candidates(args.corpus, output / 'candidates.jsonl', output / 'patient_to_cohort.parquet',
                      output / 'cohort_to_patient.parquet')
    finish(output, ['candidates.jsonl', 'rankings.jsonl', 'patient_to_cohort.parquet',
                   'cohort_to_patient.parquet', 'patients.npy', 'cohorts.npy'], pairs=len(pairs),
           candidate_scope='exhaustive cosine similarity over the full synthetic corpus'
           + (f'; each patient searches the cohorts of {args.patient_trial_sample} sampled trials'
              if args.patient_trial_sample else '')
           + (f'; {TRIAL_VERSION_POLICY}' if has_trial_versions(cohorts) else ''))


def dense_mine(args, identity):
    from matchminer_ai.training.trialspace.retrieval_workspace import run_retrieval
    from matchminer_ai.training.synthetic.pool import require_full_corpus
    require_full_corpus(args.corpus)
    return run_retrieval(args, lambda local: _dense_mine(local, identity))
