#!/usr/bin/env python3
"""Stage retrieve: exact cosine top-K of each workflow's queries under both embedders.

Patient-centric: each sampled index event searches every cohort of the corpus. Trial-centric: each sampled cohort
searches every validation index event (all of them, not only the sampled ones). Scores are exact cosine
similarities of the embed stage's normalized vectors (``matchminer_ai.matching.retrieval.score_batches``) and the
top ``--k`` are kept with ``stable_top_k`` (ties by corpus order), as training's mining selects. Both sides of an
embedder must be encoded on the same device kind (CPU float32 and CUDA bf16-autocast vectors are never compared).

``rankings.parquet``: workflow, query_id, embedder, rank (1..K), patient_id, cohort_id, score. Rankers (a) and (b)
are each embedder's top 20 (common.TOP); ranker (c) reranks TrialSpace's top K with the decision model
(lists.py). ``decision_pairs.jsonl``: every pair the decision model scores for the match questions, TrialSpace's
top K and EmbeddingGemma 2's top 20 of every query (so every merged-top-20 pair has decision scores).
"""
import argparse
import json
from pathlib import Path

from matchminer_ai.evaluation.retro_fidelity.common import (DIRECTION, EMBEDDERS, SPLIT, TOP, WORKFLOWS, log, private, read_jsonl, run_stage, stage,  # noqa: E402
                    write_json, write_jsonl)

VERSION = 'retro-fidelity-retrieve-v1'


def embedding_dir(root, embedder, side):
    return Path(root) / embedder / side


def search(queries, documents, k):
    """[(query index, [(document index, score), ...] in rank order)] by exact cosine similarity."""
    import numpy as np
    from matchminer_ai.matching.retrieval import score_batches, stable_top_k
    hits = []
    for start, scores in score_batches(np.asarray(queries), np.asarray(documents), batch_size=128, devices=('cpu',)):
        for offset, values in enumerate(scores):
            top = stable_top_k(values, min(k, len(values)))
            hits.append((start + offset, [(j, float(values[j])) for j in top]))
    return hits


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', required=True)
    p.add_argument('--embeddings', required=True, help='Root of the embed stages (<embedder>/<side>)')
    p.add_argument('--k', type=int, default=100, help='Candidates kept per query and embedder (ranker (c) reranks TrialSpace\'s)')
    p.add_argument('--output', required=True)
    stage.add_check_arg(p)
    args = p.parse_args(argv)
    private()
    corpus = stage.receipt_id(args.corpus)
    receipts = {f'{e}/{s}': stage.read_receipt(embedding_dir(args.embeddings, e, s))
                for e in EMBEDDERS for s in ('patients', 'cohorts')}
    if any(r is None for r in receipts.values()):
        raise ValueError('Every embedding stage must be complete first')
    identity = dict(version=VERSION, corpus=corpus, k=args.k, top=TOP,
                    embeddings={name: r['receipt_id'] for name, r in receipts.items()})
    output = Path(args.output)

    def work():
        import numpy as np
        import pandas as pd
        patients = read_jsonl(Path(args.corpus) / 'patients.jsonl')
        cohorts = read_jsonl(Path(args.corpus) / 'cohorts.jsonl')
        sample = json.loads((Path(args.corpus) / 'sample.json').read_text())
        pindex = {row['id']: i for i, row in enumerate(patients)}
        cindex = {row['id']: i for i, row in enumerate(cohorts)}
        rows = []
        for embedder in EMBEDDERS:
            kinds = {receipts[f'{embedder}/{s}']['identity']['device_kind'] for s in ('patients', 'cohorts')}
            if len(kinds) != 1:
                raise ValueError(f'{embedder}: patients and cohorts were encoded on different device kinds')
            vectors = {}
            for side, entities in (('patients', patients), ('cohorts', cohorts)):
                directory = embedding_dir(args.embeddings, embedder, side)
                summary = json.loads((directory / 'embedding.json').read_text())
                if summary['ids_sha256'] != stage.digest([row['id'] for row in entities]):
                    raise ValueError(f'{embedder}/{side}: vectors are not in this corpus\'s order')
                vectors[side] = np.load(directory / 'vectors.npy')
            for workflow in WORKFLOWS:
                if workflow == 'patient_centric':
                    ids = sample['patient_queries']
                    queries, documents = vectors['patients'][[pindex[i] for i in ids]], vectors['cohorts']
                else:
                    ids = sample['cohort_queries']
                    queries, documents = vectors['cohorts'][[cindex[i] for i in ids]], vectors['patients']
                for qi, hits in search(queries, documents, args.k):
                    for rank, (j, score) in enumerate(hits, 1):
                        patient = ids[qi] if workflow == 'patient_centric' else patients[j]['id']
                        cohort = cohorts[j]['id'] if workflow == 'patient_centric' else ids[qi]
                        rows.append(dict(workflow=workflow, query_id=ids[qi], embedder=embedder, rank=rank,
                                         patient_id=patient, cohort_id=cohort, score=score))
                log(f'retrieve {embedder}/{workflow}: {len(ids)} queries')
        frame = pd.DataFrame(rows)
        frame.to_parquet(output / 'rankings.parquet', index=False)
        trial_of = {row['id']: row['trial_id'] for row in cohorts}
        chosen = frame[(frame.embedder == 'trialspace_round2') | (frame['rank'] <= TOP)]
        pairs = []
        for (patient, cohort), group in chosen.groupby(['patient_id', 'cohort_id'], sort=True):
            pairs.append(dict(patient_id=patient, cohort_id=cohort, trial_id=trial_of[cohort], split=SPLIT,
                              sources=[dict(direction=DIRECTION[w], workflow=w, query_id=q, source=e, rank=int(r))
                                       for w, q, e, r in group[['workflow', 'query_id', 'embedder', 'rank']].itertuples(
                                           index=False)]))
        write_jsonl(output / 'decision_pairs.jsonl', pairs)
        summary = dict(rows=len(frame), decision_pairs=len(pairs), k=args.k,
                       queries={w: int(frame[frame.workflow == w].query_id.nunique()) for w in WORKFLOWS},
                       shorter_than_k=int((frame.groupby(['workflow', 'embedder', 'query_id']).size() < args.k).sum()))
        write_json(output / 'retrieval.json', summary)
        log(f'retrieve: {summary}')
        return summary

    run_stage(output, 'retro-retrieve', identity, ['rankings.parquet', 'decision_pairs.jsonl', 'retrieval.json'], work,
              check=args.check, upstream=dict(corpus=corpus))


if __name__ == '__main__':
    main()
