#!/usr/bin/env python3
"""Stage mine-N: TrialSpace candidates of one round (trialspace/mining.py).

Encodes the corpus's key texts with EmbeddingGemma 2 (round 1) or the previous round's fitted TrialSpace on one
or more GPUs (``--devices cuda:0,cuda:1,...``; the vectors equal a one-GPU run's) or over vLLM embedding servers
(``--embedding-backend openai``), ranks both directions exactly by cosine similarity, and adds lexical, random
and source-trial candidates. The directory can be mined on another machine and copied back: its
stage_receipt.json names the corpus receipt and the encoder, not paths or devices.
"""
import argparse
import json
from pathlib import Path

from matchminer_ai.training.trialspace.retrieval_workspace import add_retrieval_storage_args
from matchminer_ai.training.trialspace import mining, stage


def devices(value):
    return tuple(part.strip() for part in value.split(',') if part.strip())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_retrieval_storage_args(p)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', default='google/embeddinggemma-2')
    p.add_argument('--revision')
    p.add_argument('--cache-dir')
    p.add_argument('--devices', '--device', dest='devices', type=devices, default=('cuda:0',),
                   help='Comma-separated encoding devices, one worker process each (default cuda:0)')
    p.add_argument('--retrieval-devices', type=devices, help='Scoring devices (default --devices)')
    p.add_argument('--embedding-backend', choices=['sentence_transformers', 'openai'], default='sentence_transformers',
                   help='openai: encode over vLLM >= 0.32 /v1/embeddings servers (--embedding-url/--embedding-servers-file)')
    p.add_argument('--embedding-url', nargs='+')
    p.add_argument('--embedding-servers-file')
    p.add_argument('--embedding-served-model')
    p.add_argument('--max-tokens', type=int, default=8192)
    p.add_argument('--encode-batch-size', type=int, default=32)
    p.add_argument('--score-batch-size', type=int, default=128)
    p.add_argument('--patient-k', type=int, default=20, help='Top cohorts per patient, from that patient\'s sampled trials')
    p.add_argument('--cohort-k', type=int, default=40, help='Top patients per cohort, over all patients')
    p.add_argument('--lexical-k', type=int, default=5, help='TF-IDF candidates added per query, from the same search set')
    p.add_argument('--random-k', type=int, default=5, help='Random candidates added per query, from the same search set')
    p.add_argument('--patient-trial-sample', type=int, default=500,
                   help='Trials randomly sampled per patient; the patient searches only their cohorts (0 = all trials)')
    p.add_argument('--trial-sample-round', type=int, default=1,
                   help='Mining round; each round draws a fresh trial sample and version choice per patient (1 = original draws)')
    p.add_argument('--seed', type=int, default=42)
    stage.add_check_arg(p)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.encode_batch_size, args.score_batch_size, args.patient_k, args.cohort_k) < 1:
        p.error('Batch sizes and top-k counts must be positive')
    corpus_receipt = stage.receipt_id(args.corpus)
    identity = mining.mining_identity(args, corpus_receipt)
    if args.check:
        stage.exit_check(args.output, 'mine', identity)
    if stage.require_new_or_same(args.output, 'mine', identity):
        print('[skip] candidates already mined', flush=True)
        return
    mining.dense_mine(args, identity)
    pairs = json.loads((args.output / 'complete.json').read_text())['pairs']
    stage.write_receipt(args.output, 'mine', identity, [*mining.OUTPUTS, 'manifest.json'], content=dict(pairs=pairs),
                        upstream=dict(corpus=corpus_receipt))


if __name__ == '__main__':
    main()
