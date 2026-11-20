#!/usr/bin/env python3
"""Stage project-trialspace-N: one round's labeled candidates as self-contained pair tables for fitting.

Joins the mined candidates with their complete 0-6 teacher scores and the five answers (trialspace/projection.py)
into ``<output>/embeddinggemma2/{pairs,patient_to_cohort,cohort_to_patient}.parquet``, which ``trialspace.finetune``
reads. The stage receipt names the corpus, candidate and label receipts.
"""
import argparse
from pathlib import Path

from matchminer_ai.labeling import score_contract

from matchminer_ai.training.trialspace import projection, stage

OUTPUTS = ['complete.json'] + [f'embeddinggemma2/{name}' for name in (
    'manifest.json', 'pairs.parquet', 'patient_to_cohort.parquet', 'cohort_to_patient.parquet', 'scores.jsonl',
    'complete.json')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--mining', '--embeddinggemma', dest='mining', type=Path, required=True,
                        help='The round\'s mining directory (its candidates.jsonl)')
    parser.add_argument('--labels', type=Path, required=True, help='The round\'s match label directory')
    parser.add_argument('--output', type=Path, required=True)
    stage.add_check_arg(parser)
    args = parser.parse_args()
    upstream = dict(corpus=stage.receipt_id(args.corpus), mining=stage.receipt_id(args.mining),
                    labels=stage.receipt_id(args.labels))
    identity = dict(version=projection.VERSION, contract=score_contract(), **upstream)
    if args.check:
        stage.exit_check(args.output, 'project', identity)
    if stage.require_new_or_same(args.output, 'project', identity):
        print('[skip] pairs already projected', flush=True)
        return
    projection.project_labels(args.corpus, args.mining, args.labels, {'embeddinggemma2': args.mining}, args.output)
    stage.write_receipt(args.output, 'project', identity, OUTPUTS, upstream=upstream)


if __name__ == '__main__':
    main()
