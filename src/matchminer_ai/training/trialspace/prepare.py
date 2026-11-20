#!/usr/bin/env python3
"""Stage prepare-trialspace: the TrialSpace corpus every round shares (trialspace/corpus.py).

Patients get their key summary section as ``text`` and boilerplate section as ``boilerplate``; cohorts their
key criteria, boilerplate criteria and tagged criteria lines; every synthetic patient its source-trial lineage
pairs. All synthetic sources form one pool (``--merge-source-splits``). The output directory's
stage_receipt.json lets later stages, here or on another machine, name this corpus.
"""
import argparse
import json
from pathlib import Path

from matchminer_ai.labeling import score_contract

from matchminer_ai.training.cohort_format import file_stamp
from matchminer_ai.training.trialspace import corpus, stage

OUTPUTS = ('patients.jsonl', 'cohorts.jsonl', 'corpus.json', 'source_trial_pairs.jsonl', 'prepared.json')


def identity(args):
    return dict(version=corpus.VERSION, contract=score_contract(), merge_source_splits=args.merge_source_splits,
                inputs={name: file_stamp(path) for name, path in (('patients', args.patients), ('cohorts', args.cohorts))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patients', type=Path, required=True, help='Patient summaries (prepare-training output)')
    parser.add_argument('--cohorts', type=Path, required=True, help='Both versions of every trial (retrieval_cohorts.csv)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--merge-source-splits', action=argparse.BooleanOptionalAction, default=True,
                        help='Pool all synthetic patients/cohorts for retrieval and fitting; retain source_split provenance')
    stage.add_check_arg(parser)
    args = parser.parse_args()
    expected = identity(args)
    if args.check:
        stage.exit_check(args.output, 'prepare', expected)
    if stage.require_new_or_same(args.output, 'prepare', expected):
        print('[skip] corpus already prepared', flush=True)
        return
    print(corpus.prepare_corpus(args.patients, args.cohorts, args.output, merge_source_splits=args.merge_source_splits))
    summary = json.loads((args.output / 'corpus.json').read_text())
    prepared = json.loads((args.output / 'prepared.json').read_text())
    stage.write_receipt(args.output, 'prepare', expected, OUTPUTS, content=dict(
        patient_sha256=summary['patient_sha256'], cohort_sha256=summary['cohort_sha256'],
        patients=prepared['patients'], cohorts=prepared['cohorts'], source_trial_pairs=prepared['source_trial_pairs']))


if __name__ == '__main__':
    main()
