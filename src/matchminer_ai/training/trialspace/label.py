#!/usr/bin/env python3
"""Stages label-trialspace-N, label-boilerplate-N and label-criteria-N: teacher labels of one round's candidates.

The labeler is the inference package's (``matchminer_ai.labeling``): ``--task match`` asks the five match
labels in one request per pair and scores each pair 0-6; ``--task boilerplate`` labels each unique boilerplate
combination (trialspace/candidates.py) whether the patient clearly fails the cohort's boilerplate criteria;
``--task criteria`` labels every tagged criteria line of a seeded fraction of the pairs (yes = the patient does
not clearly fail the line). Later rounds import earlier match labels (``--reuse-labels``) and skip boilerplate
combinations and lines labeled before (``--skip-labeled``). Every answer keeps its reasoning and prompt; a run
resumes where it stopped, and the stage receipt names the corpus, candidates and teacher settings.
"""
import argparse
import json
from pathlib import Path

from matchminer_ai.labeling import add_label_cli_args

from matchminer_ai.training.trialspace import candidates as builders, stage

VERSION = 'trialspace-label-stage-v1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_label_cli_args(p)
    p.set_defaults(return_records=False)
    p.add_argument('--reuse-labels', type=Path,
                   help='Completed labels of an earlier round over the same corpus; its judgments of pairs in --candidates are imported')
    p.add_argument('--reuse-candidates', type=Path, help='The candidates.jsonl --reuse-labels judged')
    p.add_argument('--combine-candidates', type=Path, nargs='+', default=[],
                   help='--task boilerplate or criteria: further candidate files; each boilerplate combination or line '
                        'across all of them is labeled once')
    p.add_argument('--skip-labeled', type=Path, nargs='+', default=[],
                   help='--task boilerplate or criteria: earlier rounds\' label directories; their boilerplate combinations '
                        'or criteria lines (their candidates.jsonl) are not labeled again')
    p.add_argument('--criteria-pair-fraction', type=float, default=0.01,
                   help='--task criteria: the share of candidate pairs whose every tagged criteria line is labeled, chosen '
                        'by a seeded hash of the pair so a pair is chosen in every round or none')
    p.add_argument('--criteria-seed', type=int, default=42)
    stage.add_check_arg(p)
    return p


def candidate_receipts(paths):
    """Each candidate file by the receipt of the stage directory that holds it, and its name there."""
    return [dict(stage=stage.receipt_id(Path(path).parent), name=Path(path).name) for path in paths]


def identity(args):
    value = dict(version=VERSION, task=args.task, corpus=stage.receipt_id(args.corpus),
                 candidates=candidate_receipts([args.candidates, *args.combine_candidates]),
                 teacher_model=args.teacher_model, teacher_tokenizer=args.teacher_tokenizer,
                 max_tokens=args.max_tokens, teacher_context=args.teacher_context)
    if args.task == 'criteria':
        value.update(criteria_pair_fraction=args.criteria_pair_fraction, criteria_seed=args.criteria_seed)
    if args.skip_labeled:
        value['skip_labeled'] = [stage.receipt_id(path) for path in args.skip_labeled]
    if args.reuse_labels:
        value['reuse_labels'] = stage.receipt_id(args.reuse_labels)
    return value


def outputs(args):
    names = ['labels.jsonl', 'labeling_manifest.json', 'prompt_sources.json']
    if args.task == 'match':
        names += ['scores.jsonl', 'score_balance.json', 'scores_ready.json']
    else:
        names += ['candidates.jsonl', 'combinations.json' if args.task == 'boilerplate' else 'criteria_sample.json']
    return [name for name in names + ['storage_complete.json'] if (Path(args.output) / name).exists()]


def main():
    args = parser().parse_args()
    from matchminer_ai.training.synthetic.pool import require_full_corpus
    expected = identity(args)
    if args.check:
        stage.exit_check(args.output, 'label', expected)
    if stage.require_new_or_same(args.output, 'label', expected):
        print('[skip] labels already complete', flush=True)
        return
    require_full_corpus(args.corpus)
    from matchminer_ai.labeling import run_label
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if args.reuse_labels or args.reuse_candidates:
        builders.write_reuse_plan(args)
    if args.task == 'boilerplate':
        args.candidates = str(builders.unique_boilerplate_pairs(args.corpus, [Path(args.candidates), *args.combine_candidates],
                                                                args.output, skip=args.skip_labeled))
    elif args.task == 'criteria':
        args.candidates = str(builders.criteria_line_candidates(args.corpus, [Path(args.candidates), *args.combine_candidates],
                                                                args.output, fraction=args.criteria_pair_fraction,
                                                                seed=args.criteria_seed, skip=args.skip_labeled))
    elif args.combine_candidates or args.skip_labeled:
        raise ValueError('--combine-candidates and --skip-labeled are for boilerplate and criteria labels')
    run_label(args)
    manifest = json.loads((Path(args.output) / 'labeling_manifest.json').read_text())
    content = dict(teacher_signature=stage.digest(manifest))
    if args.task == 'match':
        content['pairs'] = json.loads((Path(args.output) / 'scores_ready.json').read_text()).get('pairs')
    stage.write_receipt(args.output, 'label', expected, outputs(args), content=content,
                        upstream=dict(corpus=expected['corpus']))


if __name__ == '__main__':
    main()
