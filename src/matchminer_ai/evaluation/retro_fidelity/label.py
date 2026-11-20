#!/usr/bin/env python3
"""Stages label-<task>-<part>: Nemotron teacher labels (the gold standard) of the evaluation's pairs and lines.

The labeler is training's: the inference package's ``matchminer_ai.labeling.run_label`` with the label stage's
command-line flags (``trialspace.label``, ``add_label_cli_args``), the same contracts as the n=5000 run
(match edbff870, boilerplate f0fc2d3c, per-criterion 7a7eb230; checked at start), Nemotron's own sampling profile
(resolved from ``--teacher-tokenizer``) and schema-constrained decoding of the JSON tasks. Every request, answer
and reasoning is kept (``labels.jsonl``, ``responses``), and an interrupted run resumes.
- ``match``: the five match labels of every pair in ``--candidates`` in one request, scored 0-6 (``scores.jsonl``).
- ``boilerplate``: one request per unique combination of patient boilerplate section and cohort boilerplate criteria
  among ``--candidates`` (trialspace.candidates.unique_boilerplate_pairs), skipping the combinations of
  ``--skip-labeled`` directories.
- ``criteria``: one request per line of ``--candidates`` (lists.py's sampled lines).

Patient text reaches the configured teacher servers only (on-prem GB300 and the LAN 8002 server); the label
workspace's local copy (MMAI_LABEL_SCRATCH_DIR) is private and removed when the labels are published.
"""
import argparse
import json
from pathlib import Path

from matchminer_ai.evaluation.retro_fidelity.common import hold_gb300, log, private, private_dir, read_jsonl, stage, write_jsonl  # noqa: E402

VERSION = 'retro-fidelity-label-v1'
CONTRACTS = {'match': 'edbff87086e7f2dfb79d5fb0ae0de9e5b2ecd48d3a02b19c4c23fe1e79b3a405',
             'boilerplate': 'f0fc2d3c948b4ebec618d70aacb8108d7e47ada0524010a25fed350c74f8cf23',
             'criteria': '7a7eb230af1f502010a5e1707d6bb46a547d1a4618d4859efc1983bb22590afc'}


def parser():
    from matchminer_ai.labeling import add_label_cli_args
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_label_cli_args(p)
    p.set_defaults(return_records=False)
    p.add_argument('--skip-labeled', type=Path, nargs='+', default=[],
                   help='boilerplate: earlier label directories whose combinations are not labeled again')
    p.add_argument('--holders', help='OA-pause holder registry (common.hold_gb300)')
    stage.add_check_arg(p)
    return p


def identity(args):
    candidates = Path(args.candidates)
    value = dict(version=VERSION, task=args.task, corpus=stage.receipt_id(args.corpus),
                 candidates=dict(stage=stage.receipt_id(candidates.parent), name=candidates.name),
                 contract=CONTRACTS[args.task], teacher_model=args.teacher_model,
                 teacher_tokenizer=args.teacher_tokenizer, max_tokens=args.max_tokens,
                 teacher_context=args.teacher_context)
    if args.skip_labeled:
        value['skip_labeled'] = [stage.receipt_id(path) for path in args.skip_labeled]
    return value


def outputs(args):
    names = ['labels.jsonl', 'labeling_manifest.json', 'prompt_sources.json', 'candidates.jsonl']
    if args.task == 'match':
        names += ['scores.jsonl', 'score_balance.json', 'scores_ready.json']
    elif args.task == 'boilerplate':
        names += ['combinations.json']
    return [name for name in names + ['storage_complete.json'] if (Path(args.output) / name).exists()]


def main(argv=None):
    args = parser().parse_args(argv)
    private()
    from matchminer_ai.labeling import contract_hash, run_label
    if contract_hash(args.task) != CONTRACTS[args.task]:
        raise ValueError(f'The {args.task} label contract differs from the n=5000 run\'s')
    expected = identity(args)
    if args.check:
        stage.exit_check(args.output, 'retro-label', expected)
    if stage.require_new_or_same(args.output, 'retro-label', expected):
        log(f'[skip] {args.task} labels already complete')
        return
    hold_gb300(args.holders, f'teacher labels ({args.task}, {Path(args.output).name})')
    output = private_dir(args.output)
    source = Path(args.candidates)
    target = output / 'candidates.jsonl'
    if args.task == 'boilerplate':
        from matchminer_ai.training.trialspace.candidates import unique_boilerplate_pairs
        args.candidates = str(unique_boilerplate_pairs(args.corpus, [source], output, skip=args.skip_labeled))
    else:  # the candidate rows this directory labels, kept beside its labels
        rows = read_jsonl(source)
        if target.exists():
            if read_jsonl(target) != rows:
                raise ValueError('This label directory was built from different candidates; use a new directory')
        else:
            write_jsonl(target, rows)
        args.candidates = str(target)
    count = sum(1 for _ in open(args.candidates))
    log(f'label {args.task} ({output.name}): {count} requests')
    run_label(args)
    manifest = json.loads((output / 'labeling_manifest.json').read_text())
    content = dict(teacher_signature=stage.digest(manifest), requests=count)
    if args.task == 'match':
        content['pairs'] = json.loads((output / 'scores_ready.json').read_text()).get('pairs')
    stage.write_receipt(output, 'retro-label', expected, outputs(args), content=content,
                        upstream=dict(corpus=expected['corpus']))
    log(f'label {args.task} ({output.name}): complete')


if __name__ == '__main__':
    main()
