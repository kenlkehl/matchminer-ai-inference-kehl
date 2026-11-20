#!/usr/bin/env python3
"""Stages decide-match, decide-boilerplate and decide-criteria: the fine-tuned decision model's P(yes).

The decision model is the n=5000 run's final fine-tune (common.DECISION_MODEL), scored through the inference
package's public functions on the texts training rendered its examples from:
- ``match``: ``score_match_quality`` (the five match questions on the patient's key section and the cohort's key
  criteria) for every pair of ``--pairs``;
- ``boilerplate``: ``exclusion_criteria_check`` (the boilerplate question on the patient's boilerplate section and
  the cohort's boilerplate criteria) once per unique combination of the two texts among ``--pairs`` whose cohort has
  boilerplate criteria (yes = potentially eligible);
- ``criteria``: ``check_eligibility_criteria`` (question 7 on the complete summary and the cohort's numbered
  criteria; every checkable line is asked, as inference asks them) for the pairs of the sampled lines in
  ``--lines``, keeping the sampled lines (yes = the patient does not clearly fail the line).

GPU: only ``--gpu`` (CUDA_VISIBLE_DEVICES; the GPU the operator allows). Before loading the model the
stage waits until that GPU has ``--min-free-gb`` free (``--wait``) or stops. Work is done in parts of ``--part-size``
pairs, saved as they finish, so an interrupted run resumes. Scores only; no clinical text is written.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from matchminer_ai.evaluation.retro_fidelity.common import DECISION_MODEL, log, private, private_dir, read_jsonl, run_stage, stage, write_json  # noqa: E402

VERSION = 'retro-fidelity-decide-v2'  # v2: match scores every pair (score_match_quality filtered low ones in v1)
LABELS = ('reasonableness', 'cancer_type', 'burden', 'treatment_history', 'biomarker')


def gpu_memory_gb(index):
    """(total, free) GB of physical GPU ``index``, as nvidia-smi reports them (every process on it)."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits',
                          '-i', str(index)], capture_output=True, text=True, check=True).stdout.strip()
    total, used = (float(v) for v in out.split(','))
    return total / 1024, (total - used) / 1024


def gpu_free_gb(index):
    return gpu_memory_gb(index)[1]


def wait_for_gpu(args, status):
    """Wait until --not-before has passed and the GPU has shown --min-free-gb free on --stable-checks consecutive
    checks (a momentary dip in another process's memory is not room)."""
    from datetime import datetime
    start = datetime.fromisoformat(args.not_before) if args.not_before else None
    streak = 0
    while True:
        free = gpu_free_gb(args.gpu)
        early = start is not None and datetime.now(start.tzinfo) < start
        streak = streak + 1 if free >= args.min_free_gb and not early else 0
        ready = streak >= args.stable_checks
        write_json(status, dict(gpu=args.gpu, free_gb=round(free, 1), needed_gb=args.min_free_gb, not_before=args.not_before,
                                checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), consecutive_checks_with_room=streak,
                                state='ready' if ready else 'waiting'))
        if ready:
            return free
        if not args.wait:
            raise SystemExit(f'GPU {args.gpu} has {free:.1f} GB free; need {args.min_free_gb} GB')
        time.sleep(args.poll_seconds if streak == 0 else 60)


def config(args):
    from matchminer_ai.config import load_default_preset
    value = load_default_preset()
    value.raw['decision_model'] = {**dict(value.raw.get('decision_model') or {}), 'model_name': str(DECISION_MODEL),
                                   'revision': None, 'device': 'cuda', 'token_budget': args.token_budget,
                                   'batch_size': args.batch_size}
    return value


def frames(args, corpus):
    """The pairs to score as a DataFrame with the public functions' columns (one row per unit of work)."""
    import pandas as pd
    patients = {row['id']: row for row in read_jsonl(Path(corpus) / 'patients.jsonl')}
    cohorts = {row['id']: row for row in read_jsonl(Path(corpus) / 'cohorts.jsonl')}
    if args.task == 'criteria':
        lines = [row for path in args.lines for row in read_jsonl(path)]
        keys = sorted({(row['patient_id'], row['cohort_id']) for row in lines})
    else:
        keys = sorted({(row['patient_id'], row['cohort_id']) for path in args.pairs for row in read_jsonl(path)})
        lines = None
    rows = []
    for patient_id, cohort_id in keys:
        p, c = patients[patient_id], cohorts[cohort_id]
        row = dict(patient_id=patient_id, cohort_trial_id=cohort_id, cancer_history_summary=p['text'],
                   general_exclusion_criteria_evidence=p.get('boilerplate') or '')
        if args.task == 'match':
            row['cohort_key_criteria'] = c['text']
        elif args.task == 'boilerplate':
            if not (c.get('boilerplate') or '').strip():
                continue
            row['cohort_boilerplate_criteria'] = c['boilerplate']
            row['combination'] = stage.digest([p.get('boilerplate') or '', c['boilerplate']])
        else:
            row['cohort_criteria_items'] = json.dumps(c['items'], ensure_ascii=False)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if args.task == 'boilerplate':  # one decision per unique combination (identical inputs, identical answer)
        frame = frame.drop_duplicates('combination').reset_index(drop=True)
    return frame, keys, lines


def score(args, frame, settings):
    import pandas as pd
    from matchminer_ai.matching import check_eligibility_criteria, exclusion_criteria_check, score_match_quality
    if args.task == 'match':
        # filter_low_quality defaults to True and would drop every pair below the score cutoff; metrics need them all.
        out = score_match_quality(frame, config=settings, filter_low_quality=False)
        return pd.DataFrame(dict(patient_id=out.patient_id, cohort_id=out.cohort_trial_id,
                                 match_quality_score=out.match_quality_score,
                                 **{f'p_{label}': out[f'match_quality_{label}'] for label in LABELS}))
    if args.task == 'boilerplate':
        out = exclusion_criteria_check(frame, config=settings)
        return pd.DataFrame(dict(combination=frame.combination.values,
                                 p_boilerplate=out.exclusion_criteria_pass_probability.values))
    out = check_eligibility_criteria(frame, config=settings)
    return pd.DataFrame(dict(patient_id=out.patient_id, cohort_id=out.cohort_trial_id, number=out.criterion_number,
                             criterion_type=out.criterion_type, p_yes=out.criterion_pass_probability))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('task', choices=['match', 'boilerplate', 'criteria'])
    p.add_argument('--corpus', required=True)
    p.add_argument('--pairs', nargs='+', default=[], help='match, boilerplate: candidate pair files')
    p.add_argument('--lines', nargs='+', default=[], help='criteria: sampled line files (lists.py)')
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', default='3')
    p.add_argument('--min-free-gb', type=float, default=16, help='Free memory the GPU must have before loading')
    p.add_argument('--max-gpu-gb', type=float, default=12, help='Hard cap on this process\'s GPU memory')
    p.add_argument('--wait', action='store_true', help='Wait for the GPU to have room instead of stopping')
    p.add_argument('--poll-seconds', type=int, default=120)
    p.add_argument('--not-before', help='ISO time before which the GPU is not used (the user frees it then)')
    p.add_argument('--stable-checks', type=int, default=2, help='Consecutive checks (60 s apart) with room')
    p.add_argument('--token-budget', type=int, default=32768, help='Padded tokens per forward pass')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--part-size', type=int, default=2048)
    p.add_argument('--reuse', type=Path, help='match only: an earlier scores.parquet of the same model and inputs; its '
                   'rows are kept as they are and only the missing pairs are scored (not part of the identity: '
                   'the model is deterministic, so reused and fresh scores are the same)')
    stage.add_check_arg(p)
    args = p.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, args.gpu):
        raise SystemExit(f'CUDA_VISIBLE_DEVICES is {os.environ["CUDA_VISIBLE_DEVICES"]}, not {args.gpu}')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # before torch initializes CUDA
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    private()
    inputs = args.lines if args.task == 'criteria' else args.pairs
    if not inputs:
        p.error('give --pairs (match, boilerplate) or --lines (criteria)')
    contract = json.loads((DECISION_MODEL / 'decision_contract.json').read_text())
    identity = dict(version=VERSION, task=args.task, corpus=stage.receipt_id(args.corpus),
                    inputs=[dict(stage=stage.receipt_id(Path(path).parent), name=Path(path).name) for path in inputs],
                    model=stage.receipt_id(DECISION_MODEL), model_fingerprint=contract['fingerprint'],
                    questions_sha256=contract['questions_sha256'], part_size=args.part_size)
    output = Path(args.output)

    def work():
        import pandas as pd
        from matchminer_ai.matching.decision_model import questions_digest
        if questions_digest() != contract['questions_sha256']:
            raise ValueError('The inference package\'s decision questions differ from those the model was trained on')
        frame, keys, lines = frames(args, args.corpus)
        reused = None
        if args.reuse is not None:
            if args.task != 'match':
                raise SystemExit('--reuse is for the match task only')
            wanted = set(zip(frame.patient_id, frame.cohort_trial_id))
            reused = pd.read_parquet(args.reuse)
            reused = reused[[key in wanted for key in zip(reused.patient_id, reused.cohort_id)]].reset_index(drop=True)
            have = set(zip(reused.patient_id, reused.cohort_id))
            frame = frame[[key not in have for key in zip(frame.patient_id, frame.cohort_trial_id)]].reset_index(drop=True)
            log(f'decide {args.task}: reusing {len(reused)} scored pairs from {args.reuse.parent.name}; {len(frame)} to score')
        parts = private_dir(output / 'parts')
        free = wait_for_gpu(args, output / 'gpu.json')
        log(f'decide {args.task}: {len(frame)} units, GPU {args.gpu} with {free:.1f} GB free')
        import torch
        total, _ = gpu_memory_gb(args.gpu)
        # Cap this process so it can never take the memory of the GPU's other (the user's) processes.
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_gpu_gb / total), 0)
        settings = config(args)
        started, done = time.time(), 0
        starts = list(range(0, len(frame), args.part_size))
        for number, start in enumerate(starts):
            path = parts / f'{number:05d}.parquet'
            if path.exists():
                continue
            chunk = frame.iloc[start:start + args.part_size].reset_index(drop=True)
            result = score(args, chunk, settings)
            result.to_parquet(path.with_suffix('.tmp'), index=False)
            path.with_suffix('.tmp').replace(path)
            done += len(chunk)
            rate = done / max(time.time() - started, 1e-9)
            log(f'decide {args.task}: part {number + 1}/{len(starts)}; {rate:.1f} units/s')
        result = pd.concat([pd.read_parquet(parts / f'{n:05d}.parquet') for n in range(len(starts))], ignore_index=True)
        if reused is not None:
            result = pd.concat([reused[list(result.columns)], result], ignore_index=True)
        if args.task == 'match' and (result.duplicated(['patient_id', 'cohort_id']).any()
                                     or set(zip(result.patient_id, result.cohort_id)) != set(keys)):
            raise ValueError('Match scores do not cover every pair exactly once')
        if args.task == 'boilerplate':  # every pair of each combination gets the combination's answer
            mapping = dict(zip(result.combination, result.p_boilerplate))
            patients = {row['id']: row for row in read_jsonl(Path(args.corpus) / 'patients.jsonl')}
            cohorts = {row['id']: row for row in read_jsonl(Path(args.corpus) / 'cohorts.jsonl')}
            rows = []
            for patient_id, cohort_id in keys:
                c = cohorts[cohort_id]
                if not (c.get('boilerplate') or '').strip():
                    continue
                combination = stage.digest([patients[patient_id].get('boilerplate') or '', c['boilerplate']])
                rows.append(dict(patient_id=patient_id, cohort_id=cohort_id, combination=combination,
                                 p_boilerplate=mapping[combination]))
            result = pd.DataFrame(rows)
        elif args.task == 'criteria':
            sampled = pd.DataFrame(lines)[['patient_id', 'cohort_id', 'number', 'criterion_type', 'criteria_digest',
                                           'boilerplate', 'line_key', 'stratum']]
            merged = sampled.merge(result, on=['patient_id', 'cohort_id', 'number'], how='left',
                                   suffixes=('', '_scored'), validate='one_to_one')
            if merged.p_yes.isna().any() or not merged.criterion_type.eq(merged.criterion_type_scored).all():
                raise ValueError('A sampled line has no decision score or a different type')
            result = merged.drop(columns='criterion_type_scored')
        result.to_parquet(output / 'scores.parquet', index=False)
        seconds = time.time() - started
        summary = dict(task=args.task, units=len(frame), rows=len(result), seconds=round(seconds, 1),
                       pairs=len(keys), gpu=args.gpu, token_budget=args.token_budget,
                       reused=0 if reused is None else len(reused))
        write_json(output / 'decide.json', summary)
        for path in parts.iterdir():
            path.unlink()
        parts.rmdir()
        return summary

    run_stage(output, f'retro-decide-{args.task}', identity, ['scores.parquet', 'decide.json'], work, check=args.check,
              upstream=dict(corpus=identity['corpus']))


if __name__ == '__main__':
    main()
