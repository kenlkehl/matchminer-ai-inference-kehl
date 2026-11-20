#!/usr/bin/env python3
"""Screen the ctgov trials training can sample: ctgov rule, DFCI holdout, cancer-treatment screen.

Production training draws its trial sample (``trials.sample_trials --screened-trials``) from the
trials this step accepts, in this order (user instruction, 2026-11-04):

1. ctgov rule: select_trials keeps interventional early phase I-IV
   trials first posted since --start-year from the parsed ClinicalTrials.gov export, and assigns
   each trial's split by NCT ID.
2. DFCI holdout: trials open at DFCI between --dfci-first-year and --dfci-last-year
   (``sample_trials.dfci_nct_ids_open_during``) are excluded.
3. Cancer-treatment screen: every remaining listing (title, brief summary and eligibility criteria
   as one document) gets the inference package's screen_trials, the first step of the cohort
   workflow, with the same prompt and settings as cohort extraction. Every attempt's answer and
   reasoning is appended to <work-dir>/screen.jsonl as it returns and reused on rerun.

Writes --output, one row per trial the ctgov rule kept (nct_id, split, dates, dfci_holdout,
cancer_treatment_trial, screen_error), and <output>.screening.json with the count and IDs removed at
each step, the prompt hash, model, settings and commit.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from matchminer_ai.training.cohort_format import file_stamp, read_clinical_csv
from matchminer_ai.training.paths import commit, no_phi
from matchminer_ai.training.trials import sample_trials as sample
from matchminer_ai.training.trials.ctgov_llm import add_llm_arguments, make_config, prepare_runtime

PHASES = {'EARLY_PHASE1', 'PHASE1', 'PHASE2', 'PHASE3', 'PHASE4'}


def select_trials(frame, *, start_year=2020, seed=42):
    """The ctgov rule: interventional early phase I-IV trials first posted since start_year; splits by NCT ID."""
    required = {'nct_id', 'study_first_post_date', 'study_type', 'phases',
                'title', 'brief_summary', 'eligibility_criteria'}
    if required - set(frame):
        raise ValueError(f'Missing parsed CT.gov columns: {sorted(required - set(frame))}')
    def eligible_phases(value):
        try:
            phases = json.loads(value)
        except (TypeError, ValueError):
            return False
        return (isinstance(phases, list) and bool(phases)
                and all(isinstance(p, str) and p in PHASES for p in phases))
    dates = pd.to_datetime(frame.study_first_post_date, format='mixed', errors='coerce')
    frame = frame.loc[dates.dt.year.ge(start_year) & frame.study_type.eq('INTERVENTIONAL')
                      & frame.phases.map(eligible_phases)].copy()
    if frame.empty:
        raise ValueError('No trials meet the date, study type and phase selection')
    if not frame.nct_id.str.fullmatch(r'NCT\d{8}', na=False).all():
        raise ValueError('Expected ClinicalTrials.gov NCT identifiers')
    frame = frame.drop_duplicates()
    if frame.nct_id.duplicated().any():
        raise ValueError('Conflicting duplicate source records for an NCT ID')
    frame = frame.sort_values('nct_id').reset_index(drop=True)
    if 'split' in frame:
        if not frame.split.isin(['train', 'val', 'validation', 'test']).all():
            raise ValueError('Missing or invalid source splits')
    else:
        frame['split'] = np.random.default_rng(seed).choice(['train', 'val', 'test'],
                                                         len(frame), p=[.8, .1, .1])
    for column in ('title', 'brief_summary', 'eligibility_criteria'):
        frame[column] = frame[column].fillna('')
    return frame


def resolve_defaults(args):
    """Default paths under the data root's no_phi tree (matchminer_ai.site)."""
    data = no_phi()
    args.input = args.input or data / 'ctgov_trials.csv'
    args.work_dir = args.work_dir or data / 'trial_screen'
    args.output = args.output or data / 'screened_trials.csv'
    args.dfci_trials_csv = args.dfci_trials_csv or data / 'dfci_trials_summary_10-18-26.csv'
    return args


def main(args):
    resolve_defaults(args)
    prepare_runtime(args)
    from matchminer_ai.trials import screen_trials
    from matchminer_ai.trials.cohort_screen import SCREEN_FILE, SCREEN_SCHEMA_VERSION

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / 'screen.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        export = read_clinical_csv(args.input)
        selected = select_trials(export, start_year=args.start_year, seed=args.seed)
        held_out = sample.dfci_nct_ids_open_during(pd.read_csv(args.dfci_trials_csv), args.dfci_first_year,
                                                   args.dfci_last_year)
        selected['dfci_holdout'] = selected.nct_id.isin(held_out)
        candidates = selected.loc[~selected.dfci_holdout]
        print(f'{len(export)} exported trials; {len(selected)} meet the ctgov rule; '
              f'{int(selected.dfci_holdout.sum())} open at DFCI {args.dfci_first_year}-{args.dfci_last_year} '
              f'are held out; screening {len(candidates)}', flush=True)
        config = make_config(args)
        decisions, metadata = screen_trials(
            candidates[['nct_id', 'title', 'brief_summary', 'eligibility_criteria']].rename(columns={'nct_id': 'trial_id'}),
            config=config, output_dir=args.work_dir, return_metadata=True)
        decisions = decisions.rename(columns={'trial_id': 'nct_id'})
        if metadata['screen_errors'] and not args.allow_failures:
            raise SystemExit(f"{len(metadata['screen_errors'])} trials have no validated screen answer "
                             f"(see {args.work_dir / SCREEN_FILE}); rerun to retry them, or pass --allow-failures")
        columns = ['nct_id', 'split', 'study_first_post_date', 'study_start_date', 'overall_status', 'phases', 'dfci_holdout']
        output = selected[columns].merge(decisions, on='nct_id', how='left', validate='one_to_one')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f'.{args.output.name}.{os.getpid()}.tmp')
        output.to_csv(temporary, index=False)
        temporary.replace(args.output)
        screened = output.loc[~output.dfci_holdout]
        receipt = {
            'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'input': str(args.input.resolve()), 'input_stamp': file_stamp(args.input),
            'output': str(args.output.resolve()), 'output_stamp': file_stamp(args.output),
            'exported_trials': len(export),
            'ctgov_rule': dict(start_year=args.start_year, study_type='INTERVENTIONAL',
                                  phases=sorted(PHASES), trials=len(selected), split_seed=args.seed),
            'dfci_holdout': dict(path=str(Path(args.dfci_trials_csv).resolve()), first_year=args.dfci_first_year,
                                 last_year=args.dfci_last_year, trials=int(selected.dfci_holdout.sum()),
                                 nct_ids=sorted(selected.nct_id[selected.dfci_holdout])),
            'screen': dict(schema_version=SCREEN_SCHEMA_VERSION, prompt_sha256=metadata['prompt_sha256'],
                           decisions=str((args.work_dir / SCREEN_FILE).resolve()), screened=len(screened),
                           accepted=int(screened.cancer_treatment_trial.eq('yes').sum()),
                           rejected=int(screened.cancer_treatment_trial.eq('no').sum()),
                           rejected_nct_ids=sorted(screened.nct_id[screened.cancer_treatment_trial.eq('no')]),
                           errors=metadata['screen_errors'], reused=metadata['reused'],
                           structured_outputs=metadata['structured_outputs']),
            'model': args.model, 'tokenizer_model': args.tokenizer_model, 'max_tokens': args.max_tokens,
            'max_attempts': args.max_attempts,
            'commit': commit(),
        }
        Path(f'{args.output}.screening.json').write_text(json.dumps(receipt, indent=2) + '\n')
        screen = receipt['screen']
        print(f"screened {screen['screened']}: {screen['accepted']} cancer treatment trials, {screen['rejected']} not, "
              f"{len(screen['errors'])} without an answer; wrote {args.output}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--input', type=Path, help='Parsed ClinicalTrials.gov export; default <data root>/no_phi/ctgov_trials.csv')
    p.add_argument('--start-year', type=int, default=2020)
    p.add_argument('--seed', type=int, default=42, help='Split assignment by NCT ID')
    p.add_argument('--work-dir', type=Path, help='Default: <data root>/no_phi/trial_screen')
    p.add_argument('--allow-failures', action='store_true',
                   help='Write the table even if some trials have no validated screen answer')
    p.add_argument('--output', type=Path, help='Default: <data root>/no_phi/screened_trials.csv')
    p.add_argument('--dfci-trials-csv', type=Path, help='Default: <data root>/no_phi/dfci_trials_summary_10-18-26.csv')
    p.add_argument('--dfci-first-year', type=int, default=2025)
    p.add_argument('--dfci-last-year', type=int, default=2026)
    return add_llm_arguments(p)


if __name__ == '__main__':
    main(parser().parse_args())
