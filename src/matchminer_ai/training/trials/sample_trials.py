#!/usr/bin/env python3
"""Sample the training trials: a seeded draw of whole trials from those ``trials.screen_trials`` accepted.

The screen applies the ctgov rule, the DFCI holdout and the cancer-treatment screen before any cohort
extraction; the sampling receipt carries the counts of every step.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from matchminer_ai.training.cohort_format import file_sha256, read_clinical_csv


def dfci_nct_ids_open_during(dfci_trials, first_year, last_year):
    """NCT IDs whose dates do not place them wholly outside first_year..last_year.

    The dates are ClinicalTrials.gov study start/completion dates at month or day
    precision, so comparing years is exact. A missing date leaves that side open,
    and a trial still open to accrual in the dump counts as open.
    """
    dfci_trials = dfci_trials.dropna(subset=['nct_id'])
    for column in ['open_date', 'close_date']:
        dates = dfci_trials[column].dropna().astype(str)
        assert dates.str.fullmatch(r'\d{4}-\d{2}(-\d{2})?').all(), f'unexpected {column} format'
    open_year = pd.to_numeric(dfci_trials.open_date.fillna('').astype(str).str[:4], errors='coerce')
    close_year = pd.to_numeric(dfci_trials.close_date.fillna('').astype(str).str[:4], errors='coerce')
    outside_window = (close_year < first_year) | (open_year > last_year)
    open_during = ~outside_window | (dfci_trials.status == 'OPEN TO ACCRUAL')
    return set(dfci_trials.nct_id[open_during])


def sample_screened_trials(screened_csv, output_csv, *, n_trials=5000, seed=42):
    """Draw n_trials of the trials ``trials.screen_trials`` accepted; write one row per trial.

    The output keeps each sampled trial's screening row (nct_id, split, dates) in sample order; its
    nct_id seeds the synthetic protocols and its split is every later stage's source split. The
    receipt records the screening receipt's counts and the sampled IDs.
    """
    screened = read_clinical_csv(screened_csv)
    screening = json.loads(Path(f'{screened_csv}.screening.json').read_text())
    if screened.nct_id.duplicated().any():
        raise ValueError('Duplicate NCT IDs in the screened trials')
    held_out = screened.dfci_holdout.astype(str).str.lower().eq('true')  # CSV text, not booleans
    eligible = sorted(screened.nct_id[~held_out & screened.cancer_treatment_trial.eq('yes')])
    if len(eligible) < n_trials:
        raise ValueError(f'Only {len(eligible)} screened cancer treatment trials; need {n_trials}')
    sampled = pd.Series(eligible).sample(n=n_trials, random_state=seed).tolist()
    output = screened.set_index('nct_id').loc[sampled].reset_index()
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    manifest = dict(input=str(Path(screened_csv).resolve()), output=str(path.resolve()), seed=seed,
                    exported_trials=screening['exported_trials'], ctgov_rule=screening.get('ctgov_rule', screening.get('registry_rule')),  # 'registry_rule' before 2026-11-05
                    dfci_holdout={k: v for k, v in screening['dfci_holdout'].items() if k != 'nct_ids'},
                    screen={k: v for k, v in screening['screen'].items() if k != 'rejected_nct_ids'},
                    eligible_trials=len(eligible), sampled_trials=n_trials, sampled_nct_ids=sampled,
                    split_counts=output.split.value_counts().to_dict(),
                    sampling_source_sha256={p.name: file_sha256(p) for p in (Path(__file__),)})
    path.with_suffix('.sampling.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'sampled {n_trials} of {len(eligible)} screened cancer treatment trials (seed {seed}) to {path}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screened-trials', required=True, help='trials.screen_trials output')
    parser.add_argument('--output-csv', required=True, help='The sampled trials (one row per trial)')
    parser.add_argument('--n-trials', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    sample_screened_trials(args.screened_trials, args.output_csv, n_trials=args.n_trials, seed=args.seed)


if __name__ == '__main__':
    main()
