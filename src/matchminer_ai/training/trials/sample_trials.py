#!/usr/bin/env python3
"""Sample 5,000 trials for training.

Production (--screened-trials, user instruction 2026-11-04): a seeded draw of whole trials from
those ``trials.screen_trials`` accepted (ctgov rule, DFCI holdout, then the cancer-treatment screen),
before any cohort extraction; the receipt carries the counts of every step. The older modes sample
already-extracted cohort tables, and the space-input adapter remains for older datasets. (This was
0c_sample_trial_spaces.py until 2026-11-05.)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from matchminer_ai._trial_space_format import has_required_space_fields
from matchminer_ai.training.cohort_format import (COHORT_REQUIRED, representation_for, validate_cohort_rows,
                                 complete_cohort_criteria, load_production_cohorts, file_sha256,
                                 read_clinical_csv)
from matchminer_ai.training.paths import module_file, no_phi

def clean_trial_spaces(line_item_space_dataframe):
    return line_item_space_dataframe.this_space.apply(has_required_space_fields)


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


def sample_trial_spaces(input_csv, output_csv, dfci_csv, *, n_trials=5000, seed=42,
                        first_year=2025, last_year=2026, chunksize=2000, extraction_dir=None):
    if n_trials < 1 or chunksize < 1 or first_year > last_year:
        raise ValueError('Invalid sample size, chunk size or holdout years')
    dfci_trials = pd.read_csv(dfci_csv)
    excluded = dfci_nct_ids_open_during(dfci_trials, first_year, last_year)
    provenance = {}
    if extraction_dir is not None:
        if input_csv is not None:
            raise ValueError('Choose an input CSV or a production extraction directory')
        source, provenance = load_production_cohorts(extraction_dir, seed=seed)
        columns = source.columns
        def chunks(fields=None):
            for start in range(0, len(source), chunksize):
                frame = source.iloc[start:start + chunksize]
                yield (frame[fields] if fields else frame).copy()
    else:
        columns = read_clinical_csv(input_csv, nrows=0).columns
        def chunks(fields=None):
            yield from read_clinical_csv(input_csv, usecols=fields, chunksize=chunksize)
    # Scan only the small columns first; the full extraction CSV can contain
    # repeated multi-megabyte reasoning traces. Sampling is by trial, not space.
    counts, invalid, non_cancer = {}, set(), set()
    native = 'cohort_eligibility_criteria' in columns
    if native:
        # Keep only identities/counts in memory, validating entire trial sets even
        # when a trial straddles chunks. A malformed cohort excludes its trial.
        fields = sorted((COHORT_REQUIRED | {'nct_id', 'split', 'trial_text',
                         'title', 'brief_summary', 'eligibility_criteria',
                         'cohort_name', 'trial_representation',
                         'cohort_specific_eligibility_criteria', 'common_eligibility_criteria'}) & set(columns))
        compact = []
        for chunk in chunks(fields):
            if chunk.nct_id.isna().any():
                raise ValueError('Missing nct_id in cohort input')
            # Screening precedes sampling and completeness checks. A trial with
            # conflicting decisions cannot enter through one accepted row.
            non_cancer.update(chunk.loc[chunk.cancer_treatment_trial.eq('no'), 'nct_id'])
            invalid.update(chunk.loc[~chunk.cancer_treatment_trial.isin(['yes', 'no']), 'nct_id'])
            chunk = chunk.loc[chunk.cancer_treatment_trial.eq('yes')].copy()
            if chunk.empty:
                continue
            chunk = complete_cohort_criteria(chunk)
            good = chunk.cohort_eligibility_criteria.map(lambda x: isinstance(x, str) and bool(x.strip()))
            invalid.update(chunk.loc[~good, 'nct_id'])
            chunk['cohort_eligibility_criteria'] = 'validated text'
            compact.append(chunk[sorted(COHORT_REQUIRED | {'nct_id', 'split'} |
                ({'cohort_name', 'trial_representation'} & set(chunk)))])
        if compact:
            compact = pd.concat(compact, ignore_index=True)
            if compact.cohort_trial_id.duplicated(keep=False).any():
                invalid.update(compact.loc[compact.cohort_trial_id.duplicated(keep=False), 'nct_id'])
            for trial, group in compact.groupby('nct_id'):
                counts[trial] = len(group)
                try:
                    validate_cohort_rows(group)
                except ValueError:
                    invalid.add(trial)
    else:
        fields = ['nct_id', 'this_space'] + (['cancer_treatment_trial'] if 'cancer_treatment_trial' in columns else [])
        for chunk in chunks(fields):
            if 'cancer_treatment_trial' in chunk:
                non_cancer.update(chunk.loc[chunk.cancer_treatment_trial.eq('no'), 'nct_id'])
                invalid.update(chunk.loc[~chunk.cancer_treatment_trial.isin(['yes', 'no']), 'nct_id'])
                chunk = chunk.loc[chunk.cancer_treatment_trial.eq('yes')]
            good = clean_trial_spaces(chunk)
            invalid.update(chunk.loc[~good, 'nct_id'])
            for trial, count in chunk.groupby('nct_id').size().items():
                counts[trial] = counts.get(trial, 0) + int(count)
    eligible = sorted(set(counts) - excluded - invalid - non_cancer)
    if len(eligible) < n_trials:
        raise ValueError(f'Only {len(eligible)} complete, non-held-out trials; need {n_trials}')
    sampled = pd.Series(eligible).sample(n=n_trials, random_state=seed).tolist()
    order = {trial: i for i, trial in enumerate(sampled)}
    frames = []
    for chunk in chunks():
        selected = chunk.loc[chunk.nct_id.isin(order)].copy()
        if not selected.empty:
            frames.append(selected)
    output = pd.concat(frames, ignore_index=True)
    output['_sample_order'] = output.nct_id.map(order)
    output = output.sort_values('_sample_order', kind='stable').drop(columns='_sample_order').reset_index(drop=True)
    output['cohort_index' if native else 'space_index'] = output.index
    assert output.groupby('nct_id').size().to_dict() == {trial: counts[trial] for trial in sampled}
    if native:
        output = complete_cohort_criteria(output)
        output['trial_representation'] = representation_for(output)
        validate_cohort_rows(output)
    else:
        assert clean_trial_spaces(output).all()
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    output.to_csv(temporary, index=False)
    temporary.replace(path)
    manifest = dict(input=str(Path(extraction_dir or input_csv).resolve()), output=str(path.resolve()),
                    dfci_holdout=str(Path(dfci_csv).resolve()), first_year=first_year, last_year=last_year,
                    seed=seed, sampled_trials=n_trials,
                    source_trials=len(counts), eligible_trials=len(eligible),
                    excluded_dfci_trials=sorted(set(counts) & excluded),
                    invalid_trials=sorted(invalid), sampled_nct_ids=sampled)
    if 'cancer_treatment_trial' in columns:
        manifest.update(cancer_treatment_filter='cancer_treatment_trial == yes',
                        excluded_non_cancer_trials=sorted(non_cancer))
    if native:
        manifest.update(sampled_cohorts=len(output), all_cohorts_retained=True, trial_representation=representation_for(output),
                        cancer_treatment_trials=len(set(counts) - non_cancer - invalid),
                        cancer_treatment_filter='cancer_treatment_trial == yes',
                        excluded_non_cancer_trials=sorted(non_cancer))
    else:
        manifest.update(sampled_spaces=len(output), all_spaces_retained=True)
    manifest.update(provenance)
    manifest['sampling_source_sha256'] = {p.name: file_sha256(p) for p in
        (Path(__file__), module_file('cohort_format'), Path(dfci_csv))}
    receipt = path.with_suffix('.sampling.json')
    receipt.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if not isinstance(v, list)}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--screened-trials', help='trials.screen_trials output: draw whole trials from those it accepted '
                                                  '(production)')
    inputs.add_argument('--input-csv', help='Cohort CSV; default <data root>/no_phi/trial_cohort_lineitems.csv')
    inputs.add_argument('--extraction-dir', help='Completed production run containing inputs.parquet, combined_outputs.json and cohorts.jsonl')
    parser.add_argument('--output-csv', help='Default: <data root>/no_phi/sample_trial_cohort_lineitems.csv')
    parser.add_argument('--dfci-trials-csv', help='Default: <data root>/no_phi/dfci_trials_summary_10-18-26.csv')
    parser.add_argument('--n-trials', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--first-year', type=int, default=2025)
    parser.add_argument('--last-year', type=int, default=2026)
    parser.add_argument('--chunksize', type=int, default=2000)
    args = parser.parse_args()
    args.output_csv = args.output_csv or str(no_phi('sample_trial_cohort_lineitems.csv'))
    args.dfci_trials_csv = args.dfci_trials_csv or str(no_phi('dfci_trials_summary_10-18-26.csv'))
    if args.screened_trials:
        sample_screened_trials(args.screened_trials, args.output_csv, n_trials=args.n_trials, seed=args.seed)
        return
    if args.input_csv is None and args.extraction_dir is None:
        args.input_csv = str(no_phi('trial_cohort_lineitems.csv'))
    sample_trial_spaces(args.input_csv, args.output_csv, args.dfci_trials_csv,
                        n_trials=args.n_trials, seed=args.seed, first_year=args.first_year,
                        last_year=args.last_year, chunksize=args.chunksize, extraction_dir=args.extraction_dir)


if __name__ == '__main__':
    main()
