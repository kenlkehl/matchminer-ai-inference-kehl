"""Explicit boundary between cohort criteria and historical training columns."""
from numbers import Real
import math
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# v5: named cohorts whose complete criteria come from one extraction request each
# (inference cohort_workflow: screen, cohort names, criteria per cohort). v4: named
# cohorts with cohort-specific and common criteria. v3: unnamed historical strings.
REPRESENTATION = 'trial-treatment-cohorts-v5'
NAMED_SPLIT_REPRESENTATION = 'trial-treatment-cohorts-v4'
LEGACY_REPRESENTATION = 'trial-treatment-cohorts-v3'
SUPPORTED_REPRESENTATIONS = {LEGACY_REPRESENTATION, NAMED_SPLIT_REPRESENTATION, REPRESENTATION}
NAMED_REPRESENTATIONS = {NAMED_SPLIT_REPRESENTATION, REPRESENTATION}
COHORT_REQUIRED = {'trial_id', 'cohort_trial_id', 'cohort_number', 'cohort_count',
                   'cohort_eligibility_criteria', 'cancer_treatment_trial', 'multiple_cohorts'}
# Model-extracted criteria whose single-cohort text is the model's own, never a
# ctgov eligibility field: full protocols and ctgov text (schemas
# trial-registry-cohorts-v1 and trial-cohorts-stepwise-v1).
EXTRACTED_SOURCES = ('protocol', 'ctgov')


def cohort_version(frame):
    """'protocol' or 'ctgov' per row: which representation of its trial a cohort belongs to.

    A retrieval table can hold both versions of a trial, each a complete cohort set.
    Historical ctgov sources ('registry' before 2026-11-05, and 'llm', 'eligibility_criteria') are 'ctgov'.
    """
    if 'cohort_criteria_source' not in frame:
        return pd.Series('ctgov', index=frame.index)
    return frame.cohort_criteria_source.map(lambda s: 'protocol' if s == 'protocol' else 'ctgov')


def read_clinical_csv(path, **kwargs):
    """Preserve literal clinical text; only empty CSV cells represent missing data.

    Pandas' default NA vocabulary includes valid text such as "None", "NA" and
    "N/A". Keep those values intact, while retaining nulls for absent single-
    cohort components and normal numeric inference for counts and indices.
    Chunked reads and column selection use the same policy.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[''], **kwargs)


def is_cohort_table(frame):
    return 'cohort_eligibility_criteria' in frame.columns


def cohort_components(value):
    """Read named objects and historical strings without inventing old names."""
    if isinstance(value, str) and value.strip():
        return None, value
    fields = {'cohort_name', 'cohort_specific_eligibility_criteria'}
    if (not isinstance(value, dict) or set(value) != fields
            or not all(isinstance(value[k], str) and value[k].strip() for k in fields)):
        raise ValueError('Invalid saved cohort name/criteria object')
    return value['cohort_name'], value['cohort_specific_eligibility_criteria']


def representation_for(frame):
    """Preserve historical identity; do not claim old strings are named objects."""
    if 'trial_representation' in frame and not frame.empty:
        versions = set(frame.trial_representation)
        if len(versions) != 1 or not versions <= SUPPORTED_REPRESENTATIONS:
            raise ValueError('Conflicting trial representation; repair/regenerate the cohort CSV before generation')
        return next(iter(versions))
    if 'cohort_name' not in frame:
        return LEGACY_REPRESENTATION
    return NAMED_SPLIT_REPRESENTATION if 'cohort_specific_eligibility_criteria' in frame else REPRESENTATION


def complete_cohort_criteria(frame):
    """Rebuild complete text from specific/common fields or original eligibility.

Older cohort CSVs containing only the already-combined criteria remain readable.
When source fields are present they are authoritative, even if combined text is
stale. Missing common text represents no shared criteria, not a missing cohort.
"""
    frame = frame.copy()
    components = {'cohort_specific_eligibility_criteria', 'common_eligibility_criteria'}
    if components & set(frame) and not components <= set(frame):
        raise ValueError('Both specific and common criteria columns are required')
    def text(row):
        if row.multiple_cohorts == 'yes' and components <= set(frame):
            specific, common = row.cohort_specific_eligibility_criteria, row.common_eligibility_criteria
            if not isinstance(specific, str) or not specific.strip():
                raise ValueError('Missing cohort-specific criteria')
            common = '' if pd.isna(common) else common
            if not isinstance(common, str):
                raise ValueError('Common criteria must be text')
            return (f'Cohort-specific eligibility criteria:\n{specific}\n\n'
                    f'Common eligibility criteria:\n{common}') if common else specific
        if row.multiple_cohorts == 'no':
            if row.get('cohort_criteria_source') in EXTRACTED_SOURCES:
                # Protocol and current ctgov extraction return the sole cohort's
                # own criteria; ctgov eligibility text must not replace them.
                eligibility = row.cohort_eligibility_criteria
            elif 'eligibility_criteria' in frame:
                eligibility = row.eligibility_criteria
            elif row.get('cohort_criteria_source') == 'eligibility_criteria':
                eligibility = row.cohort_eligibility_criteria
            else:
                raise ValueError('Single cohort requires original eligibility_criteria; full trial text is not eligible input')
            if not isinstance(eligibility, str) or not eligibility.strip():
                raise ValueError('Missing original eligibility_criteria for a single cohort')
            return eligibility
        return row.cohort_eligibility_criteria
    if not frame.empty:
        frame['cohort_eligibility_criteria'] = frame.apply(text, axis=1)
        single = frame.multiple_cohorts.eq('no')
        if 'cohort_criteria_source' in frame:
            single &= ~frame.cohort_criteria_source.isin(EXTRACTED_SOURCES)
        if single.any():
            frame.loc[single, 'cohort_criteria_source'] = 'eligibility_criteria'
    return frame


def file_stamp(path):
    """Cheap change detector: byte size and whole-second mtime, without reading content.

    Whole seconds tolerate the differing timestamp precision of NFS clients on
    other hosts. A copy gets a new mtime, so compare a copy with its source by
    size alone.
    """
    stat = Path(path).stat()
    return f'{stat.st_size}:{int(stat.st_mtime)}'


def file_sha256(path):
    """Content hash for small source/config files only; use file_stamp for data files."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_production_cohorts(directory, *, seed=42):
    """Read the completed public-API production export, without new LLM calls.

    Decisions supply authoritative expected cohort counts, so a missing JSONL
    row cannot be mistaken for a smaller but complete trial. Source rows supply
    original eligibility and ctgov metadata. Splits are assigned before filtering.
    """
    directory = Path(directory)
    status = json.loads((directory / 'status.json').read_text())
    if status.get('state') != 'complete' or status.get('errors') != 0:
        raise ValueError('Production extraction must be complete with no unresolved errors')
    decisions = json.loads((directory / 'combined_outputs.json').read_text())
    source = pd.read_parquet(directory / 'inputs.parquet').sort_values('nct_id').reset_index(drop=True)
    if (source.nct_id.isna().any() or not source.nct_id.is_unique
            or set(source.nct_id) != set(decisions) or len(source) != status['total']):
        raise ValueError('Source trials, extraction status and decisions disagree')
    if 'split' not in source:
        source['split'] = np.random.default_rng(seed).choice(['train', 'val', 'test'], len(source), p=[.8, .1, .1])
    if not source.split.isin(['train', 'val', 'validation', 'test']).all():
        raise ValueError('Missing or invalid source splits')
    counts, rejected = {}, []
    for trial, result in decisions.items():
        if result.get('cancer_treatment_trial') == 'no':
            rejected.append(trial)
        elif result.get('cancer_treatment_trial') == 'yes':
            if result.get('multiple_cohorts') == 'no':
                counts[trial] = 1
            elif result.get('multiple_cohorts') == 'yes' and isinstance(result.get('cohorts'), list) and len(result['cohorts']) >= 2:
                counts[trial] = len(result['cohorts'])
            else:
                raise ValueError(f'Invalid cohort decision for {trial}')
        else:
            raise ValueError(f'Missing cancer-treatment decision for {trial}')
    cohorts = pd.read_json(directory / 'cohorts.jsonl', lines=True)
    if set(cohorts.trial_id) != set(counts):
        raise ValueError('Cohort JSONL must cover exactly the accepted cancer-treatment trials')
    cohorts['cohort_count'] = cohorts.trial_id.map(counts)
    validate_cohort_rows(cohorts)
    if len(cohorts) != status['cohort_rows'] or len(rejected) != status['rejected']:
        raise ValueError('Cohort/rejection counts disagree with extraction status')
    for row in cohorts.itertuples():
        decision = decisions[row.trial_id]
        if row.multiple_cohorts != decision['multiple_cohorts']:
            raise ValueError(f'Cohort decision mismatch for {row.trial_id}')
        if row.multiple_cohorts == 'yes':
            name, specific = cohort_components(decision['cohorts'][row.cohort_number])
            if (row.cohort_specific_eligibility_criteria != specific
                    or row.common_eligibility_criteria != decision['common_eligibility_criteria']
                    or (name is not None and getattr(row, 'cohort_name', None) != name)):
                raise ValueError(f'Criteria or name disagree with saved combined JSON for {row.trial_id}')
    frame = cohorts.merge(source, left_on='trial_id', right_on='nct_id', validate='many_to_one')
    frame = frame.sort_values(['trial_id', 'cohort_number']).reset_index(drop=True)
    complete = complete_cohort_criteria(frame)
    if not complete.cohort_eligibility_criteria.eq(frame.cohort_eligibility_criteria).all():
        raise ValueError('Saved full criteria disagree with specific/common criteria or original eligibility_criteria; repair the export first')
    if not frame.loc[frame.multiple_cohorts.eq('no'), 'cohort_criteria_source'].eq('eligibility_criteria').all():
        raise ValueError('Single cohort provenance must identify eligibility_criteria')
    complete['trial_representation'] = representation_for(complete)
    provenance = dict(source_trials=len(source), cancer_treatment_trials=len(counts),
        excluded_non_cancer_trials=sorted(rejected), extraction_directory=str(directory.resolve()),
        extraction_run_identity=json.loads((directory / 'manifest.json').read_text())['run_identity'],
        source_stamp={name: file_stamp(directory / name) for name in
            ('inputs.parquet', 'combined_outputs.json', 'cohorts.jsonl', 'manifest.json')})
    repair = directory / 'cohort_rows_repair.json'
    if repair.exists():
        provenance['cohort_rows_repair'] = str(repair.resolve())
        provenance['source_stamp'][repair.name] = file_stamp(repair)
    return complete, provenance


def validate_cohort_rows(frame):
    """Require complete accepted trial populations before sampling or generation."""
    missing = COHORT_REQUIRED - set(frame)
    if missing:
        raise ValueError(f'Missing cohort columns: {sorted(missing)}')
    if frame[list(COHORT_REQUIRED)].isna().any().any():
        raise ValueError('Missing cohort identity, criteria, or decision')
    if frame.cohort_trial_id.duplicated().any():
        raise ValueError('Duplicate cohort_trial_id')
    for column in ('trial_id', 'cohort_trial_id'):
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'Missing {column}')
    if not frame.cancer_treatment_trial.eq('yes').all():
        raise ValueError('Only accepted cancer-treatment trials may supply cohort rows')
    if not frame.multiple_cohorts.isin(['yes', 'no']).all():
        raise ValueError('Invalid multiple_cohorts decision')
    if representation_for(frame) in NAMED_REPRESENTATIONS and 'cohort_name' not in frame:
        raise ValueError('Named-cohort representation requires cohort_name')
    if 'cohort_name' in frame:
        names = frame.loc[frame.multiple_cohorts.eq('yes'), 'cohort_name']
        if not names.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('Every extracted cohort must retain a nonempty cohort_name')
    if not frame.cohort_eligibility_criteria.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Cohort criteria must be nonempty text')
    if 'nct_id' in frame and not frame.nct_id.eq(frame.trial_id).all():
        raise ValueError('nct_id and trial_id disagree')
    for trial, group in frame.groupby('trial_id', sort=False):
        if 'split' in group and (not group.split.isin(['train', 'val', 'validation', 'test']).all()
                                 or group.split.nunique() != 1):
            raise ValueError(f'Conflicting/missing split for {trial}')
    # Each version of a trial (protocol, ctgov) is its own complete cohort set.
    for (trial, _), group in frame.groupby([frame.trial_id, cohort_version(frame)], sort=False):
        count = group.cohort_count.iloc[0]
        if isinstance(count, bool) or not isinstance(count, Real) or not math.isfinite(count):
            raise ValueError(f'Invalid cohort count for {trial}')
        if not group.cohort_number.map(lambda n: isinstance(n, Real) and not isinstance(n, bool)
                                      and n >= 0 and n % 1 == 0).all():
            raise ValueError(f'Invalid cohort numbers for {trial}')
        if (count < 1 or int(count) != count or group.cohort_count.nunique() != 1
                or len(group) != count or sorted(group.cohort_number.tolist()) != list(range(int(count)))):
            raise ValueError(f'Incomplete or duplicated cohort set for {trial}')
        if group.multiple_cohorts.nunique() != 1 or (group.multiple_cohorts.iloc[0] == 'yes') != (count > 1):
            raise ValueError(f'Cohort count and decision disagree for {trial}')
    return frame


def training_rows(frame):
    """Adapt complete cohort text; shared inclusion rules are never exclusions.

    Keep native cohort IDs/fields and mark the representation. Historical names
    are internal training vocabulary, not a claim of seven-field space output.
    """
    if not is_cohort_table(frame):
        return frame.copy()
    validate_cohort_rows(frame)
    representation = representation_for(frame)
    frame = complete_cohort_criteria(frame)
    aliases = {'nct_id': frame.trial_id, 'this_space': frame.cohort_eligibility_criteria}
    if 'cohort_index' in frame:
        aliases['space_index'] = frame.cohort_index
    for key, values in aliases.items():
        if key in frame and not frame[key].eq(values).all():
            raise ValueError(f'Conflicting native cohort and legacy {key} values')
        frame[key] = values
    if 'trial_boilerplate_text' in frame and frame.trial_boilerplate_text.fillna('').ne('').any():
        raise ValueError('Cohort output does not contain separately extracted boilerplate')
    frame['trial_boilerplate_text'] = ''
    frame['trial_representation'] = representation
    return frame


def patient_generation_criteria(item):
    if item.get('trial_representation') in SUPPORTED_REPRESENTATIONS:
        return item['cohort_eligibility_criteria']
    return item['this_space'] + '\nNo history of:\n' + item['trial_boilerplate_text']
