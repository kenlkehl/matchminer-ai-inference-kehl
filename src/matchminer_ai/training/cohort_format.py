"""Explicit boundary between cohort criteria and historical training columns."""
from numbers import Real
import math
import hashlib
from pathlib import Path

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
