"""Reading a TrialSpace corpus directory, the teacher labeler's input.

A corpus directory holds ``patients.jsonl`` and ``cohorts.jsonl`` (training's corpus preparation writes
them). An entity's ``text`` is what retrieval and the match teacher read: a patient's key summary section
(the cancer history) and a cohort's key criteria. ``boilerplate`` is what the boilerplate teacher reads: a
patient's boilerplate section (other relevant history) and a cohort's boilerplate criteria. A cohort's
``items`` are its criteria lines with their inclusion/exclusion and boilerplate tags, which the
per-criterion teacher reads with the patient's complete summary.
"""
from pathlib import Path

from .files import read_jsonl


def text(value):
    return value if isinstance(value, str) and value.strip() else ''


def split_name(value):
    value = 'val' if value == 'validation' else value
    if value not in ('train', 'val', 'test'):
        raise ValueError('Every input needs an explicit train/val/test split; no random pair splits')
    return value


def validate_corpus(patients, cohorts):
    trial_splits = {}
    for records in (patients, cohorts):
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('Duplicate entity IDs')
        for row in records:
            split_name(row['split'])
            if not text(row['text']):
                raise ValueError('Empty entity text')
    for row in cohorts:
        if trial_splits.setdefault(row['trial_id'], row['split']) != row['split']:
            raise ValueError('Sibling cohorts must remain in the same split')
    for row in patients:
        if row.get('source_trial_id') in trial_splits and trial_splits[row['source_trial_id']] != row['split']:
            raise ValueError('Synthetic patient and source trial cross splits')


def load_corpus(directory):
    directory = Path(directory)
    patients, cohorts = read_jsonl(directory / 'patients.jsonl'), read_jsonl(directory / 'cohorts.jsonl')
    for cohort in cohorts:  # corpora prepared before 2026-11-05 called the ctgov version 'registry'
        if cohort.get('source') == 'registry':
            cohort['source'] = 'ctgov'
    validate_corpus(patients, cohorts)
    return patients, cohorts


def full_summary(patient):
    """A corpus patient's complete summary: the key section, then ``Other relevant history:`` and the
    boilerplate section, as the inference package joins them."""
    from matchminer_ai.patients.postprocess import normalize_patient_summary
    return normalize_patient_summary(patient['text'], patient.get('boilerplate') or None)


def pooled_records(rows):
    """Rows of one full synthetic pool: every row in split ``train``, its original split kept as ``source_split``."""
    return [dict(row, source_split=row.get('source_split', row.get('split', 'train')), split='train')
            for row in rows]


def require_full_pool(patients, cohorts):
    if any(row.get('split') != 'train' for row in [*patients, *cohorts]):
        raise ValueError('Production retrieval/labeling/fitting requires one full synthetic pool. '
                         'Reprepare with --merge-source-splits and rerun both retrieval directions; '
                         'historical assignments belong only in source_split.')


__all__ = ['full_summary', 'load_corpus', 'pooled_records', 'require_full_pool', 'split_name', 'text',
           'validate_corpus']
