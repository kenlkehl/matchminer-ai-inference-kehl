"""Lossless adapters and provenance-preserving TrialSpace corpus files.

A corpus entity's ``text`` is what retrieval and the match teacher read: a patient's key summary
section (the cancer history) and a cohort's key criteria. ``boilerplate`` is what the boilerplate
teacher reads: a patient's boilerplate section (other relevant history) and a cohort's boilerplate
criteria. A cohort's ``items`` are its criteria lines with their inclusion/exclusion and boilerplate
tags (``cohort_criteria_items``), which the per-criterion teacher and the decision model read with
the patient's complete summary (``text``, ``Other relevant history:``, ``boilerplate``).
"""
from pathlib import Path

import pandas as pd
# The corpus reader and JSON helpers are the inference labeler's (matchminer_ai.labeling, 2026-11-06).
from matchminer_ai.labeling.corpus import full_summary, load_corpus, split_name, text, validate_corpus  # noqa: F401
from matchminer_ai.labeling.files import digest, read_jsonl, write_json, write_jsonl  # noqa: F401
from matchminer_ai.training.cohort_format import read_clinical_csv


def patient_sections(row):
    """(key, boilerplate): a summary's cancer history and its other relevant history.

    Accepts the separate training columns (``patient_summary`` key section plus
    ``patient_boilerplate_text``), the public ones (``cancer_history_summary`` plus
    ``general_exclusion_criteria_evidence``), or one complete summary with an ``Other relevant
    history:`` heading, which is split.
    """
    from matchminer_ai.patients.postprocess import normalize_patient_summary, split_patient_summary
    key = text(row.get('patient_summary')) or text(row.get('cancer_history_summary'))
    other = text(row.get('patient_boilerplate_text')) or text(row.get('general_exclusion_criteria_evidence'))
    if not key:
        raise ValueError('Missing patient summary')
    key_section, embedded = split_patient_summary(normalize_patient_summary(key, other or None))
    return key_section, embedded


def read_table(path):
    path = Path(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.jsonl':
        return pd.DataFrame(read_jsonl(path))
    return read_clinical_csv(path)


def selected_splits(value):
    """Canonical source partitions; selecting them for fitting does not rename them."""
    values = [value] if isinstance(value, str) else list(value)
    if not values:
        raise ValueError('Select at least one source split')
    chosen = {split_name(item) for item in values}
    return [name for name in ('train', 'val', 'test') if name in chosen]


def prepare(patients_path, cohorts_path, output, *, merge_source_splits=False):
    from matchminer_ai.training.cohort_format import cohort_version, training_rows
    patients, cohorts = read_table(patients_path), read_table(cohorts_path)
    if 'cohort_trial_id' not in cohorts:
        raise ValueError('Use the native complete cohort table, not historical trial spaces')
    normalized = training_rows(cohorts)
    patient_rows = []
    for row in patients.to_dict('records'):
        identifier = row.get('patient_id', row.get('pseudo_mrn'))
        if pd.isna(identifier):
            raise ValueError('Missing patient identifier')
        key, boilerplate = patient_sections(row)
        item = dict(id=str(identifier), text=key, boilerplate=boilerplate, split=split_name(row.get('split')))
        origin = row.get('source_trial_id', row.get('nct_id'))
        if isinstance(origin, str) and origin.strip():
            item['source_trial_id'] = origin
        patient_rows.append(item)
    # source: which version of its trial a cohort is ('protocol' or 'ctgov');
    # mining represents each trial by one version (retrieval.TrialSampler).
    key_column = 'cohort_key_criteria' if 'cohort_key_criteria' in normalized else 'cohort_eligibility_criteria'
    from matchminer_ai.trials.cohort_boilerplate import ITEMS_COLUMN, load_criteria_items
    cohort_rows = [dict(id=row['cohort_trial_id'], trial_id=row['nct_id'], source=version,
                       text=row[key_column], boilerplate=text(row.get('cohort_boilerplate_criteria')),
                       items=load_criteria_items(row[ITEMS_COLUMN]) if text(row.get(ITEMS_COLUMN)) else [],
                       split=split_name(row.get('split')))
                   for row, version in zip(normalized.to_dict('records'), cohort_version(normalized))]
    if 'cohort_name' in normalized:
        for row, name in zip(cohort_rows, normalized.cohort_name):
            row['cohort_name'] = name if isinstance(name, str) else None
    validate_corpus(patient_rows, cohort_rows)
    if merge_source_splits:
        # Validate the source lineage first, then make one shared retrieval and
        # fitting pool. The original assignments remain available for provenance.
        for row in patient_rows + cohort_rows:
            row['source_split'], row['split'] = row['split'], 'train'
    output = Path(output)
    write_jsonl(output / 'patients.jsonl', patient_rows)
    write_jsonl(output / 'cohorts.jsonl', cohort_rows)
    write_json(output / 'corpus.json', dict(patients=len(patient_rows), cohorts=len(cohort_rows),
        patient_sha256=digest(patient_rows), cohort_sha256=digest(cohort_rows),
        source_patients=str(Path(patients_path).resolve()), source_cohorts=str(Path(cohorts_path).resolve()),
        split_policy='all_synthetic_training' if merge_source_splits else 'preserve_source_splits'))
