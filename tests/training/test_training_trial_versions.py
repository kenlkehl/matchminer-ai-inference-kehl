"""Protocol and ctgov versions of a trial: retrieval represents each trial once."""
import random

import numpy as np
import pandas as pd
import pytest

from matchminer_ai.training.trialspace.retrieval import TrialSampler, query_cohorts
from matchminer_ai.training.cohort_format import cohort_version, validate_cohort_rows
from matchminer_ai.training.trialspace.mining import rank_dense


def corpus(trials=40, versions=('protocol', 'ctgov')):
    cohorts = []
    for t in range(trials):
        for version in versions:
            for n in range(1 + (t + len(version)) % 2):
                cohorts.append(dict(id=f'T{t}-{version}-{n}', trial_id=f'T{t}', source=version,
                                    text=f'{version} criteria {t} {n}', split='train'))
    return cohorts


def test_single_version_sample_is_unchanged_and_two_versions_give_one_per_trial():
    single = corpus(versions=('ctgov',))
    legacy = random.Random('7:patient_trial_sample:P1').sample(sorted({c['trial_id'] for c in single}), 10)
    assert TrialSampler(single, 10, 7)('P1') == sorted(i for i, c in enumerate(single) if c['trial_id'] in legacy)
    assert TrialSampler(single, 0, 7)('P1') == list(range(len(single)))
    cohorts = corpus()
    sampler, seen = TrialSampler(cohorts, 10, 7), set()
    for patient in range(30):
        chosen = [cohorts[i] for i in sampler(f'P{patient}')]
        versions = {}
        for cohort in chosen:
            versions.setdefault(cohort['trial_id'], set()).add(cohort['source'])
        assert len(versions) == 10 and all(len(v) == 1 for v in versions.values())
        assert all(sum(c['trial_id'] == t for c in chosen) == sum(c['trial_id'] == t and c['source'] == next(iter(v))
                   for c in cohorts) for t, v in versions.items())  # every cohort of the chosen version
        seen.update(next(iter(v)) for v in versions.values())
    assert seen == {'protocol', 'ctgov'}
    assert sampler('P3') == TrialSampler(cohorts, 10, 7)('P3') == TrialSampler(cohorts, 10, 7, 1)('P3')
    second = TrialSampler(cohorts, 10, 7, 2)
    assert second('P3') == TrialSampler(cohorts, 10, 7, 2)('P3')
    # Each mining round draws a fresh trial sample per patient, still one version per trial.
    assert sum(second(f'P{p}') != sampler(f'P{p}') for p in range(30)) >= 28
    assert all(len({cohorts[i]['trial_id'] for i in second(f'P{p}')}) ==
               len({(cohorts[i]['trial_id'], cohorts[i]['source']) for i in second(f'P{p}')}) == 10 for p in range(30))
    queries = query_cohorts(cohorts, 7)
    by_trial = {}
    for cohort in queries:
        by_trial.setdefault(cohort['trial_id'], set()).add(cohort['source'])
    assert len(by_trial) == 40 and all(len(v) == 1 for v in by_trial.values())
    assert {next(iter(v)) for v in by_trial.values()} == {'protocol', 'ctgov'}
    assert query_cohorts(cohorts, 7) == queries


def test_dense_mining_searches_and_queries_one_version_per_trial():
    cohorts = corpus(trials=12)
    patients = [dict(id=f'P{i}', text=f'patient {i}', split='train', source_trial_id=f'T{i}') for i in range(12)]
    rng = np.random.default_rng(0)
    vectors = {}
    for name, rows in (('patients', patients), ('cohorts', cohorts)):
        values = rng.normal(size=(len(rows), 8)).astype(np.float32)
        vectors[name] = values / np.linalg.norm(values, axis=1, keepdims=True)
    hits = rank_dense(patients, cohorts, vectors, patient_k=50, cohort_k=3, lexical_k=2, random_k=2,
                      seed=7, patient_trial_sample=5)
    source = {c['id']: (c['trial_id'], c['source']) for c in cohorts}
    for patient in patients:
        found = {source[h['cohort_id']] for h in hits
                 if h['direction'] == 'patient_to_cohort' and h['patient_id'] == patient['id']}
        assert len({t for t, _ in found}) == len(found) <= 5
    querying = {(source[h['cohort_id']]) for h in hits if h['direction'] == 'cohort_to_patient'}
    assert len({t for t, _ in querying}) == len(querying) == 12
    searched = lambda found: {p['id']: {source[h['cohort_id']][0] for h in found if h['direction'] == 'patient_to_cohort'
                                        and h['patient_id'] == p['id'] and h['source'] == 'embeddinggemma2'} for p in patients}
    second = rank_dense(patients, cohorts, vectors, patient_k=50, cohort_k=3, lexical_k=2, random_k=2,
                        seed=7, patient_trial_sample=5, trial_sample_round=2)
    assert sum(searched(second)[p] != searched(hits)[p] for p in searched(hits)) >= 10  # a fresh sample each round


def test_cohort_rows_validate_each_version_as_its_own_complete_set():
    row = dict(trial_id='NCT00000001', nct_id='NCT00000001', cancer_treatment_trial='yes', split='train',
               trial_representation='trial-treatment-cohorts-v4')
    frame = pd.DataFrame([
        dict(row, cohort_trial_id='NCT00000001-cohort-0', cohort_number=0, cohort_count=1, multiple_cohorts='no',
             cohort_name=None, cohort_eligibility_criteria='Protocol criteria.', cohort_criteria_source='protocol'),
        *[dict(row, cohort_trial_id=f'NCT00000001-ctgov-cohort-{n}', cohort_number=n, cohort_count=2,
               multiple_cohorts='yes', cohort_name=f'Arm {n}', cohort_eligibility_criteria=f'ctgov criteria {n}.',
               cohort_criteria_source='ctgov') for n in range(2)]])
    validate_cohort_rows(frame)
    # Rows written before 2026-11-05 call the ctgov version 'registry'.
    assert cohort_version(frame.replace({'cohort_criteria_source': {'ctgov': 'registry'}})).tolist() == [
        'protocol', 'ctgov', 'ctgov']
    with pytest.raises(ValueError, match='Incomplete'):
        validate_cohort_rows(frame.iloc[:2])
    with pytest.raises(ValueError, match='split'):
        validate_cohort_rows(frame.assign(split=['train', 'test', 'test']))
