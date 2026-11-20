"""Trial versions in mining (one version per trial), top-k selection, and candidate merging."""
from collections import defaultdict
import random

import numpy as np


def top_indices(scores, count):
    # Stable sorting retains the original index-order tie rule. The lexical
    # corpus can contain tens of thousands of entries per query; keep this loop
    # in NumPy rather than repeatedly constructing Python float/tuple keys.
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind='stable')[:count].tolist()


# A trial can be represented twice: by the cohorts extracted from its full
# (synthetic) protocol and by those extracted from its ctgov text. Mining
# uses one version per trial so neither direction sees a trial twice.
TRIAL_VERSION_POLICY = 'one version per trial: random per patient-trial pair; per trial for cohort queries'


def _versions(cohorts):
    """trial_id -> version ('protocol'/'ctgov') -> cohort indices, in corpus order."""
    by_trial = defaultdict(lambda: defaultdict(list))
    for index, cohort in enumerate(cohorts):
        by_trial[cohort['trial_id']][cohort.get('source', 'ctgov')].append(index)
    return by_trial


def has_trial_versions(cohorts):
    return any(len(versions) > 1 for versions in _versions(cohorts).values())


class TrialSampler:
    """Cohort indices of a random trial sample for each patient query.

    A patient searches the cohorts of ``trials`` sampled trials, approximating one
    cancer center's open-trial portfolio instead of every synthetic trial; 0 or
    None keeps every trial. A trial with both a protocol and a ctgov version
    contributes one version's cohorts, chosen at random for each patient and
    trial. Both choices depend only on the seed, the mining round and IDs: each
    round draws a fresh portfolio and version choice per patient (user
    instruction, 2026-11-02), and embedders compared within a round search the
    same one. Round 1 keeps the original draws, and single-version corpora give
    the same samples as before versions existed.
    """

    def __init__(self, cohorts, trials, seed, mining_round=1):
        if trials is not None and trials < 0:
            raise ValueError('Patient trial sample must be nonnegative')
        if mining_round < 1:
            raise ValueError('Mining round must be positive')
        self.by_trial = _versions(cohorts)
        self.trial_ids = sorted(self.by_trial)
        self.trials, self.seed = trials, seed
        self.prefix = f'{seed}:' if mining_round == 1 else f'{seed}:round{mining_round}:'

    def __call__(self, patient_id):
        chosen = self.trial_ids
        if self.trials and self.trials < len(self.trial_ids):
            chosen = sorted(random.Random(f'{self.prefix}patient_trial_sample:{patient_id}').sample(
                self.trial_ids, self.trials))
        rng = random.Random(f'{self.prefix}patient_trial_version:{patient_id}')
        indices = []
        for trial in chosen:
            versions = self.by_trial[trial]
            # Draw only for two-version trials, so other trials never shift the stream.
            version = next(iter(versions)) if len(versions) == 1 else rng.choice(sorted(versions))
            indices.extend(versions[version])
        return sorted(indices)


def query_cohorts(cohorts, seed):
    """Cohorts that query for patients: each trial's cohorts in one version, chosen per trial."""
    keep = set()
    for trial, versions in _versions(cohorts).items():
        version = (next(iter(versions)) if len(versions) == 1 else
                   random.Random(f'{seed}:cohort_query_version:{trial}').choice(sorted(versions)))
        keep.update(versions[version])
    return [cohort for index, cohort in enumerate(cohorts) if index in keep]


def merge_candidates(patients, cohorts, hits):
    """One pair across every question/source/direction; preserve every provenance hit."""
    patients, cohorts = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    pairs = {}
    for hit in hits:
        pid, cid = hit['patient_id'], hit['cohort_id']
        if patients[pid]['split'] != cohorts[cid]['split']:
            raise ValueError('Cross-split candidate')
        key = (pid, cid)
        if key not in pairs:
            pairs[key] = dict(patient_id=pid, cohort_id=cid, trial_id=cohorts[cid]['trial_id'],
                              split=patients[pid]['split'], sources=[])
        pairs[key]['sources'].append({k: v for k, v in hit.items() if k not in ('patient_id', 'cohort_id')})
    return list(pairs.values())
