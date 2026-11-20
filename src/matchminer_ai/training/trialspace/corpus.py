"""The TrialSpace corpus every round shares, and candidate pairs exported per retrieval direction.

``prepare_corpus`` adapts the patient summaries and the cohort table (trialspace.data.prepare): patients with
their key summary section as ``text`` and boilerplate section as ``boilerplate``, cohorts with their key
criteria, boilerplate criteria and tagged criteria lines, plus each synthetic patient's source-trial lineage.
"""
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

from matchminer_ai.embedding.trialspace_contract import DIRECTIONS
from matchminer_ai.labeling import score_contract

from matchminer_ai.training.cohort_format import file_stamp
from .data import digest, load_corpus, prepare, read_jsonl, write_json, write_jsonl

VERSION = 'trialspace-corpus-v3'


def identity_file(path, identity):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f'Changed TrialSpace inputs/settings at {path}; use a new run directory')
    write_json(path, identity)


def prepare_corpus(patients, cohorts, output, *, merge_source_splits=True):
    if not merge_source_splits:
        raise ValueError('Production cannot hold out synthetic source splits; use --merge-source-splits')
    output = Path(output)
    identity = dict(version=VERSION, patients=str(Path(patients).resolve()), cohorts=str(Path(cohorts).resolve()),
                    patient_stamp=file_stamp(patients), cohort_stamp=file_stamp(cohorts), contract=score_contract(),
                    merge_source_splits=merge_source_splits)
    identity_file(output / 'input_manifest.json', identity)
    complete = output / 'prepared.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        for name, expected in saved['output_stamp'].items():
            if file_stamp(output / name) != expected:
                raise ValueError('Prepared TrialSpace corpus changed')
        return saved
    prepare(patients, cohorts, output, merge_source_splits=merge_source_splits)
    ps, cs = load_corpus(output)
    by_trial = defaultdict(list)
    for cohort in cs:
        by_trial[cohort['trial_id']].append(cohort)
    lineage = [dict(patient_id=p['id'], cohort_id=c['id'], split=p['split'])
               for p in ps for c in by_trial.get(p.get('source_trial_id'), [])
               if p['split'] == c['split']]
    write_jsonl(output / 'source_trial_pairs.jsonl', lineage)
    versions = defaultdict(set)
    for cohort in cs:
        versions[cohort['trial_id']].add(cohort.get('source', 'ctgov'))
    result = dict(version=VERSION, patients=len(ps), cohorts=len(cs), trials=len(by_trial),
                  cohort_sources={name: sum(c.get('source', 'ctgov') == name for c in cs) for name in ('protocol', 'ctgov')},
                  trials_with_both_versions=sum(len(v) > 1 for v in versions.values()),
                  source_trial_pairs=len(lineage), contract=score_contract(),
                  output_stamp={name: file_stamp(output / name) for name in
                                 ('patients.jsonl', 'cohorts.jsonl', 'source_trial_pairs.jsonl')})
    write_json(complete, result)
    return result


def export_candidates(corpus, candidates, cohort_output, patient_output):
    """Candidate pairs per retrieval direction, with both key texts and both boilerplate texts."""
    patients, cohorts = load_corpus(corpus)
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    tables = {direction: [] for direction in DIRECTIONS}
    # Freeze the contract once for this export rather than per pair.
    contract = score_contract()
    for pair in read_jsonl(candidates):
        p, c = pmap[pair['patient_id']], cmap[pair['cohort_id']]
        if p['split'] != c['split'] or pair['split'] != p['split']:
            raise ValueError('Candidate export crosses an effective split')
        sources = pair['sources']
        for direction in DIRECTIONS:
            if not any(s.get('direction') in (direction, 'both') for s in sources):
                continue
            tables[direction].append(dict(pseudo_mrn=p['id'], patient_id=p['id'],
                patient_summary=p['text'], patient_boilerplate_text=p.get('boilerplate', ''),
                nct_id=c['trial_id'], trial_id=c['trial_id'], cohort_trial_id=c['id'], space_trial_id=c['id'],
                this_space=c['text'], cohort_key_criteria=c['text'], trial_boilerplate_text=c.get('boilerplate', ''),
                split=p['split'], candidate_id=digest([p['id'], c['id']]),
                patient_source_split=p.get('source_split', p['split']),
                cohort_source_split=c.get('source_split', c['split']),
                cohort_source=c.get('source', 'ctgov'),
                retrieval_sources=json.dumps(sources), trialspace_contract=contract))
    for direction, path in zip(DIRECTIONS, (cohort_output, patient_output), strict=True):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        # Full clinical strings repeat across millions of pairs. The default
        # 1 MiB dictionary spills almost immediately and duplicates that text.
        pd.DataFrame(tables[direction]).to_parquet(temporary, index=False,
            compression='zstd', use_dictionary=True, dictionary_pagesize_limit=256 * 1024**2)
        temporary.replace(path)
