#!/usr/bin/env python3
"""Stages patients, cohorts and corpus: the evaluation's patients, cohorts and query samples, in the labeler's format.

``patients``: one row per summarized validation index event (``patient_id`` is the event's pseudo_mrn) with its key
section (cancer history) as ``text``, its boilerplate section (``general_exclusion_criteria_evidence``) as
``boilerplate`` and an opaque ``person`` index (one per patient, from processed_soc_treatments.csv; no MRN is
written), the sections split exactly as training's corpus splits summaries (trialspace.data.patient_sections).

``cohorts``: the ctgov cohorts of the extraction stage (cohorts.py), as training's corpus holds ctgov cohorts:
IDs ``<NCT>-ctgov-cohort-<n>``, ``text`` the key criteria, ``boilerplate`` the boilerplate criteria and ``items``
every tagged criteria line (trialspace.data.prepare, retrieval_cohorts.build).

``corpus``: the labeler's corpus directory (patients.jsonl hard-linked from the patients stage, cohorts.jsonl) and
``sample.json``: the seeded query samples, ``--patient-queries`` index events (patient-centric) and
``--cohort-queries`` cohorts (trial-centric), with the seed and every sampled ID.
"""
import argparse
import os
from pathlib import Path
import random

from matchminer_ai.evaluation.retro_fidelity.common import SPLIT, digest, log, private, read_jsonl, run_stage, stage, stamp_inputs, write_json, write_jsonl  # noqa: E402

VERSION = 'retro-fidelity-prepare-v1'


def patients(args):
    import pandas as pd
    from matchminer_ai.training.cohort_format import read_clinical_csv
    from matchminer_ai.training.trialspace.data import patient_sections
    files = sorted(Path(args.summaries).glob('*.parquet'))
    if not files:
        raise ValueError('No summary batches found')
    identity = dict(version=VERSION, stage='patients', split=SPLIT,
                    inputs=stamp_inputs([*files, args.denominator]))

    def work():
        soc = read_clinical_csv(args.denominator, usecols=['pseudo_mrn', 'dfci_mrn'], dtype=str)
        pairs = soc.drop_duplicates()
        if pairs.pseudo_mrn.duplicated().any():
            raise ValueError('An index event (pseudo_mrn) maps to more than one patient')
        person_of = pairs.set_index('pseudo_mrn').dfci_mrn
        index = {mrn: i for i, mrn in enumerate(sorted(person_of.unique()))}
        rows = []
        for path in files:
            frame = pd.read_parquet(path, columns=['patient_id', 'cancer_history_summary',
                                                   'general_exclusion_criteria_evidence'])
            for record in frame.to_dict('records'):
                event = str(record['patient_id'])
                key, boilerplate = patient_sections(record)
                rows.append(dict(id=event, text=key, boilerplate=boilerplate, split=SPLIT,
                                 person=index[person_of[event]], batch=path.stem))
        rows.sort(key=lambda row: int(row['id']))
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate index events across summary batches')
        write_jsonl(Path(args.output) / 'patients.jsonl', rows)
        summary = dict(events=len(rows), persons=len({row['person'] for row in rows}), batches=len(files),
                       empty_boilerplate=sum(not row['boilerplate'].strip() for row in rows),
                       patient_sha256=digest(rows))
        write_json(Path(args.output) / 'patients.json', summary)
        log(f'patients: {summary["events"]} events of {summary["persons"]} patients')
        return summary

    return run_stage(args.output, 'retro-patients', identity, ['patients.jsonl', 'patients.json'], work,
                     check=args.check)


def cohort_rows(table):
    """Training's corpus rows for a ctgov cohort table (retrieval_cohorts.build, trialspace.data.prepare)."""
    from matchminer_ai.labeling.corpus import text
    from matchminer_ai.trials.cohort_boilerplate import ITEMS_COLUMN, load_criteria_items
    from matchminer_ai.training.cohort_format import cohort_version, read_clinical_csv, training_rows
    frame = read_clinical_csv(table)
    if not frame.cohort_criteria_source.isin(['ctgov', 'registry']).all():
        raise ValueError('Expected ctgov cohort rows')
    # The frozen extraction programs predate the registry -> ctgov rename (training 38cee5c). The training's v4 tables
    # were renamed the same way; without it complete_cohort_criteria would replace a sole cohort's own extracted
    # criteria with the listing's whole eligibility text (it trusts extracted text only for 'protocol' and 'ctgov').
    frame = frame.assign(cohort_criteria_source=frame.cohort_criteria_source.replace({'registry': 'ctgov'}))
    frame = frame.assign(cohort_trial_id=frame.trial_id + '-ctgov-cohort-' + frame.cohort_number.astype(int).astype(str),
                         nct_id=frame.trial_id, split=SPLIT)
    frame = frame.sort_values(['trial_id', 'cohort_number']).reset_index(drop=True)
    normalized = training_rows(frame)
    if not normalized.cohort_eligibility_criteria.eq(frame.cohort_eligibility_criteria).all():
        raise ValueError('Rebuilt cohort criteria differ from the extracted criteria')
    rows, empty = [], 0
    for row, version in zip(normalized.to_dict('records'), cohort_version(normalized), strict=True):
        if not text(row['cohort_key_criteria']):
            empty += 1
            continue
        rows.append(dict(id=row['cohort_trial_id'], trial_id=row['nct_id'], source=version,
                         text=row['cohort_key_criteria'], boilerplate=text(row.get('cohort_boilerplate_criteria')),
                         items=load_criteria_items(row[ITEMS_COLUMN]) if text(row.get(ITEMS_COLUMN)) else [],
                         split=SPLIT, cohort_name=row['cohort_name'] if isinstance(row.get('cohort_name'), str) else None))
    return rows, empty


def cohorts(args):
    table = Path(args.extraction) / 'trial_cohort_lineitems_deboilerplated.csv'
    identity = dict(version=VERSION, stage='cohorts', split=SPLIT, extraction=stage.receipt_id(args.extraction))

    def work():
        rows, empty = cohort_rows(table)
        write_jsonl(Path(args.output) / 'cohorts.jsonl', rows)
        summary = dict(cohorts=len(rows), trials=len({r['trial_id'] for r in rows}), dropped_without_key_criteria=empty,
                       without_boilerplate=sum(not r['boilerplate'] for r in rows),
                       criteria_lines=sum(len(r['items']) for r in rows),
                       exclusion_lines=sum(i.get('criterion_type') == 'exclusion' for r in rows for i in r['items']),
                       untyped_lines=sum(i.get('criterion_type') not in ('inclusion', 'exclusion')
                                         for r in rows for i in r['items']),
                       cohort_sha256=digest(rows))
        write_json(Path(args.output) / 'cohorts.json', summary)
        log(f'cohorts: {summary["cohorts"]} cohorts of {summary["trials"]} trials, {summary["criteria_lines"]} lines')
        return summary

    return run_stage(args.output, 'retro-cohorts', identity, ['cohorts.jsonl', 'cohorts.json'], work,
                     check=args.check, upstream=dict(extraction=identity['extraction']))


def draw(ids, count, seed, name):
    if count > len(ids):
        raise ValueError(f'Cannot draw {count} {name} from {len(ids)}')
    return random.Random(f'{seed}:{name}').sample(list(ids), count)


def corpus(args):
    from matchminer_ai.labeling.corpus import validate_corpus
    identity = dict(version=VERSION, stage='corpus', patients=stage.receipt_id(args.patients),
                    cohorts=stage.receipt_id(args.cohorts), seed=args.seed, patient_queries=args.patient_queries,
                    cohort_queries=args.cohort_queries)
    output = Path(args.output)

    def work():
        target = output / 'patients.jsonl'
        if not target.exists():
            os.link(Path(args.patients) / 'patients.jsonl', target)  # the same file: no copy, the same stamp
        patient_rows = read_jsonl(target)
        cohort_rows_ = read_jsonl(Path(args.cohorts) / 'cohorts.jsonl')
        validate_corpus(patient_rows, cohort_rows_)
        write_jsonl(output / 'cohorts.jsonl', cohort_rows_)
        events = [row['id'] for row in patient_rows]  # sorted by event number
        spaces = [row['id'] for row in cohort_rows_]  # sorted by trial and cohort number
        sample = dict(seed=args.seed, method='random.Random(f"{seed}:{name}").sample(IDs in corpus order, n)',
                      patient_queries=draw(events, args.patient_queries, args.seed, 'patient_queries'),
                      cohort_queries=draw(spaces, args.cohort_queries, args.seed, 'cohort_queries'),
                      patient_pool=len(events), cohort_pool=len(spaces))
        write_json(output / 'sample.json', sample)
        summary = dict(patients=len(patient_rows), cohorts=len(cohort_rows_), patient_queries=args.patient_queries,
                       cohort_queries=args.cohort_queries,
                       query_persons=len({r['person'] for r in patient_rows if r['id'] in set(sample['patient_queries'])}))
        write_json(output / 'corpus.json', summary)
        log(f'corpus: {summary}')
        return summary

    return run_stage(output, 'retro-corpus', identity, ['patients.jsonl', 'cohorts.jsonl', 'sample.json', 'corpus.json'],
                     work, check=args.check, upstream=dict(patients=identity['patients'], cohorts=identity['cohorts']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('patients')
    a.add_argument('--summaries', required=True, help='Directory of summary batch parquets')
    a.add_argument('--denominator', required=True, help='processed_soc_treatments.csv of the same pull')
    b = sub.add_parser('cohorts')
    b.add_argument('--extraction', required=True, help='The cohorts.py stage directory')
    c = sub.add_parser('corpus')
    c.add_argument('--patients', required=True)
    c.add_argument('--cohorts', required=True)
    c.add_argument('--seed', type=int, default=42)
    c.add_argument('--patient-queries', type=int, default=500)
    c.add_argument('--cohort-queries', type=int, default=200)
    for q in (a, b, c):
        q.add_argument('--output', required=True)
        stage.add_check_arg(q)
    args = p.parse_args(argv)
    private()
    dict(patients=patients, cohorts=cohorts, corpus=corpus)[args.command](args)


if __name__ == '__main__':
    main()
