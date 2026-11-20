"""Which pairs and criteria lines each round asks the teacher (its contracts are tested in the inference package)."""
import json

import pytest

from matchminer_ai.labeling import contract_hash, parse_judgment, questions, validate_row


def corpus():
    patients = [dict(id='p1', text='Key: metastatic lung cancer.', boilerplate='Dialysis.', split='train')]
    cohorts = [dict(id='c1', trial_id='T1', text='Key: metastatic NSCLC.', boilerplate='No dialysis.', split='train'),
               dict(id='c2', trial_id='T2', text='Key: sarcoma.', boilerplate='', split='train')]
    pairs = [dict(patient_id='p1', cohort_id=c, split='train') for c in ('c1', 'c2')]
    return patients, cohorts, pairs


def test_boilerplate_labels_cover_each_unique_boilerplate_combination_of_every_round_once(tmp_path):
    from matchminer_ai.training.trialspace.data import read_jsonl, write_jsonl
    from matchminer_ai.training.trialspace.candidates import unique_boilerplate_pairs
    patients, cohorts, _ = corpus()
    cohorts.append(dict(id='c3', trial_id='T1', text='Key: other arm.', boilerplate='No dialysis.', split='train'))
    write_jsonl(tmp_path / 'corpus' / 'patients.jsonl', patients)
    write_jsonl(tmp_path / 'corpus' / 'cohorts.jsonl', cohorts)
    pair = lambda c: dict(patient_id='p1', cohort_id=c, split='train', sources=[])
    write_jsonl(tmp_path / 'round1.jsonl', [pair('c1'), pair('c2')])
    write_jsonl(tmp_path / 'round3.jsonl', [pair('c1'), pair('c3')])
    path = unique_boilerplate_pairs(tmp_path / 'corpus', [tmp_path / 'round3.jsonl', tmp_path / 'round1.jsonl'], tmp_path / 'out')
    # c1 and c3 share their boilerplate criteria; c2 has none: one combination to label.
    assert [p['cohort_id'] for p in read_jsonl(path)] == ['c1']
    # A later round skips the combinations an earlier round's directory labeled.
    again = unique_boilerplate_pairs(tmp_path / 'corpus', [tmp_path / 'round3.jsonl'], tmp_path / 'out2', skip=[tmp_path / 'out'])
    assert read_jsonl(again) == []


ITEMS = [dict(number=1, heading='Inclusion criteria:', text='Age >= 18.', criterion_type='inclusion', boilerplate=False),
         dict(number=2, heading='Exclusion criteria:', text='Dialysis.', criterion_type='exclusion', boilerplate=True),
         dict(number=3, heading='Exclusion criteria:', text='Unclassified.', criterion_type=None, boilerplate=False)]


def test_criteria_labels_are_one_request_per_tagged_line_of_sampled_pairs(tmp_path):
    from matchminer_ai.training.trialspace.data import read_jsonl, write_jsonl
    from matchminer_ai.training.trialspace.candidates import criteria_line_candidates
    patients, cohorts, _ = corpus()
    for cohort in cohorts:
        cohort['items'] = ITEMS
    write_jsonl(tmp_path / 'corpus' / 'patients.jsonl', patients)
    write_jsonl(tmp_path / 'corpus' / 'cohorts.jsonl', cohorts)
    write_jsonl(tmp_path / 'round1.jsonl', [dict(patient_id='p1', cohort_id=c, split='train', sources=[]) for c in ('c1', 'c2')])
    path = criteria_line_candidates(tmp_path / 'corpus', [tmp_path / 'round1.jsonl'], tmp_path / 'out', fraction=1, seed=42)
    rows = read_jsonl(path)
    # c1 and c2 have the same lines: each tagged line of the patient once; the untyped line is skipped.
    assert [(r['cohort_id'], r['number'], r['criterion_type']) for r in rows] == [('c1', 1, 'inclusion'), ('c1', 2, 'exclusion')]
    assert read_jsonl(criteria_line_candidates(tmp_path / 'corpus', [tmp_path / 'round1.jsonl'], tmp_path / 'out2',
                                               fraction=1, seed=42, skip=[tmp_path / 'out'])) == []
    item = list(questions(rows, patients, cohorts, task='criteria'))[1]
    user = item['messages'][0]['content']
    assert 'Key: metastatic lung cancer.\n\nOther relevant history:\nDialysis.' in user  # the complete summary
    assert '[2] Dialysis.' in user and user.split('OUTPUT FORMAT')[0].rstrip().endswith('[2] Exclusion criterion: Dialysis.')
    yes = json.dumps(dict(answer='yes', criterion_status='insufficient_information', reason='Not documented.'))
    fails = json.dumps(dict(answer='no', criterion_status='met', reason='On dialysis.'))
    assert parse_judgment('criteria', fails, 'exclusion')['criterion']['answer'] == 'no'
    assert parse_judgment('criteria', yes, 'inclusion')['criterion']['criterion_status'] == 'insufficient_information'
    with pytest.raises(ValueError):  # "no" (clearly fails) an exclusion line needs the exclusion to be met
        parse_judgment('criteria', json.dumps(dict(answer='no', criterion_status='not_met', reason='x')), 'exclusion')
    validate_row('criteria', dict(criterion=json.loads(fails), criterion_type='exclusion'))
    assert contract_hash('criteria') != contract_hash('match')
