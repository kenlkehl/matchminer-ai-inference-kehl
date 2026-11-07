"""The teacher labeler's contracts: five match labels in one request per pair, boilerplate and per-criterion labels."""
import json

import pytest

from matchminer_ai.labeling import (aggregate_pair, contract_hash, judge_messages, parse_judgment, questions,
                                    score_contract, validate_row)
from matchminer_ai.labeling.scoring import MAX_SCORE, validate_score

LABELS = ('reasonableness', 'cancer_type', 'burden', 'treatment_history', 'biomarker')
ITEMS = [dict(number=1, heading='Inclusion criteria:', text='Age >= 18.', criterion_type='inclusion', boilerplate=False),
         dict(number=2, heading='Exclusion criteria:', text='Dialysis.', criterion_type='exclusion', boilerplate=True)]


def answer(**yes):
    return json.dumps({label: {'answer': 'yes' if yes.get(label) else 'no', 'evidence_status': 'supported',
                               'reason': 'Fabricated.'} for label in LABELS})


def corpus():
    patients = [dict(id='p1', text='Key: metastatic lung cancer.', boilerplate='Dialysis.', split='train')]
    cohorts = [dict(id='c1', trial_id='T1', text='Key: metastatic NSCLC.', boilerplate='No dialysis.', split='train',
                    items=ITEMS),
               dict(id='c2', trial_id='T2', text='Key: sarcoma.', boilerplate='', split='train', items=ITEMS)]
    pairs = [dict(patient_id='p1', cohort_id=c, split='train') for c in ('c1', 'c2')]
    return patients, cohorts, pairs


def test_label_identities_change_only_deliberately():
    # Label directories resume only under the same contracts; a prompt or schema change must update these.
    assert contract_hash('match') == 'edbff87086e7f2dfb79d5fb0ae0de9e5b2ecd48d3a02b19c4c23fe1e79b3a405'
    assert contract_hash('boilerplate') == 'f0fc2d3c948b4ebec618d70aacb8108d7e47ada0524010a25fed350c74f8cf23'
    assert contract_hash('criteria') == '2728713a2f83dcad8828f4c73a6e433383d9869f7b6afbae6a6711b2173198f6'
    assert score_contract() == '6ac402b126c9a08abe10a7e1758695318d2cc38cc8a42b5bfe8627050183c9e5'


def test_match_labels_are_one_request_per_pair_on_key_text_only():
    patients, cohorts, pairs = corpus()
    items = list(questions(pairs, patients, cohorts, task='match'))
    assert [(i['cohort_id'], i['task']) for i in items] == [('c1', 'match'), ('c2', 'match')]
    user = items[0]['messages'][1]['content']
    assert 'metastatic lung cancer' in user and 'metastatic NSCLC' in user
    assert 'Dialysis' not in user and 'dialysis' not in user
    assert all(f'"{label}"' in items[0]['messages'][0]['content'] for label in LABELS)


def test_boilerplate_labels_read_boilerplate_text_and_skip_cohorts_without_it():
    patients, cohorts, pairs = corpus()
    items = list(questions(pairs, patients, cohorts, task='boilerplate'))
    assert [i['cohort_id'] for i in items] == ['c1']
    user = items[0]['messages'][1]['content']
    assert 'Dialysis.' in user and 'No dialysis.' in user and 'metastatic' not in user
    assert parse_judgment('boilerplate', 'Criterion 1 is met.\nYes!') == dict(excluded=True)
    assert parse_judgment('boilerplate', 'No!') == dict(excluded=False)
    with pytest.raises(ValueError):
        parse_judgment('boilerplate', 'Probably excluded.')
    validate_row('boilerplate', dict(excluded=False))
    with pytest.raises(ValueError):
        validate_row('boilerplate', dict(excluded='no'))


def test_five_answers_give_one_validated_zero_to_six_score():
    patients, cohorts, pairs = corpus()
    item = next(questions(pairs, patients, cohorts, task='match'))
    row = dict(item, contract=contract_hash('match'), **parse_judgment('match', answer(
        reasonableness=True, cancer_type=True, biomarker=True)))
    validate_row('match', row)
    score = aggregate_pair(item, row)
    assert score['score'] == 4 and score['normalized_score'] == 4 / MAX_SCORE and MAX_SCORE == 6
    assert score['answers'] == dict(reasonableness='yes', cancer_type='yes', burden='no', treatment_history='no',
                                    biomarker='yes')
    assert score['question_keys'] == {'match': item['key']} and score['contract'] == score_contract()
    pmap, cmap = {p['id']: p for p in patients}, {c['id']: c for c in cohorts}
    validate_score(score, pmap, cmap, contract_hash('match'), score_contract())
    # Specificity without reasonableness scores zero, but every answer is kept for its head.
    zero = aggregate_pair(item, dict(row, **parse_judgment('match', answer(cancer_type=True, burden=True))))
    assert zero['score'] == 0 and zero['answers']['cancer_type'] == 'yes'
    with pytest.raises(ValueError):
        validate_score(dict(score, score=5), pmap, cmap, contract_hash('match'), score_contract())
    with pytest.raises(ValueError):
        parse_judgment('match', json.dumps({label: {'answer': 'yes'} for label in LABELS}))
    assert judge_messages('match', 'P', 'C')[1]['content'].endswith('COHORT KEY ELIGIBILITY CRITERIA\nC')


def test_criteria_labels_ask_one_tagged_line_with_the_complete_summary():
    patients, cohorts, _ = corpus()
    rows = [dict(patient_id='p1', cohort_id='c1', split='train', number=n, criterion_type=t, criteria_digest='d')
            for n, t in ((1, 'inclusion'), (2, 'exclusion'))]
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
    with pytest.raises(ValueError):  # a line whose type differs from the corpus is not asked
        list(questions([dict(rows[0], criterion_type='exclusion')], patients, cohorts, task='criteria'))
