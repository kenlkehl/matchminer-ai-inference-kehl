"""Match scores of labeled patient–cohort pairs: the teacher's five answers and their 0–6 score.

One teacher response per pair answers all five match labels (``contracts``, task ``match``).
The score is the inference package's ``match_score``: 0 unless the patient is a reasonable
candidate, else 1 plus 1 each for cancer type, burden and treatment history and 2 for biomarker.
Every label is answered for every pair, so each of the decision model's match questions trains on every pair.
"""
from collections import Counter

from matchminer_ai.embedding.trialspace_contract import INSTRUCTION
from matchminer_ai.matching.match_labels import LABELS, MAX_SCORE, POINTS, match_score

from .tasks import contract_hash
from .files import digest

VERSION = 'trialspace-match-score-v1'
TASK = 'match'


def score_contract(question_contract=None):
    return digest(dict(version=VERSION, questions=question_contract or contract_hash(TASK), labels=list(LABELS),
                       score='match_score', points=POINTS, normalization=MAX_SCORE, query_instruction=INSTRUCTION))


def aggregate_pair(pair, row, *, question_contract=None):
    """The pair's score row from its one validated five-answer response."""
    expected = question_contract or contract_hash(TASK)
    if any(row[k] != pair[k] for k in ('patient_id', 'cohort_id', 'split')):
        raise ValueError('The response must belong to the same pair and split')
    if row.get('task') != TASK or row.get('contract') != expected:
        raise ValueError('Question contract differs from the current labels')
    labels = row['labels']
    if set(labels) != set(LABELS):
        raise ValueError('Every match label must be answered')
    answers = {label: labels[label]['answer'] for label in LABELS}
    score = match_score(answers)
    return dict(patient_id=pair['patient_id'], cohort_id=pair['cohort_id'], split=pair['split'],
                score=score, normalized_score=score / MAX_SCORE, answers=answers,
                evidence_status={label: labels[label]['evidence_status'] for label in LABELS},
                question_keys={TASK: row['key']}, question_contract=expected, contract=score_contract(expected))


def validate_score(row, patients, cohorts, question_contract, aggregate_contract):
    """A compact journal row is internally consistent and belongs to this corpus."""
    p, c = patients.get(row.get('patient_id')), cohorts.get(row.get('cohort_id'))
    if p is None or c is None or row.get('split') != p['split'] or p['split'] != c['split']:
        raise ValueError('Compact score does not belong to this corpus')
    answers = row.get('answers', {})
    if set(answers) != set(LABELS) or any(value not in ('yes', 'no') for value in answers.values()):
        raise ValueError('Invalid compact-score answers')
    if (row.get('question_contract') != question_contract or row.get('contract') != aggregate_contract
            or row.get('score') != match_score(answers) or row.get('normalized_score') != row['score'] / MAX_SCORE
            or set(row.get('question_keys', {})) != {TASK}):
        raise ValueError('Compact score disagrees with its answers or contract')
    return row


def score_balance(scores):
    return dict(pairs=len(scores), distribution=dict(sorted(Counter(row['score'] for row in scores).items())),
                by_split={split: dict(sorted(Counter(row['score'] for row in scores if row['split'] == split).items()))
                          for split in sorted({row['split'] for row in scores})},
                yes_answers={label: sum(row['answers'][label] == 'yes' for row in scores) for label in LABELS})


__all__ = ['LABELS', 'MAX_SCORE', 'POINTS', 'TASK', 'VERSION', 'aggregate_pair', 'score_balance', 'score_contract',
           'validate_score']
