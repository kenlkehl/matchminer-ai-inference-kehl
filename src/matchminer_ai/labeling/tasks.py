"""The teacher's label contracts (moved from the training repository's trialspace/contracts.py, 2026-11-06).

- ``match``: the five match labels of a patient–cohort pair (reasonableness and the cancer type,
  burden, treatment history and biomarker specificity answers), asked in ONE prompt per pair
  (``matchminer_ai.matching.match_labels``; user instruction 2026-11-04). The patient text is the
  summary's key section and the cohort text its key criteria. TrialSpace and the decision model's
  five match questions train on these labels.
- ``boilerplate``: whether the patient clearly fails any of the cohort's boilerplate criteria, from
  the summary's boilerplate section (``llm_exclusion_criteria.user.txt``, the prompt of
  ``exclusion_criteria_check_with_llm``). The decision model's boilerplate question trains on these
  labels, inverted: its yes means potentially eligible (not excluded).
- ``criteria``: one request per criteria line of a sampled pair (``criterion_check.labels.user.txt``
  around the decision model's own per-criterion state and question,
  ``matchminer_ai.matching.decision_model``): the complete patient summary, the cohort's complete
  numbered criteria and the line restated with its inclusion/exclusion type. The answer is the
  decision model's target: **yes = the patient does not clearly fail the line; no = an inclusion
  line clearly not met or an exclusion line clearly met**, with the literal ``criterion_status``.
"""
import hashlib
import json

from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.matching.decision_model import (
    CRITERION_LABEL_JSON_SCHEMA, PROMPTS as DECISION_PROMPTS, criterion_teacher_messages, parse_criterion_label,
    questions_digest,
)
from matchminer_ai.matching.match_labels import (
    LABELS, MATCH_LABELS_JSON_SCHEMA, MATCH_LABELS_PROMPT, build_match_label_messages, match_labels_contract,
    parse_match_labels,
)

VERSION = 'trialspace-teacher-contracts-v1'
TASKS = ('match', 'boilerplate', 'criteria')
BOILERPLATE_PROMPT = 'llm_exclusion_criteria.user.txt'
CRITERIA_PROMPT = DECISION_PROMPTS['criterion_teacher']


def prompt_sources():
    return {name: load_prompt_text(name) for name in (MATCH_LABELS_PROMPT, BOILERPLATE_PROMPT, CRITERIA_PROMPT)}


def contract_hash(task='match', sources=None):
    sources = prompt_sources() if sources is None else sources
    if task == 'match':  # decoded under its JSON schema since 2026-11-05 (user instruction)
        detail = dict(match_labels_contract(), structured=MATCH_LABELS_JSON_SCHEMA)
    elif task == 'boilerplate':
        detail = dict(prompt=BOILERPLATE_PROMPT, answer='Yes!/No!')
    else:  # the per-criterion request is built from the decision model's state and question texts
        detail = dict(prompt=CRITERIA_PROMPT, answer='answer/criterion_status/reason', builder=questions_digest(),
                      structured=CRITERION_LABEL_JSON_SCHEMA)
    # match and boilerplate hash the two prompts they always hashed; criteria hashes its own
    names = (CRITERIA_PROMPT,) if task == 'criteria' else (MATCH_LABELS_PROMPT, BOILERPLATE_PROMPT)
    sources = {name: sources[name] for name in names}
    return hashlib.sha256(json.dumps(dict(version=VERSION, task=task, detail=detail, prompts=sources),
                                     sort_keys=True).encode()).hexdigest()


def judge_messages(task, patient, cohort, *, sources=None, number=None):
    """Messages for one pair: key texts for ``match``, boilerplate texts for ``boilerplate``; for
    ``criteria``, the complete patient summary, the cohort's tagged items and the line ``number``."""
    sources = prompt_sources() if sources is None else sources
    if task == 'criteria':
        return criterion_teacher_messages(patient, cohort, number)
    if task == 'match':
        return build_match_label_messages(patient, cohort, sources[MATCH_LABELS_PROMPT])
    if task == 'boilerplate':
        return [dict(role='system', content=''),
                dict(role='user', content=sources[BOILERPLATE_PROMPT].format(
                    patient_boilerplate=patient, trial_boilerplate=cohort))]
    raise ValueError('Unknown task')


def parse_boilerplate(answer):
    """``{"excluded": bool}`` from the final Yes!/No!; anything else is invalid, never a guess."""
    tail = answer.strip()[-12:]
    if tail.endswith('Yes!'):
        return dict(excluded=True)
    if tail.endswith('No!'):
        return dict(excluded=False)
    raise ValueError('The answer must end with "Yes!" or "No!"')


def answer_schema(task):
    """The JSON schema a task's answer is decoded under (vLLM structured outputs, after the reasoning),
    or None for free text. The JSON tasks (``match``, ``criteria``) are constrained: unconstrained,
    Nemotron 3.5 Super dropped the opening quote of a ``reason`` string in about one answer in six
    (2026-11-05 test on 172.24.216.113:8002). ``boilerplate`` ends in a free-text Yes!/No!."""
    return {'match': MATCH_LABELS_JSON_SCHEMA, 'criteria': CRITERION_LABEL_JSON_SCHEMA}.get(task)


def parse_judgment(task, answer, criterion_type=None):
    if task == 'match':
        return dict(labels=parse_match_labels(answer))
    if task == 'boilerplate':
        return parse_boilerplate(answer)
    if task == 'criteria':  # yes = does not clearly fail the line; no = clearly fails it
        return dict(criterion=parse_criterion_label(answer, criterion_type))
    raise ValueError('Unknown task')


__all__ = ['BOILERPLATE_PROMPT', 'CRITERIA_PROMPT', 'answer_schema', 'LABELS', 'TASKS', 'VERSION', 'contract_hash', 'judge_messages',
           'parse_boilerplate', 'parse_judgment', 'prompt_sources']
