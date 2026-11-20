"""Component C: the decision model's seven question families, answered blind by the judges (PHI).

Each item is one teacher request of the fidelity evaluation (``<E>/labels``): a match pair (five questions in one
request), a boilerplate pair, or one criteria line. The judges get exactly the messages the Nemotron teacher got
(``labels.jsonl`` ``messages``, under the label contract the decision model was trained on), after a short
reviewer preamble, and answer under the contract's schema, never seeing the teacher's or the decision model's
answer. The agreed or adjudicated answer is the reference for scoring the teacher and the decision model.

Polarity is the decision model's: yes = potentially eligible (reasonableness, boilerplate), yes = the cohort
specifically requires the dimension and the patient documentedly satisfies it (the four other match questions),
yes = the patient does not clearly fail the line (per-criterion).
"""
from matchminer_ai.evaluation.dual_judge.judging import prompt_text

NAME = 'decision'
PROMPT = 'judge.decision.txt'
MAX_OUTPUT_TOKENS = 32000
TASKS = ('match', 'boilerplate', 'criteria')
MATCH_LABELS = ('reasonableness', 'cancer_type', 'burden', 'treatment_history', 'biomarker')
FAMILIES = (*MATCH_LABELS, 'boilerplate', 'criterion')

BOILERPLATE_FORMAT = (
    'The task below asks you to end your answer with "Yes!" or "No!". Instead, put your reasoning in "reason" and '
    'set "excluded" to true where you would answer "Yes!" and to false where you would answer "No!".')
BOILERPLATE_SCHEMA = {'type': 'object', 'required': ['reason', 'excluded'],
                      'properties': {'reason': {'type': 'string'}, 'excluded': {'type': 'boolean'}}}

KINDS = {family: dict(label=family.replace('_', ' ').capitalize(), values=['yes', 'no']) for family in FAMILIES}
KINDS['boilerplate']['label'] = 'Boilerplate (potentially eligible)'
KINDS['criterion']['label'] = 'Criterion line (does not clearly fail)'


def render(item, instructions=None):
    """(system, user): the reviewer preamble and the teacher's own system text, then the teacher's user text."""
    system = [prompt_text(PROMPT).rstrip('\n')]
    if item['task'] == 'boilerplate':
        system.append(BOILERPLATE_FORMAT)
    stored = item['messages']
    teacher_system = '\n\n'.join(m['content'] for m in stored if m['role'] == 'system' and m['content'])
    if teacher_system:
        system += ['TASK INSTRUCTIONS', teacher_system]
    user = '\n\n'.join(m['content'] for m in stored if m['role'] == 'user')
    return '\n\n'.join(system), user


def schema(item):
    return BOILERPLATE_SCHEMA if item['task'] == 'boilerplate' else item['schema']


def expect(item):
    return dict(task=item['task'], criterion_type=item.get('criterion_type'))


def validate(answer, expected):
    """The teacher contract's own checks: nonempty reasons, and a "no" on a criteria line names its failing status
    (not_met for an inclusion line, met for an exclusion line; decision_model.parse_criterion_label)."""
    task, errors = expected.get('task'), []
    if task == 'match':
        for label in MATCH_LABELS:
            if not str(answer.get(label, {}).get('reason', '')).strip():
                errors.append(f'{label}.reason must be a nonempty explanation')
    elif task == 'criteria':
        if not str(answer.get('reason', '')).strip():
            errors.append('reason must be a nonempty explanation')
        failing = 'not_met' if str(expected.get('criterion_type', '')).lower().startswith('inclusion') else 'met'
        if answer.get('answer') == 'no' and answer.get('criterion_status') != failing:
            errors.append(f'answer "no" (clearly fails) on an {expected.get("criterion_type")} line requires '
                          f'criterion_status "{failing}"; a line that does not bind the patient is not_applicable '
                          'and answers "yes"')
    elif task == 'boilerplate' and not str(answer.get('reason', '')).strip():
        errors.append('reason must be a nonempty explanation')
    return errors


def units(item, answer):
    if item['task'] == 'match':
        return {label: (label, answer[label]['answer'], dict(answer[label])) for label in MATCH_LABELS}
    if item['task'] == 'boilerplate':
        return {'boilerplate': ('boilerplate', 'no' if answer['excluded'] else 'yes', dict(answer))}
    return {'criterion': ('criterion', answer['answer'], dict(answer))}


def view(item):
    _, user = render(item)
    return dict(kind='decision', title=f'{item["task"]}: {item["cohort_id"]}', task=item['task'],
                source=dict(kind='task', text=user), criterion_number=item.get('number'),
                criterion_type=item.get('criterion_type'))


def unit_target(item, key):
    if item['task'] == 'criteria':
        return f'Criterion line [{item["number"]}] ({item["criterion_type"]}): does the patient not clearly fail it?'
    if item['task'] == 'boilerplate':
        return 'Boilerplate criteria: is the patient potentially eligible (not clearly excluded)?'
    return f'Match question: {key.replace("_", " ")}'


# What each question asks and what its answers mean, in the decision model's polarity (the teacher prompts'
# definitions: match_quality.labels.system.txt, llm_exclusion_criteria.user.txt, criterion_check.labels.user.txt).
SPECIFIC = ('yes = the cohort specifically requires this and the patient is documented to meet it; no = the cohort has '
            'no specific requirement (not applicable), the patient does not meet it (contradicted), or the record does '
            'not say (insufficient information)')
QUESTIONS = {
    'reasonableness': ('Is the patient potentially eligible for this cohort (its key criteria)?',
                       dict(yes='potentially eligible', no='clearly ineligible')),
    'cancer_type': ('Does the cohort specifically require a cancer type or histology that the patient is documented to '
                    'have?', dict(yes='specific requirement, documented as met', no='not a specific requirement, not met, '
                                  'or not documented (see the status)')),
    'burden': ('Does the cohort specifically require a stage or disease burden that the patient is documented to meet?',
               dict(yes='specific requirement, documented as met', no='not a specific requirement, not met, or not '
                    'documented (see the status)')),
    'treatment_history': ('Does the cohort specifically require prior treatment that the patient is documented to have '
                          'had?', dict(yes='specific requirement, documented as met', no='not a specific requirement, '
                                       'not met, or not documented (see the status)')),
    'biomarker': ('Does the cohort specifically require a biomarker status that the patient is documented to have?',
                  dict(yes='specific requirement, documented as met', no='not a specific requirement, not met, or not '
                       'documented (see the status)')),
    'boilerplate': ('Is the patient potentially eligible with respect to the boilerplate criteria (not clearly '
                    'excluded)?', dict(yes='not clearly excluded', no='clearly excluded')),
    'criterion': ('Does the patient NOT clearly fail this criteria line?',
                  dict(yes='does not clearly fail the line', no='clearly fails it (an inclusion line clearly not met, '
                       'or an exclusion line clearly met)')),
}
STATUS = {
    'evidence_status': dict(
        supported='the record documents what the answer says',
        contradicted='the record conflicts with the cohort\'s requirement',
        insufficient_information='the record does not say enough to decide',
        not_applicable='the cohort has no specific requirement of this kind'),
    'criterion_status': dict(
        met='the record documents that the line is true of the patient (for an exclusion line: the excluding '
            'condition applies)',
        not_met='the record documents that the line is not true of the patient',
        insufficient_information='the record does not say enough to decide',
        not_applicable='the line does not bind this patient (an alternative another line satisfies, or a requirement '
                       'for other patients)'),
}


def question(item, key):
    """The question, what yes and no mean, and the glossary of the status field the answers carry."""
    family = 'criterion' if item['task'] == 'criteria' else 'boilerplate' if item['task'] == 'boilerplate' else key
    text, answers = QUESTIONS[family]
    if item['task'] == 'criteria':
        text = f'Line [{item["number"]}] ({item["criterion_type"]}): {text}'
    field = dict(match='evidence_status', criteria='criterion_status').get(item['task'])
    note = ('The teacher answered the inverse question, "Is this patient clearly excluded?" (Yes!/No!), shown here '
            'inverted.' if item['task'] == 'boilerplate' else
            'The status explains the answer; it is not a second verdict.' if field else '')
    return dict(text=text, answers=answers, status_field=field, statuses=STATUS.get(field, {}), note=note)


def reveal(item, key):
    """The teacher's and the decision model's answers to one question, shown after the user decides (or on request)."""
    family = 'criterion' if item['task'] == 'criteria' else 'boilerplate' if item['task'] == 'boilerplate' else key
    detail = item.get('teacher_detail') or {}
    if item['task'] == 'match':
        detail = detail.get(key) or {}
    teacher = dict(answer=item['teacher'][family])
    if item['task'] == 'match':
        teacher.update(status=detail.get('evidence_status'), reason=detail.get('reason'))
    elif item['task'] == 'criteria':
        teacher.update(status=detail.get('criterion_status'), reason=detail.get('reason'))
    else:
        teacher.update(raw='Yes! (clearly excluded)' if detail.get('excluded') else 'No! (not clearly excluded)')
    p = item['decision_p_yes'].get(family)
    model = dict(p_yes=p, answer=None if p is None else ('yes' if p >= 0.5 else 'no'), threshold=0.5)
    return dict(teacher=teacher, model=model)


# --- building (where the data lives: the fidelity evaluation's labels and decision scores) -----------------------------------------

def _scan(path, wanted=None):
    """Rows of a labels.jsonl with their byte offsets: all small fields (no messages) when ``wanted`` is None, else
    the full rows at the given offsets."""
    import json
    if wanted is not None:
        rows = {}
        with open(path, 'rb') as handle:
            for offset in sorted(wanted):
                handle.seek(offset)
                rows[offset] = json.loads(handle.readline())
        return rows
    out, offset = [], 0
    with open(path, 'rb') as handle:
        for line in handle:
            row = json.loads(line)
            out.append(dict({k: row.get(k) for k in ('patient_id', 'cohort_id', 'trial_id', 'number', 'criterion_type',
                                                       'labels', 'excluded', 'criterion', 'contract', 'prompt_sha256')},
                            first_stratum=row.get('stratum'), offset=offset))
            offset += len(line)
    return out


def build(fidelity_dir, counts, seed):
    """Seeded items per task, stratified by teacher answer x decision model P(yes) >= 0.5 (the reasonableness answer
    for match pairs), so the sample holds teacher-model disagreements in both directions. Returns (items, None,
    design).

    Weights are inverse inclusion probabilities. Match pairs (unique pairs in any merged top-20 list) and boilerplate
    combinations (unique, labeled once) are sampled once: weight = cell size / sampled. Criteria lines were sampled
    twice: the fidelity evaluation labeled every line of a sample of top-20 pairs per workflow and ranker
    (lists_*/criteria_sample.json: pairs of pairs_available list slots), then this sample draws lines from them, so
    a line's weight is (pairs_available / pairs of its first-stage stratum) x (cell size / sampled). Criteria
    estimates then describe the lines of top-20 list slots, averaged over the six workflow x ranker lists."""
    import json
    import random
    from pathlib import Path
    import pandas as pd
    from matchminer_ai.matching.decision_model import CRITERION_LABEL_JSON_SCHEMA
    from matchminer_ai.matching.match_labels import MATCH_LABELS_JSON_SCHEMA
    from matchminer_ai.labeling.tasks import contract_hash

    root = Path(fidelity_dir)
    contracts = {task: contract_hash(task) for task in TASKS}
    decide = dict(match=pd.read_parquet(root / 'decide' / 'match' / 'scores.parquet'),
                  boilerplate=pd.read_parquet(root / 'decide' / 'boilerplate' / 'scores.parquet'),
                  criteria=pd.read_parquet(root / 'decide' / 'criteria' / 'scores.parquet'))
    p_match = {(str(r.patient_id), r.cohort_id): {label: float(getattr(r, f'p_{label}')) for label in MATCH_LABELS}
               for r in decide['match'].itertuples()}
    p_boilerplate = {(str(r.patient_id), r.cohort_id): float(r.p_boilerplate) for r in decide['boilerplate'].itertuples()}
    p_criterion = {(str(r.patient_id), r.cohort_id, int(r.number)): float(r.p_yes) for r in decide['criteria'].itertuples()}
    first_stage = {}
    for part in ('ab', 'c'):
        sample = json.loads((root / f'lists_{part}' / 'criteria_sample.json').read_text())
        for name, entry in sample['strata'].items():
            first_stage[(part, name)] = entry['pairs_available'] / entry['pairs']
    rng = random.Random(f'{seed}:decision')
    items, design = [], dict(strata={}, sampled={}, contracts=contracts)
    for task in TASKS:
        pool = []
        for part in ('ab', 'c'):
            path = root / 'labels' / f'{task}_{part}' / 'labels.jsonl'
            for row in _scan(path):
                if row['contract'] != contracts[task]:
                    raise ValueError(f'{path}: a row was labeled under another contract')
                key = (str(row['patient_id']), row['cohort_id'])
                if task == 'match':
                    teacher = {label: row['labels'][label]['answer'] for label in MATCH_LABELS}
                    p = p_match.get(key)
                    first = teacher['reasonableness'], p and p['reasonableness']
                elif task == 'boilerplate':
                    teacher = {'boilerplate': 'no' if row['excluded'] else 'yes'}
                    p = p_boilerplate.get(key)
                    p = None if p is None else {'boilerplate': p}
                    first = teacher['boilerplate'], p and p['boilerplate']
                else:
                    teacher = {'criterion': row['criterion']['answer']}
                    p = p_criterion.get((*key, int(row['number'])))
                    p = None if p is None else {'criterion': p}
                    first = teacher['criterion'], p and p['criterion']
                if p is None:
                    continue
                stratum = f'teacher_{first[0]}/model_{"yes" if first[1] >= 0.5 else "no"}'
                expand = first_stage[(part, row['first_stratum'])] if task == 'criteria' else 1.0
                pool.append(dict(row, part=part, path=str(path), teacher=teacher, decision_p_yes=p, stratum=stratum,
                                 first_stage_weight=expand))
        strata = {}
        for row in pool:
            strata.setdefault(row['stratum'], []).append(row)
        names = sorted(strata)
        take = {name: 0 for name in names}
        for i in range(counts[task]):  # round-robin over the strata that still have rows
            open_ = [name for name in names if take[name] < len(strata[name])]
            if not open_:
                break
            take[open_[i % len(open_)]] += 1
        design['strata'][task] = {name: len(rows) for name, rows in strata.items()}
        design['sampled'][task] = take
        chosen = [(name, row) for name in names for row in rng.sample(strata[name], take[name])]
        by_path = {}
        for _, row in chosen:
            by_path.setdefault(row['path'], set()).add(row['offset'])
        full = {path: _scan(path, offsets) for path, offsets in by_path.items()}
        for name, row in sorted(chosen, key=lambda pair: (pair[1]['path'], pair[1]['offset'])):
            stored = full[row['path']][row['offset']]
            items.append(dict(
                item_id=f'{task}-{sum(i["task"] == task for i in items) + 1:03d}', task=task,
                patient_id=str(row['patient_id']), cohort_id=row['cohort_id'], trial_id=row['trial_id'],
                number=row.get('number'), criterion_type=row.get('criterion_type'), stratum=name,
                first_stratum=row.get('first_stratum'), first_stage_weight=row['first_stage_weight'],
                weight=row['first_stage_weight'] * len(strata[name]) / take[name], messages=stored['messages'],
                schema=dict(match=MATCH_LABELS_JSON_SCHEMA, criteria=CRITERION_LABEL_JSON_SCHEMA).get(task),
                teacher=row['teacher'],
                teacher_detail=stored.get('labels') or stored.get('criterion') or dict(excluded=stored.get('excluded')),
                decision_p_yes=row['decision_p_yes'], contract=row['contract'], source=f'labels/{task}_{row["part"]}'))
    return items, None, design
