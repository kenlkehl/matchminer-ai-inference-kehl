"""Component A: the ctgov cohort extraction of one trial (screen, cohort names, criteria, merge, line tags).

Non-PHI: a ClinicalTrials.gov listing and what training's frozen programs extracted from it
(``retro_fidelity/cohorts.py``, ``<E>/extraction``). The judges see the listing exactly as the extractor saw it
(title, brief summary and eligibility criteria as one document), the extractor's five frozen prompts (checked
against the hashes the extraction recorded), and the final cohorts with their numbered, tagged lines.
"""
from matchminer_ai.evaluation.dual_judge.judging import prompt_text, quote_found

NAME = 'extraction'
PROMPT = 'judge.extraction.txt'
MAX_OUTPUT_TOKENS = 64000
STEPS = ('screen', 'list', 'criteria', 'merge', 'deboilerplate')
SEVERITY = ['none', 'minor', 'major']

_text = {'type': 'string'}
SCHEMA = {
    'type': 'object',
    'required': ['screen', 'cohort_set', 'cohorts', 'overall', 'summary'],
    'properties': {
        'screen': {'type': 'object', 'required': ['verdict', 'reason'],
                   'properties': {'verdict': {'enum': ['correct', 'incorrect']}, 'reason': _text}},
        'cohort_set': {
            'type': 'object',
            'required': ['missing_cohorts', 'spurious_cohorts', 'wrong_merges', 'missed_merges', 'verdict', 'reason'],
            'properties': {
                'missing_cohorts': {'type': 'array', 'items': _text},
                'spurious_cohorts': {'type': 'array', 'items': {'type': 'integer'}},
                'wrong_merges': {'type': 'array', 'items': _text},
                'missed_merges': {'type': 'array', 'items': _text},
                'verdict': {'enum': ['correct', 'defensible', 'incorrect']},
                'reason': _text}},
        'cohorts': {'type': 'array', 'items': {
            'type': 'object', 'required': ['cohort', 'lines', 'missing_criteria'],
            'properties': {
                'cohort': {'type': 'integer'},
                'lines': {'type': 'array', 'items': {
                    'type': 'object', 'required': ['line', 'fidelity', 'boilerplate_tag', 'type_tag', 'severity', 'note'],
                    'properties': {
                        'line': {'type': 'integer'},
                        'fidelity': {'enum': ['verbatim', 'minor_edit', 'meaning_change', 'not_in_source',
                                              'wrong_cohort']},
                        'boilerplate_tag': {'enum': ['correct', 'should_be_key', 'should_be_boilerplate']},
                        'type_tag': {'enum': ['correct', 'wrong']},
                        'severity': {'enum': SEVERITY},
                        'note': _text}}},
                'missing_criteria': {'type': 'array', 'items': {
                    'type': 'object', 'required': ['listing_text', 'severity'],
                    'properties': {'listing_text': _text, 'severity': {'enum': ['minor', 'major']}}}}}}},
        'overall': {'enum': ['accurate', 'minor_issues', 'major_issues']},
        'summary': _text,
    },
}

NO_MAJOR = {'none': 'no_major', 'minor': 'no_major', 'major': 'major'}
KINDS = {
    'screen': dict(label='Cancer-treatment screen', values=['correct', 'incorrect']),
    'cohort_set': dict(label='Cohort set (names and merges)', values=['correct', 'defensible', 'incorrect'],
                       collapse={'correct': 'acceptable', 'defensible': 'acceptable', 'incorrect': 'incorrect'}),
    'fidelity': dict(label='Line copied faithfully', values=['ok', 'minor', 'major']),
    'boilerplate_tag': dict(label='Key/boilerplate tag', values=['correct', 'should_be_key', 'should_be_boilerplate']),
    'type_tag': dict(label='Inclusion/exclusion tag', values=['correct', 'wrong']),
    'missing': dict(label='Missing criteria', values=['none', 'minor', 'major'], collapse=NO_MAJOR),
    'overall': dict(label='Overall', values=['accurate', 'minor_issues', 'major_issues'],
                    collapse={'accurate': 'no_major', 'minor_issues': 'no_major', 'major_issues': 'major'}),
}


# --- rendering -----------------------------------------------------------------------------------------------------

def system_prompt(instructions):
    """The judge rubric, then the extractor's five frozen prompts (the same for every trial)."""
    parts = [prompt_text(PROMPT).rstrip('\n'), '', "THE EXTRACTOR'S INSTRUCTIONS"]
    titles = dict(screen='1. Screen', list='2. Cohort names', criteria='3. Criteria (one request per cohort)',
                  merge='4. Merge', deboilerplate='5. Line tags (one request per line)')
    for step in STEPS:
        parts += ['', f'=== {titles[step]} ===', instructions[step]]
    return '\n'.join(parts)


def cohort_block(cohort):
    merged = cohort.get('merged_from_names') or []
    head = f'Cohort {cohort["number"]}: "{cohort["name"]}"'
    if len(merged) > 1:
        head += ' (merged from: ' + '; '.join(f'"{name}"' for name in merged) + ')'
    lines, heading = [head], None
    for item in cohort['items']:
        if item.get('heading') and item['heading'] != heading:
            heading = item['heading']
            lines.append(heading)
        kind = 'boilerplate' if item['boilerplate'] else 'key'
        lines.append(f'[{item["number"]}] ({item["criterion_type"]}; {kind}) {item["text"]}')
    return '\n'.join(lines)


def render(item, instructions):
    """(system, user) for one trial."""
    if item['screen'] == 'yes':
        names = '\n'.join(f'- {name}' for name in item['cohort_names'])
        body = (f'Step 1, screen answer (cancer_treatment_trial): yes\n\nStep 2, cohort names:\n{names}\n\n'
                'Steps 3-5, the final cohorts after merging, each with its numbered criteria lines. Every line shows '
                '[number] (inclusion or exclusion; key or boilerplate) and the text, under the heading it was '
                'written under.\n\n' + '\n\n'.join(cohort_block(cohort) for cohort in item['cohorts']))
    else:
        body = 'Step 1, screen answer (cancer_treatment_trial): no\n\nNo cohorts were extracted.'
    user = f'<listing>\n{item["listing"]}\n</listing>\n\n<extraction>\n{body}\n</extraction>'
    return system_prompt(instructions), user


def expect(item):
    return dict(cohorts={str(c['number']): [i['number'] for i in c['items']] for c in item['cohorts']})


def validate(answer, expected):
    """Coverage errors: every final cohort and every one of its lines exactly once."""
    errors, wanted = [], expected.get('cohorts', {})
    seen = {}
    for cohort in answer.get('cohorts', []):
        key = str(cohort['cohort'])
        if key in seen:
            errors.append(f'cohort {key} appears twice')
        seen[key] = [line['line'] for line in cohort['lines']]
    if set(seen) != set(wanted):
        errors.append(f'cohorts must be exactly {sorted(wanted, key=int)}; got {sorted(seen, key=int)}')
    for key, numbers in wanted.items():
        got = seen.get(key)
        if got is None:
            continue
        if sorted(got) != sorted(numbers):
            missing, extra = sorted(set(numbers) - set(got)), sorted(set(got) - set(numbers))
            dupes = sorted({n for n in got if got.count(n) > 1})
            errors.append(f'cohort {key}: lines missing {missing}, unknown {extra}, repeated {dupes}')
    return errors


# --- units ---------------------------------------------------------------------------------------------------------

def _worst(severities):
    order = {s: i for i, s in enumerate(SEVERITY)}
    return max(severities, key=order.get, default='none')


def units(item, answer):
    """{unit key: (kind, value, detail)} of one judge's answer."""
    out = {
        'screen': ('screen', answer['screen']['verdict'], dict(reason=answer['screen']['reason'])),
        'cohort_set': ('cohort_set', answer['cohort_set']['verdict'], dict(answer['cohort_set'])),
        'overall': ('overall', answer['overall'], dict(summary=answer['summary'])),
    }
    listing = item['listing']
    for cohort in answer['cohorts']:
        c = cohort['cohort']
        missing = [dict(entry, grounded=quote_found(entry['listing_text'], listing)) for entry in cohort['missing_criteria']]
        out[f'c{c}/missing'] = ('missing', _worst([m['severity'] for m in missing]), dict(missing=missing))
        for line in cohort['lines']:
            key, detail = f'c{c}/l{line["line"]}', dict(line)
            ok = line['fidelity'] in ('verbatim', 'minor_edit')
            out[f'{key}/fidelity'] = ('fidelity', 'ok' if ok else ('major' if line['severity'] == 'major' else 'minor'),
                                      detail)
            out[f'{key}/boilerplate_tag'] = ('boilerplate_tag', line['boilerplate_tag'], detail)
            out[f'{key}/type_tag'] = ('type_tag', line['type_tag'], detail)
    return out


def view(item):
    """What the adjudication app shows: the listing and the extraction."""
    return dict(kind='extraction', title=item['trial_id'], source=dict(kind='listing', text=item['listing']),
                screen=item['screen'], cohort_names=item['cohort_names'], cohorts=item['cohorts'])


def unit_target(item, key):
    """The part of the item a unit is about, for the app (a line's text, a cohort's name)."""
    parts = key.split('/')
    if parts[0].startswith('c') and parts[0][1:].lstrip('-').isdigit():
        cohort = next((c for c in item['cohorts'] if str(c['number']) == parts[0][1:]), None)
        if cohort and len(parts) > 2:
            line = next((i for i in cohort['items'] if f'l{i["number"]}' == parts[1]), None)
            if line:
                tag = 'boilerplate' if line['boilerplate'] else 'key'
                return f'Cohort {cohort["number"]}, line [{line["number"]}] ({line["criterion_type"]}; {tag}): {line["text"]}'
        if cohort:
            return f'Cohort {cohort["number"]}: {cohort["name"]}'
    return dict(screen=f'Screen answer: {item["screen"]}', cohort_set='The set of final cohorts',
                overall='The extraction as a whole').get(parts[0], key)


# --- building (where the data lives: the extraction outputs and the frozen prompts) -------------------------------------------------

def build(fidelity_dir, frozen, n, seed):
    """Seeded, stratified trials of the extraction stage: the rejected trial(s), multi-cohort trials with a merge,
    multi-cohort trials without one, and single-cohort trials. Returns (items, instructions, design)."""
    import csv
    import gzip
    import json
    import random
    import sys
    from pathlib import Path
    from matchminer_ai.trials.cohort_workflow import ctgov_trial_text  # identical to the frozen registry_trial_text

    csv.field_size_limit(sys.maxsize)
    root = Path(fidelity_dir) / 'extraction'
    instructions, hashes = {}, {}
    for step in STEPS:
        source = frozen['step2' if step == 'deboilerplate' else 'step1'] / f'trial.cohorts.{step}.user.txt'
        instructions[step] = source.read_text().removesuffix('\n')  # load_prompt_text: only the final newline
        hashes[step] = __import__('hashlib').sha256(instructions[step].encode()).hexdigest()
    recorded = json.loads((root / 'deboilerplate_v4.json').read_text())['prompt_sha256']
    if recorded != hashes['deboilerplate']:
        raise ValueError('The frozen deboilerplate prompt is not the one the extraction used')

    listings = {row['nct_id']: row for row in map(json.loads, (root / 'trials.jsonl').open())}
    screened = {row['nct_id']: row['cancer_treatment_trial'] for row in csv.DictReader((root / 'trials_with_cohorts.csv').open())}
    cohorts = {}
    for row in csv.DictReader((root / 'trial_cohort_lineitems_deboilerplated.csv').open()):
        cohorts.setdefault(row['trial_id'], []).append(row)

    def checkpoint(trial):
        data = json.loads(gzip.open(root / 'extraction' / 'checkpoints' / f'{trial}.json.gz').read())
        steps = data['metadata']['prompt_sha256_by_step']
        for step in ('screen', 'list', 'criteria'):
            if steps[step] != hashes[step]:
                raise ValueError(f'{trial}: the frozen {step} prompt is not the one the extraction used')
        merge_sha = (data.get('merge') or {}).get('prompt_sha256')
        if merge_sha and merge_sha != hashes['merge']:
            raise ValueError(f'{trial}: the frozen merge prompt is not the one the extraction used')
        return data

    strata = {'rejected': [], 'merged': [], 'multi': [], 'single': []}
    for trial in sorted(listings):
        rows = cohorts.get(trial, [])
        if screened.get(trial) != 'yes':
            strata['rejected'].append(trial)
        elif any(len(json.loads(r['merged_from'] or '[]')) > 1 for r in rows):
            strata['merged'].append(trial)
        elif len(rows) > 1:
            strata['multi'].append(trial)
        else:
            strata['single'].append(trial)
    rng = random.Random(f'{seed}:extraction')
    take = {'rejected': min(1, len(strata['rejected']))}
    rest = n - take['rejected']
    for i, name in enumerate(('merged', 'multi', 'single')):
        take[name] = min(len(strata[name]), rest // 3 + (1 if i < rest % 3 else 0))
    items = []
    for name, size in take.items():
        for trial in sorted(rng.sample(strata[name], size)):
            data, listing = checkpoint(trial), listings[trial]
            unmerged = [r['cohort_name'] for r in data.get('unmerged_cohort_rows') or data.get('cohort_rows') or []]
            names = json.loads(data['metadata']['trial_responses'][trial]['list']['answer'])['cohorts'] \
                if screened[trial] == 'yes' else []
            final = []
            for row in sorted(cohorts.get(trial, []), key=lambda r: int(r['cohort_number'])):
                merged_from = json.loads(row['merged_from'] or '[]')
                final.append(dict(number=int(row['cohort_number']), name=row['cohort_name'],
                                  merged_from_names=[unmerged[i] for i in merged_from if i < len(unmerged)],
                                  items=json.loads(row['cohort_criteria_items'])))
            items.append(dict(item_id=trial, trial_id=trial, stratum=name, weight=len(strata[name]) / size,
                              listing=ctgov_trial_text(listing['title'], listing['brief_summary'],
                                                       listing['eligibility_criteria']),
                              screen=screened[trial], cohort_names=names, cohorts=final))
    design = dict(strata={name: len(ids) for name, ids in strata.items()}, sampled=take, prompt_sha256=hashes)
    return items, instructions, design
