"""Component B: one patient summary (index event) against the notes it was written from (PHI).

The judges see the summarizer's instructions (``patient.serial.user.primer.txt`` and ``.question.txt`` of the
inference package; unchanged since 691a6ab on 2026-11-12, before the validation summaries were made on 2026-11-16),
the event's notes exactly as the summarizer read them (sorted by date, repeated sentences removed, dated headers;
``matchminer_ai.patients.prepare``), and the final summary split into numbered statements. Per-chunk running
summaries were not kept, so the whole record is judged against the final summary.
"""
import re

from matchminer_ai.evaluation.dual_judge.judging import prompt_text, quote_found

NAME = 'summary'
PROMPT = 'judge.summary.txt'
MAX_OUTPUT_TOKENS = 64000
SEVERITY = ['none', 'minor', 'major']
SECTIONS = ['age', 'sex', 'cancer_type', 'histology', 'current_extent', 'biomarkers', 'treatment_history',
            'performance_status', 'measurable_disease', 'cns_metastases', 'laboratory_values', 'lvef_qtc',
            'treatment_toxicities', 'other_medical_conditions', 'recent_major_surgery', 'other_cancers',
            'trial_participation', 'hospice_or_stopping_treatment', 'other']
HEADINGS = {'age': 'Age', 'sex': 'Sex', 'cancer type': 'Cancer type', 'histology': 'Histology',
            'current extent': 'Current extent', 'biomarkers': 'Biomarkers', 'treatment history': 'Treatment history',
            'other relevant history': 'Other relevant history'}

_text = {'type': 'string'}
_evidence = {'type': 'array', 'items': {'type': 'object', 'required': ['date', 'quote'],
                                        'properties': {'date': _text, 'quote': _text}}}
SCHEMA = {
    'type': 'object',
    'required': ['statements', 'omissions', 'overall', 'summary'],
    'properties': {
        'statements': {'type': 'array', 'items': {
            'type': 'object', 'required': ['n', 'verdict', 'severity', 'evidence', 'note'],
            'properties': {'n': {'type': 'integer'},
                           'verdict': {'enum': ['supported', 'unsupported', 'contradicted', 'outdated', 'no_claim']},
                           'severity': {'enum': SEVERITY}, 'evidence': _evidence, 'note': _text}}},
        'omissions': {'type': 'array', 'items': {
            'type': 'object', 'required': ['section', 'severity', 'description', 'evidence'],
            'properties': {'section': {'enum': SECTIONS}, 'severity': {'enum': ['minor', 'major']},
                           'description': _text, 'evidence': _evidence}}},
        'overall': {'enum': ['acceptable', 'minor_issues', 'major_issues']},
        'summary': _text,
    },
}

NO_MAJOR = {'none': 'no_major', 'minor': 'no_major', 'major': 'major'}
KINDS = {
    'statement': dict(label='Statement', values=['ok', 'minor', 'major']),
    'omission': dict(label='Omissions in a section', values=['none', 'minor', 'major'], collapse=NO_MAJOR),
    'overall': dict(label='Overall', values=['acceptable', 'minor_issues', 'major_issues'],
                    collapse={'acceptable': 'no_major', 'minor_issues': 'no_major', 'major_issues': 'major'}),
}

_SENTENCE = re.compile(r'(?<=[.;!?])\s+(?=[A-Z#(])')


def statements(summary):
    """The summary's numbered statements: each nonblank line, a line of several sentences split into them, with the
    section (the latest heading seen; a heading line with text after its colon is itself a statement)."""
    out, section = [], 'Summary'
    for raw in summary.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = line.split(':', 1)[0].strip().lower()
        if head in HEADINGS:
            section = HEADINGS[head]
            rest = line.split(':', 1)[1].strip()
            if not rest:
                continue
            if head in ('treatment history', 'other relevant history'):
                line = rest
        for sentence in _SENTENCE.split(line):
            if sentence.strip():
                out.append(dict(n=len(out) + 1, section=section, text=sentence.strip()))
    return out


def system_prompt(instructions):
    return '\n'.join([prompt_text(PROMPT).rstrip('\n'), '', "THE SUMMARIZER'S INSTRUCTIONS",
                      '', '=== For each chunk of notes ===', instructions['primer'],
                      '', '=== Final instruction of each request ===', instructions['question']])


def render(item, instructions):
    numbered, section = [], None
    for s in item['statements']:
        if s['section'] != section:
            section = s['section']
            numbered.append(f'({section})')
        numbered.append(f'[{s["n"]}] {s["text"]}')
    user = (f'<clinical_notes>\n{item["notes"]}\n</clinical_notes>\n\n'
            '<summary>\nThe summary under review, as numbered statements under the sections they belong to:\n\n'
            + '\n'.join(numbered) + '\n</summary>')
    return system_prompt(instructions), user


def expect(item):
    return dict(statements=len(item['statements']))


def validate(answer, expected):
    numbers = [s['n'] for s in answer.get('statements', [])]
    wanted = list(range(1, expected.get('statements', 0) + 1))
    if sorted(numbers) != wanted:
        missing, extra = sorted(set(wanted) - set(numbers)), sorted(set(numbers) - set(wanted))
        dupes = sorted({n for n in numbers if numbers.count(n) > 1})
        return [f'statements must be numbered 1-{len(wanted)} exactly once: missing {missing}, unknown {extra}, '
                f'repeated {dupes}']
    return []


def _grounded(evidence, notes):
    return [dict(e, grounded=quote_found(e['quote'], notes)) for e in evidence]


def units(item, answer):
    notes, out = item['notes'], {}
    for s in answer['statements']:
        detail = dict(s, evidence=_grounded(s['evidence'], notes))
        if s['verdict'] in ('supported', 'no_claim'):
            value = 'ok'
        else:
            value = 'major' if s['severity'] == 'major' else 'minor'
        out[f's{s["n"]}'] = ('statement', value, detail)
    by_section = {section: [] for section in SECTIONS}
    for omission in answer['omissions']:
        by_section[omission['section']].append(dict(omission, evidence=_grounded(omission['evidence'], notes)))
    order = {s: i for i, s in enumerate(SEVERITY)}
    for section, found in by_section.items():
        worst = max((o['severity'] for o in found), key=order.get, default='none')
        out[f'omission/{section}'] = ('omission', worst, dict(omissions=found))
    out['overall'] = ('overall', answer['overall'], dict(summary=answer['summary']))
    return out


def view(item):
    return dict(kind='summary', title=item['item_id'], source=dict(kind='notes', text=item['notes']),
                statements=item['statements'])


def unit_target(item, key):
    if key.startswith('s') and key[1:].isdigit():
        s = item['statements'][int(key[1:]) - 1]
        return f'[{s["n"]}] ({s["section"]}) {s["text"]}'
    if key.startswith('omission/'):
        return f'Omissions in section: {key.split("/", 1)[1].replace("_", " ")}'
    return 'The summary as a whole'


# --- building (where the data lives: notes, summaries and the inference package) ----------------------------------------------------

STRATA = (  # (name, share of the sample, note volume in tokens, estimated as characters / 4); contiguous, so every
    # query event with a summary is in exactly one stratum
    ('under_20k', 0.25, 0, 20_000), ('20k_80k', 0.375, 20_000, 80_000), ('80k_300k', 0.25, 80_000, 300_000),
    ('300k_plus', 0.125, 300_000, None))


def notes_text(frame):
    """One event's notes as the summarizer read them (validate_note_inputs order, deduplicate_patient_notes, then
    concatenate_and_chunk_notes's dated blocks; a one-token stand-in tokenizer keeps them in one chunk)."""
    from types import SimpleNamespace
    import pandas as pd
    from matchminer_ai.patients.prepare import concatenate_and_chunk_notes, deduplicate_patient_notes, validate_note_inputs

    normalized = validate_note_inputs(frame).sort_values(
        'note_date', key=pd.to_datetime, kind='mergesort', na_position='last')
    notes = [(str(d) if pd.notna(d) else 'unknown date', str(t)) for d, t in zip(normalized.note_date, normalized.note_text)]
    notes = deduplicate_patient_notes(notes)
    one = lambda text, add_special_tokens=False: SimpleNamespace(input_ids=[0])  # noqa: E731
    chunks = concatenate_and_chunk_notes(notes, one, chunk_size=10, chunk_overlap=0)
    return chunks[0][0] if chunks else ''


def build(fidelity_dir, denominators, n, seed):
    """Seeded events of the evaluation's 500 query events, stratified by note volume. Returns (items, instructions,
    design); items carry the notes and the summary (PHI) and an opaque item id, never the MRN."""
    import json
    import random
    from pathlib import Path
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    from matchminer_ai.llm.prompts import load_prompt_text

    instructions = dict(primer=load_prompt_text('patient.serial.user.primer.txt'),
                        question=load_prompt_text('patient.serial.user.question.txt'))
    sample = json.loads((Path(fidelity_dir) / 'corpus' / 'sample.json').read_text())
    queries = [int(event) for event in sample['patient_queries']]
    notes_file = Path(denominators) / 'soc_validation' / 'soc_validation_note_level_dataset_minus14.parquet'
    dataset = ds.dataset(notes_file)
    table = dataset.to_table(columns=['pseudo_mrn', 'date', 'RPT_ID', 'text'],  # never dfci_mrn
                             filter=pc.field('pseudo_mrn').isin(queries))
    notes = table.to_pandas().sort_values(['pseudo_mrn', 'date', 'RPT_ID'], kind='stable')
    chars = notes.text.str.len().groupby(notes.pseudo_mrn).sum()
    summaries = pd.concat([pd.read_parquet(p, columns=['patient_id', 'patient_summary'])
                           for p in sorted((Path(denominators) / 'summaries' / 'soc_validation' / 'summaries').glob('*.parquet'))])
    summaries = summaries.set_index(summaries.patient_id.astype(str)).patient_summary
    rng = random.Random(f'{seed}:summary')
    counts = [int(share * n) for _, share, _, _ in STRATA]
    for i in range(n - sum(counts)):
        counts[i % len(counts)] += 1
    items, design = [], dict(strata={}, sampled={}, query_events=len(queries),
                             with_summary=sum(str(e) in summaries.index for e in chars.index))
    for (name, _, low, high), count in zip(STRATA, counts):
        tokens = chars / 4
        pool = sorted(int(e) for e in tokens[(tokens >= low) & ((tokens < high) if high else True)].index
                      if str(e) in summaries.index)
        design['strata'][name], design['sampled'][name] = len(pool), min(count, len(pool))
        for event in sorted(rng.sample(pool, min(count, len(pool)))):
            rows = notes[notes.pseudo_mrn == event]
            frame = pd.DataFrame(dict(patient_id=rows.pseudo_mrn.astype(str), note_text=rows.text, note_date=rows.date))
            text = notes_text(frame)
            summary = summaries[str(event)]
            items.append(dict(item_id=f'S{len(items) + 1:02d}', event=str(event), stratum=name,
                              weight=len(pool) / max(1, min(count, len(pool))), notes=text, note_rows=len(rows),
                              summary=summary, statements=statements(summary)))
    design['prompt_sha256'] = {k: __import__('hashlib').sha256(v.encode()).hexdigest() for k, v in instructions.items()}
    return items, instructions, design
