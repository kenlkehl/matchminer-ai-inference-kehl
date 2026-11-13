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
    # criteria: 2728713a... (rounds 1-3 of the n=5000 run) -> 7a7eb230... on 2026-11-13, when the per-criterion
    # state, question and teacher prompt began judging a line in its role (alternatives, conditionals, branches).
    assert contract_hash('criteria') == '7a7eb230af1f502010a5e1707d6bb46a547d1a4618d4859efc1983bb22590afc'
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
    with pytest.raises(ValueError):  # a line that does not bind the patient (not_applicable) cannot fail them
        parse_judgment('criteria', json.dumps(dict(answer='no', criterion_status='not_applicable', reason='x')), 'inclusion')
    # the state carries the role rules once per packed sequence; each question points to them
    from matchminer_ai.matching.decision_model import criteria_state, criterion_question
    state = criteria_state('Key: metastatic lung cancer.', ITEMS)
    assert 'every alternative of its group' in state and 'only some of the patients the criteria admit' in state
    assert 'an alternative fails only with its whole group' in criterion_question(ITEMS[1])
    assert criterion_question(ITEMS[1]) in user
    validate_row('criteria', dict(criterion=json.loads(fails), criterion_type='exclusion'))
    with pytest.raises(ValueError):  # a line whose type differs from the corpus is not asked
        list(questions([dict(rows[0], criterion_type='exclusion')], patients, cohorts, task='criteria'))


def test_client_processes_label_what_one_process_labels(tmp_path, monkeypatch):
    """--label-client-processes 2 (its clients run as threads here) publishes the one-process outputs; the number
    of clients enters no identity."""
    import argparse
    import hashlib
    import http.server
    import pickle
    import threading
    from pathlib import Path
    from matchminer_ai.labeling import add_label_cli_args, clients, run_label, runner
    model = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4'

    class Teacher(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(dict(object='list', data=[dict(id=model, object='model', root=model, max_model_len=262144)]))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            h = hashlib.sha256(body['prompt'].encode()).hexdigest()
            text = (answer(**{label: int(h[i], 16) % 2 for i, label in enumerate(LABELS)})
                    if 'reasonableness' in json.dumps(body.get('structured_outputs')) else ('Yes!' if int(h[0], 16) % 2 else 'No!'))
            self.reply(dict(id='cmpl-1', object='text_completion', created=0, model=model,
                            choices=[dict(index=0, text=text, finish_reason='stop', logprobs=None)]))

    class Tokenizer:
        chat_template, backend_tokenizer = 'fabricated', None

        def apply_chat_template(self, messages, **kwargs):
            return '\n'.join(message['content'] for message in messages)

        def encode(self, text, add_special_tokens=False):
            return text.split()

    class Client:  # a client process's run, in a thread, so the fakes above reach it
        def __init__(self, spec, fds):
            self.pid, self.returncode = None, None
            self.thread = threading.Thread(target=self.main, args=(spec,))
            self.thread.start()

        def main(self, spec):
            try:
                clients.run_client(pickle.loads(Path(spec).read_bytes()))
            finally:
                self.returncode = 0 if self.returncode is None else self.returncode

        def poll(self):
            return self.returncode

        def wait(self):
            self.thread.join()

        terminate = kill = lambda self: None

    monkeypatch.setattr(runner, 'load_tokenizer', lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(runner, 'resolve_parser_name', lambda *args: None)
    monkeypatch.setattr(clients, 'spawn', Client)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Teacher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    patients = [dict(id=f'p{i}', text=f'Key: metastatic cancer {i}.', boilerplate=f'History {i % 2}.', split='train')
                for i in range(4)]
    cohorts = [dict(id=f'c{i}', trial_id=f'T{i}', text=f'Key: cohort {i}.', boilerplate=f'No dialysis {i}.' if i else '',
                    split='train') for i in range(4)]
    (tmp_path / 'corpus').mkdir()
    for name, rows in (('patients.jsonl', patients), ('cohorts.jsonl', cohorts)):
        (tmp_path / 'corpus' / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    pairs = [dict(patient_id=p['id'], cohort_id=c['id'], split='train') for p in patients for c in cohorts]
    (tmp_path / 'pairs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in pairs * 2))  # duplicates too

    def published(task, count):
        output = tmp_path / f'{task}{count}'
        parser = add_label_cli_args(argparse.ArgumentParser())
        args = parser.parse_args(['--corpus', str(tmp_path / 'corpus'), '--candidates', str(tmp_path / 'pairs.jsonl'),
                                  '--output', str(output), '--task', task, '--teacher-model', model,
                                  '--teacher-tokenizer', model, '--server_urls', f'http://127.0.0.1:{server.server_port}/v1',
                                  '--max_concurrent_per_server', '3', '--label-client-processes', str(count)])
        args.label_scratch_dir = str(tmp_path / 'scratch')
        run_label(args)
        products = json.loads((output / 'storage_complete.json').read_text())['products']
        labels = [{k: v for k, v in json.loads(line).items() if k != 'generated_at'}
                  for line in (output / 'labels.jsonl').read_text().splitlines()]
        return dict(labels=labels, products=sorted(products), status=json.loads((output / 'status.json').read_text())['state'],
                    **{name: (output / name).read_text() for name in ('labeling_manifest.json', 'scores.jsonl')
                       if (output / name).exists()})

    for task, questions_ in (('match', 16), ('boilerplate', 12)):
        one, two = published(task, 1), published(task, 2)
        assert one == two and one['status'] == 'complete' and len(one['labels']) == questions_
        assert not [key for key in json.loads(one['labeling_manifest.json']) if 'client' in key or 'process' in key]
        assert not list((tmp_path / 'scratch').glob('*/clients')) and 'clients' not in str(one['products'])
    server.shutdown()
