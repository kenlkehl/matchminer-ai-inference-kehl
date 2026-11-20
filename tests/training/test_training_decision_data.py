"""Teacher labels as decision-model examples: one row per state, every question's yes/no answer."""
import json

from matchminer_ai.training.trialspace.data import read_jsonl, write_jsonl


class Tokenizer:  # one token per word: enough to measure states
    def __call__(self, texts, add_special_tokens=False):
        ids = lambda text: list(range(len(text.split())))
        return {'input_ids': ids(texts) if isinstance(texts, str) else [ids(t) for t in texts]}


ITEMS = [dict(number=1, heading='Inclusion criteria:', text='Age >= 18.', criterion_type='inclusion', boilerplate=False),
         dict(number=2, heading='Exclusion criteria:', text='Dialysis.', criterion_type='exclusion', boilerplate=True)]


def test_prepare_wrangles_every_family_with_its_polarity(tmp_path, monkeypatch):
    import transformers
    import matchminer_ai.training.decision_model as t
    from matchminer_ai.training.trialspace.candidates import criteria_digest
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    write_jsonl(tmp_path / 'corpus/patients.jsonl', [dict(id='p1', text='Lung cancer.', boilerplate='Dialysis.', split='train')])
    write_jsonl(tmp_path / 'corpus/cohorts.jsonl', [dict(id='c1', trial_id='T1', text='NSCLC.', boilerplate='No dialysis.',
                                                        items=ITEMS, split='train')])
    answers = dict(reasonableness='yes', cancer_type='yes', burden='no', treatment_history='no', biomarker='no')
    write_jsonl(tmp_path / 'r1/scores.jsonl', [dict(patient_id='p1', cohort_id='c1', answers=dict(answers, burden='yes'))])
    write_jsonl(tmp_path / 'r2/scores.jsonl', [dict(patient_id='p1', cohort_id='c1', answers=answers)])
    write_jsonl(tmp_path / 'b1/labels.jsonl', [dict(task='boilerplate', patient_id='p1', cohort_id='c1', excluded=True)])
    digest = criteria_digest(ITEMS)
    write_jsonl(tmp_path / 'k1/labels.jsonl', [
        dict(task='criteria', patient_id='p1', cohort_id='c1', criteria_digest=digest, number=n, criterion_type=k,
             criterion=dict(answer=a, criterion_status=s, reason='x'))
        for n, k, a, s in ((2, 'exclusion', 'no', 'met'), (1, 'inclusion', 'yes', 'met'))])
    args = t.parser().parse_args(['prepare', '--corpus', str(tmp_path / 'corpus'), '--match-labels', str(tmp_path / 'r1'),
                                  str(tmp_path / 'r2'), '--boilerplate-labels', str(tmp_path / 'b1'), '--criteria-labels',
                                  str(tmp_path / 'k1'), '--output-dir', str(tmp_path / 'out'), '--preview', '3'])
    t.prepare(args)
    rows = {r['family']: r for r in read_jsonl(tmp_path / 'out/examples.jsonl')}
    assert rows['match']['answers'] == answers  # the later round wins
    assert rows['boilerplate']['answers'] == {'boilerplate': 'no'}  # excluded -> not potentially eligible
    assert rows['criterion']['answers'] == {'criterion_1': 'yes', 'criterion_2': 'no'}  # one state, its lines in order
    manifest = json.loads((tmp_path / 'out/prepared.json').read_text())
    assert manifest['states'] == dict(match=1, boilerplate=1, criterion=1) and manifest['conflicting_rounds'] == {'match': 1}
    preview = {r['family']: r for r in read_jsonl(tmp_path / 'out/preview.jsonl')}
    question = preview['criterion']['questions'][1]
    assert question['answer'] == 'no' and question['instructions'].endswith('[2] Exclusion criterion: Dialysis.')
    assert 'Other relevant history:\nDialysis.' in preview['criterion']['state']


def test_the_loss_is_the_yes_no_readout_inference_reads():
    import torch
    import matchminer_ai.training.decision_model as t
    readout = [[3, 7], [5, 1, 9]]  # yes forms, no forms (prompt.readout_ids)
    logits, targets = torch.randn(4, 12), torch.tensor([0, 1, 1, 0])
    best = torch.stack([logits[:, [3, 7]].amax(1), logits[:, [5, 1, 9]].amax(1)], 1)  # each option's best form
    expected = -best.log_softmax(1)[torch.arange(4), targets].mean()
    torch.testing.assert_close(t.readout_loss(logits, targets, readout), expected)
    other = logits.clone()
    other[:, [0, 2, 4, 6, 8, 10, 11]] += 30  # probability on any other token costs nothing
    torch.testing.assert_close(t.readout_loss(other, targets, readout), expected)
