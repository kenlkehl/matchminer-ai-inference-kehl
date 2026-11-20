"""Prompt, parsing, LLM-client and scoring contracts of the dual-judge evaluation (matchminer_ai.evaluation.dual_judge), on
fabricated data: strict schemas written into the shared request, the Vertex and Azure request bodies, answer
validation, statement splitting and quote grounding, and units, agreement, adjudication and gold."""
import json
from pathlib import Path

import pytest

from matchminer_ai.evaluation.dual_judge import compare as dj_compare
from matchminer_ai.evaluation.dual_judge import judge_worker, judging, rubric_decision, rubric_extraction, rubric_summary
from matchminer_ai.evaluation.dual_judge import metrics as dj_metrics


def test_strict_schema_closes_objects_types_enums_and_rejects_optional_fields():
    schema = judging.strict_schema({'type': 'object', 'required': ['a'], 'properties': {'a': {'enum': ['x', 'y']}}})
    assert schema['additionalProperties'] is False and schema['properties']['a']['type'] == 'string'
    with pytest.raises(ValueError):
        judging.strict_schema({'type': 'object', 'required': [], 'properties': {'a': {'type': 'string'}}})
    for module in (rubric_extraction, rubric_summary):  # every rubric schema passes both strict modes
        judging.strict_schema(module.SCHEMA)
    judging.strict_schema(rubric_decision.BOILERPLATE_SCHEMA)


def test_request_writes_the_schema_into_the_shared_system_text():
    schema = {'type': 'object', 'required': ['answer'], 'properties': {'answer': {'enum': ['yes', 'no']}}}
    a = judging.make_request('decision', 'x-1', 'RULES', 'TASK', schema, max_output_tokens=100)
    b = judging.make_request('decision', 'x-1', 'RULES', 'TASK', schema, max_output_tokens=100)
    assert a == b and a['request_id'].startswith('decision:x-1:')
    assert a['system'].startswith('RULES\n\nANSWER FORMAT')
    assert json.loads(a['system'].split('fields):\n', 1)[1]) == a['schema']


def test_vertex_body_has_effort_and_no_sampling_or_structured_output_feature():
    judge = object.__new__(judge_worker.VertexJudge)
    judge.model, judge.effort = 'claude-opus-5-5', 'high'
    request = judging.make_request('summary', 'S01', 'SYS', 'USER', rubric_summary.SCHEMA, max_output_tokens=64000)
    body = judge.body(request, request['user'])
    assert body['model'] == 'claude-opus-5-5' and body['max_tokens'] == 64000
    assert body['output_config'] == {'effort': 'high'}  # the org policy forbids output_config.format on Vertex
    assert not {'temperature', 'top_p', 'top_k', 'thinking'} & set(body)
    assert body['system'][0]['text'] == request['system'] and body['system'][0]['cache_control']['type'] == 'ephemeral'
    assert body['messages'] == [{'role': 'user', 'content': 'USER'}]


def test_azure_body_enforces_the_strict_schema_and_parses_refusal_and_truncation():
    judge = object.__new__(judge_worker.AzureJudge)
    judge.model, judge.effort = 'gpt-6-1-sol-deployment', 'high'
    request = judging.make_request('decision', 'd-1', 'SYS', 'USER', rubric_decision.BOILERPLATE_SCHEMA,
                                   max_output_tokens=32000)
    body = judge.body(request, request['user'])
    assert body['text']['format'] == {'type': 'json_schema', 'name': 'mmai_judge_decision', 'strict': True,
                                      'schema': request['schema']}
    assert body['reasoning'] == {'effort': 'high'} and body['max_output_tokens'] == 32000 and body['store'] is False
    assert not {'temperature', 'top_p'} & set(body)
    assert body['input'][0] == {'role': 'system', 'content': request['system']}
    message = lambda *blocks: {'status': 'completed', 'output': [{'type': 'reasoning'},  # noqa: E731
                                                                 {'type': 'message', 'content': list(blocks)}]}
    assert judge_worker.AzureJudge.parse(message({'type': 'output_text', 'text': '{"a": 1}'}))[:2] == ('ok', '{"a": 1}')
    assert judge_worker.AzureJudge.parse(message({'type': 'refusal', 'refusal': 'no'}))[0] == 'refusal'
    assert judge_worker.AzureJudge.parse(dict(message({'type': 'output_text', 'text': '{'}), status='incomplete'))[0] \
        == 'max_tokens'


def test_answer_parsing_and_schema_check():
    assert judging.parse_answer('```json\n{"a": "yes"}\n```') == {'a': 'yes'}
    with pytest.raises(ValueError):
        judging.parse_answer('[1, 2]')
    schema = judging.strict_schema({'type': 'object', 'required': ['a'], 'properties': {'a': {'enum': ['yes', 'no']}}})
    assert judging.check_enums({'a': 'yes'}, schema) == []
    assert len(judging.check_enums({'a': 'maybe', 'b': 1}, schema)) == 2


def test_summary_statements_split_sections_lines_and_sentences():
    summary = ('Age: 61\nSex: Female\nCancer type: Breast cancer\nTreatment history:\n'
               '# 1/2020-6/2020: docetaxel/cyclophosphamide\n# 7/2020-present: letrozole\n\n'
               'Other relevant history:\nECOG 1 as of 3/2021. Hypertension. LVEF 60% on echo 1/2020.')
    parts = rubric_summary.statements(summary)
    assert [p['n'] for p in parts] == list(range(1, 9))
    assert parts[0] == {'n': 1, 'section': 'Age', 'text': 'Age: 61'}
    assert [p['section'] for p in parts[3:5]] == ['Treatment history'] * 2
    assert [p['text'] for p in parts[5:]] == ['ECOG 1 as of 3/2021.', 'Hypertension.', 'LVEF 60% on echo 1/2020.']
    assert rubric_summary.validate({'statements': [{'n': n} for n in range(1, 9)]}, {'statements': 8}) == []
    assert rubric_summary.validate({'statements': [{'n': 1}, {'n': 1}]}, {'statements': 2})


def test_quote_grounding_ignores_case_and_whitespace_and_follows_ellipsis_order():
    source = 'Started   Letrozole on 7/1/2020.\nTolerating well; no hot flashes.'
    assert judging.quote_found('started letrozole on 7/1/2020', source)
    assert judging.quote_found('Started letrozole ... no hot flashes', source)
    assert not judging.quote_found('no hot flashes ... Started letrozole', source)
    assert not judging.quote_found('anastrozole', source)


def test_extraction_coverage_and_decision_polarity_validation():
    expected = {'cohorts': {'0': [1, 2, 3]}}
    ok = {'cohorts': [{'cohort': 0, 'lines': [{'line': n} for n in (1, 2, 3)]}]}
    assert rubric_extraction.validate(ok, expected) == []
    assert rubric_extraction.validate({'cohorts': [{'cohort': 0, 'lines': [{'line': 1}, {'line': 4}]}]}, expected)
    inclusion = dict(task='criteria', criterion_type='inclusion')
    assert rubric_decision.validate({'answer': 'no', 'criterion_status': 'not_met', 'reason': 'r'}, inclusion) == []
    assert rubric_decision.validate({'answer': 'no', 'criterion_status': 'met', 'reason': 'r'}, inclusion)
    assert rubric_decision.validate({'answer': 'no', 'criterion_status': 'met', 'reason': 'r'},
                                    dict(task='criteria', criterion_type='exclusion')) == []


def _decision_item(item_id, task, teacher, p):
    return dict(item_id=item_id, task=task, cohort_id='NCT0-ctgov-cohort-0', number=None, criterion_type=None,
                stratum='s', weight=1.0, teacher=teacher, decision_p_yes=p,
                messages=[{'role': 'user', 'content': 'synthetic task'}])


def _requests(run_dir, items, ids):
    judging.write_jsonl(run_dir / 'requests' / 'decision.jsonl',
                        [dict(request_id=rid, item_id=item['item_id']) for rid, item in zip(ids, items)])


def _answer(row_id, judge, item_id, answer, status='ok'):
    return dict(request_id=row_id, item_id=item_id, component='decision', judge=judge, attempt=1, final=True,
                status=status, answer=answer)


def test_units_agreement_adjudication_and_gold(tmp_path):
    items = [_decision_item('b-1', 'boilerplate', {'boilerplate': 'yes'}, {'boilerplate': 0.9}),
             _decision_item('b-2', 'boilerplate', {'boilerplate': 'yes'}, {'boilerplate': 0.6}),
             _decision_item('b-3', 'boilerplate', {'boilerplate': 'no'}, {'boilerplate': 0.2})]
    judging.write_jsonl(tmp_path / 'items' / 'decision.jsonl', items)
    _requests(tmp_path, items, ['1', '2', '3'])
    no, yes = {'reason': 'r', 'excluded': True}, {'reason': 'r', 'excluded': False}
    judging.write_jsonl(tmp_path / 'responses' / 'vertex' / 'decision.jsonl',
                        [_answer('1', 'vertex', 'b-1', yes), _answer('2', 'vertex', 'b-2', yes),
                         _answer('3', 'vertex', 'b-3', no)])
    judging.write_jsonl(tmp_path / 'responses' / 'azure' / 'decision.jsonl',
                        [_answer('1', 'azure', 'b-1', yes), _answer('2', 'azure', 'b-2', no),
                         _answer('3', 'azure', 'b-3', None, status='refusal')])
    units, summary = dj_compare.compare(tmp_path, audit_fraction=0.0)
    status = {u['item_id']: u['status'] for u in units}
    assert status == {'b-1': 'agreed', 'b-2': 'disputed', 'b-3': 'single'}
    assert summary['components']['decision']['queued'] == {'disputed': 1, 'single': 1, 'audit': 0}
    # the user decides b-2 "no" (excluded), then changes their mind on b-3 and withdraws it
    journal = judging.Journal(tmp_path / 'adjudications.jsonl')
    journal.append(dict(unit_id='decision:b-2:boilerplate', choice='judge2', value='no', at='2026-11-18T10:00:00'))
    journal.append(dict(unit_id='decision:b-3:boilerplate', choice='other', value='no', at='2026-11-18T10:01:00'))
    journal.append(dict(unit_id='decision:b-3:boilerplate', choice='skip', value=None, at='2026-11-18T10:02:00'))
    decisions = judging.latest_decisions(judging.read_jsonl(tmp_path / 'adjudications.jsonl'))
    assert set(decisions) == {'decision:b-2:boilerplate'}
    by_item = {u['item_id']: u for u in units}
    assert judging.gold(by_item['b-1'], decisions) == ('yes', 'agreed')
    assert judging.gold(by_item['b-2'], decisions) == ('no', 'adjudicated')
    assert judging.gold(by_item['b-3'], decisions) == (None, 'pending')
    result = dj_metrics.metrics(tmp_path)['components']['decision']
    family = result['results']['boilerplate']
    assert family['n'] == 2 and family['teacher_accuracy']['rate'] == 0.5  # the teacher said yes on b-2
    assert family['model_auroc']['value'] == 1.0  # P 0.9 (gold yes) above P 0.6 (gold no)
    judges = result['agreement']['boilerplate']['judges']
    assert judges['azure']['accuracy_adjudicated'] == 1.0 and judges['vertex']['accuracy_adjudicated'] == 0.0


def test_awaiting_units_are_not_queued_until_the_other_judge_runs(tmp_path):
    items = [_decision_item('b-1', 'boilerplate', {'boilerplate': 'yes'}, {'boilerplate': 0.9})]
    judging.write_jsonl(tmp_path / 'items' / 'decision.jsonl', items)
    _requests(tmp_path, items, ['1'])
    judging.write_jsonl(tmp_path / 'responses' / 'vertex' / 'decision.jsonl',
                        [_answer('1', 'vertex', 'b-1', {'reason': 'r', 'excluded': False})])
    units, _ = dj_compare.compare(tmp_path, audit_fraction=1.0)
    assert [(u['status'], u['queued']) for u in units] == [('awaiting', False)]


def test_collapsed_kinds_compare_major_versus_not_major_and_kappa():
    kinds = rubric_summary.KINDS
    assert judging.compared(kinds, 'omission', 'minor') == judging.compared(kinds, 'omission', 'none') == 'no_major'
    assert judging.compared(kinds, 'statement', 'minor') == 'minor'
    assert judging.unit_status({'vertex': 'no_major', 'azure': 'no_major'}) == 'agreed'
    assert judging.cohen_kappa([('a', 'a'), ('b', 'b')]) == 1.0
    assert judging.cohen_kappa([('a', 'a'), ('a', 'a')]) is None


def test_answers_to_superseded_requests_are_not_used(tmp_path):
    """A rebuilt component can reuse item ids for new items; only answers to the current requests count."""
    items = [_decision_item('b-1', 'boilerplate', {'boilerplate': 'yes'}, {'boilerplate': 0.9})]
    judging.write_jsonl(tmp_path / 'items' / 'decision.jsonl', items)
    _requests(tmp_path, items, ['new'])
    for judge in judging.JUDGES:
        judging.write_jsonl(tmp_path / 'responses' / judge / 'decision.jsonl',
                            [_answer('old', judge, 'b-1', {'reason': 'r', 'excluded': True})])
    units, _ = dj_compare.compare(tmp_path, audit_fraction=0.0)
    assert units == []  # no current answers, so no units yet


def test_usage_cost_bills_anthropic_cache_tokens():
    usage = dict(input_tokens=1_000_000, cache_write_tokens=1_000_000, cache_read_tokens=1_000_000, output_tokens=0)
    assert judging.usage_cost(usage, dict(usd_in=4.0, usd_out=20.0)) == pytest.approx(4.0 * (1 + 1.25 + 0.1))
