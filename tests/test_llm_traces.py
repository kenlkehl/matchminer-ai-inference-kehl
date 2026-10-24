import json
from types import SimpleNamespace

import pytest

from matchminer_ai.llm.endpoint_pool import EndpointPool
from matchminer_ai.llm.traces import TRACE_VERSION, response_record


def response(finish='stop', **message):
    return SimpleNamespace(model='served-teacher', choices=[SimpleNamespace(finish_reason=finish,
        message=SimpleNamespace(content='Final answer', **message))], usage=SimpleNamespace(total_tokens=30))


def test_trace_preserves_reasoning_channels_and_raw_response():
    messages = [dict(role='user', content='Fabricated clinical input')]
    for field in ('reasoning', 'reasoning_content'):
        record = response_record(response(**{field: 'Teacher reasoning'}), messages=messages)
        assert record['reasoning'] == 'Teacher reasoning'
        assert record['messages'] == messages and record['trace_version'] == TRACE_VERSION
        assert record['raw_response']['choices'][0]['message'][field] == 'Teacher reasoning'
        assert record['raw_response']['usage']['total_tokens'] == 30


def test_endpoint_archives_token_limited_output_before_application_validation(tmp_path, monkeypatch):
    monkeypatch.setenv('MMAI_LLM_TRACE_DIR', str(tmp_path))
    monkeypatch.setenv('MMAI_LLM_TRACE_STAGE', 'catalog')
    def factory(**kwargs):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **request: response('length', reasoning='Partial reasoning'))))
    pool = EndpointPool(['http://teacher/v1'], client_factory=factory)
    messages = [dict(role='user', content='Fabricated trial')]
    result, metadata = pool.request_with_metadata(model='teacher', messages=messages, temperature=1,
        extra_body={'top_k': 64, 'chat_template_kwargs': {'enable_thinking': True}, 'api_key': 'DO-NOT-ARCHIVE'},
        extra_headers={'Authorization': 'DO-NOT-ARCHIVE'})
    assert result.choices[0].finish_reason == 'length'
    path = next(tmp_path.rglob('*.json'))
    record = json.loads(path.read_text())
    assert metadata['trace_path'] == str(path.resolve())
    assert record['reasoning'] == 'Partial reasoning'
    assert record['provenance']['endpoint'] == 'http://teacher/v1'
    assert record['stage'] == 'catalog' and record['messages'] == messages
    assert record['raw_response']['choices'][0]['finish_reason'] == 'length'
    assert record['request_parameters']['extra_body']['top_k'] == 64
    assert 'DO-NOT-ARCHIVE' not in path.read_text()


def test_archive_failure_cannot_silently_return_an_unrecorded_answer(tmp_path, monkeypatch):
    calls = []
    def complete(**request):
        calls.append(request)
        return response(reasoning='Teacher reasoning')
    def cannot_write(*args, **kwargs):
        raise OSError('Archive disk is full')
    monkeypatch.setattr('matchminer_ai.llm.traces.write_response_trace', cannot_write)
    pool = EndpointPool(['http://teacher/v1'], trace_dir=tmp_path,
        client_factory=lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete))))
    with pytest.raises(OSError, match='disk is full'):
        pool.request(model='teacher', messages=[dict(role='user', content='Fabricated input')])
    assert len(calls) == 1
