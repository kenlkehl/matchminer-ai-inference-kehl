"""Backup destinations are opt-in, session scoped, and captured with each job."""

from threading import Event

import pytest
from fastapi.testclient import TestClient
from .test_app import catalog, wait
from .test_connections import settings_for
from .test_note_compression_cache import SOURCE, job
from .test_task_connections import assert_config, selection
from .test_workup import WorkupInference

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Inference


@pytest.mark.parametrize('method,task', [('full_notes', 'workup_full_backup'), ('agentic', 'workup_agent_backup')])
def test_only_selected_method_receives_enabled_backup(tmp_path, monkeypatch, method, task):
    from matchminer_ai import patients

    catalog(tmp_path)
    calls = []

    def review(frame, items, **kwargs):
        calls.append(kwargs)
        assert_config(kwargs['config'], 'workup', False)
        assert frame.note_text.tolist() == [n['note_text'] for n in SOURCE['notes']]
        return {'assessments': [], 'metadata': {}, 'notice': 'Fabricated review.'}

    monkeypatch.setattr(patients, 'review_patient_workup', review)
    monkeypatch.setattr(patients, 'review_patient_workup_with_note_search', review)
    monkeypatch.setattr(Inference, 'retrieve', WorkupInference.retrieve)
    with TestClient(create_app(settings_for(tmp_path))) as client:
        initial = client.get('/api/config').json()['tasks']
        assert not initial['workup_full_backup']['backup_enabled']
        assert not initial['workup_agent_backup']['backup_enabled']
        assert initial[task]['failure_threshold'] == 3
        client.post('/api/config', json=selection('workup'))
        body = {**selection(task, True), 'backup_enabled': True, 'failure_threshold': 4}
        assert client.post('/api/config', json=body).status_code == 200
        retrieval = job(client, '/api/retrieve', {'summary': 'Fabricated summary'})['id']
        for chosen in ('full_notes', 'agentic'):
            job(client, '/api/workup', {**SOURCE, 'method': chosen,
                'retrieval_job_id': retrieval, 'space_index': 0})
            options = calls[-1]
            if chosen == method:
                assert_config(options['backup_config'], task, True)
                assert options['max_consecutive_failures'] == 4
                assert options['backup_config'].remote['request_timeout'] == 7200
            else:
                assert 'backup_config' not in options
        client.post('/api/config', json={**body, 'backup_enabled': False})
        job(client, '/api/workup', {**SOURCE, 'method': method,
            'retrieval_job_id': retrieval, 'space_index': 0})
        assert 'backup_config' not in calls[-1]


def test_backup_snapshot_and_session_isolation(tmp_path, monkeypatch):
    from matchminer_ai import patients

    catalog(tmp_path)
    entered, release = Event(), Event()
    captured = []

    def review(frame, items, **kwargs):
        captured.append(kwargs)
        entered.set()
        assert release.wait(5)
        return {'assessments': [], 'metadata': {}, 'notice': 'Fabricated review.'}

    monkeypatch.setattr(patients, 'review_patient_workup', review)
    monkeypatch.setattr(Inference, 'retrieve', WorkupInference.retrieve)
    api = create_app(settings_for(tmp_path))
    with TestClient(api) as a, TestClient(api) as b:
        body = {**selection('workup_full_backup'), 'backup_enabled': True}
        a.post('/api/config', json=body)
        assert b.get('/api/config').json()['tasks']['workup_full_backup']['backup_enabled'] is False
        retrieval = job(a, '/api/retrieve', {'summary': 'Fabricated summary'})['id']
        try:
            started = a.post('/api/workup', json={**SOURCE,
                'retrieval_job_id': retrieval, 'space_index': 0}).json()['job_id']
            assert entered.wait(5)
            a.post('/api/config', json={**body, 'endpoint': 'http://changed.invalid/v1', 'model': 'new-model'})
        finally:
            release.set()
        assert wait(a, started).json()['status'] == 'complete'
        assert captured[0]['backup_config'].remote['server_urls'] == ['http://workup_full_backup.invalid/v1']


@pytest.mark.parametrize('overrides', [
    {'task': 'workup', 'backup_enabled': True},
    {'task': 'workup_full_backup', 'failure_threshold': 0},
    {'task': 'workup_agent_backup', 'failure_threshold': 11},
])
def test_invalid_backup_settings_rejected(tmp_path, overrides):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        before = client.get('/api/config').json()
        assert client.post('/api/config', json={**selection('workup_full_backup'), **overrides}).status_code == 422
        assert client.get('/api/config').json() == before


def test_backup_changes_preserve_saved_compressions(tmp_path, monkeypatch):
    from matchminer_ai import patients

    catalog(tmp_path)
    monkeypatch.setattr(patients, 'compress_patient_notes',
        lambda frame, **kw: frame.assign(compressed_note_text='Fabricated compressed note.'))
    with TestClient(create_app(settings_for(tmp_path))) as client:
        job(client, '/api/compress', SOURCE)
        saved = client.get('/api/note-compression').json()['note_compression']
        for task in ('workup_full_backup', 'workup_agent_backup'):
            client.post('/api/config', json={**selection(task), 'backup_enabled': True})
            assert client.get('/api/note-compression').json()['note_compression'] == saved
