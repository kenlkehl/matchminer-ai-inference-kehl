"""One atomic model copy, independent later edits, and no cross-session changes."""

from threading import Event

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from .test_app import catalog, wait
from .test_connections import settings_for
from .test_note_compression_cache import SOURCE, job
from .test_task_connections import selection

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.settings import GEMINI4_PROFILE, GEMINI_PROFILE, GEMMA_PROFILE, LLM_TASKS

CONNECTION_FIELDS = ("llm_backend", "endpoint", "model", "tokenizer", "reasoning")


def shared(task="workup", **overrides):
    return {
        "task": task,
        "apply_to_all": True,
        "llm_backend": "local",
        "endpoint": "shared:8000",
        "model": "Qwen/Qwen3.5-4B",
        "tokenizer": "matching-tokenizer",
        "reasoning": True,
        **overrides,
    }


def connection(task):
    return {key: task[key] for key in CONNECTION_FIELDS}


@pytest.mark.parametrize("source", list(LLM_TASKS))
def test_all_tasks_copy_source_connection_and_later_edits_remain_independent(tmp_path, source):
    api = create_app(settings_for(tmp_path))
    with TestClient(api) as client, TestClient(api) as other:
        defaults = other.get("/api/config").json()
        for task in LLM_TASKS:
            settings = {
                **selection(task),
                "model": f"previous-{task}",
                "backup_enabled": task == "workup_full_backup",
                "failure_threshold": 7 if task == "workup_agent_backup" else 4,
            }
            assert client.post("/api/config", json=settings).status_code == 200
        prior = client.get("/api/config").json()["tasks"]
        response = client.post("/api/config", json=shared(source))
        assert response.status_code == 200
        applied = response.json()
        assert applied["model"] == "Qwen/Qwen3.5-4B"
        expected = {
            "llm_backend": "local",
            "endpoint": "http://shared:8000/v1",
            "model": "Qwen/Qwen3.5-4B",
            "tokenizer": "matching-tokenizer",
            "reasoning": True,
        }
        for task, saved in applied["tasks"].items():
            assert connection(saved) == expected
            assert saved["sampling"]["temperature"] == 1.0
            assert saved["sampling"]["top_p"] == 0.95
            if task != source:
                assert saved["backup_enabled"] == prior[task]["backup_enabled"]
                assert saved["failure_threshold"] == prior[task]["failure_threshold"]
        assert client.get("/api/config").json() == applied
        assert other.get("/api/config").json() == defaults

        edited = client.post(
            "/api/config",
            json={
                **selection("summarize"),
                "model": "one-task-model",
            },
        ).json()["tasks"]
        assert edited["summarize"]["model"] == "one-task-model"
        assert all(edited[t] == applied["tasks"][t] for t in LLM_TASKS if t != "summarize")


def test_copy_from_backup_saves_only_its_own_backup_controls(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        client.post(
            "/api/config",
            json={
                **selection("workup_full_backup"),
                "backup_enabled": True,
                "failure_threshold": 4,
            },
        )
        result = client.post(
            "/api/config",
            json=shared(
                "workup_agent_backup",
                backup_enabled=True,
                failure_threshold=9,
            ),
        ).json()["tasks"]
        assert result["workup_agent_backup"]["backup_enabled"]
        assert result["workup_agent_backup"]["failure_threshold"] == 9
        assert result["workup_full_backup"]["backup_enabled"]
        assert result["workup_full_backup"]["failure_threshold"] == 4
        assert not any(result[t]["backup_enabled"] for t in ("summarize", "compress", "workup"))


@pytest.mark.parametrize(
    "provider,profile",
    [
        ("gemini4_vertex", GEMINI4_PROFILE),
        ("gemini_vertex", GEMINI_PROFILE),
        ("gemma_vertex", GEMMA_PROFILE),
    ],
)
def test_bulk_cloud_selection_uses_server_owned_destination(tmp_path, provider, profile):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        response = client.post(
            "/api/config",
            json=shared(
                llm_backend=provider,
                endpoint="http://untrusted.invalid/v1",
                model="untrusted",
            ),
        )
        assert response.status_code == 200
        for saved in response.json()["tasks"].values():
            assert all(saved[k] == profile[k] for k in ("endpoint", "model", "tokenizer"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": None},
        {"endpoint": "file:///tmp/model"},
        {"failure_threshold": 0},
    ],
)
def test_invalid_bulk_request_changes_no_task(tmp_path, overrides):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        before = client.get("/api/config").json()
        assert client.post("/api/config", json=shared(**overrides)).status_code == 422
        assert client.get("/api/config").json() == before


def test_bulk_model_change_only_invalidates_compression_if_connection_changed(
    tmp_path, monkeypatch
):
    from matchminer_ai import patients

    catalog(tmp_path)
    monkeypatch.setattr(
        patients,
        "compress_patient_notes",
        lambda frame, **kw: frame.assign(compressed_note_text="Fabricated compact note."),
    )
    with TestClient(create_app(settings_for(tmp_path))) as client:
        client.post("/api/config", json=shared(reasoning=False))
        job(client, "/api/compress", SOURCE)
        saved = client.get("/api/note-compression").json()["note_compression"]
        client.post("/api/config", json=shared("summarize", reasoning=False))
        assert client.get("/api/note-compression").json()["note_compression"] == saved
        client.post("/api/config", json=shared(model="changed-model", reasoning=False))
        assert client.get("/api/note-compression").json()["note_compression"] is None


def test_bulk_update_does_not_redirect_running_job(tmp_path, monkeypatch):
    from matchminer_ai import patients

    entered, release = Event(), Event()
    captured = []

    def summarize(frame, **kwargs):
        captured.append(kwargs["config"])
        entered.set()
        assert release.wait(5)
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    monkeypatch.setattr(patients, "summarize_patients", summarize)
    with TestClient(create_app(settings_for(tmp_path))) as client:
        client.post("/api/config", json=selection("summarize"))
        try:
            key = client.post("/api/summarize", json=SOURCE).json()["job_id"]
            assert entered.wait(5)
            assert client.post("/api/config", json=shared()).status_code == 200
        finally:
            release.set()
        assert wait(client, key).json()["status"] == "complete"
        assert captured[0].remote["server_urls"] == ["http://summarize.invalid/v1"]
        assert captured[0].patient["local"]["chat_template_kwargs"]["enable_thinking"] is False
