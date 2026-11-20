"""Task connections remain independent while sharing completed note compressions."""

from threading import Event

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from .test_app import catalog, wait
from .test_connections import settings_for
from .test_note_compression_cache import SOURCE, job
from .test_workup import WorkupInference

from matchminer_ai.apps.guideline_fetcher import app as app_module
from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Inference


def selection(task, thinking=False):
    return {"task": task, "endpoint": f"http://{task}.invalid/v1",
            "model": "Qwen/Qwen3.5-4B", "reasoning": thinking}


def assert_config(config, task, thinking):
    from matchminer_ai.llm.backends import build_llm_runtime_config
    from matchminer_ai.llm.remote_inference import build_remote_request_config

    assert config.remote["server_urls"] == [f"http://{task}.invalid/v1"]
    runtime = build_llm_runtime_config("patient", config.patient, config=config)
    params, extra = build_remote_request_config(runtime)
    assert params["temperature"] == (1.0 if thinking else 0.7)
    assert params["top_p"] == (0.95 if thinking else 0.8)
    assert params["presence_penalty"] == 1.5
    assert extra["chat_template_kwargs"] == {"enable_thinking": thinking}
    assert extra["top_k"] == 20 and extra["min_p"] == 0.0
    assert extra["repetition_penalty"] == 1.0
    assert "reasoning_effort" not in params and "reasoning_effort" not in extra


@pytest.mark.parametrize("thinking", [False, True])
def test_each_task_routes_to_its_endpoint_and_reuses_compression_after_other_changes(
    tmp_path, monkeypatch, thinking
):
    from matchminer_ai import patients

    catalog(tmp_path)
    calls = []

    def compress(frame, **kwargs):
        assert_config(kwargs["config"], "compress", thinking)
        assert kwargs["thinking"] == ("on" if thinking else "off")
        assert kwargs["max_concurrent_requests"] == 8
        calls.append("compress")
        return frame.assign(compressed_note_text="CT done.")

    def summarize(frame, **kwargs):
        assert_config(kwargs["config"], "summarize", thinking)
        assert frame.note_text.tolist() == ["CT done.", "CT done."]
        calls.append("summarize")
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    def review(frame, items, **kwargs):
        assert_config(kwargs["config"], "workup", thinking)
        assert frame.note_text.tolist() == ["CT done.", "CT done."]
        if "thinking" in kwargs:
            assert kwargs["thinking"] is None  # inherit the selected task mode
        calls.append("workup")
        return {"assessments": [], "metadata": {}, "notice": "Fabricated review."}

    monkeypatch.setattr(patients, "compress_patient_notes", compress)
    monkeypatch.setattr(patients, "summarize_patients", summarize)
    monkeypatch.setattr(patients, "review_patient_workup", review)
    monkeypatch.setattr(patients, "review_patient_workup_with_note_search", review)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    with TestClient(create_app(settings_for(tmp_path))) as client:
        config = client.post("/api/config", json=selection("compress", thinking)).json()
        assert config["tasks"]["summarize"]["model"] == "default-model"
        job(client, "/api/compress", SOURCE)
        saved = client.get("/api/note-compression").json()["note_compression"]
        for task in ("summarize", "workup"):
            updated = client.post("/api/config", json=selection(task, thinking)).json()
            assert updated["tasks"][task]["reasoning"] is thinking
            assert updated["tasks"][task]["sampling"]["temperature"] == (1.0 if thinking else 0.7)
            assert client.get("/api/note-compression").json()["note_compression"] == saved
        job(client, "/api/summarize", {**SOURCE, "note_source": "compressed"})
        retrieval = job(client, "/api/retrieve", {"summary": "Fabricated summary"})["id"]
        for method in ("full_notes", "agentic"):
            job(client, "/api/workup", {**SOURCE, "note_source": "compressed",
                "method": method, "retrieval_job_id": retrieval, "space_index": 0})
        assert calls == ["compress", "summarize", "workup", "workup"]
        assert client.get("/api/note-compression").json()["note_compression"] == saved
        # An unchanged compression configuration is safe to retain.
        client.post("/api/config", json=selection("compress", thinking))
        assert client.get("/api/note-compression").json()["note_compression"] == saved
        client.post("/api/config", json=selection("compress", not thinking))
        assert client.get("/api/note-compression").json()["note_compression"] is None
        job(client, "/api/summarize", {**SOURCE, "note_source": "compressed"}, status="failed")
        assert calls == ["compress", "summarize", "workup", "workup"]


def test_task_settings_are_private_and_running_job_keeps_submitted_endpoint_and_thinking(
    tmp_path, monkeypatch
):
    from matchminer_ai import patients

    entered, release = Event(), Event()
    configs = []

    def summarize(frame, **kwargs):
        configs.append(kwargs["config"])
        entered.set()
        assert release.wait(5)
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    monkeypatch.setattr(patients, "summarize_patients", summarize)
    with TestClient(create_app(settings_for(tmp_path))) as a, TestClient(a.app) as b:
        defaults = b.get("/api/config").json()["tasks"]
        assert all(task["reasoning"] is False for task in defaults.values())
        a.post("/api/config", json=selection("summarize", True))
        try:
            key = a.post("/api/summarize", json=SOURCE).json()["job_id"]
            assert entered.wait(5)
            a.post("/api/config", json={**selection("summarize"), "endpoint": "changed.invalid"})
            a.post("/api/config", json=selection("workup", True))
        finally:
            release.set()
        assert wait(a, key).json()["status"] == "complete"
        assert_config(configs[0], "summarize", True)
        assert b.get("/api/config").json()["tasks"] == defaults
        assert a.get("/api/config").json()["tasks"]["compress"] == defaults["compress"]
        assert a.post("/api/config", json={**selection("summarize"), "task": "invalid"}).status_code == 422


def test_discovery_uses_selected_task_without_modifying_other_tasks(tmp_path, monkeypatch):
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.5-4B"}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda **kwargs:
        original(transport=httpx.MockTransport(handle), **kwargs))
    with TestClient(create_app(settings_for(tmp_path))) as client:
        client.post("/api/config", json=selection("compress"))
        before = client.get("/api/config").json()
        result = client.post("/api/endpoint-check", json={"task": "compress"}).json()
        assert result["available"] is True
        assert client.get("/api/config").json() == before
    assert len(seen) == 1 and str(seen[0].url) == "http://compress.invalid/v1/models"
    assert seen[0].method == "GET" and not seen[0].content
