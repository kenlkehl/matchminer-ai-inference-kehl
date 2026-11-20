import threading

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from .test_app import catalog, wait
from .test_connections import settings_for

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Inference
from matchminer_ai.apps.guideline_fetcher.settings import GEMINI4_PROFILE, GEMINI_PROFILE, GEMINI_PROJECT, LLM_TASKS, Settings


def gemini_selection():
    # Even valid browser-supplied destinations must not redirect Google ADC.
    return {"llm_backend": "gemini_vertex", "endpoint": "http://untrusted.test/v1",
            "model": "untrusted-model", "tokenizer": "untrusted-tokenizer"}


def assert_gemini(config, profile=GEMINI_PROFILE):
    from matchminer_ai.llm.backends import build_llm_runtime_config
    from matchminer_ai.llm.structured import resolve_structured_config

    runtime = build_llm_runtime_config("patient", config.patient, config=config)
    assert runtime["provider"] == "google_agent_platform"
    assert runtime["google_project_id"] == GEMINI_PROJECT
    assert runtime["server_urls"] == [profile["endpoint"]]
    assert runtime["model_name"] == profile["model"]
    assert runtime["tokenizer_name"] == profile["tokenizer"]
    assert runtime["max_model_len"] == runtime["context_window"] == 1048576
    assert runtime["sampling_params"] == {"max_tokens": 65536, "reasoning_effort": "high"}
    assert runtime["remote"]["extra_body"] == {}
    assert runtime["chat_template_kwargs"] == {}
    structured, _ = resolve_structured_config(runtime, cache_dir=None)
    assert structured.provider == "google_agent_platform"
    assert structured.tokenizer_mode == "bytes"
    assert structured.max_tokens == 65536
    assert structured.request_params == {"reasoning_effort": "high"}


def test_gemini_summarization_keeps_snapshot_and_session_isolation(tmp_path, monkeypatch):
    import matchminer_ai.patients

    entered, release = threading.Event(), threading.Event()
    configs = []

    def summarize(frame, *, config):
        configs.append(config)
        entered.set()
        assert release.wait(10)
        return pd.DataFrame([{"cancer_history_summary": "Synthetic summary",
                              "general_exclusion_criteria_evidence": ""}])

    monkeypatch.setattr(matchminer_ai.patients, "summarize_patients", summarize)
    with TestClient(create_app(settings_for(tmp_path))) as a, TestClient(a.app) as b:
        response = a.post("/api/config", json=gemini_selection())
        assert response.status_code == 200
        assert response.json()["endpoint"] == GEMINI_PROFILE["endpoint"]
        assert b.get("/api/config").json()["llm_backend"] == "local"
        try:
            job = a.post("/api/summarize", json={"text": "Fabricated note"}).json()["job_id"]
            assert entered.wait(10)
            a.post("/api/config", json={"endpoint": "localhost:8000", "model": "local-model"})
        finally:
            release.set()
        assert wait(a, job).json()["status"] == "complete"
        assert_gemini(configs[0])
        assert a.get("/api/config").json()["llm_backend"] == "local"


@pytest.mark.parametrize("method,api_name", [
    ("full_notes", "review_patient_workup"),
    ("agentic", "review_patient_workup_with_note_search"),
])
def test_gemini_workup_uses_raw_notes_and_same_provider(tmp_path, monkeypatch, method, api_name):
    import matchminer_ai.patients

    catalog(tmp_path)
    captured = []
    monkeypatch.setattr(Inference, "retrieve", lambda *args: {
        "summary": "Synthetic summary", "spaces": [{"clinical_space_summary": "Population",
            "diagnostic_workup": [{"name": "Synthetic workup"}]}],
    })

    def review(notes, recommendations, **kwargs):
        captured.append((notes, kwargs["config"]))
        return {"assessments": []}

    monkeypatch.setattr(matchminer_ai.patients, api_name, review)
    with TestClient(create_app(settings_for(tmp_path))) as client:
        client.post("/api/config", json=gemini_selection())
        key = client.post("/api/retrieve", json={"summary": "Synthetic summary"}).json()["job_id"]
        wait(client, key)
        job = client.post("/api/workup", json={"text": "Raw fabricated notes",
            "retrieval_job_id": key, "space_index": 0, "method": method}).json()["job_id"]
        assert wait(client, job).json()["status"] == "complete"
    assert captured[0][0] == "Raw fabricated notes"
    assert_gemini(captured[0][1])


def test_google_check_has_fixed_destination_and_does_not_save(tmp_path, monkeypatch):
    calls = []

    def check(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return [(endpoint, 200, "OK")]

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.check_openai_endpoint", check)
    with TestClient(create_app(settings_for(tmp_path))) as client:
        result = client.post("/api/endpoint-check", json={
            "llm_backend": "gemini_vertex", "endpoint": "http://untrusted.test",
            "model": "untrusted-model",
        }).json()
        assert result["available"] is True
        assert result["models"] == [GEMINI_PROFILE["model"]]
        assert client.get("/api/config").json()["llm_backend"] == "local"
    assert calls == [(GEMINI_PROFILE["endpoint"], {
        "provider": "google_agent_platform", "google_project_id": GEMINI_PROJECT,
        "model_name": GEMINI_PROFILE["model"], "timeout": 30,
    })]


def test_google_auth_failure_is_sanitized(tmp_path, monkeypatch):
    def check(*args, **kwargs):
        raise RuntimeError("secret-token and internal credential path")

    monkeypatch.setattr("matchminer_ai.llm.vllm_server.check_openai_endpoint", check)
    with TestClient(create_app(settings_for(tmp_path))) as client:
        result = client.post("/api/endpoint-check", json={"llm_backend": "gemini_vertex"})
        assert result.json()["available"] is False
        assert "secret-token" not in result.text
        assert "Application Default Credentials" in result.text
        assert client.post("/api/config", json={**gemini_selection(),
                                               "llm_backend": "unknown"}).status_code == 422


def test_default_gemini4_applies_to_every_task_and_matches_display(tmp_path, monkeypatch):
    # Default since 2026-11-14 (user: the Gemma 4 26B MaaS endpoint "is very janky").
    monkeypatch.delenv("MMAI_LLM_BACKEND", raising=False)
    settings = Settings(device="cpu", catalog_root=tmp_path)
    for task in LLM_TASKS:
        selected = settings.for_task(task)
        assert_gemini(selected.inference_config(), GEMINI4_PROFILE)
        assert selected.backup_enabled is False
    with TestClient(create_app(settings)) as client:
        displayed = client.get("/api/config").json()
        assert set(displayed["tasks"]) == set(LLM_TASKS)
        for connection in (displayed, *displayed["tasks"].values()):
            assert connection["llm_backend"] == "gemini4_vertex"
            assert all(connection[key] == value for key, value in GEMINI4_PROFILE.items())
            assert connection["sampling"] == {}
            assert connection["backup_enabled"] is False


def test_operator_gemini_default_matches_display_and_local_remains_selectable(tmp_path, monkeypatch):
    monkeypatch.setenv("MMAI_LLM_BACKEND", "gemini_vertex")
    settings = Settings(device="cpu", catalog_root=tmp_path)
    assert_gemini(settings.inference_config())
    with TestClient(create_app(settings)) as client:
        displayed = client.get("/api/config").json()
        assert displayed["llm_backend"] == "gemini_vertex"
        assert all(displayed[key] == value for key, value in GEMINI_PROFILE.items())
        local = client.post("/api/config", json={
            "llm_backend": "local", **displayed["local_defaults"],
        }).json()
        assert local["llm_backend"] == "local"
        assert local["endpoint"] == settings.endpoint
        assert local["model"] == settings.model


def test_operator_can_set_local_default(monkeypatch):
    monkeypatch.setenv("MMAI_LLM_BACKEND", "local")
    config = Settings(device="cpu").inference_config()
    assert config.remote.get("provider", "openai") == "openai"
    assert config.patient["reasoning_effort"] == "xhigh"
