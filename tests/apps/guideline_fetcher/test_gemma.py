"""Selectable Gemma MaaS uses a fixed Vertex destination and shared inference defaults."""

import pytest
from fastapi.testclient import TestClient

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Inference
from matchminer_ai.apps.guideline_fetcher.settings import GEMMA_PROFILE, GEMINI_PROJECT, LLM_TASKS, Settings
from .test_app import catalog, wait
from .test_connections import settings_for


def selection():
    return {
        "llm_backend": "gemma_vertex",
        "endpoint": "http://untrusted.test/v1",
        "model": "untrusted-model",
        "tokenizer": "untrusted-tokenizer",
    }


def assert_gemma(config):
    from matchminer_ai.llm.backends import build_llm_runtime_config
    from matchminer_ai.llm.structured import resolve_structured_config

    runtime = build_llm_runtime_config("patient", config.patient, config=config)
    resolved, _ = resolve_structured_config(runtime, cache_dir=None)
    assert resolved.provider == "google_agent_platform"
    assert resolved.google_project_id == GEMINI_PROJECT
    assert resolved.base_url == GEMMA_PROFILE["endpoint"]
    assert resolved.model == GEMMA_PROFILE["model"]
    assert resolved.context_window == 262144 and resolved.max_tokens == 65536
    assert resolved.tokenizer_mode == "bytes"
    assert resolved.thinking == "off" and resolved.top_k == 64
    assert resolved.temperature == 1.0 and resolved.top_p == 0.95
    assert "reasoning_effort" not in resolved.request_params
    assert resolved.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert resolved.request_start_interval_seconds == 2.0
    assert resolved.capacity_retry_initial_seconds == 15.0
    assert resolved.capacity_retry_max_seconds == 60.0


def test_gemma_selection_is_fixed_and_session_scoped(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as a, TestClient(a.app) as b:
        result = a.post("/api/config", json=selection())
        assert result.status_code == 200
        assert all(result.json()[key] == value for key, value in GEMMA_PROFILE.items())
        assert a.get("/api/config").json()["llm_backend"] == "gemma_vertex"
        assert b.get("/api/config").json()["llm_backend"] == "local"
        assert result.json()["vertex_profiles"]["gemma_vertex"]["model"] == GEMMA_PROFILE["model"]


@pytest.mark.parametrize(
    "method,api_name",
    [
        ("full_notes", "review_patient_workup"),
        ("agentic", "review_patient_workup_with_note_search"),
    ],
)
def test_gemma_workup_uses_selected_model(tmp_path, monkeypatch, method, api_name):
    import matchminer_ai.patients

    catalog(tmp_path)
    configs = []
    monkeypatch.setattr(
        Inference,
        "retrieve",
        lambda *args: {
            "summary": "Synthetic summary",
            "spaces": [
                {
                    "clinical_space_summary": "Population",
                    "diagnostic_workup": [{"name": "Synthetic workup"}],
                }
            ],
        },
    )

    def review(notes, recommendations, **kwargs):
        assert notes == "Raw fabricated notes"
        configs.append(kwargs["config"])
        return {"assessments": []}

    monkeypatch.setattr(matchminer_ai.patients, api_name, review)
    with TestClient(create_app(settings_for(tmp_path))) as client:
        client.post("/api/config", json=selection())
        key = client.post("/api/retrieve", json={"summary": "Synthetic summary"}).json()["job_id"]
        wait(client, key)
        key = client.post(
            "/api/workup",
            json={
                "text": "Raw fabricated notes",
                "retrieval_job_id": key,
                "space_index": 0,
                "method": method,
            },
        ).json()["job_id"]
        assert wait(client, key).json()["status"] == "complete"
    assert_gemma(configs[0])


def test_gemma_connection_ignores_browser_destination(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "matchminer_ai.llm.vllm_server.check_openai_endpoint",
        lambda endpoint, **kwargs: calls.append((endpoint, kwargs)) or [(endpoint, 200, "OK")],
    )
    with TestClient(create_app(settings_for(tmp_path))) as client:
        result = client.post(
            "/api/endpoint-check",
            json={
                "llm_backend": "gemma_vertex",
                "endpoint": "http://untrusted.test",
                "model": "bad",
            },
        ).json()
        assert result["available"] and result["models"] == [GEMMA_PROFILE["model"]]
        assert client.get("/api/config").json()["llm_backend"] == "local"
    assert calls == [
        (
            GEMMA_PROFILE["endpoint"],
            {
                "provider": "google_agent_platform",
                "google_project_id": GEMINI_PROJECT,
                "model_name": GEMMA_PROFILE["model"],
                "timeout": 30,
            },
        )
    ]


def test_operator_can_select_gemma_as_default(monkeypatch):
    monkeypatch.setenv("MMAI_LLM_BACKEND", "gemma_vertex")
    assert_gemma(Settings(device="cpu").inference_config())


def test_operator_gemma_applies_to_every_task_and_matches_display(tmp_path, monkeypatch):
    # Gemma was the default until 2026-11-14; Gemini 4 Hydrogen is now (see test_gemini).
    monkeypatch.setenv("MMAI_LLM_BACKEND", "gemma_vertex")
    settings = Settings(device="cpu", catalog_root=tmp_path)
    for task in LLM_TASKS:
        selected = settings.for_task(task)
        assert_gemma(selected.inference_config())
        assert selected.backup_enabled is False
    with TestClient(create_app(settings)) as client:
        displayed = client.get("/api/config").json()
        assert set(displayed["tasks"]) == set(LLM_TASKS)
        for connection in (displayed, *displayed["tasks"].values()):
            assert connection["llm_backend"] == "gemma_vertex"
            assert all(connection[key] == value for key, value in GEMMA_PROFILE.items())
            assert connection["reasoning"] is False
            assert connection["backup_enabled"] is False
