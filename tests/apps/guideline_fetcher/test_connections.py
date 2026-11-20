import threading

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from matchminer_ai.apps.guideline_fetcher import app as app_module
from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Inference
from matchminer_ai.apps.guideline_fetcher.settings import Settings, normalize_endpoint
from .test_app import catalog, wait


@pytest.mark.parametrize("address,expected", [
    ("gpu-server:8060", "http://gpu-server:8060/v1"),
    (" http://localhost:8000/ ", "http://localhost:8000/v1"),
    ("https://example.test/prefix/v1/", "https://example.test/prefix/v1"),
    ("http://[::1]:8000", "http://[::1]:8000/v1"),
])
def test_endpoint_normalization(address, expected):
    assert normalize_endpoint(address) == expected


@pytest.mark.parametrize("address", [
    "", "file:///etc/passwd", "http://", "http://host:99999",
    "http://user:secret@host/v1", "http://host/v1?key=secret",
    "http://host/#fragment", "http://some host/v1",
])
def test_invalid_endpoint_rejected(address):
    with pytest.raises(ValueError):
        normalize_endpoint(address)


def settings_for(tmp_path):
    return Settings(llm_backend="local", catalog_root=tmp_path, device="cpu", endpoint="http://default:8000/v1",
                    model="default-model", tokenizer="default-tokenizer")


def test_settings_are_session_scoped_and_jobs_keep_their_snapshot(tmp_path, monkeypatch):
    import matchminer_ai.patients

    entered, release = threading.Event(), threading.Event()
    configs = []

    def summarize(frame, *, config):
        configs.append(config)
        entered.set()
        assert release.wait(5)
        return pd.DataFrame([{"cancer_history_summary": "Synthetic summary",
                              "general_exclusion_criteria_evidence": ""}])

    monkeypatch.setattr(matchminer_ai.patients, "summarize_patients", summarize)
    app = create_app(settings_for(tmp_path))
    with TestClient(app) as a, TestClient(app) as b:
        assert a.post("/api/config", json={"endpoint": "gpu-server:8060", "model": "first-model",
                       "tokenizer": "matching-tokenizer"}).status_code == 200
        assert a.get("/api/config").json()["model"] == "first-model"
        assert b.get("/api/config").json()["model"] == "default-model"
        try:
            first = a.post("/api/summarize", json={"text": "Fabricated note"}).json()["job_id"]
            assert entered.wait(5)
            a.post("/api/config", json={"endpoint": "other:8000", "model": "second-model"})
            second = b.post("/api/summarize", json={"text": "Other fabricated note"}).json()["job_id"]
        finally:
            release.set()
        assert wait(a, first).json()["status"] == "complete"
        assert wait(b, second).json()["status"] == "complete"
        assert configs[0].remote["server_urls"] == ["http://gpu-server:8060/v1"]
        assert configs[0].patient["remote"]["model_name"] == "first-model"
        assert configs[0].patient["remote"]["tokenizer_name"] == "matching-tokenizer"
        assert configs[0].patient["reasoning_effort"] == "xhigh"
        assert configs[0].patient["sampling_profile"] == "auto"
        assert configs[1].remote["server_urls"] == ["http://default:8000/v1"]
        assert a.get("/api/config").json()["tokenizer"] == "second-model"
        assert b.get("/api/jobs/" + first).status_code == 404


def test_workup_uses_saved_connection_with_raw_notes(tmp_path, monkeypatch):
    import matchminer_ai.patients

    catalog(tmp_path)
    captured = []
    monkeypatch.setattr(Inference, "retrieve", lambda *args: {
        "summary": "Synthetic summary", "spaces": [{"clinical_space_summary": "Population",
            "diagnostic_workup": [{"name": "Synthetic workup"}]}],
    })

    def review(notes, recommendations, **kwargs):
        captured.append((notes, recommendations, kwargs["config"]))
        return {"assessments": []}

    monkeypatch.setattr(matchminer_ai.patients, "review_patient_workup", review)
    with TestClient(create_app(settings_for(tmp_path))) as client:
        client.post("/api/config", json={"endpoint": "gpu-server:8060", "model": "served-model"})
        key = client.post("/api/retrieve", json={"summary": "Synthetic summary"}).json()["job_id"]
        wait(client, key)
        job = client.post("/api/workup", json={"text": "Raw fabricated notes",
            "retrieval_job_id": key, "space_index": 0}).json()["job_id"]
        assert wait(client, job).json()["status"] == "complete"
    assert captured[0][0] == "Raw fabricated notes"
    assert captured[0][2].remote["server_urls"] == ["http://gpu-server:8060/v1"]
    remote = captured[0][2].patient["remote"]
    assert remote["model_name"] == "served-model"
    assert remote["tokenizer_name"] == "served-model"


@pytest.mark.parametrize("payload,expected", [
    ({"data": [{"id": "model-a"}, {"id": "model-b"}, {"id": "model-a"}]}, ["model-a", "model-b"]),
    ({"data": []}, []),
    ({"data": [{"id": None}, "bad", {"id": ""}]}, []),
    ({"data": None}, []),
])
def test_discovery_uses_entered_endpoint_without_saving_or_sending_notes(tmp_path, monkeypatch,
                                                                      payload, expected):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    original = httpx.AsyncClient
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda **kwargs:
        original(transport=httpx.MockTransport(handle), **kwargs))
    with TestClient(create_app(settings_for(tmp_path))) as client:
        response = client.post("/api/endpoint-check", json={"endpoint": "gpu-server:8060",
                                                          "model": "model-b"})
        result = response.json()
        assert result["models"] == expected
        assert result["model"] == ("model-b" if expected else None)
        assert result["available"] == bool(expected)
        assert client.get("/api/config").json()["model"] == "default-model"
        if expected:
            result = client.post("/api/endpoint-check", json={"endpoint": "gpu-server:8060"}).json()
            assert result["model"] == "model-a"
    assert all(str(r.url) == "http://gpu-server:8060/v1/models" for r in calls)
    assert all(r.method == "GET" and not r.content for r in calls)


def test_discovery_failure_does_not_echo_provider_details(tmp_path, monkeypatch):
    def fail(request):
        raise httpx.ConnectError("INTERNAL PROVIDER DETAIL", request=request)

    original = httpx.AsyncClient
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda **kwargs:
        original(transport=httpx.MockTransport(fail), **kwargs))
    with TestClient(create_app(settings_for(tmp_path))) as client:
        response = client.post("/api/endpoint-check", json={})
        assert response.json()["models"] == []
        assert "INTERNAL PROVIDER DETAIL" not in response.text


def test_config_rejects_invalid_fields_and_cross_origin_writes(tmp_path):
    with TestClient(create_app(settings_for(tmp_path))) as client:
        body = {"endpoint": "gpu-server:8060", "model": "new-model"}
        assert client.post("/api/config", json={**body, "model": "   "}).status_code == 422
        assert client.post("/api/config", json={**body, "api_key": "secret"}).status_code == 422
        assert client.post("/api/config", json={**body, "endpoint": "file:///tmp"}).status_code == 422
        assert client.post("/api/config", json=body,
                           headers={"Origin": "https://outside.invalid"}).status_code == 403
        assert client.get("/api/config").json()["model"] == "default-model"
