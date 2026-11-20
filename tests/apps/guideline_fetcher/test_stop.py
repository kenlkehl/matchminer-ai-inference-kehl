"""Server-owned cancellation survives refresh and retains workup inputs."""

import json
from threading import Event, Lock
import time

from fastapi.testclient import TestClient
import pytest

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.jobs import Jobs
from matchminer_ai.apps.guideline_fetcher.services import Inference
from matchminer_ai.apps.guideline_fetcher.settings import Settings
from .test_app import FakeInference, catalog, wait


def stopped(client, key):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{key}").json()
        if job["status"] == "cancelled":
            return job
        time.sleep(0.02)
    pytest.fail("Stop did not release the inference worker")


def test_queued_job_stop_never_invokes_callback():
    jobs, release, started = Jobs(), Event(), Event()
    calls = []
    jobs.submit("first", "retrieve", lambda _: started.set() or release.wait(3))
    try:
        assert started.wait(1)
        second = jobs.submit("second", "workup", lambda _: calls.append(1))
        assert jobs.cancel("another", second) is None
        assert jobs.cancel("second", second)["status"] == "cancelled"
        assert jobs.latest("second")["status"] == "cancelled"
        jobs.clear("second")
        assert jobs.get("second", second) is None
        assert not calls
    finally:
        release.set()
        jobs.shutdown()


def test_stop_is_session_owned_idempotent_and_refreshable(tmp_path):
    from matchminer_ai.cancellation import cancel_sleep

    catalog(tmp_path)
    inference = FakeInference()
    started = Event()

    def slow(*args):
        started.set()
        cancel_sleep(30)
        pytest.fail("Cancelled summarizer continued")

    inference.summarize = slow
    app = create_app(Settings(catalog_root=tmp_path), inference)
    with TestClient(app) as client, TestClient(app) as other:
        key = client.post("/api/summarize", json={"text": "Synthetic notes"}).json()["job_id"]
        assert started.wait(1)
        assert client.get("/api/session").json()["job"]["id"] == key
        assert other.post(f"/api/jobs/{key}/stop").status_code == 404
        assert (
            client.post(
                f"/api/jobs/{key}/stop", headers={"Origin": "https://outside.invalid"}
            ).status_code
            == 403
        )
        assert client.post(f"/api/jobs/{key}/stop").status_code == 200
        assert stopped(client, key)["result"] is None
        assert client.post(f"/api/jobs/{key}/stop").json()["status"] == "cancelled"
        assert client.get("/api/session").json()["job"]["status"] == "cancelled"
        inference.summarize = FakeInference().summarize
        next_key = client.post("/api/summarize", json={"text": "Synthetic notes"}).json()["job_id"]
        assert wait(client, next_key).json()["status"] == "complete"


def test_agentic_stop_closes_all_question_repls_and_preserves_inputs(tmp_path, monkeypatch):
    from matchminer_ai.llm.structured import StructuredClient
    from matchminer_ai.patients import _note_repl

    catalog(tmp_path)
    settings = Settings(
        llm_backend="local",
        catalog_root=tmp_path,
        device="cpu",
        endpoint="http://synthetic.invalid/v1",
        model="synthetic",
    )
    original_config = Settings.inference_config

    def config(self):
        runtime = original_config(self)
        runtime.patient.update(tokenizer_mode="bytes", context_window=32768)
        runtime.patient["remote"].setdefault("request_params", {})["max_tokens"] = 1024
        return runtime

    monkeypatch.setattr(Settings, "inference_config", config)
    retrieval = {
        "summary": "Synthetic summary",
        "spaces": [
            {
                "name": "Synthetic population",
                "diagnostic_workup": [{"name": name} for name in ["CT", "ECG", "LDH"]],
            }
        ],
    }
    monkeypatch.setattr(Inference, "retrieve", lambda *args: retrieval)
    all_waiting, release = Event(), Event()
    lock, processes, calls = Lock(), [], []
    original_popen = _note_repl.subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(_note_repl.subprocess, "Popen", popen)

    def http(self, endpoint, body=None, **kwargs):
        payload = json.loads(body["messages"][1]["content"].rsplit("Request:\n", 1)[-1])
        if "last_cell_result" not in payload:
            value = {"terms": ["scan"]}
        elif payload["search_only"]:
            value = dict(
                action="python",
                code=["notes"],
                memory="",
                status="unknown",
                answer={},
                limitations=[],
                needs_review=False,
            )
        else:
            with lock:
                calls.append(1)
                if len(calls) == 3:
                    all_waiting.set()
            assert release.wait(10)
            value = {}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(StructuredClient, "_http_request", http)
    notes = [
        {"note_text": "Fabricated CT ordered.", "note_date": "2026-01-01", "note_type": "Oncology"}
    ]
    with TestClient(create_app(settings)) as client:
        key = client.post("/api/retrieve", json={"summary": "Synthetic summary"}).json()["job_id"]
        wait(client, key)
        key2 = client.post(
            "/api/workup",
            json={
                "text": "Fabricated CT ordered.",
                "notes": notes,
                "retrieval_job_id": key,
                "space_index": 0,
                "method": "agentic",
            },
        ).json()["job_id"]
        try:
            assert all_waiting.wait(6)
            assert client.post(f"/api/jobs/{key2}/stop").status_code == 200
            job = stopped(client, key2)
            assert job["result"] is None
            assert job["restore"]["notes_input"]["notes"] == notes
            assert job["restore"]["retrieval"] == retrieval
            assert all(p.poll() is not None for p in processes)
            assert len(calls) == 3
            assert client.get("/api/session").json()["job"]["status"] == "cancelled"
        finally:
            release.set()
