"""Session reuse keeps source identity and optional raw-note workflows intact."""

from copy import deepcopy

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from .test_app import catalog, wait
from .test_workup import WorkupInference

from matchminer_ai.apps.guideline_fetcher.app import SESSION_COOKIES, create_app
from .conftest import mode_settings
from matchminer_ai.apps.guideline_fetcher.note_compressions import NoteCompressions
from matchminer_ai.apps.guideline_fetcher.services import Inference
from matchminer_ai.apps.guideline_fetcher.settings import Settings

SOURCE = {
    "text": "Fabricated longitudinal record",
    "notes": [
        {
            "note_text": "Computed tomography completed.",
            "note_date": "2026-01-01",
            "note_type": "Radiology",
        },
        {"note_text": "No fever documented.", "note_date": None},
    ],
}


@pytest.fixture(params=[False, True], ids=["standard", "phi"])
def compressed_app(tmp_path, monkeypatch, request):
    from matchminer_ai import patients

    catalog(tmp_path)
    settings = mode_settings(
        request.param,
        catalog_root=tmp_path,
        llm_backend="local",
        device="cpu",
        endpoint="http://fabricated.invalid/v1",
        model="fabricated-model",
    )
    calls = {"compression": [], "summary": [], "review": []}

    def compress(frame, **kwargs):
        calls["compression"].append(frame.to_dict("records"))
        kwargs["progress_callback"](len(frame), len(frame))
        return frame.assign(
            compressed_note_text=[f"Compressed {n + 1}." for n in range(len(frame))]
        )

    def summarize(frame, **kwargs):
        calls["summary"].append(frame.note_text.tolist())
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    def review(frame, items, **kwargs):
        calls["review"].append(
            frame.note_text.tolist() if isinstance(frame, pd.DataFrame) else frame
        )
        return {
            "assessments": [],
            "notice": "Fabricated review.",
            "metadata": {"note_count": len(frame), "requests": 1},
        }

    monkeypatch.setattr(patients, "compress_patient_notes", compress)
    monkeypatch.setattr(patients, "summarize_patients", summarize)
    monkeypatch.setattr(patients, "review_patient_workup", review)
    monkeypatch.setattr(patients, "review_patient_workup_with_note_search", review)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    return create_app(settings), calls, settings


def job(client, route, body, status="complete"):
    response = client.post(route, json=body)
    assert response.status_code == 202, response.text
    done = wait(client, response.json()["job_id"]).json()
    assert done["status"] == status, done["progress"]
    return done


@pytest.mark.parametrize("dated", [False, True])
def test_standalone_compression_survives_retrieval_and_is_optional_for_both_reviews(
    compressed_app, dated
):
    api, calls, _ = compressed_app
    source = SOURCE if dated else {"text": "Computed tomography completed.", "notes": None}
    compressed = ["Compressed 1.", "Compressed 2."] if dated else ["Compressed 1."]
    with TestClient(api) as client:
        result = job(client, "/api/compress", source)["result"]
        assert result["compression"]["reused"] is False
        assert calls["summary"] == [] and calls["review"] == []
        job(client, "/api/summarize", {**source, "note_source": "compressed"})
        saved = client.get("/api/note-compression").json()["note_compression"]
        assert saved["notes_input"] == source
        assert [row["compressed_note_text"] for row in saved["notes"]] == compressed
        assert saved["model"] == "fabricated-model"
        if dated:
            assert [
                {k: v for k, v in row.items() if k != "compressed_note_text"}
                for row in saved["notes"]
            ] == SOURCE["notes"]
        retrieval = job(client, "/api/retrieve", {"summary": "Edited fabricated summary"})["id"]
        assert client.get("/api/session").json()["note_compression"] == saved
        for method in ("full_notes", "agentic"):
            payload = {**source, "retrieval_job_id": retrieval, "space_index": 0, "method": method}
            reused = job(client, "/api/workup", {**payload, "note_source": "compressed"})["result"]
            assert reused["review"]["metadata"]["compression"]["reused"] is True
            assert calls["review"][-1] == compressed
            assert reused["notes_input"] == source
            raw = job(client, "/api/workup", {**payload, "note_source": "original"})["result"]
            assert "compression" not in raw["review"]["metadata"]
            assert calls["review"][-1] == (
                [row["note_text"] for row in SOURCE["notes"]] if dated else source["text"]
            )
            assert client.get("/api/note-compression").json()["note_compression"] == saved
        repeated = job(client, "/api/summarize", {**source, "note_source": "compressed"})["result"]
        assert repeated["compression"]["reused"] is True
        assert calls["summary"] == [compressed, compressed]
        assert len(calls["compression"]) == 1


@pytest.mark.parametrize("changed", ["text", "note_text", "note_date", "note_type", "order"])
def test_changed_patient_record_invalidates_and_recompresses(compressed_app, changed):
    api, calls, _ = compressed_app
    source = deepcopy(SOURCE)
    if changed == "text":
        source["text"] += " Another record."
    elif changed == "order":
        source["notes"].reverse()
    else:
        source["notes"][0][changed] += "-changed" if changed != "note_date" else "T12:00:00"
    with TestClient(api) as client:
        job(client, "/api/compress", SOURCE)
        failed = job(
            client, "/api/summarize", {**source, "note_source": "compressed"}, status="failed"
        )
        assert "Run Compress notes first" in failed["progress"]
        assert calls["summary"] == [] and len(calls["compression"]) == 1
        job(client, "/api/summarize", {**source, "note_source": "original"})
        assert client.get("/api/note-compression").json()["note_compression"] is None
        fresh = job(client, "/api/compress", source)["result"]
        assert fresh["compression"]["reused"] is False
        assert len(calls["compression"]) == 2
        assert client.get("/api/session").json()["note_compression"]["notes_input"] == source


def test_cache_is_session_private_and_clear_and_connection_changes_invalidate(compressed_app):
    api, calls, settings = compressed_app
    with TestClient(api) as client:
        job(client, "/api/compress", SOURCE)
        session_cookie = SESSION_COOKIES[settings.mode]
        cookie = client.cookies.get(session_cookie)
        assert cookie
        client.cookies.clear()
        assert client.get("/api/note-compression").json()["note_compression"] is None
        assert client.delete("/api/note-compression").status_code == 204
        client.cookies.clear()
        client.cookies.set(session_cookie, cookie)
        assert client.get("/api/note-compression").json()["note_compression"] is not None
        assert (
            client.post(
                "/api/config",
                json={
                    "llm_backend": "local",
                    "endpoint": "http://fabricated.invalid/v1",
                    "model": "new-model",
                },
            ).status_code
            == 200
        )
        assert client.get("/api/note-compression").json()["note_compression"] is None
        job(client, "/api/compress", SOURCE)
        assert (
            client.get("/api/note-compression").json()["note_compression"]["model"] == "new-model"
        )
        assert client.delete("/api/session").status_code == 204
        assert client.get("/api/session").json() == {"job": None, "note_compression": None}
        job(client, "/api/compress", SOURCE)
        assert len(calls["compression"]) == 3


def test_complete_compression_is_available_after_downstream_failure(compressed_app, monkeypatch):
    from matchminer_ai import patients

    api, calls, _ = compressed_app

    def fail(*args, **kwargs):
        raise RuntimeError("Fabricated downstream failure")

    monkeypatch.setattr(patients, "summarize_patients", fail)
    with TestClient(api) as client:
        job(client, "/api/compress", SOURCE)
        job(client, "/api/summarize", {**SOURCE, "note_source": "compressed"}, status="failed")
        assert client.get("/api/session").json()["note_compression"] is not None
        retrieval = job(client, "/api/retrieve", {"summary": "Fabricated summary"})["id"]
        job(
            client,
            "/api/workup",
            {
                **SOURCE,
                "note_source": "compressed",
                "retrieval_job_id": retrieval,
                "space_index": 0,
                "method": "agentic",
            },
        )
        assert len(calls["compression"]) == 1


def test_failed_compression_does_not_cache_partial_text(compressed_app, monkeypatch):
    from matchminer_ai import patients

    api, calls, _ = compressed_app

    def fail(*args, **kwargs):
        raise RuntimeError("Fabricated compression failure")

    monkeypatch.setattr(patients, "compress_patient_notes", fail)
    with TestClient(api) as client:
        job(client, "/api/compress", SOURCE, status="failed")
        assert client.get("/api/session").json()["note_compression"] is None
        assert calls["summary"] == []


@pytest.mark.parametrize("workflow", ["summarize", "full_notes", "agentic"])
@pytest.mark.parametrize("stale", [False, True])
def test_unavailable_compressed_notes_never_start_compression_or_clinical_inference(
    compressed_app, workflow, stale
):
    api, calls, _ = compressed_app
    with TestClient(api) as client:
        if stale:
            job(client, "/api/compress", SOURCE)
            client.delete("/api/note-compression")
        before = deepcopy(calls)
        payload = {**SOURCE, "note_source": "compressed"}
        route = "/api/summarize"
        if workflow != "summarize":
            retrieval = job(client, "/api/retrieve", {"summary": "Fabricated summary"})["id"]
            payload.update(retrieval_job_id=retrieval, space_index=0, method=workflow)
            route = "/api/workup"
        failed = job(client, route, payload, status="failed")
        assert "Run Compress notes first" in failed["progress"]
        assert calls == before
        assert client.get("/api/note-compression").json()["note_compression"] is None


def test_compression_job_preserves_current_summary_and_owned_retrieval(compressed_app):
    api, calls, _ = compressed_app
    with TestClient(api) as client:
        retrieval = job(client, "/api/retrieve", {"summary": "Fabricated summary"})
        payload = {
            **SOURCE,
            "summary": "Edited fabricated summary",
            "retrieval_job_id": retrieval["id"],
        }
        compression = job(client, "/api/compress", payload)
        assert compression["kind"] == "compress"
        assert compression["restore"]["summary"] == payload["summary"]
        assert compression["result"]["retrieval"] == retrieval["result"]
        assert compression["result"]["retrieval_job_id"] == retrieval["id"]
        assert client.get("/api/session").json()["job"] == compression
        assert calls["summary"] == [] and calls["review"] == []
        again = job(client, "/api/compress", payload)
        assert again["result"]["compression"]["reused"] is True
        assert len(calls["compression"]) == 1
        client.cookies.clear()
        assert client.post("/api/compress", json=payload).status_code == 400


def test_old_implicit_compression_argument_is_rejected(compressed_app):
    api, calls, _ = compressed_app
    with TestClient(api) as client:
        assert (
            client.post("/api/summarize", json={**SOURCE, "compress_notes": True}).status_code
            == 422
        )
        assert (
            client.post("/api/summarize", json={**SOURCE, "note_source": "unknown"}).status_code
            == 422
        )
        assert calls == {"compression": [], "summary": [], "review": []}


def test_independent_compression_keeps_matching_summary_provenance_and_workup(compressed_app):
    api, _, _ = compressed_app
    with TestClient(api) as client:
        summary = job(client, "/api/summarize", SOURCE)["result"]
        compression = job(client, "/api/compress", {**SOURCE, "summary": summary["summary"]})
        assert compression["restore"]["summary_result"] == summary
        retrieval = job(client, "/api/retrieve", {"summary": summary["summary"]})["id"]
        workup = job(
            client,
            "/api/workup",
            {
                **SOURCE,
                "retrieval_job_id": retrieval,
                "space_index": 0,
                "note_source": "compressed",
            },
        )["result"]
        compression = job(
            client,
            "/api/compress",
            {
                **SOURCE,
                "summary": summary["summary"],
                "retrieval_job_id": retrieval,
                "note_source": "original",
            },
        )
        assert compression["restore"]["workup_result"] == workup
        assert compression["restore"]["note_source"] == "original"


def test_cache_expiry_and_late_producer_cannot_restore_cleared_patient(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("matchminer_ai.apps.guideline_fetcher.note_compressions.time.monotonic", lambda: clock[0])
    config = Settings().inference_config()
    store = NoteCompressions(ttl=10)
    bound = store.bind("owner")
    bound.save(**SOURCE, config=config, compressed_notes=[], metadata={})
    assert store.latest("owner") is not None
    clock[0] += 11
    assert store.latest("owner") is None
    old = store.bind("owner")
    store.clear("owner")
    new = store.bind("owner")
    old.save(**SOURCE, config=config, compressed_notes=[], metadata={})
    assert new.lookup(**SOURCE, config=config) is None
    assert store.latest("owner") is None


def test_configuration_identity_and_snapshot_copies(compressed_app):
    _, _, settings = compressed_app
    config = settings.inference_config()
    store = NoteCompressions()
    bound = store.bind("owner")
    rows = [{**SOURCE["notes"][0], "compressed_note_text": "CT done."}]
    bound.save(**SOURCE, config=config, compressed_notes=rows, metadata={})
    rows[0]["compressed_note_text"] = "Changed outside cache"
    snapshot = bound.lookup(**SOURCE, config=config)
    assert snapshot["notes"][0]["compressed_note_text"] == "CT done."
    snapshot["notes"][0]["compressed_note_text"] = "Changed returned copy"
    assert store.latest("owner")["notes"][0]["compressed_note_text"] == "CT done."
    config.remote["server_urls"] = ["http://different.invalid/v1"]
    assert bound.lookup(**SOURCE, config=config) is None
    assert store.latest("owner") is None
