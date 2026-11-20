"""Compressed-note review retains original inputs and labels transformed evidence."""

import json
import re
from threading import Barrier, Event

import pytest
from fastapi.testclient import TestClient
from .test_app import catalog, wait
from .test_stop import stopped
from .test_workup import WorkupInference

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Inference
from matchminer_ai.apps.guideline_fetcher.settings import Settings


@pytest.mark.parametrize("dated", [False, True])
@pytest.mark.parametrize("model,thinking", [
    ("fabricated-model", False),
    ("Qwen/Qwen3.5-4B", False),
    ("Qwen/Qwen3.5-4B", True),
])
def test_compression_and_full_review_through_app_jobs(tmp_path, monkeypatch, dated, model, thinking):
    from matchminer_ai.llm.structured import StructuredClient
    from transformers import AutoTokenizer

    catalog(tmp_path)
    settings = Settings(
        catalog_root=tmp_path,
        llm_backend="local",
        device="cpu",
        endpoint="http://fabricated.invalid/v1",
        model=model, reasoning=thinking,
    )
    original_config = Settings.inference_config

    def config(self):
        value = original_config(self)
        value.remote.update(context_window=16384, tokenizer_mode="bytes", stream=False)
        value.patient["remote"]["request_params"] = {"max_tokens": 2048}
        return value

    monkeypatch.setattr(Settings, "inference_config", config)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)

    class Tokenizer:
        is_fast = True

        def __call__(self, text, **kwargs):
            return {"offset_mapping": [match.span() for match in re.finditer(r"\S+", text)]}

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    calls = []
    barrier = Barrier(2) if dated else None

    def respond(self, path, body=None, **kwargs):
        assert path == "/chat/completions"
        calls.append(body)
        if "Qwen3.5" in model:
            assert body["chat_template_kwargs"] == {"enable_thinking": thinking}
            assert body["temperature"] == (1.0 if thinking else 0.7)
            assert body["top_p"] == (0.95 if thinking else 0.8)
            assert body["top_k"] == 20 and body["min_p"] == 0.0
            assert body["presence_penalty"] == 1.5 and body["repetition_penalty"] == 1.0
            assert "reasoning_effort" not in body
        if body["messages"][0]["content"].startswith("Compress this note"):
            assert body["chat_template_kwargs"]["enable_thinking"] is thinking
            assert "reasoning_effort" not in body
            assert self.config.max_concurrent_requests == 8
            # With dated rows, both independent compression calls must overlap.
            if barrier:
                barrier.wait(timeout=3)
            text = body["messages"][1]["content"]
            assert "\n" not in text and "  " not in text
            compressed = "CT done." if "tomography" in text else "No fever."
        else:
            assert self.config.max_concurrent_requests == 1
            payload = json.loads(body["messages"][1]["content"])
            fragments = payload["raw_note_fragments"]
            assert fragments[0]["text"] == "CT done."
            assert "tomography" not in json.dumps(payload)
            compressed = json.dumps(
                {
                    "assessments": [
                        {
                            "name": "Imaging",
                            "applicability": "applies",
                            "status": "completed",
                            "bottom_line": "Imaging documented in the compressed note.",
                        }
                    ]
                }
            )
        return {"choices": [{"message": {"content": compressed}, "finish_reason": "stop"}]}

    monkeypatch.setattr(StructuredClient, "_http", respond)
    original_text = "Computed  tomography\n\nhas been completed."
    notes = (
        [
            {"note_text": "There is no fever.", "note_date": None, "note_type": "Visit"},
            {"note_text": original_text, "note_date": "2026-01-01", "note_type": "Radiology"},
        ]
        if dated
        else None
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/config").json()["note_compression_available"]
        compression = client.post("/api/compress", json={"text": original_text, "notes": notes})
        compressed = wait(client, compression.json()["job_id"]).json()
        assert compressed["status"] == "complete", compressed["progress"]
        assert compressed["kind"] == "compress"
        assert len(calls) == (2 if dated else 1)
        retrieval = client.post("/api/retrieve", json={"summary": "Fabricated summary"}).json()[
            "job_id"
        ]
        wait(client, retrieval)
        response = client.post(
            "/api/workup",
            json={
                "text": original_text,
                "notes": notes,
                "retrieval_job_id": retrieval,
                "space_index": 0,
                "method": "full_notes",
                "note_source": "compressed",
            },
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        done = wait(client, job_id).json()
        assert done["status"] == "complete", done["progress"]
        result = done["result"]
        assert result["notes_input"] == {"text": original_text, "notes": notes}
        assert result["method"] == "full_notes" and result["note_source"] == "compressed"
        assert client.get("/api/session").json()["job"]["result"] == result
        assert done["restore"]["notes_input"] == result["notes_input"]
        assert (
            done["restore"]["method"] == "full_notes"
            and done["restore"]["note_source"] == "compressed"
        )
        review = result["review"]
        assert review["metadata"]["method"] == "full_notes"
        assert review["metadata"]["evidence_selection"] == "automatic_reviewed_excerpts"
        assert review["metadata"]["scope"] == "all_supplied_compressed_notes"
        assert review["metadata"]["compression"]["thinking"] == ("on" if thinking else "off")
        assert review["metadata"]["compression"]["losslessness_verified"] is False
        assert review["metadata"]["compression"]["output_characters"] == (17 if dated else 8)
        assert review["metadata"]["note_count"] == (2 if dated else 1)
        evidence = review["assessments"][0]["evidence"][0]
        assert evidence["quote"] == "CT done." and evidence["quote"] not in original_text
        assert evidence["note_date"] == ("2026-01-01T00:00:00+00:00" if dated else None)
        assert [e["quote"] for e in review["assessments"][0]["evidence"]] == (
            ["CT done.", "No fever."] if dated else ["CT done."]
        )
        assert "may differ from the original notes" in review["notice"]
        assert len(calls) == (3 if dated else 2)


@pytest.mark.parametrize("workflow", ["summarize", "full_notes", "agentic"])
def test_gemini_compression_runs_without_thinking_switch(tmp_path, monkeypatch, workflow):
    import matchminer_ai.patients
    import pandas as pd
    from matchminer_ai.llm.structured import StructuredClient

    catalog(tmp_path)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    original_config = Settings.inference_config

    def config(self):
        value = original_config(self)
        value.remote.update(context_window=16384, tokenizer_mode="bytes", stream=False)
        value.patient["remote"]["request_params"]["max_tokens"] = 2048
        return value

    monkeypatch.setattr(Settings, "inference_config", config)
    calls, downstream = [], []

    def respond(self, path, body=None, **kwargs):
        assert path == "/chat/completions"
        assert self.config.model == "google/gemini-3.8-flash"
        assert "chat_template_kwargs" not in body
        assert body["reasoning_effort"] == "high"
        calls.append(body)
        return {
            "choices": [
                {
                    "message": {"content": "CT done.", "reasoning_content": "Discarded reasoning."},
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(StructuredClient, "_http", respond)

    def summarize(frame, **kwargs):
        downstream.append(frame.copy())
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    def review(frame, items, **kwargs):
        downstream.append(frame.copy())
        if workflow == "agentic":
            assert kwargs["history"] is None
        return {
            "assessments": [],
            "notice": "Documentation review.",
            "metadata": {"method": workflow, "note_count": 1},
        }

    monkeypatch.setattr(matchminer_ai.patients, "summarize_patients", summarize)
    monkeypatch.setattr(matchminer_ai.patients, "review_patient_workup", review)
    monkeypatch.setattr(matchminer_ai.patients, "review_patient_workup_with_note_search", review)
    settings = Settings(catalog_root=tmp_path, llm_backend="gemini_vertex", device="cpu")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/config").json()["note_compression_available"] is True
        payload = {"text": "Fabricated CT completed.", "note_source": "compressed"}
        compression = client.post("/api/compress", json={"text": payload["text"]}).json()["job_id"]
        assert wait(client, compression).json()["status"] == "complete"
        assert len(calls) == 1 and downstream == []
        route = "/api/summarize"
        if workflow != "summarize":
            key = client.post("/api/retrieve", json={"summary": "Fabricated summary"}).json()[
                "job_id"
            ]
            wait(client, key)
            payload.update(retrieval_job_id=key, space_index=0, method=workflow)
            route = "/api/workup"
        response = client.post(route, json=payload)
        assert response.status_code == 202
        done = wait(client, response.json()["job_id"]).json()
        assert done["status"] == "complete", done["progress"]
        result = done["result"]
        metadata = (
            result["compression"]
            if workflow == "summarize"
            else result["review"]["metadata"]["compression"]
        )
        assert metadata["thinking_requested"] == "off"
        assert metadata["thinking"] == "default"
        assert len(calls) == 1
        assert downstream[0].note_text.tolist() == ["CT done."]
        assert result["notes_input"]["text"] == payload["text"]


@pytest.mark.parametrize("workflow", ["summarize", "full_notes", "agentic"])
@pytest.mark.parametrize("enabled", [False, True])
def test_selected_source_never_compresses_inside_the_clinical_workflow(
    tmp_path, monkeypatch, workflow, enabled
):
    import matchminer_ai.patients
    import pandas as pd

    catalog(tmp_path)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    source = [
        {
            "note_text": "Fabricated computed tomography completed.",
            "note_date": "2026-01-01",
            "note_type": "Radiology",
        }
    ]
    compression_calls, received = [], []

    def compress(frame, **kwargs):
        assert frame.to_dict("records") == source
        assert kwargs["thinking"] == "off"
        assert kwargs["max_concurrent_requests"] == 8
        assert kwargs["config"].remote["server_urls"] == ["http://selected.invalid/v1"]
        kwargs["progress_callback"](0, 1)
        kwargs["progress_callback"](1, 1)
        compression_calls.append(1)
        return frame.assign(compressed_note_text="CT done.")

    def summarize(frame, **kwargs):
        assert kwargs["config"].remote["max_concurrent_requests"] == 1
        received.append(frame.copy())
        return pd.DataFrame([{"cancer_history_summary": "Fabricated summary"}])

    def review(raw, items, **kwargs):
        assert kwargs["config"].remote["max_concurrent_requests"] == 1
        if workflow == "agentic":
            assert kwargs["history"] == (None if enabled else "Original concatenated text")
            assert kwargs["patient_summary"] == "Fabricated summary"
        received.append(raw.copy())
        return {
            "assessments": [],
            "notice": "Documentation review.",
            "metadata": {"method": workflow, "note_count": 1},
        }

    monkeypatch.setattr(matchminer_ai.patients, "compress_patient_notes", compress)
    monkeypatch.setattr(matchminer_ai.patients, "summarize_patients", summarize)
    api_name = (
        "review_patient_workup_with_note_search"
        if workflow == "agentic"
        else "review_patient_workup"
    )
    monkeypatch.setattr(matchminer_ai.patients, api_name, review)
    settings = Settings(catalog_root=tmp_path, llm_backend="local", device="cpu")
    with TestClient(create_app(settings)) as client:
        client.post("/api/config", json={"endpoint": "selected.invalid", "model": "fabricated"})
        payload = {
            "text": "Original concatenated text",
            "notes": source,
            "note_source": "compressed" if enabled else "original",
        }
        if enabled:
            compression = client.post(
                "/api/compress", json={"text": payload["text"], "notes": source}
            ).json()["job_id"]
            assert wait(client, compression).json()["status"] == "complete"
            assert received == [] and compression_calls == [1]
        route = "/api/summarize"
        if workflow != "summarize":
            key = client.post("/api/retrieve", json={"summary": "Fabricated summary"}).json()[
                "job_id"
            ]
            wait(client, key)
            payload.update(retrieval_job_id=key, space_index=0, method=workflow)
            route = "/api/workup"
        key = client.post(route, json=payload).json()["job_id"]
        done = wait(client, key).json()
        assert done["status"] == "complete", done["progress"]
        result = done["result"]
        assert result["notes_input"] == {"text": payload["text"], "notes": source}
        assert result["note_source"] == payload["note_source"]
        assert client.get("/api/session").json()["job"]["result"] == result
        assert done["restore"]["note_source"] == payload["note_source"]
        metadata = (
            result.get("compression")
            if workflow == "summarize"
            else result["review"]["metadata"].get("compression")
        )
        assert bool(metadata) is enabled
        assert received[0]["note_text"].tolist() == (
            ["CT done."] if enabled else [source[0]["note_text"]]
        )
        assert received[0]["note_type"].tolist() == ["Radiology"]
        assert compression_calls == ([1] if enabled else [])


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_stopped_standalone_compression_never_starts_review(
    tmp_path, monkeypatch, cancel
):
    import matchminer_ai.patients
    from matchminer_ai.cancellation import cancel_sleep
    from matchminer_ai.patients import NoteCompressionError

    catalog(tmp_path)
    monkeypatch.setattr(Inference, "retrieve", WorkupInference.retrieve)
    review_calls = []
    monkeypatch.setattr(
        matchminer_ai.patients, "review_patient_workup", lambda *a, **kw: review_calls.append(1)
    )
    entered = Event()

    def compress(*args, **kwargs):
        entered.set()
        if cancel:
            cancel_sleep(30)
            pytest.fail("Cancelled compression continued")
        raise NoteCompressionError("Fabricated private provider detail")

    monkeypatch.setattr(matchminer_ai.patients, "compress_patient_notes", compress)
    settings = Settings(catalog_root=tmp_path, llm_backend="local", device="cpu")
    with TestClient(create_app(settings)) as client:
        key = client.post("/api/retrieve", json={"summary": "Fabricated summary"}).json()["job_id"]
        wait(client, key)
        job = client.post(
            "/api/compress",
            json={
                "text": "Fabricated note.",
                "retrieval_job_id": key,
                "summary": "Fabricated edited summary",
            },
        ).json()["job_id"]
        assert entered.wait(3)
        if cancel:
            assert client.post(f"/api/jobs/{job}/stop").status_code == 200
            result = stopped(client, job)
        else:
            result = wait(client, job).json()
            assert result["status"] == "failed"
        assert result["result"] is None
        assert "private" not in result["progress"]
        assert result["restore"]["notes_input"]["text"] == "Fabricated note."
        assert result["restore"]["summary"] == "Fabricated edited summary"
        assert result["restore"]["retrieval_job_id"] == key
        assert result["restore"]["retrieval"]["spaces"]
        assert review_calls == []
        assert client.get("/api/note-compression").json()["note_compression"] is None
