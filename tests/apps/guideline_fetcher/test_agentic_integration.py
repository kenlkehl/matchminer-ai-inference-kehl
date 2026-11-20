"""App jobs exercise the sibling package and real isolated workers with fake LLM replies."""

import json
import time
from collections import OrderedDict
from threading import Lock

import pytest
from fastapi.testclient import TestClient

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Inference
from matchminer_ai.apps.guideline_fetcher.settings import Settings
from .test_app import catalog


@pytest.mark.parametrize("provider,model,thinking", [
    ("local", "Qwen/Qwen3.8-27B-FP8", False),
    ("local", "Qwen/Qwen3.5-4B", False),
    ("local", "Qwen/Qwen3.5-4B", True),
    ("gemini_vertex", "unused", False),
    ("gemma_vertex", "unused", False),
])
def test_enhanced_agentic_review_through_app_jobs(tmp_path, monkeypatch, provider, model, thinking):
    from matchminer_ai.llm.structured import EndpointError, StructuredClient
    from matchminer_ai.patients import workup_search_review

    # Run the public API, real configuration resolver and REPL. Only network
    # responses and unrelated population retrieval are replaced.
    monkeypatch.setattr(workup_search_review, "_VOCABULARY", OrderedDict())
    original_config = Settings.inference_config

    def bounded_config(self):
        config = original_config(self)
        config.patient.update(tokenizer_mode="bytes", context_window=32768)
        config.patient["remote"]["request_params"]["max_tokens"] = 1024
        return config

    monkeypatch.setattr(Settings, "inference_config", bounded_config)
    items = [{"name": name, "conditions": "For the fabricated population"}
             for name in ["CT", "ECG", "LDH"]]
    monkeypatch.setattr(Inference, "retrieve", lambda *args: {
        "summary": "SUMMARY FOR NAVIGATION ONLY",
        "spaces": [{"name": "Fabricated population", "clinical_space_summary": "Population",
                    "diagnostic_workup": items}],
    })
    calls = []
    lock = Lock()

    def assessment(status="planned", more=False):
        return {"status": "answered", "answer": {"status": status,
                "applicability": "applies", "bottom_line": "Fabricated finding."},
                "limitations": [], "needs_more_evidence": more}

    def http(self, endpoint, body=None, **kwargs):
        assert endpoint == "/chat/completions"
        if provider == "gemini_vertex":
            assert "chat_template_kwargs" not in body
        else:
            assert self.config.thinking == ("on" if thinking else "off")
            assert body["chat_template_kwargs"]["enable_thinking"] is thinking
            assert body["temperature"] == (0.7 if provider == "local" and not thinking else 1.0)
            assert body["top_p"] == (0.8 if provider == "local" and not thinking else 0.95)
            if "Qwen3.5" in model:
                assert body["top_k"] == 20 and body["min_p"] == 0.0
                assert body["presence_penalty"] == 1.5 and body["repetition_penalty"] == 1.0
                assert "reasoning_effort" not in body
                assert "reasoning_effort" not in body["chat_template_kwargs"]
        request = next(message for message in body["messages"] if message["role"] == "user")
        payload = json.loads(request["content"].rsplit("Request:\n", 1)[-1])
        with lock:
            calls.append((self.config.provider, payload))
        if "last_cell_result" in payload:
            assert payload["patient_summary"] == "SUMMARY FOR NAVIGATION ONLY"
            name = json.loads(payload["question"])["recommendation"]["name"]
            value = assessment()
            value.pop("needs_more_evidence")
            value.update(action="python" if payload["search_only"] else "final",
                         code=[f"notes.loc[notes.note_type.eq('Oncology').fillna(False) & notes.note_text.str.contains('{name} ordered', na=False)]"]
                         if payload["search_only"] else [], memory="", needs_review=name == "ECG")
        else:
            name = payload["workup_item"]["recommendation"]["name"]
            if "unsuccessful_terms" in payload:
                assert "provisional_assessment" not in payload
                assert "source_excerpts" not in payload
                assert "SUMMARY FOR NAVIGATION ONLY" not in json.dumps(payload)
                if name == "LDH":
                    raise EndpointError("PRIVATE PROVIDER FAILURE MUST NOT LEAK")
                value = {"terms": [name]}
            elif name == "CT":
                assert any("CT completed" in e["quote"] for e in payload["source_excerpts"])
                value = assessment("completed")
            else:
                assert name == "ECG"
                full_record = payload["review_reasons"][0].startswith("Serial")
                completed = full_record and any(
                    "EKG performed" in e["quote"] for e in payload["source_excerpts"]
                )
                value = assessment("completed" if completed else "planned", more=not completed)
        return {"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(value), "reasoning_content": "MUST NOT BE REPLAYED"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}

    monkeypatch.setattr(StructuredClient, "_http", http)
    catalog(tmp_path)
    settings = Settings(catalog_root=tmp_path, llm_backend=provider, device="cpu",
                        endpoint="http://fabricated.invalid/v1", model=model, reasoning=thinking)
    notes = [{"note_text": "CT completed. EKG performed.", "note_date": "2026-02-01", "note_type": "Radiology"},
             {"note_text": "CT ordered. ECG ordered. LDH ordered.", "note_date": "2026-01-01", "note_type": "Oncology"}]

    def done(client, key):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{key}").json()
            if job["status"] in {"complete", "failed"}:
                assert job["status"] == "complete", job.get("error")
                return job["result"]
            time.sleep(0.02)
        pytest.fail("Fabricated agentic job did not finish")

    with TestClient(create_app(settings)) as client:
        key = client.post("/api/retrieve", json={"summary": "Fabricated summary"}).json()["job_id"]
        done(client, key)
        response = client.post("/api/workup", json={"text": "Not the dated source",
            "notes": notes, "retrieval_job_id": key, "space_index": 0, "method": "agentic"})
        assert response.status_code == 202
        output = done(client, response.json()["job_id"])
        assert client.get("/api/session").json()["job"]["result"] == output

    assert output["method"] == "agentic" and output["notes_input"]["notes"] == notes
    result = output["review"]
    ct, ecg, ldh = result["assessments"]
    assert [a["status"] for a in [ct, ecg, ldh]] == ["completed", "completed", "planned"]
    assert ct["metadata"]["followup_review"]["rounds"][0]["changed"]
    assert ct["metadata"]["followup_review"]["full_record"] is None
    assert ecg["metadata"]["followup_review"]["full_record"] == {"chunks": 2, "complete": True}
    assert ldh["metadata"]["followup_review"]["status"] == "incomplete"
    assert any("last validated assessment" in limitation for limitation in ldh["limitations"])
    assert result["metadata"]["incomplete_review_items"] == 1
    assert result["metadata"]["failed_items"] == 0
    assert result["metadata"]["requests"] == len(calls)
    assert result["metadata"]["max_parallel_questions"] == 3
    for row in result["assessments"]:
        assert row["metadata"]["initial_search_requests"] <= 16
        assert row["metadata"]["followup_review"]["requests"] <= 12
        for evidence in row["evidence"]:
            source = sorted(notes, key=lambda n: n["note_date"])[evidence["note_number"] - 1]
            assert evidence["quote"] in source["note_text"]
            assert evidence["note_date"].startswith(source["note_date"])
            assert evidence["note_type"] == source["note_type"]
    assert "PRIVATE PROVIDER FAILURE" not in json.dumps(output)
    assert "MUST NOT BE REPLAYED" not in json.dumps([payload for _, payload in calls])
    assert {actual_provider for actual_provider, _ in calls} == {
        "google_agent_platform" if provider != "local" else "openai"
    }
