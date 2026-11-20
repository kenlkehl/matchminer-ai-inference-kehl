"""Real public review API with fabricated source notes and mocked LLM transport."""

import json

import pytest
from fastapi.testclient import TestClient
from .test_app import catalog, wait

from matchminer_ai.apps.guideline_fetcher.app import create_app
from matchminer_ai.apps.guideline_fetcher.services import Inference
from matchminer_ai.apps.guideline_fetcher.settings import Settings


@pytest.mark.parametrize("source", ["original", "compressed"])
def test_full_note_job_uses_findings_only_and_returns_code_owned_context(
    tmp_path, monkeypatch, source
):
    from matchminer_ai import patients
    from matchminer_ai.llm import structured
    from matchminer_ai.patients import workup

    catalog(tmp_path)
    settings = Settings(
        catalog_root=tmp_path,
        llm_backend="local",
        model="fabricated",
        tokenizer="fabricated",
        endpoint="http://model.invalid/v1",
        device="cpu",
    )
    monkeypatch.setattr(
        Inference,
        "retrieve",
        lambda self, summary, *a: {
            "summary": summary,
            "report": "Fabricated guideline report",
            "spaces": [
                {
                    "name": "Fabricated space",
                    "diagnostic_workup": [{"name": "Imaging"}],
                    "clinical_space_summary": "Fabricated population",
                }
            ],
        },
    )
    monkeypatch.setattr(
        patients,
        "compress_patient_notes",
        lambda frame, **kw: frame.assign(compressed_note_text="Fabricated CT done."),
    )
    monkeypatch.setattr(
        workup,
        "resolve_structured_config",
        lambda *a, **kw: (
            structured.StructuredConfig(
                model="fabricated",
                tokenizer_mode="bytes",
                response_format="json_schema",
                max_tokens=2048,
                context_window=32768,
            ),
            {},
        ),
    )

    class Tokenizer:
        is_fast = True

        def __call__(self, text, **kw):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: Tokenizer())
    calls = []

    def http(self, endpoint, body, **kw):
        assert endpoint == "/chat/completions"
        schema = body["response_format"]["json_schema"]["schema"]
        fields = schema["properties"]["assessments"]["items"]["properties"]
        assert set(fields) == {"name", "status", "applicability", "bottom_line"}
        payload = json.loads(body["messages"][1]["content"])
        assert all("evidence" not in row for row in payload["prior_assessments"])
        calls.append(payload)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "assessments": [
                                    {
                                        "name": "Imaging",
                                        "status": "completed",
                                        "applicability": "uncertain",
                                        "bottom_line": "Fabricated finding.",
                                    }
                                ],
                            }
                        )
                    },
                }
            ]
        }

    monkeypatch.setattr(structured.StructuredClient, "_http", http)
    raw = "FABRICATED EXAM\n\nCT: completed.\nResult: <5 & unchanged."
    input_notes = [{"note_text": raw, "note_date": "2026-01-01", "note_type": "Exam"}]
    with TestClient(create_app(settings)) as client:
        retrieval = client.post("/api/retrieve", json={"summary": "Fabricated summary"}).json()[
            "job_id"
        ]
        assert wait(client, retrieval).json()["status"] == "complete"
        if source == "compressed":
            compression = client.post(
                "/api/compress", json={"text": raw, "notes": input_notes}
            ).json()["job_id"]
            assert wait(client, compression).json()["status"] == "complete"
        key = client.post(
            "/api/workup",
            json={
                "text": raw,
                "notes": input_notes,
                "retrieval_job_id": retrieval,
                "space_index": 0,
                "method": "full_notes",
                "note_source": source,
            },
        ).json()["job_id"]
        done = wait(client, key).json()
        assert done["status"] == "complete", done["progress"]
        review = done["result"]["review"]
        assert review["metadata"]["evidence_selection"] == "automatic_reviewed_excerpts"
        assert review["assessments"][0]["evidence"] == [
            {
                "note_number": 1,
                "note_date": "2026-01-01T00:00:00+00:00",
                "note_type": "Exam",
                "quote": raw if source == "original" else "Fabricated CT done.",
            }
        ]
        assert len(calls) == 1
        assert client.get("/api/session").json()["job"]["result"]["review"] == review
