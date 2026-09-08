import io
import json

import pandas as pd
import pytest

from matchminer_ai.cli import llm_trial_check


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self, amount: int = -1) -> bytes:
        return self._payload if amount == -1 else self._payload[:amount]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


def test_discover_model_normalizes_endpoint_and_uses_api_key(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.headers, timeout))
        return _Response({"data": [{"id": "served-model"}]})

    monkeypatch.setattr(llm_trial_check.request, "urlopen", fake_urlopen)

    base_url, model_ids = llm_trial_check.discover_model(
        "localhost:8000/v1/models",
        api_key="secret",
        timeout=3,
    )

    assert base_url == "http://localhost:8000/v1"
    assert model_ids == ["served-model"]
    assert calls == [
        (
            "http://localhost:8000/v1/models",
            {"Authorization": "Bearer secret"},
            3,
        )
    ]


def test_discover_model_rejects_empty_model_list(monkeypatch):
    monkeypatch.setattr(
        llm_trial_check.request,
        "urlopen",
        lambda req, timeout: _Response({"data": []}),
    )

    with pytest.raises(RuntimeError, match="returned no model IDs"):
        llm_trial_check.discover_model("http://localhost:8000/v1")


def test_read_multiline_reprompts_after_empty_input():
    lines = iter(["", ".done", "patient line 1", "patient line 2", ".done"])
    output = io.StringIO()

    result = llm_trial_check.read_multiline(
        "patient summary",
        input_fn=lambda: next(lines),
        output=output,
    )

    assert result == "patient line 1\npatient line 2"
    assert "cannot be empty" in output.getvalue()


def test_main_discovers_model_and_scores_with_public_package_api(monkeypatch):
    monkeypatch.setattr(
        llm_trial_check,
        "discover_model",
        lambda endpoint: ("http://localhost:8000/v1", ["served-model"]),
    )
    captured = {}

    def fake_score(candidate, *, config):
        captured["candidate"] = candidate.copy()
        captured["config"] = config
        return pd.DataFrame(
            [
                {
                    "llm_match_quality_score": 4,
                    "llm_match_quality_reasoning_text": "Complete reasoning",
                    "llm_match_quality_answer_text": (
                        "Complete final answer\nFinal score: 4"
                    ),
                    "llm_match_quality_parse_status": "parsed",
                }
            ]
        )

    monkeypatch.setattr(
        llm_trial_check,
        "_score_candidate",
        fake_score,
    )
    lines = iter(
        [
            "Patient summary line 1",
            "Patient summary line 2",
            ".done",
            "Trial space line 1",
            ".done",
        ]
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    exit_code = llm_trial_check.main(
        ["localhost:8000"],
        input_fn=lambda: next(lines),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "Model: served-model" in stdout.getvalue()
    assert "Reasoning:\nComplete reasoning" in stdout.getvalue()
    assert "Final output:\nComplete final answer\nFinal score: 4" in stdout.getvalue()
    assert "LLM trial match score: 4/5" in stdout.getvalue()
    assert captured["candidate"].to_dict("records") == [
        {
            "patient_id": "interactive-patient",
            "space_trial_id": "interactive-trial-space",
            "cancer_history_summary": (
                "Patient summary line 1\nPatient summary line 2"
            ),
            "clinical_space_summary": "Trial space line 1",
        }
    ]
    config = captured["config"]
    assert config.debug_mode is True
    assert config.remote["enabled"] is True
    assert config.remote["server_urls"] == ["http://localhost:8000/v1"]
    assert config.llm_match_quality["remote"]["model_name"] == "served-model"
