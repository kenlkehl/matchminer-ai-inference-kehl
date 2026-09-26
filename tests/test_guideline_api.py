"""Public API and HTTP-wire integration using invented source text only."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_guideline_extraction import catalog, extraction, make_library, state

from matchminer_ai import load_config, load_default_preset
from matchminer_ai._storage import directory_lock, read_json
from matchminer_ai.config import config_snapshot
from matchminer_ai.llm.structured import EndpointError
from matchminer_ai.trials import _guideline_prompts as prompts
from matchminer_ai.trials import (
    audit_guideline_catalog,
    list_guidelines,
    summarize_guidelines,
)
from matchminer_ai.trials._guideline_canonical import TASK as CATALOG_TASK


@pytest.mark.parametrize("change", ["unused_only", "clinical_prompt", "corrupt_digest"])
def test_resume_ignores_only_legacy_unused_patient_prompt(
    guideline_run, fake_endpoint, change
):
    from matchminer_ai._storage import digest

    root, output, config = guideline_run
    summarize_guidelines(root, disease="fictional", output_dir=output, config=config)
    path = output / "run_config.json"
    saved = json.loads(path.read_text())
    saved["prompt_resources_sha256"]["structured.memory_retry.txt"] = "unused-patient-prompt"
    if change == "clinical_prompt":
        saved["prompt_resources_sha256"]["guideline.system.txt"] = "changed-clinical-prompt"
    identity = {k: v for k, v in saved.items() if k not in {"config_sha256", "runtime", "stage_versions"}}
    identity = json.loads(json.dumps(identity))
    for key in ("timeout", "attempts", "api_key_env", "stream", "max_concurrent_requests"):
        identity["llm"].pop(key)
    saved["config_sha256"] = "invalid" if change == "corrupt_digest" else digest(identity)
    path.write_text(json.dumps(saved))
    fake_endpoint.clear()
    if change == "unused_only":
        result = summarize_guidelines(root, disease="fictional", output_dir=output, config=config)
        assert len(result) == 1
        assert not fake_endpoint
        assert "structured.memory_retry.txt" not in json.loads(path.read_text())["prompt_resources_sha256"]
    else:
        with pytest.raises(ValueError, match="changed"):
            summarize_guidelines(root, disease="fictional", output_dir=output, config=config)


@pytest.fixture
def guideline_run(tmp_path):
    config = load_default_preset()
    config.remote.update(
        enabled=True,
        server_urls=["http://synthetic.invalid:8002/v1"],
        max_concurrent_requests=2,
        request_timeout=7200,
        max_retries=1,
    )
    return make_library(tmp_path), tmp_path / "catalog", config


@pytest.fixture
def fake_endpoint(monkeypatch):
    calls = []

    def respond(request, timeout):
        body = json.loads(request.data) if request.data else None
        calls.append((request.full_url, body, timeout))
        if request.full_url.endswith("/v1/models"):
            return io.BytesIO(
                json.dumps(
                    {"data": [{"id": "synthetic-gemma4", "max_model_len": 262144}]}
                ).encode()
            )
        if request.full_url.endswith("/tokenize"):
            count = sum(len(m["content"].encode()) for m in body["messages"]) // 4 + 100
            return io.BytesIO(
                json.dumps({"count": count, "max_model_len": 262144}).encode()
            )
        assert request.full_url.endswith("/v1/chat/completions")
        assert timeout == 7200
        assert body["model"] == "synthetic-gemma4"
        assert body["max_tokens"] == 100000
        assert body["temperature"] == 1.0
        assert body["top_p"] == 0.95
        assert body["top_k"] == 64
        assert body["chat_template_kwargs"]["enable_thinking"] is True
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        prompt = body["messages"][1]["content"]
        value = (
            extraction()
            if prompt.startswith(prompts.EXTRACT_TASK)
            else catalog()
            if prompt.startswith(CATALOG_TASK)
            else state()
        )
        content = json.dumps(value)
        events = [
            {
                "model": "synthetic-gemma4",
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": "Synthetic reasoning."}}
                ],
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": content[: len(content) // 2]}}
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content[len(content) // 2 :]},
                        "finish_reason": "stop",
                    }
                ]
            },
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 1000}},
        ]
        assert body["stream"] is True
        return io.BytesIO(
            (
                "".join("data: " + json.dumps(e) + "\n\n" for e in events)
                + "data: [DONE]\n\n"
            ).encode()
        )

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    return calls


def test_public_roundtrip_wire_settings_exports_provenance_and_offline_resume(
    guideline_run, fake_endpoint, monkeypatch
):
    root, output, config = guideline_run
    progress = []
    result, metadata, qc = summarize_guidelines(
        root,
        disease="fictional",
        output_dir=output,
        config=config,
        return_metadata=True,
        return_qc=True,
        progress_callback=progress.append,
    )
    assert len(result) == 1
    assert result.iloc[0]["clinical_space_number"] == 1
    assert result.iloc[0]["general_exclusion_criteria"] == "NA"
    assert result.iloc[0]["trial_id"].startswith("guideline:fictional:")
    assert result.iloc[0]["space"] == state()["space"]
    assert result.iloc[0]["evidence"][0]["quote"].startswith("Fictional disease")
    assert result.iloc[0]["diagnostic_workup"][0]["name"] == "Synthetic test A"
    assert list(qc.columns) == ["metric", "value", "denominator", "percent", "ids"]
    assert metadata["validation"]["status"] == "passed"
    assert metadata["validation"]["verified_raw_provider_responses"] == 3
    assert (
        metadata["model_metadata"]["guideline_summarizer"]["context_window"] == 262144
    )
    assert metadata["config_snapshot"]["guideline"] == config.guideline
    assert metadata["run_config"]["llm"]["thinking"] == "on"
    assert metadata["run_config"]["prompt_resources_sha256"][
        "guideline.repair_focus.txt"
    ]
    assert "structured.memory_retry.txt" not in metadata["run_config"]["prompt_resources_sha256"]
    assert any("canonicalizing" in line for line in progress)
    assert any("populating" in line for line in progress)
    assert sum(url.endswith("/chat/completions") for url, _, _ in fake_endpoint) == 3
    assert all(
        body["chat_template_kwargs"] == {"enable_thinking": True}
        for url, body, _ in fake_endpoint
        if url.endswith("/tokenize")
    )
    for path in (output / "checkpoints").glob("*/attempt-*.json"):
        assert (
            read_json(path)["choices"][0]["message"]["reasoning"]
            == "Synthetic reasoning."
        )
    csv = pd.read_csv(output / "trial_spaces.csv", keep_default_na=False)
    assert (
        csv["clinical_space_summary"].tolist()
        == result["clinical_space_summary"].tolist()
    )

    def unexpected(*args, **kwargs):
        raise AssertionError(
            "Accepted complete runs must resume without endpoint access"
        )

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", unexpected)
    config.remote["max_concurrent_requests"] = 4
    config.remote["request_timeout"] = 8000
    config.remote["max_retries"] = 2
    for want_metadata, want_qc in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        resumed = summarize_guidelines(
            root,
            disease="fictional",
            output_dir=output,
            config=config,
            return_metadata=want_metadata,
            return_qc=want_qc,
        )
        frame = resumed[0] if isinstance(resumed, tuple) else resumed
        pd.testing.assert_frame_equal(frame, result)
        if want_metadata or want_qc:
            assert len(resumed) == 1 + want_metadata + want_qc
    assert (
        audit_guideline_catalog(root, disease="fictional", output_dir=output)["status"]
        == "passed"
    )


def test_inventory_and_config_yaml_roundtrip(guideline_run, tmp_path):
    root, _, config = guideline_run
    assert list_guidelines(root).iloc[0]["disease"] == "fictional"
    assert list_guidelines(root).iloc[0]["status"] == "present_unverified"
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(config_snapshot(config)))
    assert load_config(path).guideline == config.guideline


@pytest.mark.parametrize(
    "location", ["collection", "markdown", "code", "parent", "symlink"]
)
def test_sources_and_code_cannot_receive_artifacts(guideline_run, tmp_path, location):
    root, _, config = guideline_run
    paths = {
        "collection": root.parent / "results",
        "markdown": root / "results",
        "code": Path(__file__).resolve().parents[1] / "never-created-catalog",
        "parent": tmp_path,
    }
    link = tmp_path / "linked-source"
    link.symlink_to(root, target_is_directory=True)
    paths["symlink"] = link / "results"
    with pytest.raises(ValueError, match="outside the source"):
        summarize_guidelines(
            root, disease="fictional", output_dir=paths[location], config=config
        )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.remote.update(enabled=False), "requires.*enabled"),
        (lambda c: c.remote.update(max_concurrent_requests=0), "positive integer"),
        (lambda c: c.guideline.update(packet_pages=0), "positive integer"),
        (
            lambda c: c.guideline.update(context_window=100001),
            "Context window must exceed",
        ),
        (lambda c: c.guideline.update(context_window=-1), "context_window"),
        (
            lambda c: c.remote.update(server_urls=["http://one/v1", "http://two/v1"]),
            "exactly one",
        ),
        (
            lambda c: c.guideline["remote"]["extra_body"].update(max_tokens=1),
            "duplicate|Put max_tokens",
        ),
        (
            lambda c: c.guideline["remote"]["request_params"].update(messages=[]),
            "cannot override",
        ),
        (
            lambda c: c.guideline.update(tokenizer_mode="silent-fallback"),
            "tokenizer_mode",
        ),
    ],
)
def test_invalid_configuration_fails_before_generation(guideline_run, change, message):
    root, output, config = guideline_run
    change(config)
    with pytest.raises(ValueError, match=message):
        summarize_guidelines(
            root, disease="fictional", output_dir=output, config=config
        )
    assert not (output / "paradigms.jsonl").exists()


def test_partial_copy_fails_before_model_requests(guideline_run, fake_endpoint):
    root, output, config = guideline_run
    (root.parent / "fictional.pdf").write_bytes(b"unfinished")
    with pytest.raises(ValueError, match="Source PDF"):
        summarize_guidelines(
            root, disease="fictional", output_dir=output, config=config
        )
    assert fake_endpoint == []


def test_extra_body_cannot_override_the_output_reserve(guideline_run, fake_endpoint):
    root, output, config = guideline_run
    config.guideline["remote"]["extra_body"]["max_completion_tokens"] = 1
    with pytest.raises(ValueError, match="do not set max_completion_tokens"):
        summarize_guidelines(
            root, disease="fictional", output_dir=output, config=config
        )
    assert fake_endpoint == []


def test_request_extensions_and_tokenizer_template_stay_consistent(
    guideline_run, fake_endpoint
):
    root, output, config = guideline_run
    config.guideline["remote"]["request_params"]["seed"] = 42
    config.guideline["remote"]["extra_body"]["chat_template_kwargs"][
        "custom_template_flag"
    ] = True
    summarize_guidelines(root, disease="fictional", output_dir=output, config=config)
    for url, body, _ in fake_endpoint:
        if body is not None:
            assert body["chat_template_kwargs"]["custom_template_flag"] is True
        if url.endswith("/chat/completions"):
            assert body["seed"] == 42
    config.guideline["remote"]["request_params"]["seed"] = 43
    with pytest.raises(ValueError, match="changed"):
        summarize_guidelines(
            root, disease="fictional", output_dir=output, config=config
        )


def test_audit_detects_reasoning_effort_changed_on_a_retry(guideline_run, fake_endpoint):
    root, output, config = guideline_run
    config.guideline["remote"]["request_params"]["reasoning_effort"] = "xhigh"
    summarize_guidelines(root, disease="fictional", output_dir=output, config=config)
    path = next((output / "checkpoints").glob("*/request-attempt-*.json"))
    request = read_json(path)
    request["body"]["reasoning_effort"] = "low"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="Retry changed reasoning_effort"):
        audit_guideline_catalog(root, disease="fictional", output_dir=output)


def test_callback_failure_does_not_discard_completed_work(guideline_run, fake_endpoint):
    root, output, config = guideline_run

    def callback(message):
        raise RuntimeError("Synthetic UI disconnected")

    result = summarize_guidelines(
        root,
        disease="fictional",
        output_dir=output,
        config=config,
        progress_callback=callback,
    )
    assert len(result) == 1
    assert read_json(output / "status.json")["status"] == "complete"


def test_unknown_endpoint_context_is_not_guessed(guideline_run, monkeypatch):
    root, output, config = guideline_run
    monkeypatch.setattr(
        "matchminer_ai.llm.structured.urlopen",
        lambda *a, **k: io.BytesIO(b'{"data":[{"id":"synthetic-gemma4"}]}'),
    )
    with pytest.raises(EndpointError, match="set guideline.context_window explicitly"):
        summarize_guidelines(
            root, disease="fictional", output_dir=output, config=config
        )


def test_run_lock_rejects_concurrent_writer(tmp_path):
    with directory_lock(tmp_path):
        with pytest.raises(ValueError, match="Another process"):
            with directory_lock(tmp_path):
                pytest.fail("Second writer acquired lock")


def test_audit_failure_cannot_be_reported_as_complete(
    guideline_run, fake_endpoint, monkeypatch
):
    root, output, config = guideline_run

    def reject(*args, **kwargs):
        raise ValueError("Synthetic provenance mismatch")

    monkeypatch.setattr(
        "matchminer_ai.trials.guidelines.audit_guideline_catalog", reject
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        summarize_guidelines(
            root, disease="fictional", output_dir=output, config=config
        )
    status = read_json(output / "status.json")
    assert status["status"] == "failed"
    assert status["stage"] == "audit"


def test_model_failure_returns_no_partial_catalog(
    guideline_run, fake_endpoint, monkeypatch
):
    root, output, config = guideline_run

    def reject(*args, **kwargs):
        raise EndpointError("Synthetic endpoint unavailable")

    monkeypatch.setattr(
        "matchminer_ai.trials._guideline_generation.Client.complete", reject
    )
    with pytest.raises(RuntimeError, match="extraction packets failed"):
        summarize_guidelines(
            root, disease="fictional", output_dir=output, config=config
        )
    assert read_json(output / "status.json")["status"] == "failed"
    assert not (output / "paradigms.jsonl").exists()


def test_explicit_model_and_context_support_endpoint_without_discovery(
    guideline_run, fake_endpoint
):
    root, output, config = guideline_run
    config.guideline["remote"]["model_name"] = "synthetic-gemma4"
    config.guideline["context_window"] = 262144
    summarize_guidelines(root, disease="fictional", output_dir=output, config=config)
    assert not any(url.endswith("/models") for url, _, _ in fake_endpoint)


def test_converter_requires_explicit_paths_and_rejects_code_output(tmp_path):
    pytest.importorskip("pypdf")
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "guideline_converter", repo / "scripts/convert_guidelines.py"
    )
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    with pytest.raises(SystemExit) as missing:
        converter.main([])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as blocked:
        converter.main(
            ["--input-dir", str(tmp_path), "--output-dir", str(repo / "never-create")]
        )
    assert blocked.value.code == 2
    assert not (repo / "never-create").exists()
