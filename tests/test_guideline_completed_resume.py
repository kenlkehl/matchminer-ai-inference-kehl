"""A recovered, audited synthetic catalog must survive collection restarts."""

import json
from dataclasses import replace

import pytest

from matchminer_ai._storage import atomic_json, read_json
from matchminer_ai.trials import _guideline_pipeline as pipeline
from matchminer_ai.trials import _guideline_prompts as prompts
from matchminer_ai.trials._guideline_audit import audit_catalog
from matchminer_ai.trials._guideline_canonical import TASK as CATALOG_TASK
from matchminer_ai.trials._guideline_generation import Client
from matchminer_ai.trials._guideline_sources import load_guideline
from matchminer_ai.llm.structured import StructuredConfig
from test_guideline_extraction import catalog, extraction, make_library, quoted_state


@pytest.fixture
def completed(tmp_path, monkeypatch):
    guideline = load_guideline(make_library(tmp_path), "fictional")
    output = tmp_path / "derived"
    config = StructuredConfig(model="synthetic", tokenizer_mode="bytes")

    def respond(self, endpoint, body, **kwargs):
        prompt = body["messages"][1]["content"]
        value = (
            extraction() if prompt.startswith(prompts.EXTRACT_TASK)
            else catalog() if prompt.startswith(CATALOG_TASK) else quoted_state()
        )
        return {"choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps(value)},
        }]}

    monkeypatch.setattr(Client, "_http", respond)

    def finalize():
        atomic_json(output / "validation.json", audit_catalog(guideline, output))

    pipeline.run_guideline(guideline, output, config, workers=1, finalize=finalize)
    return guideline, output, config, finalize


def test_completed_resume_preserves_artifacts_and_still_runs_full_audit(completed, monkeypatch):
    guideline, output, config, finalize = completed
    before = {p.name: p.read_bytes() for p in output.glob("*.json*")}
    messages, audits = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError("An audited completed catalog must not restart generation")

    monkeypatch.setattr(pipeline, "Client", forbidden)

    def checked_finalize():
        audits.append(True)
        finalize()

    result = pipeline.run_guideline(
        guideline, output, replace(config, attempts=12), workers=32,
        progress_callback=messages.append, finalize=checked_finalize,
    )
    assert result["status"] == "complete"
    assert audits == [True]
    assert messages == ["fictional: reusing audited completed catalog"]
    assert {p.name: p.read_bytes() for p in output.glob("*.json*")} == before

    # The shortcut cannot hide damaged intermediate provenance behind an intact
    # export hash. The public API's final audit must reject it and mark failure.
    ledger = read_json(output / "extraction_ownership.json")
    ledger["branches"][0]["owner_job"] = "tampered-owner"
    atomic_json(output / "extraction_ownership.json", ledger)
    with pytest.raises(ValueError, match="ownership ledger differs"):
        pipeline.run_guideline(guideline, output, config, finalize=finalize)
    assert read_json(output / "status.json")["status"] == "failed"


@pytest.mark.parametrize("change", ["export", "audit", "source", "count", "running", "missing"])
def test_unverified_outputs_do_not_bypass_normal_resume(completed, monkeypatch, change):
    guideline, output, config, finalize = completed
    audit = read_json(output / "validation.json")
    if change == "export":
        with (output / "paradigms.jsonl").open("a") as stream:
            stream.write("\n")
    elif change == "audit":
        audit["status"] = "failed"
    elif change == "source":
        audit["source_fingerprint"] = "different-source"
    elif change == "count":
        audit["paradigms"] += 1
    elif change == "running":
        status = read_json(output / "status.json")
        status["status"] = "running"
        atomic_json(output / "status.json", status)
    atomic_json(output / "validation.json", audit)
    if change == "missing":
        (output / "validation.json").unlink()

    def normal_resume(*args, **kwargs):
        raise RuntimeError("Normal checkpoint resume reached")

    monkeypatch.setattr(pipeline, "Client", normal_resume)
    with pytest.raises(RuntimeError, match="Normal checkpoint resume reached"):
        pipeline.run_guideline(guideline, output, config, finalize=finalize)


def test_completed_resume_does_not_accept_a_changed_model(completed):
    guideline, output, config, finalize = completed
    before = (output / "status.json").read_bytes()
    with pytest.raises(ValueError, match="settings changed"):
        pipeline.run_guideline(
            guideline, output, replace(config, model="different"), finalize=finalize,
        )
    assert (output / "status.json").read_bytes() == before


def test_export_stays_running_until_full_audit_succeeds(completed, monkeypatch):
    guideline, output, config, finalize = completed
    (output / "validation.json").unlink()
    observed = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Accepted checkpoint resume must not send LLM requests")

    monkeypatch.setattr(Client, "_http", forbidden)

    def checked_finalize():
        status = read_json(output / "status.json")
        observed.append((status["status"], status["stage"]))
        assert (output / "paradigms.jsonl").exists()
        assert not (output / "validation.json").exists()
        finalize()

    result = pipeline.run_guideline(
        guideline, output, config, workers=1, finalize=checked_finalize,
    )
    assert observed == [("running", "audit")]
    assert result["status"] == "complete"
    assert result["stage"] == "export"
    assert read_json(output / "status.json") == result
    assert read_json(output / "validation.json")["status"] == "passed"


@pytest.mark.parametrize("exception, expected", [(ValueError, "failed"), (KeyboardInterrupt, "interrupted")])
def test_audit_failure_or_interruption_never_reports_completion(completed, exception, expected):
    guideline, output, config, _ = completed
    (output / "validation.json").unlink()

    def fail_audit():
        status = read_json(output / "status.json")
        assert (status["status"], status["stage"]) == ("running", "audit")
        raise exception("synthetic audit stop")

    with pytest.raises(exception, match="synthetic audit stop"):
        pipeline.run_guideline(guideline, output, config, workers=1, finalize=fail_audit)
    status = read_json(output / "status.json")
    assert (status["status"], status["stage"]) == (expected, "audit")
    assert "synthetic audit stop" in status["error"]
    assert not (output / "validation.json").exists()


@pytest.mark.parametrize("stage", ["extract", "canonicalize", "details", "export"])
def test_audit_rejects_other_running_stages(completed, stage):
    guideline, output, _, _ = completed
    status = read_json(output / "status.json")
    status.update(status="running", stage=stage)
    atomic_json(output / "status.json", status)
    with pytest.raises(ValueError, match="not complete"):
        audit_catalog(guideline, output)


def test_pending_audit_still_rejects_corrupt_provenance(completed):
    guideline, output, _, _ = completed
    status = read_json(output / "status.json")
    status.update(status="running", stage="audit")
    atomic_json(output / "status.json", status)
    ledger = read_json(output / "extraction_ownership.json")
    ledger["branches"][0]["owner_job"] = "tampered-owner"
    atomic_json(output / "extraction_ownership.json", ledger)
    with pytest.raises(ValueError, match="ownership ledger differs"):
        audit_catalog(guideline, output)
