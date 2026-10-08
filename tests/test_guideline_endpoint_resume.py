"""Interrupt/resume and explicit transport migration with invented clinical text."""

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from matchminer_ai._storage import read_json
from matchminer_ai.cancellation import CancellationToken, InferenceCancelled, cancellation_scope
from matchminer_ai.llm.structured import StructuredConfig
from matchminer_ai.trials import audit_guideline_catalog, summarize_guidelines
from matchminer_ai.trials._guideline_endpoints import resume_transport
from matchminer_ai.trials._guideline_generation import Client
from test_guideline_api import fake_endpoint, guideline_run  # noqa: F401
from test_pdq_catalog import downloads, pdq_endpoint, URL  # noqa: F401


@pytest.mark.parametrize("source_kind", ["nccn", "pdq"])
def test_interrupt_then_migrate_retains_raw_checkpoints_and_passes_audit(
    source_kind, guideline_run, fake_endpoint, tmp_path, downloads, pdq_endpoint, monkeypatch,  # noqa: F811
):
    from matchminer_ai.trials import prepare_pdq_library, summarize_pdq
    import matchminer_ai.llm.structured as wire

    root, output, config = guideline_run
    # Each fixture installs its own endpoint; choose the matching synthetic source.
    if source_kind == "pdq":
        root = tmp_path / "pdq_source"
        prepare_pdq_library({"fictional": URL}, output_dir=root)
        summarize = summarize_pdq
        calls = pdq_endpoint
    else:
        fake_endpoint = globals()["fake_endpoint"].__wrapped__(monkeypatch)
        calls = fake_endpoint
        summarize = summarize_guidelines
    config.remote["max_concurrent_requests"] = 1
    token = CancellationToken()
    original_complete = Client.complete

    def complete(client, job, *args, **kwargs):
        value = original_complete(client, job, *args, **kwargs)
        if job.startswith("extract-"):
            token.cancel()
        return value

    with monkeypatch.context() as m:
        m.setattr(Client, "complete", complete)
        with cancellation_scope(token), pytest.raises(InferenceCancelled):
            summarize(root, disease="fictional", output_dir=output, config=config)
    assert read_json(output / "status.json")["status"] == "interrupted"
    accepted = list((output / "checkpoints").glob("*/accepted.json"))
    assert accepted
    preserved = {p: p.read_bytes() for p in (output / "checkpoints").glob("*/attempt-*.json")}
    config_id = read_json(output / "run_config.json")["config_sha256"]
    original_url = config.remote["server_urls"][0]
    config.remote["server_urls"] = ["http://replacement.invalid:8001/v1"]
    with pytest.raises(ValueError, match="allow_endpoint_change"):
        summarize(root, disease="fictional", output_dir=output, config=config)
    assert not (output / "endpoint_history.json").exists()
    calls.clear()
    frame = summarize(root, disease="fictional", output_dir=output, config=config,
                      allow_endpoint_change=True)
    assert len(frame) == 1
    assert all(p.read_bytes() == data for p, data in preserved.items())
    saved = read_json(output / "run_config.json")
    assert saved["config_sha256"] == config_id
    assert saved["llm"]["base_url"] == original_url
    assert saved["llm"]["transport_base_url"] == config.remote["server_urls"][0]
    assert [h["endpoint"] for h in read_json(output / "endpoint_history.json")] == [
        original_url, config.remote["server_urls"][0],
    ]
    assert all(url.startswith("http://replacement.invalid:8001/") for url, *_ in calls)
    requests = [read_json(p) for p in (output / "checkpoints").glob("*/request.json")]
    assert sum(r["job"].startswith("extract-") for r in requests) == 1
    assert any(r.get("transport_base_url") for r in requests)
    assert audit_guideline_catalog(root, disease="fictional", output_dir=output)["status"] == "passed"
    with monkeypatch.context() as m:
        m.setattr(wire, "urlopen", lambda *a, **k: pytest.fail("Completed resume is offline"))
        assert len(summarize(root, disease="fictional", output_dir=output, config=config)) == 1
    # Recorded actual destinations are audited, including retries.
    attempt = next(p for p in (output / "checkpoints").glob("*/request-attempt-*.json")
                   if read_json(p).get("transport_base_url"))
    changed = read_json(attempt)
    changed["transport_base_url"] = "http://unrecorded.invalid/v1"
    attempt.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="endpoint history"):
        audit_guideline_catalog(root, disease="fictional", output_dir=output)


@pytest.mark.parametrize("change", ["model", "context", "sampling", "reasoning"])
def test_migration_rejects_incompatible_settings_even_with_opt_in(change, monkeypatch):
    old = StructuredConfig(base_url="http://old.invalid/v1", model="synthetic-gemma4")
    new = replace(old, base_url="http://new.invalid/v1")
    info = {"id": old.model, "max_model_len": old.context_window}
    if change == "model":
        info["id"] = "different-model"
    elif change == "context":
        info["max_model_len"] = 131072
    elif change == "sampling":
        new = replace(new, temperature=0.5)
    else:
        new = replace(new, thinking="off")
    monkeypatch.setattr(Client, "discover", lambda self: info)
    with pytest.raises(ValueError, match="requires"):
        resume_transport(new, {"llm": old.public_dict()}, allow_endpoint_change=True)


def test_migrated_request_tokens_are_counted_at_new_transport(monkeypatch, tmp_path):
    calls = []
    old = StructuredConfig(base_url="http://old.invalid/v1", model="synthetic-gemma4")
    def http(client, endpoint, body=None, **kwargs):
        calls.append(client.config.transport_base_url or client.config.base_url)
        return {"count": 123, "max_model_len": 262144}
    monkeypatch.setattr(Client, "_http", http)
    messages = [{"role": "user", "content": "Invented text."}]
    Client(old, tmp_path).count_tokens(messages)
    Client(replace(old, transport_base_url="http://new.invalid/v1"), tmp_path).count_tokens(messages)
    assert calls == ["http://old.invalid/v1", "http://new.invalid/v1"]


def test_pdq_batch_interruption_status_and_resume(tmp_path, downloads, monkeypatch):  # noqa: F811
    from matchminer_ai import load_default_preset
    from matchminer_ai.trials import prepare_pdq_library
    from matchminer_ai.cancellation import check_cancelled

    spec = importlib.util.spec_from_file_location(
        "pdq_cli", Path(__file__).parents[1] / "examples/extract_pdq.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "source", tmp_path / "output"
    inventory = prepare_pdq_library({"fictional": URL}, output_dir=source)
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=["http://chosen.invalid/v1"],
                         max_concurrent_requests=2)
    token = CancellationToken()
    def interrupted(*args, **kwargs):
        token.cancel()
        check_cancelled()
    monkeypatch.setattr(module, "summarize_pdq", interrupted)
    assert module.run_batch(inventory, source, output, config, token=token) == 130
    assert read_json(output / "collection.json")["status"] == "interrupted"
    monkeypatch.setattr(module, "summarize_pdq", lambda *a, **k: (
        inventory, {"validation": {"status": "passed"}}
    ))
    assert module.run_batch(inventory, source, output, copy.deepcopy(config)) == 0
    assert read_json(output / "collection.json")["status"] == "complete"
