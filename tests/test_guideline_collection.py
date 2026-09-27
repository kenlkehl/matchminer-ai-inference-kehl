"""Collection orchestration tests use only fabricated source manifests."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_guideline_extraction import make_library


def runner_module():
    path = (
        Path(__file__).resolve().parents[1] / "examples/extract_guideline_collection.py"
    )
    spec = importlib.util.spec_from_file_location("collection_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collection_records_failures_and_continues_other_diseases(
    tmp_path, monkeypatch
):
    root = make_library(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    second = dict(manifest["guidelines"][0], folder="another")
    manifest["guidelines"].append(second)
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "collection-results"
    module = runner_module()
    calls = []

    def summarize(source, *, disease, output_dir, config, progress_callback):
        calls.append(disease)
        assert config.remote["max_concurrent_requests"] == 32
        assert config.guideline["sampling_profile"] == "auto"
        if disease == "fictional":
            raise RuntimeError("Synthetic failure")
        return [1, 2]

    monkeypatch.setattr(module, "summarize_guidelines", summarize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--source",
            str(root),
            "--output-dir",
            str(output),
            "--endpoint",
            "http://synthetic.invalid/v1",
        ],
    )
    assert module.main() is True
    state = json.loads((output / "collection.json").read_text())
    assert set(calls) == {"fictional", "another"}
    assert state["status"] == "failed"
    assert state["diseases"]["another"]["status"] == "complete"
    assert state["diseases"]["another"]["spaces"] == 2
    assert state["failed_diseases"] == ["fictional"]
    original = (output / "collection.json").read_bytes()
    recovered = []

    def recover(*args, **kwargs):
        recovered.append(kwargs["disease"])
        return [1]

    monkeypatch.setattr(module, "summarize_guidelines", recover)
    sys.argv.append("--retry-failed")
    assert module.main() is False
    assert recovered == ["fictional"]
    assert (output / "collection.json").read_bytes() == original
    assert json.loads((output / "recovery.json").read_text())["status"] == "complete"
    sys.argv.pop()
    # Retrying uses the same disease output directories and changes the summary
    # only after the underlying public pipeline has resumed/audited each disease.
    monkeypatch.setattr(module, "summarize_guidelines", lambda *args, **kwargs: [1])
    assert module.main() is False
    assert json.loads((output / "collection.json").read_text())["status"] == "complete"


def test_new_catalogs_use_requested_server_and_checkpoints_keep_owner(tmp_path):
    module = runner_module()
    first, second = "http://original.invalid/v1", "http://replica.invalid/v1"
    for disease in ("active", "complete"):
        (tmp_path / disease).mkdir()
        (tmp_path / disease / "run_config.json").write_text(
            json.dumps({"llm": {"base_url": first}})
        )
    previous = {
        "active": {"status": "running"},
        "complete": {"status": "complete"},
        "pending": {"status": "pending", "endpoint": first},
    }
    assigned = module.assign_endpoints(
        ["active", "complete", "pending", "new"],
        tmp_path,
        [first, second],
        previous,
        new_endpoint=second,
    )
    assert assigned == {
        "active": first,
        "complete": first,
        "pending": second,
        "new": second,
    }
    with pytest.raises(ValueError, match="require endpoint"):
        module.assign_endpoints(
            ["active"], tmp_path, [second], previous, new_endpoint=second
        )


@pytest.mark.parametrize("difference", ["model", "context_window"])
def test_replica_preflight_rejects_different_model_or_context(monkeypatch, difference):
    module = runner_module()

    def discover(client):
        other = client.config.base_url.endswith("replica/v1")
        return {
            "id": "different" if other and difference == "model" else "same",
            "max_model_len": 131072
            if other and difference == "context_window"
            else 262144,
        }

    monkeypatch.setattr(module.StructuredClient, "discover", discover)
    with pytest.raises(ValueError, match="same model and context"):
        module.validate_endpoints(["http://primary/v1", "http://replica/v1"], None)


def test_collection_dispatches_unstarted_catalog_to_replica(tmp_path, monkeypatch):
    module = runner_module()
    root = make_library(tmp_path)
    output = tmp_path / "collection-results"
    first, second = "http://original.invalid/v1", "http://replica.invalid/v1"
    calls = []
    monkeypatch.setattr(
        module,
        "validate_endpoints",
        lambda endpoints, model: {
            e: {"model": "synthetic", "context_window": 262144} for e in endpoints
        },
    )

    def summarize(*args, **kwargs):
        c = kwargs["config"]
        calls.append(c.remote["server_urls"])
        assert c.remote["max_concurrent_requests"] == 32
        assert c.guideline["reasoning_effort"] == "xhigh"
        return [1]

    monkeypatch.setattr(module, "summarize_guidelines", summarize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--source",
            str(root),
            "--output-dir",
            str(output),
            "--endpoint",
            first,
            "--additional-endpoint",
            second,
            "--new-disease-endpoint",
            second,
        ],
    )
    assert module.main() is False
    state = json.loads((output / "collection.json").read_text())
    assert calls == [[second]]
    assert state["identity"]["endpoint"] == first
    assert state["new_disease_endpoint"] == second
    assert state["diseases"]["fictional"]["endpoint"] == second


def test_verified_source_override_preserves_endpoint_and_requires_same_edition(tmp_path, monkeypatch):
    root = make_library(tmp_path / "original")
    alternate = make_library(tmp_path / "verified")
    for library in (root, alternate):
        source_hash = json.loads((library / "fictional/manifest.json").read_text())["source_sha256"]
        manifest_path = library / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["guidelines"][0]["source_sha256"] = source_hash
        manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "fresh-results"
    module = runner_module()
    calls = []

    def summarize(source, *, disease, output_dir, config, progress_callback):
        calls.append((str(source), disease, config.remote["server_urls"]))
        return [1]

    monkeypatch.setattr(module, "summarize_guidelines", summarize)
    monkeypatch.setattr(sys, "argv", [
        "runner", "--source", str(root), "--output-dir", str(output),
        "--endpoint", "http://chosen.invalid:8001/v1", "--source-override", "fictional", str(alternate),
    ])
    assert module.main() is False
    assert calls == [(str(alternate.resolve()), "fictional", ["http://chosen.invalid:8001/v1"])]
    identity = json.loads((output / "collection.json").read_text())["identity"]
    assert identity["source_overrides"] == {"fictional": str(alternate.resolve())}
    assert identity["skipped"] == {}
    manifest_path = alternate / "fictional/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = "different-edition"
    manifest_path.write_text(json.dumps(manifest))
    calls.clear()
    with pytest.raises(SystemExit):
        module.main()
    assert not calls
