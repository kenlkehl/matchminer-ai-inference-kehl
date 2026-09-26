"""Synthetic end-to-end citation review and tamper/resume coverage."""

import copy
import io
import json

import pytest

from matchminer_ai._storage import read_json, atomic_json
from matchminer_ai.trials import (
    summarize_guidelines,
    review_guideline_citations,
    audit_guideline_catalog,
    load_guideline_catalog,
)
from matchminer_ai.trials._guideline_citation_review import apply_review, REVIEW
from matchminer_ai.trials._guideline_sources import load_guideline
from test_guideline_api import guideline_run, fake_endpoint  # noqa: F401
from test_guideline_extraction import TEXT, state


def review(unsupported=False):
    def item(name):
        return {
            "name": name,
            "evidence": [{"page_id": "p0002", "source_text": TEXT}],
            "issues": [],
        }

    value = {
        "population": item(state()["name"]),
        "diagnostic_workup": [item("Synthetic test A")],
        "treatment_options": [],
    }
    if unsupported:
        value["diagnostic_workup"][0].update(
            evidence=[], issues=["Synthetic source does not support the panel."]
        )
    return value


@pytest.fixture
def completed(guideline_run, fake_endpoint, monkeypatch):  # noqa: F811
    root, output, config = guideline_run
    frame = summarize_guidelines(
        root, disease="fictional", output_dir=output, config=config
    )
    import matchminer_ai.llm.structured as structured

    previous = structured.urlopen
    calls = []
    responses = [review()]

    def respond(request, timeout):
        body = json.loads(request.data) if request.data else {}
        if (
            body.get("response_format", {}).get("json_schema", {}).get("schema")
            != REVIEW
        ):
            return previous(request, timeout)
        calls.append(body)
        event = {
            "model": "synthetic-gemma4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": json.dumps(responses[0])},
                    "finish_reason": "stop",
                }
            ],
        }
        return io.BytesIO(
            ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
        )

    monkeypatch.setattr(structured, "urlopen", respond)
    return root, output, config, frame, calls, responses


@pytest.mark.parametrize("unsupported", [False, True])
def test_public_review_preserves_clinical_content_and_resumes_offline(
    completed, tmp_path, monkeypatch, unsupported
):
    _, original, config, frame, calls, responses = completed
    responses[0] = review(unsupported)
    output = tmp_path / "reviewed"
    before = (original / "paradigms.jsonl").read_bytes()
    result, metadata = review_guideline_citations(
        original, output_dir=output, config=config, return_metadata=True
    )
    assert len(calls) == 1
    old = frame.iloc[0].to_dict()
    new = result.iloc[0].to_dict()
    for key in old:
        if key not in {"evidence", "diagnostic_workup", "treatment_options"}:
            assert old[key] == new[key]
    for key in {"diagnostic_workup", "treatment_options"}:
        for a, b in zip(old[key], new[key]):
            assert {k: v for k, v in a.items() if k != "evidence"} == {
                k: v for k, v in b.items() if k not in {"evidence", "citation_review"}
            }
    assert (original / "paradigms.jsonl").read_bytes() == before
    assert metadata["validation"]["unresolved_items"] == int(unsupported)
    item = result.iloc[0]["diagnostic_workup"][0]
    assert bool(item["evidence"]) != unsupported
    assert item["citation_review"]["status"] == (
        "unresolved" if unsupported else "supported_by_model_review"
    )
    loaded = load_guideline_catalog(output)
    assert loaded.iloc[0]["diagnostic_workup"] == result.iloc[0]["diagnostic_workup"]
    if unsupported:
        assert "Source support unresolved" in (output / "report.md").read_text()
    monkeypatch.setattr(
        "matchminer_ai.llm.structured.urlopen",
        lambda *a, **k: pytest.fail("resume contacted endpoint"),
    )
    resumed = review_guideline_citations(original, output_dir=output, config=config)
    assert resumed.to_dict("records") == result.to_dict("records")
    config.guideline["remote"]["request_params"]["temperature"] = 0.3
    with pytest.raises(ValueError, match="changed"):
        review_guideline_citations(original, output_dir=output, config=config)


@pytest.mark.parametrize("change", ["quote", "clinical", "review", "original", "csv"])
def test_audit_rejects_tampering(completed, tmp_path, change):
    root, original, config, _, _, _ = completed
    output = tmp_path / "reviewed"
    review_guideline_citations(original, output_dir=output, config=config)
    if change in {"quote", "clinical"}:
        path = output / "paradigms.jsonl"
        row = json.loads(path.read_text())
        if change == "quote":
            row["evidence"][0]["quote"] = "Unrelated synthetic source text"
        else:
            row["diagnostic_workup"][0]["name"] = "Changed test"
        path.write_text(json.dumps(row) + "\n")
    elif change == "review":
        path = output / "reviews/citation-0001.json"
        value = read_json(path)
        value["result"]["population"]["issues"] = ["Invented reviewer finding"]
        atomic_json(path, value)
    elif change == "original":
        with (output / "original_paradigms.jsonl").open("a") as stream:
            stream.write("\n")
    else:
        (output / "trial_spaces.csv").write_text("space_trial_id\nchanged\n")
    with pytest.raises(ValueError):
        audit_guideline_catalog(root, disease="fictional", output_dir=output)


def test_review_rejects_dropped_renamed_unsupported_and_invented_evidence(completed):
    root, _, _, frame, _, _ = completed
    pages = load_guideline(root, "fictional").pages
    original = frame.iloc[0].to_dict()
    changes = []
    value = review()
    value["diagnostic_workup"] = []
    changes.append(value)
    value = review()
    value["diagnostic_workup"][0]["name"] = "Other"
    changes.append(value)
    value = review()
    value["population"]["evidence"] = []
    changes.append(value)
    value = review()
    value["population"]["evidence"][0]["source_text"] = "Invented"
    changes.append(value)
    for value in changes:
        with pytest.raises(ValueError):
            apply_review(original, value, pages)
    assert original == frame.iloc[0].to_dict()


def test_loader_rejects_empty_evidence_without_explicit_issue(completed):
    _, _, _, frame, _, _ = completed
    records = copy.deepcopy(frame.to_dict("records"))
    records[0]["diagnostic_workup"][0]["evidence"] = []
    import pandas as pd

    with pytest.raises(ValueError, match="retain evidence"):
        load_guideline_catalog(pd.DataFrame(records))


def test_collection_continues_after_failure_and_reconciles_resume(
    tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path
    from matchminer_ai import load_default_preset
    from test_guideline_extraction import make_library

    path = (
        Path(__file__).resolve().parents[1] / "examples/review_guideline_collection.py"
    )
    spec = importlib.util.spec_from_file_location("citation_collection", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = make_library(tmp_path)
    catalogs = {}
    for name in ["one", "two"]:
        folder = tmp_path / name
        folder.mkdir()
        atomic_json(
            folder / "sources.json", {"input_directory": str(root / "fictional")}
        )
        catalogs[name] = str(folder)
    destinations = []

    def first(path, **kwargs):
        destinations.append(kwargs["output_dir"])
        if Path(path).name == "one":
            raise RuntimeError("Synthetic failure")
        return None, {"validation": {"paradigms": 2, "unresolved_items": 1}}

    monkeypatch.setattr(runner, "review_guideline_citations", first)
    result = runner.run_collection(
        catalogs, tmp_path / "reviews", load_default_preset(), disease_workers=2
    )
    assert result["status"] == "failed"
    assert result["diseases"]["two"]["status"] == "complete"

    def second(path, **kwargs):
        assert kwargs["output_dir"] in destinations
        return None, {"validation": {"paradigms": 2, "unresolved_items": 0}}

    monkeypatch.setattr(runner, "review_guideline_citations", second)
    result = runner.run_collection(
        catalogs, tmp_path / "reviews", load_default_preset(), disease_workers=2
    )
    assert result["status"] == "complete"
    assert read_json(tmp_path / "reviews/collection.json")["counts"] == {"complete": 2}
