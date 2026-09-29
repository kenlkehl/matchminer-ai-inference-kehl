"""Exhausted population rewrites split safely and resume their successful leaves."""

import copy
import json

import pytest

from matchminer_ai._storage import atomic_json, read_json
from matchminer_ai.llm.structured import StructuredConfig
from matchminer_ai.trials import _guideline_canonical as canonical
from matchminer_ai.trials._guideline_batching import can_split
from matchminer_ai.trials._guideline_completeness import TASK, validate_report
from matchminer_ai.trials._guideline_generation import Client
from matchminer_ai.trials._guideline_pipeline import parallel_jobs
from matchminer_ai.trials._guideline_sources import load_guideline
from test_guideline_completeness import judge, populations
from test_guideline_extraction import TEXT, make_library


@pytest.mark.parametrize("saved_failure", [False, True])
@pytest.mark.parametrize("failure_kind", ["coverage", "blank", "nonexistent"])
def test_failed_batch_splits_without_repeating_successful_or_exhausted_batches(
    tmp_path, monkeypatch, saved_failure, failure_kind,
):
    guideline = load_guideline(
        make_library(tmp_path, TEXT + "\n\nAdditional synthetic source"), "fictional",
    )
    inputs = populations()
    third = copy.deepcopy(inputs[0])
    third.update(candidate_id="third-opaque-id", name="Fictional gamma")
    third["space"]["cancer_burden_allowed"] = "Fictional burden gamma"
    inputs.append(third)
    output = tmp_path / "results"
    client = Client(
        StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=1),
        output / "checkpoints",
    )
    monkeypatch.setattr(canonical, "MAX_CANDIDATES_PER_BATCH", 2)
    calls = []

    def respond(endpoint, body, **kwargs):
        assert body["max_tokens"] == 100000
        messages = body["messages"]
        assert "opaque" not in json.dumps(messages)
        if messages[0]["content"] == TASK:
            value = judge(messages)
        elif messages[1]["content"].startswith(canonical.SELECT_TASK):
            text = messages[1]["content"].split("\n\nRequired JSON schema:\n", 1)[1]
            _, end = json.JSONDecoder().raw_decode(text)
            states = json.JSONDecoder().raw_decode(text[end:].lstrip())[0]["source_backed_definitions"]
            value = {"spaces": [s["space"] for s in states], "context_only_topics": [], "uncertainties": []}
        else:
            text = messages[1]["content"].split("FINAL TASK AND DISEASE-STATE DESCRIPTIONS:\n", 1)[1]
            descriptions = json.JSONDecoder().raw_decode(text[len(canonical.TASK) + 1:])[0]
            calls.append([s["name"] for s in descriptions])
            # Larger requests lose a population or invent a citation; singleton
            # responses retain every original definition and valid citation.
            value = {"states": [
                {key: s[key] for key in canonical.LEAN_STATE["properties"]}
                for s in (descriptions[:1] if failure_kind == "coverage" else descriptions)
            ], "context_only_topics": [], "uncertainties": []}
            if len(descriptions) > 1 and failure_kind != "coverage":
                value["states"][0]["evidence"] = [{
                    "page_id": "p0002", "line_ids": [2 if failure_kind == "blank" else 99],
                }]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(client, "_http", respond)
    if saved_failure:
        atomic_json(output / "run_config.json", {"source_fingerprint": guideline.fingerprint})
        atomic_json(output / "candidates.json", inputs)
        atomic_json(output / "canonical_batches.json", {
            "batches": {}, "failures": {
                "catalog-content-0001": "EndpointError: exhausted 6 attempts: " + {
                    "coverage": "Catalog omitted or broadened 1 of 2 input populations",
                    "blank": "p0002: evidence may not consist entirely of blank lines",
                    "nonexistent": "p0002: nonexistent source line IDs [99]; maximum is 3",
                }[failure_kind],
            },
        })
    result = canonical.consolidate(
        client, guideline, inputs, output, 1, parallel_jobs, lambda _: None,
    )
    assert {s["name"] for s in result["groups"]} == {s["name"] for s in inputs}
    assert {s["name"]: s["space"] for s in result["groups"]} == {
        s["name"]: s["space"] for s in inputs
    }
    for item in result["groups"]:
        canonical.validate_catalog(
            {
                "states": [{k: item[k] for k in canonical.LEAN_STATE["properties"]}],
                "context_only_topics": [], "uncertainties": [],
            },
            guideline.pages, "Fictional",
        )
    validate_report(read_json(output / "canonical_coverage.json"), inputs, result["groups"])
    assert calls.count([third["name"]]) == 1
    assert calls.count([inputs[0]["name"], inputs[1]["name"]]) == (0 if saved_failure else 1)
    batches = read_json(output / "canonical_batches.json")
    assert not batches["failures"]
    assert set(batches["batches"]) == {
        "catalog-content-0001-part0001", "catalog-content-0001-part0002", "catalog-content-0002",
    }
    assert sorted(i for b in batches["batches"].values() for i in b["input_candidate_ids"]) == sorted(s["candidate_id"] for s in inputs)
    assert read_json(output / "canonical_context.json")["batch_count"] == 3
    assert (output / "canonical_selection.json").exists()
    ledger = read_json(output / "canonical_batch_splits.json")
    assert ledger["splits"]["catalog-content-0001"]["child_size"] == 1

    monkeypatch.setattr(client, "_http", lambda *a, **k: pytest.fail("Resume must reuse successful children"))
    assert canonical.consolidate(client, guideline, inputs, output, 1, parallel_jobs, lambda _: None) == result
    ledger["splits"]["catalog-content-0001"]["input_sha256"] = "tampered"
    atomic_json(output / "canonical_batch_splits.json", ledger)
    with pytest.raises(ValueError, match="receipt differs"):
        canonical.consolidate(client, guideline, inputs, output, 1, parallel_jobs, lambda _: None)


@pytest.mark.parametrize("error,size,expected", [
    ("EndpointError: exhausted 6 attempts: Catalog omitted or broadened 2", 77, True),
    ("EndpointError: exhausted 6 attempts: field mixes AND and OR", 2, True),
    ("EndpointError: exhausted 6 attempts: Catalog omitted or broadened 1", 1, False),
    ("EndpointError: exhausted 6 attempts: HTTP 503", 77, False),
    ("EndpointError: exhausted 6 attempts: p0002: evidence may not consist entirely of blank lines", 62, True),
    ("EndpointError: exhausted 6 attempts: p0002: nonexistent source line IDs [99]; maximum is 3", 62, True),
    ("EndpointError: exhausted 6 attempts: Unknown or unsupplied evidence page: p9999", 2, True),
    ("EndpointError: exhausted 6 attempts: Each state and option needs at least one source-line citation", 2, True),
    ("EndpointError: exhausted 6 attempts: p0002: evidence may not consist entirely of blank lines", 1, False),
    ("EndpointError: exhausted 6 attempts: evidence server unavailable", 62, False),
    ("Catalog omitted or broadened 2", 77, False),
])
def test_splitting_is_bounded_and_does_not_retry_transport_failures(error, size, expected):
    assert can_split(error, size) is expected


@pytest.mark.parametrize("error", [
    "Catalog omitted or broadened population",
    "p0002: evidence may not consist entirely of blank lines",
])
def test_split_stops_at_unresolved_singletons_without_claiming_success(tmp_path, error):
    from matchminer_ai.trials._guideline_batching import run_batches

    guideline = load_guideline(make_library(tmp_path), "fictional")
    inputs, calls = populations(), []

    def fail(key, data):
        calls.append(key)
        raise RuntimeError("exhausted 6 attempts: " + error)

    values, failures = run_batches(
        [(inputs, None)], guideline, inputs, tmp_path / "derived", 1,
        parallel_jobs, fail, lambda members: None, lambda message: None,
    )
    assert not values
    assert len(calls) == 3
    assert set(failures) == {"catalog-content-0001-part0001", "catalog-content-0001-part0002"}
