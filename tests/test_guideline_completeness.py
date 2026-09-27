"""Completeness regressions use fabricated populations, never guideline text."""

import copy
import json
from types import SimpleNamespace

import pytest

from matchminer_ai._storage import atomic_json, digest, read_json
from matchminer_ai.trials._guideline_canonical import (
    CATALOG, SELECT_TASK, build_call, consolidate, validate_catalog,
)
from matchminer_ai.trials._guideline_completeness import (
    COVERAGE, PROMPT_FILES, TASK, CoverageError, require_coverage,
    review_coverage, validate_report, validate_reviews,
)
from matchminer_ai.trials._guideline_generation import Client
from matchminer_ai.llm.structured import StructuredConfig
from matchminer_ai.trials._guideline_pipeline import run_guideline
from matchminer_ai.trials._guideline_sources import load_guideline
from test_guideline_extraction import (
    CATALOG_TASK, catalog, extraction, make_library, prompts, quoted_state, state,
)


def populations():
    first = {"candidate_id": "opaque-do-not-send", **state()}
    second = copy.deepcopy(first)
    second["candidate_id"] = "another-opaque-record"
    second["name"] = "Fictional beta after prior treatment"
    second["space"]["prior_treatment_required"] = "Prior fictional therapy"
    return [first, second]


def judge(messages):
    payload = json.loads(messages[1]["content"])
    known = {s["name"] for s in payload["proposed_catalog"]}
    return {"reviews": [
        {"input_name": row["name"],
         "status": "represented" if row["name"] in known else "missing",
         "matched_population_names": [row["name"]] if row["name"] in known else [],
         "reason": "Synthetic review of retained treatment distinction"}
        for row in payload["input_populations"]
    ]}


def test_drop_is_rejected_and_review_never_receives_internal_identifiers():
    inputs = populations()
    calls = []

    class Reviewer:
        config = SimpleNamespace(max_concurrent_requests=32)

        def fits(self, messages):
            return True

        def complete(self, job, messages, schema, validator):
            rendered = json.dumps(messages)
            assert "candidate_id" not in rendered
            assert "opaque" not in rendered
            assert schema == COVERAGE
            calls.append(messages)
            value = judge(messages)
            validator(value)
            return value

    report = review_coverage(Reviewer(), inputs, inputs[:1])
    assert len(report["reviews"]) == 2
    assert len(calls) == 1
    with pytest.raises(CoverageError, match="1 of 2"):
        require_coverage(report)
    report = review_coverage(Reviewer(), inputs, inputs)
    validate_report(report, inputs, inputs, set())
    assert len(calls) == 1  # Exact clinical definitions need no model bookkeeping.


def test_review_must_account_for_every_input_and_reference_existing_populations():
    inputs = populations()
    value = {"reviews": [{"input_name": inputs[0]["name"], "status": "represented",
                           "matched_population_names": [inputs[0]["name"]], "reason": "Same state"}]}
    with pytest.raises(ValueError, match="every input"):
        validate_reviews(value, inputs, inputs)
    value["reviews"].append({**value["reviews"][0], "input_name": inputs[1]["name"]})
    value["reviews"][1]["matched_population_names"] = ["Invented population"]
    with pytest.raises(ValueError, match="absent"):
        validate_reviews(value, inputs, inputs)
    value["reviews"][1]["matched_population_names"] = []
    with pytest.raises(ValueError, match="must name"):
        validate_reviews(value, inputs, inputs)
    value["reviews"][1]["status"] = "missing"
    validate_reviews(value, inputs, inputs)


def test_cached_single_state_cannot_bypass_guard_and_full_catalog_is_retried(tmp_path, monkeypatch):
    guideline = load_guideline(make_library(tmp_path), "fictional")
    inputs = populations()
    client = Client(StructuredConfig(model="synthetic", tokenizer_mode="bytes"), tmp_path / "cache")
    output = tmp_path / "results"
    calls = []

    def respond(endpoint, body, **kwargs):
        if body["messages"][0]["content"] == TASK:
            value = judge(body["messages"])
        else:
            calls.append(body)
            value = catalog()
            if len(calls) > 1:
                assert "failed its population-coverage review" in body["messages"][-1]["content"]
                value["states"].append({k: inputs[1][k] for k in value["states"][0]})
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(client, "_http", respond)
    _, _, messages = build_call(client, guideline, inputs)
    # Simulate an old accepted response with no completeness validator.
    client.complete("catalog-content-0001", messages, CATALOG,
                    lambda v: validate_catalog(v, guideline.pages))

    def run_jobs(jobs, workers, function):
        return {key: function(key, data) for key, data in jobs}, {}

    result = consolidate(client, guideline, inputs, output, 1, run_jobs, lambda _: None)
    assert len(result["groups"]) == 2
    assert len(calls) == 2  # Original cache reused, then one corrected full response.
    assert all(r["status"] == "represented" for r in read_json(output / "canonical_coverage.json")["reviews"])


def test_nonidentical_review_requires_accepted_response_provenance():
    inputs = populations()[:1]
    proposed = copy.deepcopy(inputs)
    proposed[0]["name"] = "Equivalent fictional wording"
    row = {"input_name": inputs[0]["name"], "status": "represented",
           "matched_population_names": [proposed[0]["name"]], "reason": "Equivalent wording"}
    result = {"reviews": [row]}
    from matchminer_ai.trials._guideline_completeness import VERSION, population
    report = {"version": VERSION, "input_sha256": digest([population(c) for c in inputs]),
              "catalog_sha256": digest([population(c) for c in proposed]), "reviews": [row],
              "review_batches": [{"input_positions": [0], "job": "coverage", "result_sha256": digest(result)}]}
    with pytest.raises(ValueError, match="accepted review"):
        validate_report(report, inputs, proposed, set())
    validate_report(report, inputs, proposed, {("coverage", digest(result))})
    report["reviews"][0]["reason"] = "Altered review"
    with pytest.raises(ValueError, match="accepted review"):
        validate_report(report, inputs, proposed, {("coverage", digest(result))})


def test_cross_batch_selection_cannot_drop_a_distinct_population(tmp_path, monkeypatch):
    guideline = load_guideline(make_library(tmp_path), "fictional")
    inputs = populations()
    client = Client(StructuredConfig(model="synthetic", tokenizer_mode="bytes"), tmp_path / "cache")
    from matchminer_ai.trials import _guideline_canonical as module
    monkeypatch.setattr(module, "MAX_CANDIDATES_PER_BATCH", 1)
    selections = []

    def respond(endpoint, body, **kwargs):
        messages = body["messages"]
        if messages[0]["content"] == TASK:
            value = judge(messages)
        elif messages[1]["content"].startswith(SELECT_TASK):
            selections.append(body)
            chosen = inputs[:1] if len(selections) == 1 else inputs
            value = {"spaces": [s["space"] for s in chosen],
                     "context_only_topics": [], "uncertainties": []}
        else:
            item = inputs[1] if inputs[1]["name"] in messages[1]["content"] else inputs[0]
            value = catalog()
            value["states"] = [{k: item[k] for k in value["states"][0]}]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(client, "_http", respond)
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda _: None)

    def run_jobs(jobs, workers, function):
        return {key: function(key, data) for key, data in jobs}, {}

    result = consolidate(client, guideline, inputs, tmp_path / "results", 1, run_jobs, lambda _: None)
    assert len(selections) == 2
    assert len(result["groups"]) == 2


def test_coverage_upgrade_preserves_exact_old_requests_but_rejects_changed_settings(tmp_path, monkeypatch):
    guideline = load_guideline(make_library(tmp_path), "fictional")
    output = tmp_path / "results"
    config = StructuredConfig(model="synthetic", tokenizer_mode="bytes")

    def respond(self, endpoint, body, **kwargs):
        prompt = body["messages"][1]["content"]
        value = extraction() if prompt.startswith(prompts.EXTRACT_TASK) else (
            catalog() if prompt.startswith(CATALOG_TASK) else quoted_state())
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(Client, "_http", respond)
    run_guideline(guideline, output, config, workers=1)
    old = read_json(output / "run_config.json")
    for name in PROMPT_FILES:
        old["prompt_resources_sha256"].pop(name)
    old["stage_versions"].pop("catalog_coverage")
    identity = {k: copy.deepcopy(v) for k, v in old.items() if k not in ("config_sha256", "runtime", "stage_versions")}
    for key in ("timeout", "attempts", "api_key_env", "stream", "max_concurrent_requests"):
        identity["llm"].pop(key)
    old["config_sha256"] = digest(identity)
    atomic_json(output / "run_config.json", old)

    def no_request(*args, **kwargs):
        raise AssertionError("Compatible saved work should not be regenerated")

    monkeypatch.setattr(Client, "_http", no_request)
    run_guideline(guideline, output, config, workers=1)
    assert "catalog_coverage" in read_json(output / "run_config.json")["stage_versions"]
    old["llm"]["temperature"] = 0.77
    identity["llm"]["temperature"] = 0.77
    old["config_sha256"] = digest(identity)
    atomic_json(output / "run_config.json", old)
    with pytest.raises(ValueError, match="changed"):
        run_guideline(guideline, output, config, workers=1)
