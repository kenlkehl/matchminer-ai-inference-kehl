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

        def complete(self, job, messages, schema, validator, **kwargs):
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


def test_coverage_keeps_all_full_findings_for_retry_feedback():
    rows = [{"input_name": f"Synthetic population {i}", "status": "missing",
             "matched_population_names": [], "reason": "Long synthetic explanation. " * 20 + f"Exact restriction {i}"}
            for i in range(9)]
    with pytest.raises(CoverageError) as error:
        require_coverage({"reviews": rows})
    assert json.loads(str(error.value).split("Complete coverage findings:\n")[1]) == rows


def test_equivalent_comparison_notation_is_exact_but_threshold_changes_are_not():
    from matchminer_ai.trials._guideline_completeness import identical_population_key

    inputs = populations()[:1]
    inputs[0]["space"]["age_range_allowed"] = ">=18 years AND <=80 years"
    proposed = copy.deepcopy(inputs)
    proposed[0]["space"]["age_range_allowed"] = "≥18 years AND ≤80 years"

    class Reviewer:
        config = SimpleNamespace(max_concurrent_requests=1)

        def fits(self, messages):
            return True

        def complete(self, *args, **kwargs):
            pytest.fail("Identical notation needs no model judgment")

    original = copy.deepcopy(inputs)
    report = review_coverage(Reviewer(), inputs, proposed)
    validate_report(report, inputs, proposed, set())
    assert inputs == original
    for changed in (">18 years AND <=80 years", ">=19 years AND <=80 years", ">=18 years OR <=80 years"):
        altered = copy.deepcopy(proposed)
        altered[0]["space"]["age_range_allowed"] = changed
        assert identical_population_key(inputs[0]) != identical_population_key(altered[0])
        bad_report = copy.deepcopy(report)
        from matchminer_ai.trials._guideline_completeness import population
        bad_report["catalog_sha256"] = digest([population(s) for s in altered])
        with pytest.raises(ValueError, match="Nonidentical"):
            validate_report(bad_report, inputs, altered, set())


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


@pytest.mark.parametrize("failure", ["name", "count", "matched_name"])
@pytest.mark.parametrize("missing", [False, True])
def test_exhausted_review_bookkeeping_splits_without_losing_judgments(
    tmp_path, monkeypatch, failure, missing,
):
    from matchminer_ai.trials._guideline_audit import audit_accepted_response

    inputs = populations()
    # Same clinical names can have different definitions; no name-keyed remapping.
    inputs[1]["name"] = inputs[0]["name"]
    proposed = copy.deepcopy(inputs)
    for row in proposed:
        row["name"] = "Proposed " + row["name"]
    client = Client(StructuredConfig(
        model="synthetic", tokenizer_mode="bytes", attempts=2,
        max_concurrent_requests=2,
    ), tmp_path / "cache")
    calls = []

    def respond(endpoint, body, **kwargs):
        payload = json.loads(body["messages"][1]["content"])
        original = payload["input_populations"]
        calls.append(copy.deepcopy(original))
        rows = [{"input_name": row["name"], "status": "represented",
                 "matched_population_names": [proposed[0]["name"]],
                 "reason": row["space"]["prior_treatment_required"]}
                for row in original]
        if len(original) > 1:
            if failure == "name":
                rows[0]["input_name"] = "Abbreviated fictional wording"
            elif failure == "matched_name":
                rows[0]["matched_population_names"] = ["Abbreviated proposed wording"]
            else:
                rows.pop()
        elif missing:
            rows[0].update(status="missing", matched_population_names=[])
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"reviews": rows}),
        }}]}

    monkeypatch.setattr(client, "_http", respond)
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda _: None)
    report = review_coverage(client, inputs, proposed)
    if missing:
        with pytest.raises(CoverageError, match="2 of 2"):
            require_coverage(report)
    else:
        require_coverage(report)
    assert [r["reason"] for r in report["reviews"]] == [
        row["space"]["prior_treatment_required"] for row in inputs
    ]
    assert [b["input_positions"] for b in report["review_batches"]] == [[0], [1]]
    assert [len(c) for c in calls] == [2, 2, 1, 1]
    accepted = set()
    for path in (tmp_path / "cache").glob("*/accepted.json"):
        value = read_json(path)
        audit_accepted_response(path.parent, value)
        accepted.add((value["job"], digest(value["result"])))
    if missing:
        with pytest.raises(CoverageError, match="2 of 2"):
            validate_report(report, inputs, proposed, accepted)
    else:
        validate_report(report, inputs, proposed, accepted)
    assert review_coverage(client, inputs, proposed) == report
    assert len(calls) == 4  # Exhausted parent and accepted children are reused.
    altered = copy.deepcopy(report)
    altered["reviews"][1]["reason"] = "Invented clinical judgment"
    with pytest.raises(ValueError, match="accepted review"):
        validate_report(altered, inputs, proposed, accepted)


@pytest.mark.parametrize("count,error", [
    (2, "saved attempts exhausted: HTTP 503"),  # Never split a transport failure.
    (1, "exhausted: Coverage reviews must retain input clinical names in input order"),
    (2, "exhausted: Catalog omitted or broadened 1 of 2 input populations"),
])
def test_review_transport_or_singleton_failure_remains_terminal(count, error):
    from matchminer_ai.llm.structured import EndpointError

    inputs = populations()[:count]
    proposed = copy.deepcopy(inputs)
    for row in proposed:
        row["name"] = "Other " + row["name"]

    class Reviewer:
        config = SimpleNamespace(max_concurrent_requests=1)

        def fits(self, messages):
            return True

        def complete(self, *args, **kwargs):
            raise EndpointError(error)

    with pytest.raises(EndpointError, match="exhausted"):
        review_coverage(Reviewer(), inputs, proposed)


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


@pytest.mark.parametrize("fix", [True, False])
def test_final_coverage_repairs_only_affected_source_batches_and_stops_unchanged_failures(
    tmp_path, monkeypatch, fix,
):
    from matchminer_ai.trials import _guideline_canonical as module

    guideline = load_guideline(make_library(tmp_path), "fictional")
    inputs = populations()
    client = Client(StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=2), tmp_path / "cache")
    monkeypatch.setattr(module, "MAX_CANDIDATES_PER_BATCH", 1)
    calls = []
    selection_catalogs = []

    def respond(endpoint, body, **kwargs):
        messages = body["messages"]
        if messages[0]["content"] == TASK:
            payload = json.loads(messages[1]["content"])
            # An intermediate review missed the broadening. The final review
            # catches it against the original inputs and the whole catalog.
            missing = len(payload["proposed_catalog"]) > 1
            value = {"reviews": [{
                "input_name": row["name"],
                "status": "missing" if missing else "represented",
                "matched_population_names": [] if missing else [payload["proposed_catalog"][0]["name"]],
                "reason": "Synthetic burden restriction was broadened" if missing else "Synthetic intermediate review missed distinction",
            } for row in payload["input_populations"]]}
        elif messages[1]["content"].startswith(SELECT_TASK):
            text = messages[1]["content"].split("\n\nRequired JSON schema:\n", 1)[1]
            _, end = json.JSONDecoder().raw_decode(text)
            states = json.JSONDecoder().raw_decode(text[end:].lstrip())[0]["source_backed_definitions"]
            selection_catalogs.append(states)
            value = {"spaces": [s["space"] for s in states], "context_only_topics": [], "uncertainties": []}
        else:
            item = inputs[1] if inputs[1]["name"] in messages[1]["content"] else inputs[0]
            restored = "FINAL COVERAGE FINDINGS" in messages[-1]["content"]
            calls.append((item["name"], restored))
            value = catalog()
            value["states"] = [{k: copy.deepcopy(item[k]) for k in value["states"][0]}]
            if item is inputs[0] and not (restored and fix):
                value["states"][0]["space"]["cancer_burden_allowed"] = "Broader fictional burden"
            if restored:
                assert "Synthetic burden restriction was broadened" in messages[-1]["content"]
                assert "candidate_id" not in messages[-1]["content"]
                assert body["max_tokens"] == 100000
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(client, "_http", respond)

    def run_jobs(jobs, workers, function):
        return {key: function(key, data) for key, data in jobs}, {}

    output = tmp_path / "results"
    if fix:
        result = consolidate(client, guideline, inputs, output, 1, run_jobs, lambda _: None)
        assert [g["space"] for g in result["groups"]] == [s["space"] for s in inputs]
        validate_report(read_json(output / "canonical_coverage.json"), inputs, result["groups"])
        assert len(selection_catalogs) == 2
        monkeypatch.setattr(client, "_http", lambda *a, **k: pytest.fail("Resume should reuse saved calls"))
        assert consolidate(client, guideline, inputs, output, 1, run_jobs, lambda _: None) == result
    else:
        with pytest.raises(CoverageError, match="omitted or broadened"):
            consolidate(client, guideline, inputs, output, 1, run_jobs, lambda _: None)
    assert calls == [(inputs[0]["name"], False), (inputs[1]["name"], False), (inputs[0]["name"], True)]
    assert list((output / "population_coverage").glob("final-*.json"))
