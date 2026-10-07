"""Exhausted population rewrites split safely and resume their successful leaves."""

import copy
import json

import pytest

from matchminer_ai._storage import atomic_json, read_json
from matchminer_ai.llm.structured import StructuredConfig
from matchminer_ai.trials import _guideline_canonical as canonical
from matchminer_ai.trials._guideline_audit import audit_accepted_response
from matchminer_ai.trials._guideline_batching import can_split
from matchminer_ai.trials._guideline_completeness import TASK, validate_report
from matchminer_ai.trials._guideline_generation import Client
from matchminer_ai.trials._guideline_pipeline import parallel_jobs
from matchminer_ai.trials._guideline_sources import load_guideline
from test_guideline_completeness import judge, populations
from test_guideline_extraction import TEXT, make_library


@pytest.mark.parametrize("saved_failure", [False, True])
@pytest.mark.parametrize("failure_kind", [
    "coverage", "restriction", "histology", "diagnostic_criterion", "diagnostic_threshold",
    "prior_exclusion", "biomarker_exclusion", "treatment_phase", "response_assessment",
    "joint_primary_history", "stage_morphology_union", "compound_required", "compound_excluded",
    "grouping", "treatment_prose",
    "blank", "nonexistent", "length",
])
def test_failed_batch_splits_without_repeating_successful_or_exhausted_batches(
    tmp_path, monkeypatch, saved_failure, failure_kind,
):
    guideline = load_guideline(
        make_library(tmp_path, TEXT + "\n\nAdditional synthetic source"), "fictional",
    )
    inputs = populations()
    if failure_kind == "histology":
        inputs[0]["space"]["histology_allowed"] = "Fictional carcinoma grade 1"
    elif failure_kind == "diagnostic_criterion":
        inputs[0]["space"]["cancer_burden_allowed"] = (
            "Clinical presentation suggestive AND Imaging suggestive"
        )
    elif failure_kind == "diagnostic_threshold":
        inputs[0]["space"]["cancer_burden_allowed"] = (
            "Fictional pattern A OR (Fictional lesion at least 30 mm AND Fictional morphology B)"
        )
    elif failure_kind == "prior_exclusion":
        inputs[0]["space"]["prior_treatment_excluded"] = "Prior fictional adjuvant therapy"
    elif failure_kind == "biomarker_exclusion":
        inputs[0]["space"]["biomarkers_excluded"] = "Fictional marker amplification"
    elif failure_kind == "treatment_phase":
        inputs[0]["space"]["prior_treatment_required"] = (
            "Initial systemic fictional therapy AND (Complete response OR Partial response)"
        )
        inputs[0]["space"]["prior_treatment_excluded"] = "Maintenance fictional therapy"
    elif failure_kind == "response_assessment":
        inputs[0]["space"]["prior_treatment_required"] = (
            "Initial fictional therapy AND complete metabolic response (PET-negative) "
            "at 2 months after treatment"
        )
    elif failure_kind == "joint_primary_history":
        inputs[0]["space"]["cancer_type_allowed"] = (
            "Fictional primary A OR Fictional primary B OR Fictional primary C"
        )
        inputs[0]["space"]["prior_treatment_required"] = "Prior fictional procedure"
    elif failure_kind == "stage_morphology_union":
        inputs[0]["space"]["histology_allowed"] = "Fictional favorable histology with complete necrosis"
        inputs[0]["space"]["cancer_burden_allowed"] = (
            "Fictional stage I OR Fictional stage II OR Fictional stage III"
        )
    elif failure_kind in {"compound_required", "compound_excluded"}:
        field = "biomarkers_required" if failure_kind == "compound_required" else "biomarkers_excluded"
        inputs[0]["space"][field] = (
            "Fictional marker A OR (combined Fictional marker B AND Fictional marker C) "
            "OR Fictional marker D"
        )
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
        truncated = False
        assert "opaque" not in json.dumps(messages)
        if messages[0]["content"] == TASK:
            value = judge(messages)
            if failure_kind in {
                "restriction", "histology", "diagnostic_criterion", "diagnostic_threshold",
                "prior_exclusion", "biomarker_exclusion", "treatment_phase",
                "response_assessment", "joint_primary_history", "stage_morphology_union",
            }:
                payload = json.loads(messages[1]["content"])
                proposed = {s["name"]: s for s in payload["proposed_catalog"]}
                inputs_by_name = {s["name"]: s for s in payload["input_populations"]}
                for review in value["reviews"]:
                    name = review["input_name"]
                    if name in proposed and proposed[name]["space"] != inputs_by_name[name]["space"]:
                        review.update(status="missing", matched_population_names=[],
                            reason={
                                "histology": "Histologic grade restriction absent from proposed fields",
                                "restriction": "Additional response-probability restriction narrows the original population",
                                "diagnostic_criterion": "Conjoined clinical diagnostic criterion absent from proposed fields",
                                "diagnostic_threshold": "Diagnostic morphology alternatives and size threshold absent from proposed fields",
                                "prior_exclusion": "Prior adjuvant treatment exclusion absent from proposed fields",
                                "biomarker_exclusion": "Biomarker exclusion absent from proposed fields",
                                "treatment_phase": "Initial therapy, response and pre-maintenance restrictions absent from proposed fields",
                                "response_assessment": "End-of-treatment metabolic response and exact assessment timing absent from proposed fields",
                                "joint_primary_history": "No single proposed population retains both the primary-site allowance and required prior procedure",
                                "stage_morphology_union": "Stage II and III complete-necrosis branches are absent from the proposed stage union",
                            }[failure_kind])
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
                {key: copy.deepcopy(s[key]) for key in canonical.LEAN_STATE["properties"]}
                for s in (descriptions[:1] if failure_kind == "coverage" else descriptions)
            ], "context_only_topics": [], "uncertainties": []}
            if len(descriptions) > 1 and failure_kind == "restriction":
                value["states"][0]["space"]["cancer_burden_allowed"] += "; additional response-probability restriction"
            elif len(descriptions) > 1 and failure_kind == "histology":
                value["states"][0]["space"]["histology_allowed"] = "Fictional carcinoma"
            elif len(descriptions) > 1 and failure_kind == "diagnostic_criterion":
                value["states"][0]["space"]["cancer_burden_allowed"] = "Imaging suggestive"
            elif len(descriptions) > 1 and failure_kind == "diagnostic_threshold":
                value["states"][0]["space"]["cancer_burden_allowed"] = "Fictional lesion at risk"
            elif len(descriptions) > 1 and failure_kind == "prior_exclusion":
                value["states"][0]["space"]["prior_treatment_excluded"] = "NA"
            elif len(descriptions) > 1 and failure_kind == "biomarker_exclusion":
                value["states"][0]["space"]["biomarkers_excluded"] = "NA"
            elif len(descriptions) > 1 and failure_kind == "treatment_phase":
                value["states"][0]["space"]["prior_treatment_required"] = "Systemic fictional therapy"
                value["states"][0]["space"]["prior_treatment_excluded"] = "NA"
            elif len(descriptions) > 1 and failure_kind == "response_assessment":
                value["states"][0]["space"]["prior_treatment_required"] = (
                    "Initial fictional therapy AND complete response"
                )
            elif len(descriptions) > 1 and failure_kind == "joint_primary_history":
                # Two complementary partial states do not preserve one full
                # population: one narrows the site, the other drops history.
                alternative = copy.deepcopy(value["states"][0])
                alternative["name"] += "; without the required procedure"
                alternative["space"]["prior_treatment_required"] = "NA"
                value["states"][0]["space"]["cancer_type_allowed"] = "Fictional primary A"
                value["states"].append(alternative)
            elif len(descriptions) > 1 and failure_kind == "stage_morphology_union":
                value["states"][0]["space"]["cancer_burden_allowed"] = "Fictional stage I"
                alternative = copy.deepcopy(value["states"][0])
                alternative["name"] += "; without the necrosis restriction"
                alternative["space"]["histology_allowed"] = "Fictional favorable histology"
                alternative["space"]["cancer_burden_allowed"] = "Fictional stage II OR Fictional stage III"
                value["states"].append(alternative)
            elif len(descriptions) > 1 and failure_kind in {"compound_required", "compound_excluded"}:
                field = "biomarkers_required" if failure_kind == "compound_required" else "biomarkers_excluded"
                value["states"][0]["space"][field] = (
                    "(Fictional marker A OR combined Fictional marker B AND Fictional marker C "
                    "OR Fictional marker D)"
                )
            elif len(descriptions) > 1 and failure_kind == "grouping":
                value["states"][0]["space"]["cancer_burden_allowed"] = "(Fictional burden A OR Fictional burden B AND Fictional burden C)"
            elif len(descriptions) > 1 and failure_kind == "treatment_prose":
                # Lowercase alternatives in explanatory treatment prose still
                # require grouping; an exhausted rewrite must split without
                # accepting the malformed restriction or repeating its parent.
                value["states"][0]["space"]["cancer_burden_allowed"] = (
                    "Localized fictional cancer AND considered for focal or "
                    "whole-region fictional therapy"
                )
            elif len(descriptions) > 1 and failure_kind == "length":
                # Even parseable JSON cannot be accepted when the provider
                # says the answer exhausted its output allowance.
                truncated = True
            elif len(descriptions) > 1 and failure_kind != "coverage":
                value["states"][0]["evidence"] = [{
                    "page_id": "p0002", "line_ids": [2 if failure_kind == "blank" else 99],
                }]
        return {"choices": [{"finish_reason": "length" if truncated else "stop",
                             "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(client, "_http", respond)
    if saved_failure:
        atomic_json(output / "run_config.json", {"source_fingerprint": guideline.fingerprint})
        atomic_json(output / "candidates.json", inputs)
        atomic_json(output / "canonical_batches.json", {
            "batches": {}, "failures": {
                "catalog-content-0001": "EndpointError: exhausted 6 attempts: " + {
                    "coverage": "Catalog omitted or broadened 1 of 2 input populations",
                    "restriction": "Catalog omitted or broadened 1 of 2 input populations: added response-probability restriction",
                    "histology": "Catalog omitted or broadened 1 of 2 input populations: histologic grade omitted from fields",
                    "diagnostic_criterion": "Catalog omitted or broadened 1 of 2 input populations: conjoined clinical diagnostic criterion lost",
                    "diagnostic_threshold": "Catalog omitted or broadened 1 of 2 input populations: diagnostic morphology alternatives and size threshold lost",
                    "prior_exclusion": "Catalog omitted or broadened 1 of 2 input populations: prior adjuvant treatment exclusion lost",
                    "biomarker_exclusion": "Catalog omitted or broadened 1 of 2 input populations: biomarker exclusion lost",
                    "treatment_phase": "Catalog omitted or broadened 1 of 2 input populations: initial therapy, response and pre-maintenance restrictions lost",
                    "response_assessment": "Catalog omitted or broadened 1 of 2 input populations: metabolic response and assessment timing lost",
                    "joint_primary_history": "Catalog omitted or broadened 1 of 2 input populations: site allowance and required prior procedure not retained together",
                    "stage_morphology_union": "Catalog omitted or broadened 1 of 2 input populations: stage II/III complete-necrosis branches lost",
                    "compound_required": "biomarkers_required mixes AND and OR at the same parenthesis level",
                    "compound_excluded": "biomarkers_excluded mixes AND and OR at the same parenthesis level",
                    "grouping": "cancer_burden_allowed mixes AND and OR at the same parenthesis level",
                    "treatment_prose": "cancer_burden_allowed mixes AND and OR at the same parenthesis level. Rejected field text: 'Localized fictional cancer AND considered for focal or whole-region fictional therapy'. Keep only population criteria in this field",
                    "blank": "p0002: evidence may not consist entirely of blank lines",
                    "nonexistent": "p0002: nonexistent source line IDs [99]; maximum is 3",
                    "length": "Incomplete response: finish_reason='length'; increase max_tokens or reduce the packet",
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
    for path in (output / "checkpoints").glob("*/accepted.json"):
        accepted = read_json(path)
        audit_accepted_response(path.parent, accepted)
    if failure_kind == "length" and not saved_failure:
        truncated = [p for p in (output / "checkpoints").glob("*/attempt-*.json")
                     if read_json(p)["choices"][0]["finish_reason"] == "length"]
        assert len(truncated) == 1
        assert not truncated[0].with_name("accepted.json").exists()
        assert "finish_reason='length'" in read_json(truncated[0].with_name("failure-1.json"))["error"]
        original_truncated = truncated[0].read_bytes()
    ledger = read_json(output / "canonical_batch_splits.json")
    assert ledger["splits"]["catalog-content-0001"]["child_size"] == 1

    monkeypatch.setattr(client, "_http", lambda *a, **k: pytest.fail("Resume must reuse successful children"))
    assert canonical.consolidate(client, guideline, inputs, output, 1, parallel_jobs, lambda _: None) == result
    if failure_kind == "length" and not saved_failure:
        assert truncated[0].read_bytes() == original_truncated
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
    ("EndpointError: exhausted 6 attempts: Incomplete response: finish_reason='length'", 83, True),
    ("EndpointError: exhausted 6 attempts: Incomplete response: finish_reason='length'", 1, False),
    ("Incomplete response: finish_reason='length'", 83, False),
    ("EndpointError: exhausted 6 attempts: Incomplete response: finish_reason='client_aborted_repetition'", 83, False),
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
