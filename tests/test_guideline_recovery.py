"""Synthetic regressions for failures discovered in the expanded PDQ run."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from matchminer_ai._storage import atomic_json, read_json
from matchminer_ai.llm.structured import EndpointError, StructuredConfig
from matchminer_ai.trials import _guideline_canonical as canonical
from matchminer_ai.trials._guideline_audit import audit_catalog
from matchminer_ai.trials._guideline_completeness import review_coverage, require_coverage, CoverageError, TASK
from matchminer_ai.trials._guideline_extraction_batches import run_extractions, validate_ledger, can_split
from matchminer_ai.trials._guideline_generation import Client
from matchminer_ai.trials._guideline_pipeline import parallel_jobs, run_guideline
from matchminer_ai.trials._guideline_sources import load_guideline
from matchminer_ai.trials._guideline_specificity import has_ungrouped_mixed_logic
from test_guideline_completeness import judge, populations
from test_guideline_extraction import make_library, catalog, extraction, quoted_state, prompts, CATALOG_TASK


@pytest.mark.parametrize("phrase", ["spine and pelvis MRI", "MRI of the spine and pelvis", "MRI of spine and pelvis"])
def test_imaging_region_is_atomic_but_surrounding_logic_is_checked(phrase):
    assert not has_ungrouped_mixed_logic(f"modality A OR {phrase} OR modality B")
    assert has_ungrouped_mixed_logic(f"finding A AND modality A OR {phrase}")
    assert has_ungrouped_mixed_logic("spine AND pelvis involvement OR brain involvement")


def test_extraction_splits_only_failed_packets_and_preserves_exact_ownership(tmp_path):
    g = load_guideline(make_library(tmp_path), "fictional")
    pages = [replace(g.pages["p0002"], id=f"p{i:04d}", number=i) for i in range(2, 8)]
    guideline = SimpleNamespace(fingerprint=g.fingerprint, pages={p.id: p for p in pages}, primary=pages, disease="fictional")
    output = tmp_path / "results"
    atomic_json(output / "extraction.json", {"packets": {}, "failures": {
        "extract-0001": "EndpointError: exhausted: page_coverage must account for every PRIMARY page exactly once",
    }})
    calls, cache = [], {}

    def extract(key, primary):
        if key in cache:
            return cache[key]
        calls.append(key)
        if key.startswith("extract-0001") and len(primary) > 1:
            raise EndpointError("exhausted: states_extracted requires state or option evidence from each page")
        cache[key] = {"primary_page_ids": [p.id for p in primary], "result": {"synthetic": key}}
        return cache[key]

    jobs = [("extract-0001", pages[:4]), ("extract-0002", pages[4:])]
    result, failures = run_extractions(jobs, guideline, output, 3, parallel_jobs, extract, lambda _: None)
    assert not failures and "extract-0001" not in calls
    assert sorted(p for r in result.values() for p in r["primary_page_ids"]) == [p.id for p in pages]
    assert calls.count("extract-0002") == 1
    count = len(calls)
    atomic_json(output / "extraction.json", {"packets": result, "failures": failures})
    resumed, errors = run_extractions(jobs, guideline, output, 3, parallel_jobs, extract, lambda _: None)
    assert resumed == result and not errors and len(calls) == count
    ledger = read_json(output / "extraction_packet_splits.json")
    ledger["splits"]["extract-0001"]["children"][0].pop()
    with pytest.raises(ValueError, match="ownership"):
        validate_ledger(ledger, guideline)
    assert not can_split("exhausted: HTTP 503", 8)
    assert not can_split("exhausted: page_coverage must account", 1)


@pytest.mark.parametrize("missing", [False, True])
def test_singleton_coverage_uses_exact_clinical_names_without_forcing_pass(tmp_path, monkeypatch, missing):
    inputs = populations()[:1]
    states = copy.deepcopy(inputs)
    states[0]["name"] = "Exact proposed clinical name"
    client = Client(StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=1, response_format="json_schema"), tmp_path / "cache")
    calls = []

    def respond(endpoint, body, **kwargs):
        fields = body["response_format"]["json_schema"]["schema"]["properties"]["reviews"]["items"]["properties"]
        constrained = "enum" in fields["input_name"]
        calls.append(constrained)
        value = {"reviews": [{"input_name": inputs[0]["name"],
            "status": "missing" if missing and constrained else "represented",
            "matched_population_names": ([] if missing else [states[0]["name"]]) if constrained else ["Wrong capitalization"],
            "reason": "Synthetic clinical comparison"}]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(client, "_http", respond)
    report = review_coverage(client, inputs, states)
    assert calls == [False, True]
    if missing:
        with pytest.raises(CoverageError):
            require_coverage(report)
    else:
        require_coverage(report)


def test_immutable_detail_schema_preserves_raw_model_provenance(tmp_path, monkeypatch):
    guideline = load_guideline(make_library(tmp_path), "fictional")
    output = tmp_path / "results"
    calls = []

    def respond(self, endpoint, body, **kwargs):
        prompt = body["messages"][1]["content"]
        if prompt.startswith(prompts.EXTRACT_TASK):
            value = extraction()
        elif prompt.startswith(CATALOG_TASK):
            value = catalog()
        else:
            value = quoted_state()
            fields = body["response_format"]["json_schema"]["schema"]["properties"]["space"]["properties"]
            constrained = "enum" in fields["cancer_burden_allowed"]
            calls.append(constrained)
            if not constrained:
                value["space"]["cancer_burden_allowed"] += " with rewritten wording"
            else:
                assert {k: v["enum"][0] for k, v in fields.items()} == value["space"]
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(Client, "_http", respond)
    llm = StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=1, response_format="json_schema")
    run_guideline(guideline, output, llm, workers=1)
    assert calls == [False, True]
    assert audit_catalog(guideline, output)["status"] == "passed"
    monkeypatch.setattr(Client, "_http", lambda *a, **k: pytest.fail("Resume should reuse accepted content"))
    run_guideline(guideline, output, llm, workers=1)


def test_failed_selection_retains_all_validated_definitions_and_audits(tmp_path, monkeypatch):
    guideline = load_guideline(make_library(tmp_path), "fictional")
    output = tmp_path / "results"
    inputs = populations()
    monkeypatch.setattr(canonical, "MAX_CANDIDATES_PER_BATCH", 1)

    def respond(self, endpoint, body, **kwargs):
        messages = body["messages"]
        prompt = messages[1]["content"]
        if messages[0]["content"] == TASK:
            value = judge(messages)
        elif prompt.startswith(prompts.EXTRACT_TASK):
            value = extraction()
            value["candidates"] = [{**c, "defining_branch": value["candidates"][0]["defining_branch"]} for c in inputs]
            for c in value["candidates"]:
                c.pop("candidate_id")
        elif prompt.startswith(canonical.SELECT_TASK):
            value = {"spaces": [inputs[0]["space"]], "context_only_topics": [], "uncertainties": []}
        elif prompt.startswith(CATALOG_TASK):
            selected = inputs[1] if inputs[1]["name"] in prompt else inputs[0]
            value = catalog()
            value["states"] = [{k: selected[k] for k in canonical.LEAN_STATE["properties"]}]
        else:
            text = prompt.split("FINAL FIXED POPULATION AND CANDIDATE MENUS TO REVIEW:\n")[1]
            group = json.JSONDecoder().raw_decode(text)[0]["canonical_group"]
            value = quoted_state()
            value.update(name=group["name"], space=group["space"])
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(Client, "_http", respond)
    run_guideline(guideline, output, StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=1), workers=2)
    assert read_json(output / "canonical_selection.json")["mode"] == "retain-validated-definitions"
    assert len(read_json(output / "canonical_groups.json")["groups"]) == 2
    assert audit_catalog(guideline, output)["status"] == "passed"
    selection = read_json(output / "canonical_selection.json")
    selection["available_states"].pop()
    atomic_json(output / "canonical_selection.json", selection)
    with pytest.raises(ValueError, match="Selection inputs"):
        audit_catalog(guideline, output)


def test_final_coverage_narrows_affected_leaf_without_rewriting_other_batches(tmp_path):
    from matchminer_ai.trials._guideline_batching import run_batches, validate_ledger
    guideline = load_guideline(make_library(tmp_path), "fictional")
    candidates = [dict(populations()[0], candidate_id=f"c{i}", name=f"Synthetic {i}") for i in range(6)]
    report = {"reviews": [{"status": "missing" if i == 1 else "represented"} for i in range(6)]}
    calls = []

    def local(key, data):
        members, packed = data
        calls.append([c["candidate_id"] for c in members])
        return {"input_candidate_ids": calls[-1]}

    output = tmp_path / "results"
    values, failed = run_batches([(candidates[:4], None), (candidates[4:], None)],
        guideline, candidates, output, 2, parallel_jobs, local, lambda _: None,
        lambda _: None, coverage_repairs=[report])
    assert not failed
    assert ["c1"] in calls and ["c4", "c5"] in calls
    assert sorted(x for row in values.values() for x in row["input_candidate_ids"]) == [f"c{i}" for i in range(6)]
    validate_ledger(read_json(output / "canonical_batch_splits.json"), guideline, candidates)


def test_literal_selection_uses_original_typography_and_keeps_quote_validation(tmp_path, monkeypatch):
    from matchminer_ai.trials._guideline_quote_selection import select_literal_passages
    from matchminer_ai.trials._guideline_quotes import resolve_excerpt
    text = "Fictional anti-X therapy is an option for fictional disease."
    guideline = load_guideline(make_library(tmp_path, text), "fictional")
    assertion = {"population": {"name": "Fictional disease", "space": populations()[0]["space"]},
        "assertion": {"name": "Fictional anti-X therapy", "conditions": "NA", "category": "NA"},
        "existing_evidence": [], "rejected_evidence": {"page_id": "p0002", "source_text": text.replace("anti-X", "anti–X")}}
    client = Client(StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=1), tmp_path / "cache")

    def respond(endpoint, body, **kwargs):
        prompt = body["messages"][1]["content"]
        assert "faulty draft, not an authoritative quotation" in prompt
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"selected_passages": [1], "unresolved": []})}}]}

    monkeypatch.setattr(client, "_http", respond)
    patch, receipt = select_literal_passages(client, guideline, guideline.pages, assertion)
    assert patch["evidence"] == [{"page_id": "p0002", "source_text": text}]
    with pytest.raises(ValueError, match="verbatim"):
        resolve_excerpt(guideline.pages["p0002"], assertion["rejected_evidence"]["source_text"])
