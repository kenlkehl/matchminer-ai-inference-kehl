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


@pytest.mark.parametrize("phrase", [
    "double trisomies of chromosomes 4 and 10",
    "trisomies of chromosomes 8 and 21",
    "triple trisomies of chromosomes 4, 10, and 17",
    "loss of chromosomes X and Y",
])
def test_chromosome_list_is_atomic_without_rewriting_or_hiding_other_logic(phrase):
    from matchminer_ai.trials._guideline_specificity import validate_decision_fields

    candidate = quoted_state()
    candidate["space"]["biomarkers_required"] = f"(marker A OR {phrase}) AND marker B"
    original = copy.deepcopy(candidate)
    validate_decision_fields(candidate, "Synthetic leukemia guideline")
    assert candidate == original
    assert not has_ungrouped_mixed_logic(f"marker A OR {phrase}.")
    assert has_ungrouped_mixed_logic(f"marker A OR {phrase} AND marker B")
    assert has_ungrouped_mixed_logic(f"marker A AND {phrase} OR marker B")


@pytest.mark.parametrize("value", [
    "marker A OR chromosomes 4 and marker B",
    "marker A OR chromosome 4 and chromosome 10",
    "marker A OR chromosomes 4 and 10q abnormality",
    "marker A OR chromosomes 4 and 10.5",
    "marker A OR chromosomes 4 and 100",
])
def test_chromosome_list_mask_requires_complete_identifiers(value):
    assert has_ungrouped_mixed_logic(value)


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


@pytest.mark.parametrize("failure", ["local-loss", "repetition", "final-loss"])
def test_failed_selection_retains_all_validated_definitions_and_audits(tmp_path, monkeypatch, failure):
    guideline = load_guideline(make_library(tmp_path), "fictional")
    output = tmp_path / "results"
    inputs = populations()
    monkeypatch.setattr(canonical, "MAX_CANDIDATES_PER_BATCH", 1)

    def respond(self, endpoint, body, **kwargs):
        messages = body["messages"]
        prompt = messages[1]["content"]
        if messages[0]["content"] == TASK:
            value = judge(messages)
            if failure == "final-loss":
                payload = json.loads(messages[1]["content"])
                known = {s['name'] for s in payload['proposed_catalog']}
                for row in value['reviews']:
                    match = None
                    if row['input_name'] == inputs[1]['name'] and 'Validated beta' in known:
                        match = 'Validated beta'
                    elif row['input_name'] == 'Validated beta':
                        # Simulate a locally accepted broad substitute. The
                        # final review of original populations must still catch it.
                        match = inputs[0]['name']
                    if match:
                        row.update(status='represented', matched_population_names=[match])
        elif prompt.startswith(prompts.EXTRACT_TASK):
            value = extraction()
            value["candidates"] = [{**c, "defining_branch": value["candidates"][0]["defining_branch"]} for c in inputs]
            for c in value["candidates"]:
                c.pop("candidate_id")
        elif prompt.startswith(canonical.SELECT_TASK):
            if failure == "repetition":
                raise EndpointError("exhausted: Repeated 24 complete clinical definitions without any new definition.")
            value = {"spaces": [inputs[0]["space"]], "context_only_topics": ["Discarded beta"] if failure == "final-loss" else [], "uncertainties": []}
        elif prompt.startswith(CATALOG_TASK):
            selected = inputs[1] if inputs[1]["name"] in prompt else inputs[0]
            value = catalog()
            value["states"] = [{k: selected[k] for k in canonical.LEAN_STATE["properties"]}]
            if failure == "final-loss" and selected == inputs[1]:
                value['states'][0]['name'] = 'Validated beta'
        else:
            text = prompt.split("FINAL FIXED POPULATION AND CANDIDATE MENUS TO REVIEW:\n")[1]
            group = json.JSONDecoder().raw_decode(text)[0]["canonical_group"]
            value = quoted_state()
            value.update(name=group["name"], space=group["space"])
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(Client, "_http", respond)
    run_guideline(guideline, output, StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=1), workers=2)
    mode = "retain-after-final-coverage" if failure == "final-loss" else "retain-validated-definitions"
    assert read_json(output / "canonical_selection.json")["mode"] == mode
    assert len(read_json(output / "canonical_groups.json")["groups"]) == 2
    assert "Discarded beta" not in read_json(output / "canonical_groups.json")["context_only_topics"]
    assert audit_catalog(guideline, output)["status"] == "passed"
    selection = read_json(output / "canonical_selection.json")
    if failure == "final-loss":
        assert selection['rejected_coverage']['reviews'][1]['status'] == 'missing'
        corrupt = copy.deepcopy(selection)
        corrupt['rejected_coverage']['catalog_sha256'] = 'corrupted'
        atomic_json(output / "canonical_selection.json", corrupt)
        with pytest.raises(ValueError, match='Population coverage report differs'):
            audit_catalog(guideline, output)
        atomic_json(output / "canonical_selection.json", selection)
    selection["available_states"].pop()
    atomic_json(output / "canonical_selection.json", selection)
    with pytest.raises(ValueError, match="Selection inputs"):
        audit_catalog(guideline, output)


@pytest.mark.parametrize("address_failure", [None, "invalid-lines", "other-owner", "unresolved-lines"])
def test_extraction_grouping_retry_keeps_source_ownership_and_audits(tmp_path, monkeypatch, address_failure):
    g = load_guideline(make_library(tmp_path), 'fictional')
    output = tmp_path/'results'
    good = extraction()
    good['candidates'][0]['space']['biomarkers_required'] = 'Marker A OR (combined marker B AND marker C)'
    bad = copy.deepcopy(good)
    bad['candidates'][0]['space']['biomarkers_required'] = 'Marker A OR combined marker B AND marker C'
    calls = []

    def respond(self, endpoint, body, **kwargs):
        prompt = body['messages'][1]['content']
        if prompt.startswith(prompts.EXTRACT_TASK):
            focused = 'Complete ambiguous parenthesis groups:' in prompt
            addressing = 'SOURCE-ADDRESS RETRY' in prompt
            calls.append((focused, addressing))
            assert 'PRIMARY' in prompt and 'p0002' in prompt
            value = copy.deepcopy(good if focused else bad)
            if addressing:
                assert 'OWNED SOURCE WITH ORIGINAL LINE LABELS:' in prompt
                assert '"enum": [1]' in prompt
            if focused and address_failure and (not addressing or address_failure == 'unresolved-lines'):
                if address_failure == 'other-owner':
                    value['candidates'][0]['defining_branch']['page_id'] = 'p9999'
                else:
                    value['candidates'][0]['defining_branch']['line_ids'] = [20]
                    value['candidates'][0]['evidence'][0]['line_ids'] = [20]
        elif prompt.startswith(CATALOG_TASK):
            value = catalog()
            value['states'][0]['space'] = good['candidates'][0]['space']
        else:
            value = quoted_state()
            value['space'] = good['candidates'][0]['space']
        return {'choices':[{'finish_reason':'stop', 'message':{'content':json.dumps(value)}}]}

    monkeypatch.setattr(Client, '_http', respond)
    config = StructuredConfig(model='synthetic', tokenizer_mode='bytes', attempts=1)
    if address_failure in ('other-owner', 'unresolved-lines'):
        with pytest.raises(RuntimeError, match='extraction packets failed'):
            run_guideline(g, output, config, workers=1)
        assert calls == ([(False, False), (True, False)] if address_failure == 'other-owner'
                         else [(False, False), (True, False), (True, True)])
        assert not (output/'paradigms.jsonl').exists()
        return
    run_guideline(g, output, config, workers=1)
    assert calls == ([(False, False), (True, False), (True, True)] if address_failure
                     else [(False, False), (True, False)])
    packet = read_json(output/'extraction.json')['packets']['extract-0001']
    assert packet['primary_page_ids'] == ['p0002']
    assert packet['result'] == good
    assert packet['reserved_output_tokens'] == 100000
    assert audit_catalog(g, output)['status'] == 'passed'
    monkeypatch.setattr(Client, '_http', lambda *a, **k: pytest.fail('Reuse accepted grouping repair'))
    run_guideline(g, output, config, workers=1)


def test_owned_address_schema_does_not_constrain_clinical_text_or_supporting_evidence():
    from matchminer_ai.trials._guideline_extraction_retry import owned_address_schema
    from matchminer_ai.trials._guideline_schema import EXTRACTION
    before = copy.deepcopy(EXTRACTION)
    schema = owned_address_schema([SimpleNamespace(id='p0002', text='Heading\n\nSynthetic table on one line.')])
    fields = schema['properties']['candidates']['items']['properties']
    original = EXTRACTION['properties']['candidates']['items']['properties']
    for key in original:
        if key != 'defining_branch':
            assert fields[key] == original[key]
    assert schema['properties']['uncertainties'] == EXTRACTION['properties']['uncertainties']
    branch = fields['defining_branch']['properties']
    assert branch['page_id']['enum'] == ['p0002']
    assert branch['line_ids']['items']['enum'] == [1, 3]
    assert schema['properties']['page_coverage']['items']['properties']['reason'] == {'type': 'string', 'minLength': 1}
    assert EXTRACTION == before


def test_restoring_selection_does_not_override_failed_full_union_coverage(tmp_path, monkeypatch):
    g = load_guideline(make_library(tmp_path), 'fictional')
    inputs = populations()
    client = Client(StructuredConfig(model='synthetic', tokenizer_mode='bytes', attempts=1), tmp_path/'cache')
    batches = {f'batch-{i}': {'result': {'states': [{k: c[k] for k in canonical.LEAN_STATE['properties']}],
                'context_only_topics': [], 'uncertainties': []}, 'context': {}}
               for i, c in enumerate(inputs)}
    monkeypatch.setattr(canonical, 'partition_calls', lambda *a: [])
    monkeypatch.setattr(canonical, 'run_batches', lambda *a, **k: (batches, {}))
    monkeypatch.setattr(canonical, 'build_selection_call', lambda *a: ([], [], []))
    def complete(job, messages, schema, validator, **kwargs):
        result = {'spaces': [inputs[0]['space']], 'context_only_topics': [], 'uncertainties': []}
        validator(result)
        return result
    monkeypatch.setattr(client, 'complete', complete)
    final_sizes = []
    def review(client, candidates, states):
        final = candidates is inputs
        if final:
            final_sizes.append(len(states))
        return {'reviews': [{'status': 'missing' if final else 'represented'} for c in candidates]}
    monkeypatch.setattr(canonical, 'review_coverage', review)
    with pytest.raises(CoverageError, match='omitted or broadened'):
        canonical.consolidate(client, g, inputs, tmp_path/'results', 1, parallel_jobs,
                              lambda _: None, coverage_repairs=({'reviews': []},))
    assert final_sizes == [1, 2]
    assert not (tmp_path/'results/canonical_groups.json').exists()


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


def test_exact_definition_union_preserves_all_clinical_names_without_extra_spaces():
    from matchminer_ai.trials._guideline_completeness import validate_report, exact_match_names
    first = populations()[0]
    second = copy.deepcopy(first)
    second['name'] = 'Fictional disease; prognostic workup'
    original = copy.deepcopy([first, second])
    merged = canonical.combine_identical([
        {**first, 'source_batch_numbers': [1]}, {**second, 'source_batch_numbers': [2]},
    ], preserve_names=True)
    assert len(merged) == 1
    assert merged[0]['population_names'] == [first['name'], second['name']]
    assert merged[0]['source_batch_numbers'] == [1, 2]
    assert [first, second] == original

    class NoModel:
        config = SimpleNamespace(max_concurrent_requests=1)
        def complete(self, *args, **kwargs):
            pytest.fail('No model bookkeeping needed for retained exact clinical definitions')

    report = review_coverage(NoModel(), original, merged)
    validate_report(report, original, merged, set())
    altered = copy.deepcopy(second)
    altered['space']['prior_treatment_required'] = 'Required prior synthetic therapy'
    assert not exact_match_names(altered, merged)
    unknown = {**second, 'name': 'Unretained clinical population'}
    assert not exact_match_names(unknown, merged)
    # Selection chooses definitions; code must carry their full name provenance.
    chosen = canonical.selected_states({'spaces': [merged[0]['space']],
        'context_only_topics': [], 'uncertainties': []}, merged)
    assert chosen == merged


def test_boolean_feedback_identifies_late_offending_group_without_changing_logic():
    from matchminer_ai.trials._guideline_specificity import mixed_logic_segments, validate_decision_fields
    value = populations()[0]
    value['space']['biomarkers_required'] = '(' + 'Long synthetic criterion; ' * 40 + ') OR (A AND B OR C)'
    original = copy.deepcopy(value)
    assert mixed_logic_segments(value['space']['biomarkers_required']) == ['A AND B OR C']
    with pytest.raises(ValueError, match="Complete ambiguous parenthesis groups:.*A AND B OR C"):
        validate_decision_fields(value, 'Fictional')
    assert value == original
    assert not mixed_logic_segments('(A AND B) OR C')


def test_output_limited_coverage_splits_complete_inputs(tmp_path, monkeypatch):
    client = Client(StructuredConfig(model='synthetic', tokenizer_mode='bytes', attempts=1), tmp_path/'cache')
    inputs = populations()
    proposed = copy.deepcopy(inputs)
    for p in proposed:
        p['space']['age_range_allowed'] = 'Any age'
    sizes = []
    def complete(job, messages, schema, validator, **kwargs):
        size = len(json.loads(messages[1]['content'])['input_populations']);sizes.append(size)
        if size > 1:
            raise EndpointError("exhausted: Incomplete response: finish_reason='length'")
        result = judge(messages);validator(result);return result
    monkeypatch.setattr(client, 'complete', complete)
    report = review_coverage(client, inputs, proposed)
    require_coverage(report)
    assert sizes == [2, 1, 1]
    assert [r['input_name'] for r in report['reviews']] == [p['name'] for p in inputs]


def test_retry_headroom_removes_only_whole_optional_source_pages(tmp_path):
    from matchminer_ai.trials._guideline_context import pack_messages
    g = load_guideline(make_library(tmp_path), 'fictional')
    primary = g.pages['p0002']
    extra = replace(primary, id='p0003', number=3, text=primary.text + ' Supplementary text.' * 100)
    # Copy immutable source objects; this is synthetic packing, not source alteration.
    g = replace(g, pages={primary.id: primary, extra.id: extra})
    client = Client(StructuredConfig(model='synthetic', tokenizer_mode='bytes'), tmp_path/'cache')
    initial = pack_messages(client, g, {'p0002'}, '', 'Synthetic task', {}, canonical.CATALOG)
    assert {p.id for p in initial[0]} == {'p0002', 'p0003'}
    full_size = client.count_tokens(initial[2])
    reserve = client.prompt_budget - full_size + 1
    packed = pack_messages(client, g, {'p0002'}, '', 'Synthetic task', {}, canonical.CATALOG,
                           retry_headroom_tokens=reserve)
    assert [p.id for p in packed[0]] == ['p0002']
    assert packed[1] == ['p0003']
    assert client.config.max_tokens == 100000


def test_source_owner_refresh_is_bounded_provenance_checked_and_resumable(tmp_path):
    from matchminer_ai._storage import digest
    from matchminer_ai.trials._guideline_completeness import population
    from matchminer_ai.trials._guideline_source_refresh import (
        prepare_refresh, load_refreshes, review_job, archive_changed_candidates, audit_refreshes,
    )
    g = load_guideline(make_library(tmp_path), 'fictional')
    output = tmp_path/'results'
    original = extraction()
    c = {'candidate_id':'extract-0001-c001', **original['candidates'][0]}
    packet = {'result':original, 'primary_page_ids':['p0002']}
    report = {'input_sha256':digest([population(c)]), 'reviews':[{
        'input_name':c['name'], 'status':'missing', 'matched_population_names':[],
        'reason':'Synthetic dispute about restriction scope',
    }]}
    atomic_json(output/'status.json', {'status':'failed'})
    atomic_json(output/'candidates.json', [c])
    atomic_json(output/'extraction.json', {'packets':{'extract-0001':packet}, 'failures':{}})
    atomic_json(output/'canonical_batches.json', {'failures':{
        'catalog-content-0001':'exhausted: Catalog omitted or broadened 1 of 1 input populations',
    }})
    atomic_json(output/'population_coverage/catalog-content-0001.json',report)
    assert prepare_refresh(output,g)
    assert not prepare_refresh(output,g)  # Never repeat the same source review unchanged.
    record = load_refreshes(output,g)['packets']['extract-0001']
    assert record['original_packet'] == packet
    replacement = copy.deepcopy(original)
    replacement['candidates'][0]['space']['biomarkers_excluded'] = 'Fictional A OR Fictional B'
    new = [{'candidate_id':c['candidate_id'], **replacement['candidates'][0]}]
    original_checkpoint = output/'checkpoints/original/accepted.json'
    atomic_json(original_checkpoint, {'job':'extract-0001','result_sha256':digest(original)})
    atomic_json(output/'checkpoints/review/accepted.json',
        {'job':review_job('extract-0001',record),'result_sha256':digest(replacement)})
    audit_refreshes(output,g,{'extract-0001':{**packet,'result':replacement}})
    with pytest.raises(ValueError,match='replacement extraction'):
        audit_refreshes(output,g,{'extract-0001':packet})
    # Simulate interruption after moving the first derived artifact.
    archive = output/'superseded_candidate_artifacts'/digest([c]);archive.mkdir(parents=True)
    (output/'candidates.json').rename(archive/'candidates.json')
    atomic_json(output/'candidate_archive_pending.json',{'old_sha256':digest([c]),'new_sha256':digest(new)})
    archive_changed_candidates(output,new)
    assert read_json(archive/'candidates.json') == [c]
    assert (archive/'canonical_batches.json').exists()
    assert original_checkpoint.exists()
    assert not (output/'candidate_archive_pending.json').exists()


def test_irreconcilable_population_is_reextracted_by_model_with_original_preserved(tmp_path, monkeypatch):
    source = 'Fictional disease excludes either marker A or marker B. Consider synthetic test A when finding B is present.'
    g = load_guideline(make_library(tmp_path, source), 'fictional')
    output = tmp_path/'results'
    original = extraction()
    original['candidates'][0]['space']['biomarkers_excluded'] = 'Marker A AND Marker B'
    corrected = copy.deepcopy(original)
    corrected['candidates'][0]['space']['biomarkers_excluded'] = 'Marker A OR Marker B'
    source_reviews = []

    def respond(self, endpoint, body, **kwargs):
        prompt = body['messages'][1]['content']
        if body['messages'][0]['content'] == TASK:
            payload = json.loads(prompt)
            reviews = []
            for item in payload['input_populations']:
                matches = [s['name'] for s in payload['proposed_catalog'] if s['space'] == item['space']]
                reviews.append({'input_name':item['name'], 'status':'represented' if matches else 'missing',
                    'matched_population_names':matches, 'reason':'Synthetic difference in exclusion scope'})
            value = {'reviews':reviews}
        elif prompt.startswith(prompts.EXTRACT_TASK):
            review = 'DISPUTED EXTRACTION AND COVERAGE FINDING' in prompt
            source_reviews.append(review)
            value = corrected if review else original
        elif prompt.startswith(CATALOG_TASK):
            value = catalog()
            value['states'] = [{k:corrected['candidates'][0][k] for k in canonical.LEAN_STATE['properties']}]
        else:
            value = quoted_state()
            value['space'] = corrected['candidates'][0]['space']
            for owner in [value, *value['diagnostic_workup'], *value['treatment_options']]:
                owner['evidence'] = [{'page_id':'p0002', 'source_text':source}]
        return {'choices':[{'finish_reason':'stop', 'message':{'content':json.dumps(value)}}]}

    monkeypatch.setattr(Client,'_http',respond)
    config = StructuredConfig(model='synthetic', tokenizer_mode='bytes', attempts=1)
    run_guideline(g,output,config,workers=1)
    assert source_reviews == [False, True]
    responses = [json.loads(p.read_text()) for p in (output/'checkpoints').glob('*/attempt-*.json')]
    assert any(json.loads(r['choices'][0]['message']['content']) == original for r in responses)
    assert read_json(output/'candidates.json')[0]['space']['biomarkers_excluded'] == 'Marker A OR Marker B'
    assert list((output/'superseded_candidate_artifacts').glob('*/candidates.json'))
    assert audit_catalog(g,output)['status'] == 'passed'
    canonical_groups = read_json(output/'canonical_groups.json')
    canonical_groups.pop('preserve_population_names')
    atomic_json(output/'canonical_groups.json',canonical_groups)
    with pytest.raises(ValueError,match='population name provenance contract'):
        audit_catalog(g,output)
