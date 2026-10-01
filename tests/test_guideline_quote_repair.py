"""Focused repair resumes real pipeline checkpoints without weakening citation checks."""

import copy
import json
from dataclasses import replace

import pytest

from matchminer_ai._storage import atomic_json, digest, read_json
from matchminer_ai.llm.structured import EndpointError, StructuredConfig
from matchminer_ai.trials._guideline_audit import audit_catalog
from matchminer_ai.trials._guideline_generation import Client, clinical_content
from matchminer_ai.trials._guideline_pipeline import run_guideline
from matchminer_ai.trials._guideline_quote_repair import (
    PROMPT_FILES,
    REPAIR,
    apply_replacements,
    audit_repair,
    repair_quoted_response,
    validate_replacement,
)
from matchminer_ai.trials._guideline_quotes import materialize_quoted_state
from matchminer_ai.trials._guideline_sources import load_guideline
from test_guideline_extraction import (
    CATALOG_TASK,
    TEXT,
    catalog,
    extraction,
    make_library,
    prompts,
    quoted_state,
)


def response(value):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(value),
                    "reasoning_content": "Private reasoning is not a draft",
                },
            }
        ]
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda _: None)
    guideline = load_guideline(make_library(tmp_path), "fictional")
    client = Client(
        StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=2),
        tmp_path / "checkpoints",
    )
    return guideline, client


def test_every_invalid_citation_is_repaired_without_losing_valid_content(
    setup, tmp_path, monkeypatch
):
    guideline, client = setup
    original = quoted_state()
    valid = copy.deepcopy(original["diagnostic_workup"][0]["evidence"])
    original["diagnostic_workup"][0]["evidence"] += [
        {"page_id": "p0002", "source_text": f"Absent invented excerpt {i}"}
        for i in range(12)
    ]
    calls = []

    def respond(endpoint, body, **kwargs):
        calls.append(copy.deepcopy(body))
        assert kwargs["output_schema"] == REPAIR
        content = json.dumps(body["messages"])
        assert "owner_index" not in content and "evidence_index" not in content
        assert "Private reasoning" not in content
        assert "AFFECTED SOURCE PAGES" in content
        return response({"evidence": valid, "unresolved": []})

    monkeypatch.setattr(client, "_http", respond)
    before = copy.deepcopy(original)
    repaired = repair_quoted_response(
        client,
        guideline,
        guideline.pages,
        "detail-0001",
        original,
        "Correct every invalid excerpt in the draft: " + "x" * 3000,
        lambda v: materialize_quoted_state(v, guideline.pages),
        output=tmp_path,
    )
    assert len(calls) == 12
    assert original == before
    assert clinical_content(repaired) == clinical_content(original)
    assert repaired["evidence"] == original["evidence"]
    assert repaired["diagnostic_workup"][0]["evidence"][0] == valid[0]
    assert all(call["max_tokens"] == 100000 for call in calls)
    assert all(call["chat_template_kwargs"]["enable_thinking"] for call in calls)
    receipt = read_json(next((tmp_path / "quote_repairs").glob("*.json")))
    accepted = {
        (a["job"], a["result_sha256"])
        for a in (read_json(f) for f in client.cache_dir.glob("*/accepted.json"))
    }
    audit_repair(original, repaired, receipt, accepted, guideline.pages)
    with pytest.raises(ValueError, match="accepted model"):
        audit_repair(original, repaired, receipt, set(), guideline.pages)
    tampered = copy.deepcopy(repaired)
    tampered["diagnostic_workup"][0]["conditions"] = "Changed population"
    receipt["result_sha256"] = digest(tampered)
    with pytest.raises(ValueError, match="unapproved"):
        audit_repair(original, tampered, receipt, accepted, guideline.pages)


def test_unsupported_or_ambiguous_replacements_never_pass(setup):
    guideline, _ = setup
    page = replace(guideline.pages["p0002"], text="Branch A test.\nBranch B test.")
    for value, message in [
        ({"evidence": [], "unresolved": ["Assertion unsupported"]}, "unsupported"),
        (
            {
                "evidence": [{"page_id": "p0002", "source_text": "test."}],
                "unresolved": [],
            },
            "occurs 2 times",
        ),
        (
            {
                "evidence": [{"page_id": "p0002", "source_text": "New test."}],
                "unresolved": [],
            },
            "does not occur",
        ),
        (
            {
                "evidence": [{"page_id": "unknown", "source_text": "test."}],
                "unresolved": [],
            },
            "unsupplied",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_replacement(value, {page.id: page})
    validate_replacement(
        {
            "evidence": [{"page_id": page.id, "source_text": "Branch B test."}],
            "unresolved": [],
        },
        {page.id: page},
    )


def test_duplicate_rejected_excerpts_share_one_checkpoint_call(
    setup, tmp_path, monkeypatch
):
    guideline, client = setup
    value = quoted_state()
    value["diagnostic_workup"][0]["evidence"] = [
        {"page_id": "p0002", "source_text": "Same absent excerpt"}
    ] * 2
    calls = []

    def respond(endpoint, body, **kwargs):
        calls.append(body)
        return response({"evidence": quoted_state()["evidence"], "unresolved": []})

    monkeypatch.setattr(client, "_http", respond)
    result = repair_quoted_response(
        client,
        guideline,
        guideline.pages,
        "detail-0001",
        value,
        "Correct every invalid excerpt in the draft:",
        lambda v: materialize_quoted_state(v, guideline.pages),
        output=tmp_path,
    )
    assert len(calls) == 1
    assert len(result["diagnostic_workup"][0]["evidence"]) == 2
    receipt = read_json(next((tmp_path / "quote_repairs").glob("*.json")))
    assert len(receipt["replacements"]) == 2
    assert len({p["job"] for p in receipt["replacements"]}) == 1


def test_context_expands_only_after_focused_repair_fails(setup, tmp_path, monkeypatch):
    guideline, client = setup
    extra = replace(
        guideline.pages["p0002"],
        id="p0003",
        number=3,
        text="Different supplied page supports synthetic test A.",
    )
    guideline.pages[extra.id] = extra
    client.config = replace(client.config, attempts=1)
    value = quoted_state()
    value["diagnostic_workup"][0]["evidence"][0]["source_text"] = "Wrong cited page"
    calls = []

    def respond(endpoint, body, **kwargs):
        calls.append(body)
        content = json.dumps(body["messages"])
        if len(calls) == 1:
            assert extra.text not in content
        else:
            assert extra.text in content
        # The first response must be rejected: this page wasn't shown yet.
        return response(
            {
                "evidence": [{"page_id": extra.id, "source_text": extra.text}],
                "unresolved": [],
            }
        )

    monkeypatch.setattr(client, "_http", respond)
    repaired = repair_quoted_response(
        client,
        guideline,
        guideline.pages,
        "detail-0001",
        value,
        "Correct every invalid excerpt in the draft:",
        lambda v: materialize_quoted_state(v, guideline.pages),
        output=tmp_path,
    )
    assert len(calls) == 2
    assert all(body["max_tokens"] == 100000 for body in calls)
    assert clinical_content(repaired) == clinical_content(value)
    assert repaired["diagnostic_workup"][0]["evidence"][0]["page_id"] == extra.id


def test_non_citation_errors_cannot_enter_citation_repair(setup, tmp_path, monkeypatch):
    guideline, client = setup
    monkeypatch.setattr(
        client, "complete", lambda *a, **k: pytest.fail("No repair call")
    )
    with pytest.raises(ValueError, match="fixed population"):
        repair_quoted_response(
            client,
            guideline,
            guideline.pages,
            "detail-0001",
            quoted_state(),
            "Changed fixed population",
            lambda _: None,
            output=tmp_path,
        )


def test_patches_preserve_valid_quotes_and_reject_unknown_positions():
    original = quoted_state()
    replacement = {
        "owner_index": 0,
        "evidence_index": 0,
        "result": {"evidence": original["evidence"]},
    }
    assert apply_replacements(original, [replacement]) == original
    with pytest.raises(ValueError, match="Duplicate"):
        apply_replacements(original, [replacement, replacement])
    with pytest.raises(ValueError, match="Unknown"):
        apply_replacements(original, [{**replacement, "evidence_index": 5}])


def test_pipeline_resumes_old_failed_draft_and_audits_repair_provenance(
    setup, tmp_path, monkeypatch
):
    import matchminer_ai.trials._guideline_pipeline as pipeline

    guideline, client = setup
    output = tmp_path / "catalog"
    calls = []
    repairing = False
    original_repair = pipeline.repair_quoted_response

    def unavailable(*args, **kwargs):
        raise ValueError(args[5])

    def respond(self, endpoint, body, **kwargs):
        prompt = body["messages"][1]["content"]
        if kwargs.get("output_schema") == REPAIR:
            calls.append("repair")
            return response(
                {
                    "evidence": [{"page_id": "p0002", "source_text": TEXT}],
                    "unresolved": [],
                }
            )
        assert (
            not repairing
        ), "Resume must reuse the original raw draft and accepted work"
        if prompt.startswith(prompts.EXTRACT_TASK):
            value = extraction()
        elif prompt.startswith(CATALOG_TASK):
            value = catalog()
        else:
            value = quoted_state()
            value["diagnostic_workup"][0]["evidence"][0]["source_text"] = (
                "Missing source quotation"
            )
        return response(value)

    monkeypatch.setattr(Client, "_http", respond)
    monkeypatch.setattr(pipeline, "repair_quoted_response", unavailable)
    with pytest.raises(RuntimeError, match="detail jobs failed"):
        run_guideline(guideline, output, replace(client.config, attempts=1), workers=1)
    saved_attempts = {
        f: f.read_bytes() for f in (output / "checkpoints").glob("*/attempt-*.json")
    }
    old = read_json(output / "run_config.json")
    for name in PROMPT_FILES:
        old["prompt_resources_sha256"].pop(name)
    old["stage_versions"].pop("detail_citation_repairs")
    identity = {
        k: copy.deepcopy(v)
        for k, v in old.items()
        if k not in ("config_sha256", "runtime", "stage_versions")
    }
    for key in (
        "timeout",
        "attempts",
        "api_key_env",
        "stream",
        "max_concurrent_requests",
    ):
        identity["llm"].pop(key)
    old["config_sha256"] = digest(identity)
    atomic_json(output / "run_config.json", old)
    monkeypatch.setattr(pipeline, "repair_quoted_response", original_repair)
    repairing = True
    run_guideline(guideline, output, client.config, workers=1)
    assert calls == ["repair"]
    assert all(f.read_bytes() == raw for f, raw in saved_attempts.items())
    assert audit_catalog(guideline, output)["status"] == "passed"
    # Repair checkpoints are reusable; no repeated inference on a successful resume.
    monkeypatch.setattr(
        Client, "_http", lambda *a, **k: pytest.fail("No repeated inference")
    )
    run_guideline(guideline, output, client.config, workers=1)
    receipt_path = next((output / "quote_repairs").glob("*.json"))
    receipt = read_json(receipt_path)
    receipt["replacements"][0]["result_sha256"] = "tampered"
    atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="accepted model"):
        audit_catalog(guideline, output)


@pytest.mark.parametrize("diagnostic,expected", [
    ("quote-select: exhausted 6 attempts: No supporting literal passage selected: synthetic wrong phase", ValueError),
    ("quote-select: exhausted 6 attempts: HTTP 503", EndpointError),
])
def test_unsupported_fixed_assertion_returns_to_generation_but_transport_does_not(
    setup, tmp_path, monkeypatch, diagnostic, expected,
):
    from matchminer_ai.llm.structured import EndpointError
    import matchminer_ai.trials._guideline_quote_repair as repair

    guideline, client = setup
    value = quoted_state()
    value["diagnostic_workup"][0]["evidence"][0]["source_text"] = "Synthetic unsupported components"
    original = copy.deepcopy(value)

    def exhausted(*args, **kwargs):
        raise EndpointError("citation repair exhausted 6 attempts")

    def unsupported(*args, **kwargs):
        raise EndpointError(diagnostic)

    monkeypatch.setattr(client, "complete", exhausted)
    monkeypatch.setattr(repair, "select_literal_passages", unsupported)
    with pytest.raises(expected) as error:
        repair_quoted_response(client, guideline, guideline.pages, "detail-0001", value,
            "Correct every invalid excerpt in the draft: unsupported", lambda v: None,
            output=tmp_path)
    assert value == original
    if expected is ValueError:
        assert "Fixed draft assertion" in str(error.value)
        assert "Synthetic unsupported components" in str(error.value)
        assert not list((tmp_path / "quote_repairs").glob("*.json"))


@pytest.mark.parametrize("issue", ["components", "category"])
def test_pipeline_regenerates_unsupported_detail_and_audits_original_population(
    setup, tmp_path, monkeypatch, issue,
):
    import matchminer_ai.trials._guideline_quote_repair as repair
    from matchminer_ai.trials._guideline_quotes import QUOTED_DETAIL

    guideline, client = setup
    output = tmp_path / "catalog"
    detail_calls = []

    def no_support(*args, **kwargs):
        raise EndpointError(
            "quote-select: exhausted 6 attempts: No supporting literal passage selected: "
            + ("synthetic wrong phase" if issue == "components" else "category annotation belongs to adjacent option")
        )

    def respond(self, endpoint, body, **kwargs):
        schema = kwargs.get("output_schema")
        if schema == REPAIR:
            raise EndpointError("citation repair exhausted 6 attempts")
        prompt = body["messages"][1]["content"]
        if prompt.startswith(prompts.EXTRACT_TASK):
            value = extraction()
        elif prompt.startswith(CATALOG_TASK):
            value = catalog()
        else:
            assert schema == QUOTED_DETAIL
            detail_calls.append(copy.deepcopy(body))
            value = quoted_state()
            if len(detail_calls) == 1:
                value["diagnostic_workup"][0]["evidence"][0]["source_text"] = "Synthetic unsupported components from another phase"
                if issue == "category":
                    value["diagnostic_workup"][0]["category"] = "Category 2B"
            else:
                assert "Citation-only repair found no supporting passage" in body["messages"][-1]["content"]
                assert "exact unchanged canonical space" in body["messages"][-1]["content"]
                assert "attached to an adjacent option" in body["messages"][-1]["content"]
                assert body["max_tokens"] == 100000
        return response(value)

    monkeypatch.setattr(Client, "_http", respond)
    monkeypatch.setattr(repair, "select_literal_passages", no_support)
    run_guideline(guideline, output, client.config, workers=1)
    assert len(detail_calls) == 2
    assert read_json(output / "details.json")["records"]["detail-0001"]["space"] == quoted_state()["space"]
    assert read_json(output / "details.json")["records"]["detail-0001"]["diagnostic_workup"][0]["category"] == quoted_state()["diagnostic_workup"][0]["category"]
    assert audit_catalog(guideline, output)["status"] == "passed"
    monkeypatch.setattr(Client, "_http", lambda *a, **k: pytest.fail("No regeneration on successful resume"))
    run_guideline(guideline, output, client.config, workers=1)
