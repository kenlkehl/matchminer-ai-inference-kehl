"""Literal passage selection uses synthetic sources and mocked provider responses."""

import copy
import json
from dataclasses import replace

import pytest

from matchminer_ai._storage import atomic_json, digest, read_json
from matchminer_ai.llm.structured import EndpointError, StructuredConfig
from matchminer_ai.trials._guideline_audit import audit_catalog
from matchminer_ai.trials._guideline_generation import Client
from matchminer_ai.trials._guideline_pipeline import run_guideline
from matchminer_ai.trials._guideline_quote_repair import REPAIR
from matchminer_ai.trials._guideline_quote_selection import (
    MARKER,
    PROMPT,
    SELECTION,
    literal_choices,
    selected_patch,
)
from matchminer_ai.trials._guideline_quotes import resolve_excerpt
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
from test_guideline_quote_repair import response


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda _: None)
    guideline = load_guideline(make_library(tmp_path), "fictional")
    client = Client(
        StructuredConfig(model="synthetic", tokenizer_mode="bytes", attempts=2),
        tmp_path / "checkpoints",
    )
    return guideline, client


def assertion(text):
    return {
        "assertion": {"name": "Synthetic test A"},
        "rejected_evidence": {"page_id": "p0002", "source_text": text},
    }


def test_duplicate_table_cells_gain_literal_branch_context(setup):
    guideline, _ = setup
    page = replace(
        guideline.pages["p0002"],
        text=("Branch alpha\nSynthetic test A\n\n" "Branch beta\nSynthetic test A\n"),
    )
    pages = {page.id: page}
    choices = literal_choices(pages, assertion("Synthetic test A"))
    assert {c["source_text"] for c in choices} == {
        "Branch alpha\nSynthetic test A",
        "Branch beta\nSynthetic test A",
    }
    selected = next(c["choice"] for c in choices if "beta" in c["source_text"])
    patch = selected_patch(
        {"selected_passages": [selected], "unresolved": []},
        choices,
        pages,
    )
    assert patch["evidence"][0]["source_text"] == "Branch beta\nSynthetic test A"
    for choice in choices:
        resolve_excerpt(page, choice["source_text"])


@pytest.mark.parametrize("rejected", ["with an ", "and or", ""])
def test_stopword_only_rejected_fragment_searches_fixed_assertion(setup, rejected):
    guideline, _ = setup
    page = replace(
        guideline.pages["p0002"],
        text="Branch alpha\nSynthetic test A\n\nBranch beta\nOther procedure B\n",
    )
    choices = literal_choices({page.id: page}, assertion(rejected))
    assert choices
    assert all("Synthetic test A" in item["source_text"] for item in choices)
    for item in choices:
        resolve_excerpt(page, item["source_text"])
    # Candidate retrieval is not clinical approval: unsupported selections still fail.
    with pytest.raises(ValueError, match="No supporting literal passage"):
        selected_patch(
            {"selected_passages": [], "unresolved": ["No supported assertion"]},
            choices,
            {page.id: page},
        )


def test_interleaved_columns_stay_separate_literal_fragments(setup):
    guideline, _ = setup
    page = replace(
        guideline.pages["p0002"],
        text=(
            "Branch alpha requires     Other discussion here\n"
            "synthetic test A          unrelated footnote\n"
            "only with finding B.      More discussion\n"
        ),
    )
    pages = {page.id: page}
    choices = literal_choices(
        pages, assertion("Branch alpha requires synthetic test A only with finding B.")
    )
    texts = ["Branch alpha requires", "synthetic test A", "only with finding B."]
    selected = [
        next(c["choice"] for c in choices if c["source_text"] == t) for t in texts
    ]
    patch = selected_patch(
        {"selected_passages": selected, "unresolved": []},
        choices,
        pages,
    )
    assert [r["source_text"] for r in patch["evidence"]] == texts
    with pytest.raises(ValueError, match="does not occur"):
        resolve_excerpt(page, " ".join(texts))


@pytest.mark.parametrize(
    "selected,unresolved",
    [
        ([], []),
        ([0], []),
        ([True], []),
        ([1.0], []),
        ([2], []),
        ([1, 1], []),
        ([1], ["No supporting branch"]),
    ],
)
def test_selection_rejects_invalid_choices(setup, selected, unresolved):
    guideline, _ = setup
    choices = [{"choice": 1, "page_id": "p0002", "source_text": TEXT}]
    with pytest.raises(ValueError):
        selected_patch(
            {"selected_passages": selected, "unresolved": unresolved},
            choices,
            guideline.pages,
        )


def test_exhausted_strategy_reuses_valid_raw_response_before_refusing_new_calls(
    setup,
    monkeypatch,
):
    _, client = setup
    calls = []
    messages = [{"role": "user", "content": "Pick a passage"}]
    value = {"selected_passages": [1], "unresolved": []}

    def respond(*args, **kwargs):
        calls.append(1)
        return response(value)

    def validate(_):
        raise ValueError("Synthetic validation failure")

    monkeypatch.setattr(client, "_http", respond)
    with pytest.raises(EndpointError, match="exhausted"):
        client.complete("test", messages, SELECTION, validate, reuse_exhausted=True)
    assert len(calls) == client.config.attempts
    with pytest.raises(EndpointError, match="saved attempts exhausted"):
        client.complete("test", messages, SELECTION, validate, reuse_exhausted=True)
    assert len(calls) == client.config.attempts
    # The option is explicit: other workflows retain their normal resume retries.
    with pytest.raises(EndpointError, match="exhausted"):
        client.complete("test", messages, SELECTION, validate)
    assert len(calls) == client.config.attempts * 2
    monkeypatch.setattr(
        client, "_http", lambda *a, **k: pytest.fail("Saved answer exists")
    )
    assert (
        client.complete(
            "test",
            messages,
            SELECTION,
            lambda _: None,
            reuse_exhausted=True,
        )
        == value
    )


@pytest.mark.parametrize("rejected", ["Synthetic test Z", "and", "with an"])
def test_pipeline_resumes_exhausted_copy_jobs_selects_and_audits_literal_passages(
    setup,
    tmp_path,
    monkeypatch,
    rejected,
):
    import matchminer_ai.trials._guideline_quote_repair as repair

    guideline, client = setup
    output = tmp_path / "catalog"
    calls = []
    selecting = False
    original_select = repair.select_literal_passages

    def unavailable(*args, **kwargs):
        raise EndpointError("Selection fallback was not installed yet")

    def respond(self, endpoint, body, **kwargs):
        calls.append(kwargs["output_schema"])
        if selecting:
            assert (
                kwargs["output_schema"] == SELECTION
            ), "Do not repeat exhausted copy jobs"
            assert body["max_tokens"] == 100000
            assert body["chat_template_kwargs"]["enable_thinking"]
            content = next(
                m["content"] for m in body["messages"] if MARKER in m["content"]
            )
            payload = json.JSONDecoder().raw_decode(content.split(MARKER)[1])[0]
            index = next(
                c["choice"] for c in payload["candidates"] if c["source_text"] == TEXT
            )
            return response({"selected_passages": [index], "unresolved": []})
        if kwargs["output_schema"] == REPAIR:
            return response(
                {
                    "evidence": [
                        {"page_id": "p0002", "source_text": "Synthetic test Z"}
                    ],
                    "unresolved": [],
                }
            )
        prompt = body["messages"][1]["content"]
        if prompt.startswith(prompts.EXTRACT_TASK):
            value = extraction()
        elif prompt.startswith(CATALOG_TASK):
            value = catalog()
        else:
            value = quoted_state()
            value["diagnostic_workup"][0]["evidence"][0]["source_text"] = (
                rejected
            )
        return response(value)

    monkeypatch.setattr(Client, "_http", respond)
    monkeypatch.setattr(repair, "select_literal_passages", unavailable)
    with pytest.raises(RuntimeError, match="detail jobs failed"):
        run_guideline(guideline, output, client.config, workers=1)
    saved_attempts = {
        f: f.read_bytes() for f in (output / "checkpoints").glob("*/attempt-*.json")
    }
    # A real existing catalog has the copy repair prompt but not the new selector.
    old = read_json(output / "run_config.json")
    old["prompt_resources_sha256"].pop(PROMPT)
    identity = {
        k: copy.deepcopy(v)
        for k, v in old.items()
        if k
        not in (
            "config_sha256",
            "runtime",
            "stage_versions",
        )
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
    calls.clear()
    selecting = True
    monkeypatch.setattr(repair, "select_literal_passages", original_select)
    run_guideline(guideline, output, client.config, workers=1)
    assert calls == [SELECTION]
    assert all(f.read_bytes() == raw for f, raw in saved_attempts.items())
    assert audit_catalog(guideline, output)["status"] == "passed"
    monkeypatch.setattr(
        Client, "_http", lambda *a, **k: pytest.fail("No repeated inference")
    )
    run_guideline(guideline, output, client.config, workers=1)
    receipt_path = next((output / "quote_repairs").glob("*.json"))
    original_receipt = read_json(receipt_path)
    for change in ("payload", "response", "patch"):
        receipt = copy.deepcopy(original_receipt)
        item = receipt["replacements"][0]
        if change == "payload":
            item["selection"]["payload"]["candidates"][0]["nearby_source_text"] = (
                "Different branch"
            )
        elif change == "response":
            item["selection"]["result"]["selected_passages"] = [999]
        else:
            item["result"]["evidence"][0]["source_text"] = "Synthetic test Z"
        atomic_json(receipt_path, receipt)
        with pytest.raises(ValueError):
            audit_catalog(guideline, output)
