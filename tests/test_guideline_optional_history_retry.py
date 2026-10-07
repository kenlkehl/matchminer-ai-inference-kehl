"""Synthetic optional-history retries keep clinical fields and evidence intact."""

import copy
import json
from dataclasses import replace

import pytest

from matchminer_ai._storage import read_json
from matchminer_ai.llm.structured import EndpointError, StructuredConfig
from matchminer_ai.trials._guideline_audit import audit_accepted_response
from matchminer_ai.trials._guideline_generation import Client
from matchminer_ai.trials._guideline_schema import EXTRACTION, validate_extraction
from matchminer_ai.trials._guideline_sources import load_guideline, render_pages
from matchminer_ai.trials._guideline_specificity import validate_decision_field_batch
from test_guideline_extraction import extraction, make_library


@pytest.mark.parametrize("saved_failure", [False, True])
@pytest.mark.parametrize("field,bad_history,good_history,guidance", [
    (
        "prior_treatment_required",
        "Prior procedure A with or without procedure B and staging C",
        "(Prior procedure A with or without procedure B) AND staging C",
        "This retains both receipt and nonreceipt of B",
    ),
    (
        "prior_treatment_required",
        "one or more prior systemic therapy lines AND progression following prior therapy "
        "AND no satisfactory alternative options",
        "(one or more prior systemic therapy lines) AND progression following prior therapy "
        "AND no satisfactory alternative options",
        "Group the complete count phrase, including intervening qualifiers",
    ),
    *[(field,
        "(Fictional marker A OR combined Fictional marker B and Fictional marker C OR Fictional marker D)",
        "Fictional marker A OR (combined Fictional marker B AND Fictional marker C) OR Fictional marker D",
        "Preserve a source-defined combined abnormality as one alternative",
    ) for field in ("biomarkers_required", "biomarkers_excluded")],
])
def test_extraction_retries_optional_history_without_dropping_populations(
    tmp_path, monkeypatch, saved_failure, field, bad_history, good_history, guidance,
):
    source = (
        "Five fictional states require procedure A and staging C; procedure B is optional. "
        "A separate fictional therapy decision requires at least one prior systemic therapy "
        "line, subsequent progression, and no satisfactory alternative options. "
        "The fictional biomarker criteria are marker A, the combination of markers B and C, "
        "or marker D, as alternatives in either a required or excluded field. "
        "Consider synthetic test A when finding B is present."
    )
    guideline = load_guideline(make_library(tmp_path, source), "fictional")
    messages = [{"role": "user", "content": render_pages(guideline.primary, ["p0002"])}]
    bad = extraction()
    base = bad["candidates"].pop()
    for number in range(5):
        item = copy.deepcopy(base)
        item["name"] = f"Fictional population {number}"
        item["space"]["cancer_burden_allowed"] = f"Fictional state {number}"
        item["space"][field] = bad_history
        bad["candidates"].append(item)
    good = copy.deepcopy(bad)
    for item in good["candidates"]:
        item["space"][field] = good_history
    original = copy.deepcopy(bad)

    def validate(value):
        validate_extraction(value, guideline.pages, ["p0002"])
        validate_decision_field_batch(value["candidates"], "Fictional guideline")

    with pytest.raises(ValueError, match="mixes AND and OR") as rejected:
        validate(bad)
    assert all(item["name"] in str(rejected.value) for item in bad["candidates"])
    assert bad == original
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda _: None)
    config = StructuredConfig(
        model="synthetic", attempts=2, tokenizer_mode="bytes",
        request_params={"reasoning_effort": "xhigh"},
    )
    client = Client(replace(config, attempts=1) if saved_failure else config, tmp_path / "cache")
    calls = []

    def generate(endpoint, body, **kwargs):
        calls.append(copy.deepcopy(body))
        value = bad if len(calls) == 1 else good
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(value),
        }}]}

    monkeypatch.setattr(client, "_http", generate)
    raw = None
    if saved_failure:
        with pytest.raises(EndpointError, match="mixes AND and OR"):
            client.complete("extract-0001", messages, EXTRACTION, validate)
        raw = next(client.cache_dir.glob("*/attempt-1.json"))
        raw_bytes = raw.read_bytes()
        client = Client(config, client.cache_dir)
        monkeypatch.setattr(client, "_http", generate)
    assert client.complete("extract-0001", messages, EXTRACTION, validate) == good
    retry = calls[1]
    assert retry["messages"][:-2] == messages
    assert json.loads(retry["messages"][-2]["content"]) == bad
    feedback = retry["messages"][-1]["content"]
    assert guidance in feedback
    assert all(item["name"] in feedback for item in bad["candidates"])
    assert retry["max_tokens"] == calls[0]["max_tokens"] == 100000
    assert client.config.context_window == 262144
    assert retry["chat_template_kwargs"] == {"enable_thinking": True}
    assert retry["reasoning_effort"] == "xhigh"
    if raw is not None:
        assert raw.read_bytes() == raw_bytes
    accepted_path = next(client.cache_dir.glob("*/accepted.json"))
    accepted = read_json(accepted_path)
    assert accepted["result"] == good
    assert accepted["repair_applied"] is False
    audit_accepted_response(accepted_path.parent, accepted)
    for before, after in zip(bad["candidates"], accepted["result"]["candidates"]):
        assert {k: v for k, v in before.items() if k != "space"} == {
            k: v for k, v in after.items() if k != "space"
        }
        assert {k: v for k, v in before["space"].items() if k != field} == {
            k: v for k, v in after["space"].items() if k != field
        }
    monkeypatch.setattr(client, "_http", lambda *a, **k: pytest.fail("Accepted resume regenerates nothing"))
    assert client.complete("extract-0001", messages, EXTRACTION, validate) == good


def test_unchanged_ambiguous_optional_history_stays_failed(tmp_path, monkeypatch):
    value = extraction()
    value["candidates"][0]["space"]["prior_treatment_required"] = (
        "Prior procedure A with or without procedure B and staging C"
    )
    client = Client(
        StructuredConfig(model="synthetic", attempts=2, tokenizer_mode="bytes"), tmp_path,
    )
    monkeypatch.setattr("matchminer_ai.llm.structured.time.sleep", lambda _: None)
    monkeypatch.setattr(client, "_http", lambda *a, **k: {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
    })
    with pytest.raises(EndpointError, match="mixes AND and OR"):
        client.complete("extract-0001", [{"role": "user", "content": "Fictional source"}], EXTRACTION,
                        lambda draft: validate_decision_field_batch(draft["candidates"], "Fictional guideline"))
    assert not list(tmp_path.glob("*/accepted.json"))
    assert len(list(tmp_path.glob("*/attempt-*.json"))) == 2
