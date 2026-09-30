"""Prompt examples must satisfy the actual standalone/workup response contracts."""

import ast
import json

import pytest

from matchminer_ai.patients import note_search_qa as qa, workup_search as workup


@pytest.mark.parametrize("mode", ["standalone", "workup", "workup_review"])
def test_rendered_search_examples_match_schema_and_final_validator(mode):
    history, spans = workup._history(workup._notes("Chest CT completed."))
    limits = qa.NoteSearchLimits(max_scan_patterns=24)
    contract = (
        None
        if mode == "standalone"
        else workup._answer_format(spans, request_review=mode == "workup_review")
    )
    prompt = qa._search_system(limits, contract)
    schema = qa._schema(limits, contract)
    evidence = [
        {"start": n["start"], "end": n["end"], "quote": n["text"]} for n in spans
    ]
    assert "1 to 24 regex strings" in prompt
    for placeholder in (
        "{task_instructions}",
        "{response_examples}",
        "{assessment_contract}",
        "{max_scan_patterns}",
    ):
        assert placeholder not in prompt
    if contract:
        assert "Your final deliverable is an answer OBJECT" in prompt
        assert "answer field is an empty string" not in prompt
    else:
        assert "Your final deliverable is a concise plain-text answer" in prompt

    for label in ("python", "final", "unknown final"):
        suffix = prompt.split(f"\n{label} action:\n", 1)[1]
        example, _ = json.JSONDecoder().raw_decode(suffix)
        assert set(example) == set(schema["required"])
        assert isinstance(example["answer"], dict if contract else str)
        assert ("needs_review" in example) == (mode == "workup_review")
        qa._validate(
            example,
            history,
            limits,
            final_only=label != "python",
            cells=1,
            answer_format=contract,
            evidence=evidence if label == "final" else (),
        )
        if label == "python":
            ast.parse("\n".join(example["code"]))
        else:
            assert example["code"] == []
