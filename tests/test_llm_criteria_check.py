"""check_eligibility_criteria_with_llm with a fake backend: one request per tagged line with the per-criterion teacher
prompt, decoded under the label contract's schema, the question-7 polarity, retries, and outputs shaped like
check_eligibility_criteria's."""
from copy import deepcopy
import json
import re

import pandas as pd
import pytest

from conftest import prompt_text
from matchminer_ai import load_default_preset
from matchminer_ai._texts import patient_full_texts
from matchminer_ai.labeling import contract_hash
from matchminer_ai.matching import check_eligibility_criteria_with_llm
from matchminer_ai.matching.decision_model import CRITERION_LABEL_JSON_SCHEMA, criterion_teacher_messages

ITEMS = [
    {"number": 1, "heading": "Inclusion criteria:", "text": "Metastatic non-small cell lung cancer.",
     "criterion_type": "inclusion", "boilerplate": False},
    {"number": 2, "heading": "Inclusion criteria:", "text": "ECOG 0-1.", "criterion_type": "inclusion",
     "boilerplate": True},
    {"number": 3, "heading": "Exclusion criteria:", "text": "Prior osimertinib.", "criterion_type": "exclusion",
     "boilerplate": False},
    {"number": 4, "heading": "Exclusion criteria:", "text": "A line whose type never validated.",
     "criterion_type": None, "boilerplate": False},
]
QUESTION = re.compile(r"CRITERION TO ASSESS\n\[(\d+)\] (?:Inclusion|Exclusion) criterion:")


def answer(value, status, reason="Fabricated reason."):
    return json.dumps({"answer": value, "criterion_status": status, "reason": reason})


class LineBackend:
    """Answers each line's request from ``script[number]``, consumed in order (the last repeats); an
    ``(answer, finish_reason)`` pair sets the finish reason."""

    def __init__(self, script):
        self.script, self.calls = {n: list(v) for n, v in script.items()}, []

    def generate_llm_outputs(self, *, prompt_list, llm_config, **kwargs):
        from matchminer_ai.llm.backends import LLMGenerationResult

        outputs = []
        for prompt in prompt_list:
            text = prompt_text(prompt)
            number = int(QUESTION.search(text).group(1))
            self.calls.append((number, text, deepcopy(llm_config)))
            queue = self.script[number]
            value = queue.pop(0) if len(queue) > 1 else queue[0]
            outputs.append(value if isinstance(value, tuple) else (value, "stop"))
        return LLMGenerationResult(final_outputs=[a for a, _ in outputs], finish_reasons=[f for _, f in outputs],
                                   reasoning_outputs=["fabricated reasoning"] * len(outputs), raw_outputs=[],
                                   model_metadata={"model_sha": "test"})


@pytest.fixture
def config():
    config = load_default_preset()
    config.remote["enabled"] = True
    config.remote["server_urls"] = ["http://fake/v1"]
    return config


def pairs(rows=1):
    return pd.DataFrame([{"patient_id": "P1", "cohort_trial_id": "NCT00000001-cohort-0",
                          "cancer_history_summary": "Age: 60\nCancer type: Lung cancer\nBiomarkers: EGFR L858R",
                          "general_exclusion_criteria_evidence": "ECOG 1.",
                          "cohort_criteria_items": json.dumps(ITEMS)}] * rows)


def use(monkeypatch, backend):
    monkeypatch.setattr("matchminer_ai.matching.llm_checks.get_llm_backend", lambda _config: backend)


def test_each_line_gets_the_teacher_request_under_its_schema(monkeypatch, config):
    backend = LineBackend({1: [answer("yes", "met")], 2: [answer("yes", "insufficient_information")],
                           3: [answer("no", "met", "Received osimertinib.")]})
    use(monkeypatch, backend)
    seen = []
    result, metadata = check_eligibility_criteria_with_llm(pairs(), config=config, return_metadata=True,
                                                           progress=lambda done, total: seen.append((done, total)))
    # The polarity of question 7: yes = does not clearly fail the line; no = clearly fails it.
    assert result["criterion_number"].tolist() == [1, 2, 3]
    assert result["llm_criterion_answer"].tolist() == ["yes", "yes", "no"]
    assert result["llm_criterion_clearly_fails"].tolist() == [False, False, True]
    assert result["llm_criterion_status"].tolist() == ["met", "insufficient_information", "met"]
    assert result["llm_criterion_reason"].tolist()[2] == "Received osimertinib."
    assert result["llm_criterion_parse_status"].eq("parsed").all() and result["llm_criterion_attempts"].eq(1).all()
    assert result["boilerplate"].tolist() == [False, True, False] and "llm_criterion_reasoning_text" not in result
    # The training teacher's request, line by line, decoded under the label contract's answer schema.
    summary = patient_full_texts(pairs())[0]
    for number, text, runtime in backend.calls:
        assert text == criterion_teacher_messages(summary, ITEMS, number)[0]["content"]
        schema = runtime["remote"]["request_params"]["response_format"]["json_schema"]["schema"]
        assert schema == CRITERION_LABEL_JSON_SCHEMA
        assert runtime["remote"]["request_params"]["temperature"] == 1.0  # Nemotron's own sampling profile
    assert sorted(n for n, _, _ in backend.calls) == [1, 2, 3]
    details = metadata["model_metadata"]["check_eligibility_criteria_with_llm"]
    assert metadata["criteria_contract"] == contract_hash("criteria")
    assert details["unchecked_lines"] == 1 and details["lines"] == 3 and details["requests"] == 3
    assert details["retries"] == 0 and details["parse_failures"] == 0 and details["structured_outputs"] is True
    assert seen[-1] == (3, 3)


def test_invalid_answers_are_retried_with_the_error_then_reported(monkeypatch, config):
    backend = LineBackend({
        # "no" on an inclusion line needs not_met: invalid, then a valid answer
        1: [answer("no", "insufficient_information"), answer("no", "not_met", "Has breast cancer.")],
        2: [(answer("yes", "met"), "length")],  # a truncated answer never validates
        3: ["not JSON"],
    })
    use(monkeypatch, backend)
    result = check_eligibility_criteria_with_llm(pairs(), config=config)
    first, second, third = result.to_dict("records")
    assert first["llm_criterion_answer"] == "no" and first["llm_criterion_clearly_fails"] is True
    assert first["llm_criterion_attempts"] == 2 and first["llm_criterion_parse_status"] == "parsed"
    retry = next(text for number, text, _ in backend.calls if number == 1 and "failed validation" in text)
    assert 'requires criterion_status "not_met"' in retry
    assert second["llm_criterion_answer"] is None and second["llm_criterion_clearly_fails"] is None
    assert second["llm_criterion_finish_reason"] == "length" and second["llm_criterion_attempts"] == 3
    assert second["llm_criterion_parse_status"].startswith("parse_failed: Incomplete response")
    assert third["llm_criterion_parse_status"].startswith("parse_failed: Response must be valid JSON")
    assert third["llm_criterion_reason"] == ""
    failing = check_eligibility_criteria_with_llm(pairs(), config=config, filter_failing=True)
    assert failing["criterion_number"].tolist() == [1]  # unanswered lines are not reported as failing


def test_identical_pairs_are_asked_once_and_debug_mode_returns_the_texts(monkeypatch, config):
    backend = LineBackend({n: [answer("yes", "met")] for n in (1, 2, 3)})
    use(monkeypatch, backend)
    config.debug_mode = True
    result = check_eligibility_criteria_with_llm(pairs(rows=2), config=config)
    assert len(result) == 6 and len(backend.calls) == 3
    assert result["llm_criterion_reasoning_text"].eq("fabricated reasoning").all()
    assert json.loads(result["llm_criterion_answer_text"].iloc[0])["answer"] == "yes"


def test_no_checkable_line_sends_nothing(monkeypatch, config):
    backend = LineBackend({})
    use(monkeypatch, backend)
    frame = pairs()
    frame["cohort_criteria_items"] = json.dumps([ITEMS[3] | {"number": 1}])
    result, metadata = check_eligibility_criteria_with_llm(frame, config=config, return_metadata=True)
    assert result.empty and backend.calls == []
    assert metadata["model_metadata"]["check_eligibility_criteria_with_llm"]["unchecked_lines"] == 1
    with pytest.raises(ValueError, match="cohort_criteria_items"):
        check_eligibility_criteria_with_llm(frame.drop(columns="cohort_criteria_items"), config=config)
