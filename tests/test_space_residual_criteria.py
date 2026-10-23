import asyncio
import json
from pathlib import Path

import pytest

from matchminer_ai.trials.space_residual_criteria import (
    build_residual_criteria_messages, core_space_primer, extract_residual_criteria,
    load_residual_criteria_prompts, parse_residual_additions, parse_residual_criteria,
)

PROMPT_DIR = Path(__file__).parents[1] / "src/matchminer_ai/prompts"
SOURCE = "No prior cisplatin. ECOG must be 0 or 1."
SPACE = dict(age="18 years or older", sex="no restriction", cancer_primary_site="lung",
             histology="adenocarcinoma", disease_state="advanced/metastatic",
             prior_treatment_requirements="NA", biomarker_requirements="NA")
CORE = dict(kind="exclusion", criterion="Prior cisplatin.", source_quote="No prior cisplatin.",
            belongs_in_space=True, field="prior_treatment_requirements", reason="Excluded prior treatment.")
RESIDUAL = dict(kind="inclusion", criterion="ECOG must be 0 or 1.", source_quote="ECOG must be 0 or 1.",
                belongs_in_space=False, field=None, reason="Performance status is outside core fields.")


def test_authored_core_rules_verbatim_without_boilerplate_task():
    prompts = load_residual_criteria_prompts(PROMPT_DIR)
    primer = (PROMPT_DIR / "trial.user.primer.txt").read_text()
    assert "Then, list exclusion criteria described" not in prompts["core_primer"]
    assert 'Once you define the list of trial spaces' not in prompts["core_primer"]
    for line in primer.splitlines():
        if line.startswith(("-Re ", "Examples of valid", "Examples of invalid", "CRITICAL:")):
            assert line in prompts["core_primer"]
    for additions in (None, [CORE]):
        text = build_residual_criteria_messages(SOURCE, SPACE, prompts, additions)[0]["content"]
        assert text.index("AUTHORED CORE SPACE DEFINITIONS:") < text.index("TRIAL DOCUMENT:") < text.index("CURRENT COMPLETED SPACE:")
        assert "Reasoning: high" not in text
        assert text.endswith(json.dumps(SPACE))
    with pytest.raises(ValueError, match="boilerplate section"):
        core_space_primer("Unexpected primer format")


def test_classification_validates_grounding_polarity_and_field_contract():
    parsed = parse_residual_criteria(json.dumps({"criteria": [CORE, RESIDUAL]}), SOURCE)
    assert parsed["criteria"][1]["kind"] == "inclusion"
    assert parse_residual_criteria('```json\n{"criteria": []}\n```', SOURCE) == {"criteria": []}
    for update in ({"source_quote": "Not in source"}, {"belongs_in_space": "true"},
                   {"field": "boilerplate"}, {"belongs_in_space": False}, {"reason": ""},
                   {"kind": "other"}, {"criterion": "line\nbreak"}):
        with pytest.raises(ValueError):
            parse_residual_criteria(json.dumps({"criteria": [{**CORE, **update}]}), SOURCE)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_residual_criteria(json.dumps({"criteria": [CORE, CORE]}), SOURCE)
    with pytest.raises(ValueError, match="Incomplete"):
        parse_residual_criteria('{"criteria": []}', SOURCE, "length")
    with pytest.raises(ValueError, match="Duplicate JSON"):
        parse_residual_criteria('{"criteria": [], "criteria": []}', SOURCE)
    with pytest.raises(ValueError):
        parse_residual_additions('{"space": {"age": "18"}}')


def test_source_quote_accepts_markdown_rendering_but_not_changed_thresholds():
    source = r"QTc \<= 480 ms \[QTcF\]."
    criterion = {**RESIDUAL, "criterion": "QTc <= 480 ms.", "source_quote": "QTc <= 480 ms [QTcF]."}
    assert parse_residual_criteria(json.dumps({"criteria": [criterion]}), source)["criteria"] == [criterion]
    rendered = {**criterion, "source_quote": "QTc ≤ 480 ms [QTcF]."}
    assert parse_residual_criteria(json.dumps({"criteria": [rendered]}), source.rstrip("."))["criteria"] == [rendered]
    for changed in ("QTc > 480 ms [QTcF].", "QTc <= 500 ms [QTcF]."):
        with pytest.raises(ValueError, match="unmatched quote"):
            parse_residual_criteria(json.dumps({"criteria": [{**criterion, "source_quote": changed}]}), source)


def test_complete_bare_space_is_normalized_without_accepting_partial_or_extra_fields():
    assert parse_residual_additions(json.dumps(SPACE)) == {"space": SPACE}
    assert parse_residual_additions("```json\n" + json.dumps(SPACE) + "\n```") == {"space": SPACE}
    for value in ({key: item for key, item in SPACE.items() if key != "age"},
                  {**SPACE, "extra": "unsupported"}, {**SPACE, "age": ""}):
        with pytest.raises(ValueError):
            parse_residual_additions(json.dumps(value))


def run_loop(responses, limit=3):
    calls = []
    async def request(stage, messages, parser):
        calls.append((stage, messages))
        return parser(json.dumps(responses[len(calls) - 1]), "stop")
    result = asyncio.run(extract_residual_criteria(
        SOURCE, SPACE, load_residual_criteria_prompts(PROMPT_DIR), request, limit))
    return result, calls


def test_add_missing_core_then_return_only_remaining_criteria():
    revised = {**SPACE, "prior_treatment_requirements": "No prior cisplatin"}
    result, calls = run_loop([{"criteria": [CORE, RESIDUAL]}, {"space": revised}, {"criteria": [RESIDUAL]}])
    assert [name for name, _ in calls] == ["check_00", "add_01", "check_01"]
    assert result["space"] == revised
    assert result["residual_criteria"] == [RESIDUAL]
    assert result["converged"] and result["refinements_applied"] == 1
    assert json.dumps(revised) in calls[-1][1][0]["content"]


def test_cap_returns_latest_space_and_all_remaining_criteria_without_review_queue():
    versions = [{**SPACE, "prior_treatment_requirements": f"No prior cisplatin; qualification {i}"} for i in range(3)]
    responses = []
    for version in versions:
        responses += [{"criteria": [CORE, RESIDUAL]}, {"space": version}]
    responses += [{"criteria": [CORE, RESIDUAL]}]
    result, calls = run_loop(responses)
    assert len(calls) == 7  # Three refinements plus four checks, never eight.
    assert result["space"] == versions[-1]
    assert result["residual_criteria"] == [CORE, RESIDUAL]
    assert not result["converged"] and result["stop_reason"] == "refinement_limit"
    assert result["refinements_applied"] == 3
    assert "review" not in json.dumps(result)


def test_zero_limit_still_extracts_residuals():
    result, calls = run_loop([{"criteria": [CORE, RESIDUAL]}], limit=0)
    assert len(calls) == 1 and result["space"] == SPACE
    assert result["stop_reason"] == "refinement_limit"
    assert result["residual_criteria"] == [CORE, RESIDUAL]


def test_no_change_and_cycle_stop_without_replacing_latest_valid_space():
    result, calls = run_loop([{"criteria": [CORE]}, {"space": SPACE}])
    assert result["stop_reason"] == "no_change" and len(calls) == 2
    revised = {**SPACE, "prior_treatment_requirements": "No prior cisplatin"}
    result, calls = run_loop([{"criteria": [CORE]}, {"space": revised},
                             {"criteria": [CORE]}, {"space": SPACE}])
    assert result["stop_reason"] == "cycle" and len(calls) == 4
    assert result["space"] == revised
    assert result["residual_criteria"] == [CORE]


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_limits_fail_before_sending(limit):
    with pytest.raises(ValueError, match="nonnegative integer"):
        run_loop([], limit=limit)
