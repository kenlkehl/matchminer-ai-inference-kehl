"""Named anatomic regions do not introduce population Boolean operators."""

import copy

import pytest

from matchminer_ai.trials._guideline_schema import FIELDS
from matchminer_ai.trials._guideline_specificity import validate_decision_fields


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("text", [
    "Head and neck cancer OR synthetic disease B",
    "Prior therapy for head and neck cancer with recurrence or persistence",
    "(Prior therapy for head and neck cancer) AND (response A OR response B)",
    "HEAD AND NECK cancer OR synthetic disease B",
])
def test_named_region_preserves_fields_and_population_alternatives(field, text):
    value = {"name": "Synthetic population", "space": dict.fromkeys(FIELDS, "NA")}
    value["space"][field] = text
    original = copy.deepcopy(value)
    validate_decision_fields(value, "Fictional guideline")
    assert value == original


@pytest.mark.parametrize("text", [
    "Head and neck cancer AND response A OR response B",
    "Prior therapy AND (head and neck cancer OR disease B AND disease C)",
    "Head and neck cancer OR disease B AND prior therapy",
    "Head disease AND neck disease OR synthetic disease B",
    "Head and cervical cancer OR disease B AND prior therapy",
])
def test_named_region_does_not_hide_ungrouped_population_logic(text):
    value = {"name": "Synthetic population", "space": dict.fromkeys(FIELDS, "NA")}
    value["space"]["prior_treatment_required"] = text
    original = copy.deepcopy(value)
    with pytest.raises(ValueError, match="mixes AND and OR"):
        validate_decision_fields(value, "Fictional guideline")
    assert value == original
