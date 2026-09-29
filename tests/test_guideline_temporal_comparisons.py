"""Inclusive numeric time boundaries do not change population Boolean logic."""

import copy

import pytest

from matchminer_ai.trials._guideline_schema import FIELDS
from matchminer_ai.trials._guideline_specificity import validate_decision_fields


@pytest.mark.parametrize("text", [
    "Prior initial therapy AND complete response AND relapse at or after 2 years after therapy",
    "Prior therapy AND assessment at or before 3 months after therapy",
    "Prior therapy AND relapse at or after 1.5 years",
    "Prior therapy AND relapse AT OR AFTER 1 YEAR",
    "Prior therapy AND (relapse at or after 2 weeks OR relapse at or before 1 day)",
])
def test_temporal_boundaries_preserve_original_population(text):
    value = {"name": "Synthetic population", "space": dict.fromkeys(FIELDS, "NA")}
    value["space"]["prior_treatment_required"] = text
    original = copy.deepcopy(value)
    validate_decision_fields(value, "Fictional guideline")
    assert value == original


@pytest.mark.parametrize("text", [
    "Prior therapy AND relapse at or after 2 years OR untreated disease",
    "Prior therapy AND (relapse at or after 2 years OR disease B AND disease C)",
    "Prior therapy AND assessment at OR after treatment",
    "Prior therapy AND relapse at OR after 2nd therapy",
    "Prior therapy AND relapse at OR after 2 cycles",
    "Prior therapy AND relapse at OR after 2 years-old population",
])
def test_temporal_comparisons_do_not_hide_population_alternatives(text):
    value = {"name": "Synthetic population", "space": dict.fromkeys(FIELDS, "NA")}
    value["space"]["prior_treatment_required"] = text
    original = copy.deepcopy(value)
    with pytest.raises(ValueError, match="mixes AND and OR"):
        validate_decision_fields(value, "Fictional guideline")
    assert value == original
