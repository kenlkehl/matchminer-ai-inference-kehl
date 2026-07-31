from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.patients.ontology import NCItDrugRecord, OncoTreeNode
from matchminer_ai.trials import structure_trial_space


class _FakeBackend:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.prompts = []

    def generate_llm_outputs(self, *, prompt_list, **_kwargs):
        self.prompts.append(prompt_list[0])
        if not self.responses:
            raise AssertionError("Unexpected extra LLM request")
        return LLMGenerationResult(
            final_outputs=[self.responses.pop(0)],
            model_metadata={"model_name": "synthetic-trial-space-structurer"},
            finish_reasons=["stop"],
            reasoning_outputs=[""],
            raw_outputs=[],
        )


class _FakeNCItIndex:
    def __init__(self) -> None:
        self.record = NCItDrugRecord(
            code="C106432",
            preferred_name="Pembrolizumab",
            synonyms=("Pembrolizumab", "Keytruda"),
            definition="SELECTED_NCIT_DEFINITION: binds PD-1 and blocks it.",
            semantic_types=("Pharmacologic Substance",),
        )

    def search(self, query: str, *, limit: int = 8):
        del limit
        return [self.record] if query.casefold() == "pembrolizumab" else []


def _fake_oncotree() -> OncoTreeNode:
    luad = OncoTreeNode(
        code="LUAD",
        name="Lung Adenocarcinoma",
        main_type="Non-Small Cell Lung Cancer",
        tissue="Lung",
        children=(),
    )
    lung = OncoTreeNode(
        code="LUNG",
        name="Lung",
        main_type="Lung Cancer",
        tissue="Lung",
        children=(luad,),
    )
    breast = OncoTreeNode(
        code="BREAST",
        name="Breast",
        main_type="Breast Cancer",
        tissue="Breast",
        children=(),
    )
    return OncoTreeNode(
        code="TISSUE",
        name="Tissue",
        main_type=None,
        tissue=None,
        children=(breast, lung),
    )


def test_structures_trial_space_with_bounded_ontology_prompts() -> None:
    responses = [
        {
            "minimum_age": 18,
            "maximum_age": None,
            "sex_allowed": ["female", "male", "other"],
            "cancer_description": "Non-small cell lung cancer",
            "histology_description": "adenocarcinoma",
            "cancer_burden_allowed": ["advanced_or_palliative_intent"],
            "prior_treatment_required": [
                {
                    "requirement": (
                        "Prior pembrolizumab and subsequent disease progression"
                    ),
                    "drug_mentions": ["pembrolizumab"],
                    "response_requirement": "disease progression",
                }
            ],
            "prior_treatment_excluded": [],
            "biomarkers_required": [
                {
                    "marker": "EGFR",
                    "type": "mutation",
                    "result": "activating mutation",
                    "screening_assessment_required": True,
                }
            ],
            "biomarkers_excluded": [],
        },
        {"selected_index": 1, "stop": False},
        {"selected_index": 0, "stop": True},
        {"selected_indices": [0], "retry_queries": []},
        {
            "status": "matched",
            "selected_index": 0,
            "target": "PD-1",
            "mechanism_of_action": "PD-1 blockade",
        },
    ]
    backend = _FakeBackend(responses)
    config = load_default_preset()
    config.remote["enabled"] = True
    progress_updates = []

    with (
        patch(
            "matchminer_ai.patients.structure.get_llm_backend",
            return_value=backend,
        ),
        patch(
            "matchminer_ai.trials.structure.load_oncotree",
            return_value=_fake_oncotree(),
        ),
        patch(
            "matchminer_ai.trials.structure.load_ncit_drug_index",
            return_value=_FakeNCItIndex(),
        ),
    ):
        structured = structure_trial_space(
            "Synthetic package-format trial space",
            trial_id="NCT00000001",
            space_trial_id="NCT00000001-1",
            config=config,
            progress_callback=lambda *args: progress_updates.append(args),
        )

    assert structured["trial_id"] == "NCT00000001"
    assert structured["space_trial_id"] == "NCT00000001-1"
    assert structured["age_range"] == {
        "minimum_age": 18,
        "maximum_age": None,
    }
    assert structured["sex_allowed"] == ["female", "male", "other"]
    assert structured["cancer_type"] == {
        "name": "Lung Cancer",
        "oncotree_code": "LUNG",
    }
    assert structured["histology"] == {
        "name": "Lung Adenocarcinoma",
        "oncotree_code": "LUAD",
    }
    assert structured["cancer_burden_allowed"] == [
        "advanced_or_palliative_intent"
    ]
    required = structured["prior_treatment_required"][0]
    assert required["response_requirement"] == "disease progression"
    assert required["drugs"][0]["ncit_code"] == "C106432"
    assert required["drugs"][0]["target"] == "PD-1"
    assert structured["biomarkers_required"][0] == {
        "marker": "EGFR",
        "type": "mutation",
        "result": "activating mutation",
        "screening_assessment_required": True,
    }

    prompt_texts = [
        prompt.messages[-1]["content"]
        for prompt in backend.prompts
        if prompt.messages is not None
    ]
    root_prompt = next(text for text in prompt_texts if '"code": "LUNG"' in text)
    assert '"code": "BREAST"' in root_prompt
    assert "LUAD" not in root_prompt
    candidate_prompt = next(
        text for text in prompt_texts if '"preferred_name": "Pembrolizumab"' in text
    )
    assert "SELECTED_NCIT_DEFINITION" not in candidate_prompt
    assert any("SELECTED_NCIT_DEFINITION" in text for text in prompt_texts)
    assert progress_updates[-1][0] == "complete"
    assert backend.responses == []


def test_rejects_empty_trial_space() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        structure_trial_space(" ")
