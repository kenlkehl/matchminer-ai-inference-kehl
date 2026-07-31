from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.patients import structure_patient_summary
from matchminer_ai.patients.ontology import (
    NCItDrugRecord,
    OncoTreeNode,
    load_ncit_drug_index,
    load_oncotree,
)


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
            model_metadata={"model_name": "synthetic-structurer"},
            finish_reasons=["stop"],
            reasoning_outputs=[""],
            raw_outputs=[],
        )


class _FakeNCItIndex:
    def __init__(self) -> None:
        self.pembrolizumab = NCItDrugRecord(
            code="C106432",
            preferred_name="Pembrolizumab",
            synonyms=("Pembrolizumab", "Keytruda"),
            definition="PEMBROLIZUMAB_DEFINITION_ONLY: binds PD-1 and blocks it.",
            semantic_types=("Pharmacologic Substance",),
        )
        self.ipilimumab = NCItDrugRecord(
            code="C2052",
            preferred_name="Ipilimumab",
            synonyms=("Ipilimumab", "Yervoy"),
            definition="IPILIMUMAB_DEFINITION_ONLY: binds CTLA-4 and blocks it.",
            semantic_types=("Pharmacologic Substance",),
        )

    def search(self, query: str, *, limit: int = 8):
        del limit
        normalized = query.casefold()
        if normalized == "pembrolizumab":
            return [self.pembrolizumab]
        if normalized == "yervoy":
            return [self.ipilimumab]
        return []


def _fake_oncotree() -> OncoTreeNode:
    luad = OncoTreeNode(
        code="LUAD",
        name="Lung Adenocarcinoma",
        main_type="Non-Small Cell Lung Cancer",
        tissue="Lung",
        children=(),
    )
    melanoma = OncoTreeNode(
        code="MEL",
        name="Melanoma",
        main_type="Melanoma",
        tissue="Skin",
        children=(),
    )
    lung = OncoTreeNode(
        code="LUNG",
        name="Lung",
        main_type="Lung Cancer",
        tissue="Lung",
        children=(luad,),
    )
    skin = OncoTreeNode(
        code="SKIN",
        name="Skin",
        main_type="Skin Cancer",
        tissue="Skin",
        children=(melanoma,),
    )
    return OncoTreeNode(
        code="TISSUE",
        name="Tissue",
        main_type=None,
        tissue=None,
        children=(lung, skin),
    )


def test_structures_every_active_cancer_with_bounded_ontology_prompts() -> None:
    responses = [
        {
            "age": 70,
            "sex": "Female",
            "cancers": [
                {
                    "cancer_description": "Lung cancer",
                    "histology_description": "adenocarcinoma",
                    "biomarkers": [
                        {
                            "marker": "KRAS",
                            "type": "mutation",
                            "result": "G12C",
                        }
                    ],
                    "treatment_history": [
                        {
                            "treatment": "Pembrolizumab",
                            "start_date": "2025-01",
                            "end_date": "2025-06",
                            "drug_mentions": ["pembrolizumab"],
                            "response": "partial response",
                        }
                    ],
                    "cancer_burden": "advanced_or_palliative_intent",
                },
                {
                    "cancer_description": "Melanoma",
                    "histology_description": "cutaneous melanoma",
                    "biomarkers": [
                        {
                            "marker": "PD-L1",
                            "type": "expression",
                            "result": "10%",
                        }
                    ],
                    "treatment_history": [
                        {
                            "treatment": "Ipilimumab",
                            "start_date": "2024",
                            "end_date": None,
                            "drug_mentions": ["ipi"],
                            "response": "stable disease",
                        }
                    ],
                    "cancer_burden": "early_or_curative_intent",
                },
            ],
        },
        {"selected_index": 0, "stop": False},
        {"selected_index": 0, "stop": True},
        {"selected_index": 1, "stop": False},
        {"selected_index": 0, "stop": True},
        {"selected_indices": [0], "retry_queries": []},
        {
            "status": "matched",
            "selected_index": 0,
            "target": "PD-1",
            "mechanism_of_action": "PD-1 blockade",
        },
        {"selected_indices": [], "retry_queries": ["Yervoy"]},
        {"selected_indices": [0], "retry_queries": []},
        {
            "status": "matched",
            "selected_index": 0,
            "target": "CTLA-4",
            "mechanism_of_action": "CTLA-4 blockade",
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
            "matchminer_ai.patients.structure.load_oncotree",
            return_value=_fake_oncotree(),
        ),
        patch(
            "matchminer_ai.patients.structure.load_ncit_drug_index",
            return_value=_FakeNCItIndex(),
        ),
    ):
        structured = structure_patient_summary(
            "Synthetic two-cancer summary",
            config=config,
            progress_callback=lambda *args: progress_updates.append(args),
        )

    assert structured["age"] == 70
    assert structured["sex"] == "female"
    assert len(structured["cancers"]) == 2
    lung, melanoma = structured["cancers"]
    assert lung["cancer_type"] == {
        "name": "Lung Cancer",
        "oncotree_code": "LUNG",
    }
    assert lung["histology"] == {
        "name": "Lung Adenocarcinoma",
        "oncotree_code": "LUAD",
    }
    assert lung["biomarkers"][0] == {
        "marker": "KRAS",
        "type": "mutation",
        "result": "G12C",
    }
    assert lung["treatment_history"][0]["response"] == "partial response"
    assert lung["treatment_history"][0]["drugs"][0] == {
        "source_name": "pembrolizumab",
        "normalized_name": "Pembrolizumab",
        "ncit_code": "C106432",
        "target": "PD-1",
        "mechanism_of_action": "PD-1 blockade",
        "normalization_status": "matched",
    }
    assert melanoma["cancer_type"] == {
        "name": "Skin Cancer",
        "oncotree_code": "SKIN",
    }
    assert melanoma["histology"]["oncotree_code"] == "MEL"
    assert melanoma["treatment_history"][0]["response"] == "stable disease"
    assert melanoma["treatment_history"][0]["drugs"][0]["ncit_code"] == "C2052"

    prompt_texts = [
        prompt.messages[-1]["content"]
        for prompt in backend.prompts
        if prompt.messages is not None
    ]
    root_prompt = next(text for text in prompt_texts if '"code": "LUNG"' in text)
    assert '"code": "SKIN"' in root_prompt
    assert "LUAD" not in root_prompt
    lung_prompt = next(text for text in prompt_texts if '"code": "LUAD"' in text)
    assert "SKIN" not in lung_prompt
    candidate_prompt = next(
        text for text in prompt_texts if '"preferred_name": "Pembrolizumab"' in text
    )
    assert "PEMBROLIZUMAB_DEFINITION_ONLY" not in candidate_prompt
    detail_prompt = next(
        text for text in prompt_texts if "PEMBROLIZUMAB_DEFINITION_ONLY" in text
    )
    assert "IPILIMUMAB_DEFINITION_ONLY" not in detail_prompt
    assert progress_updates[-1][0] == "complete"
    assert backend.responses == []


def test_bundled_ontologies_load_and_find_synonymous_ncit_drug() -> None:
    config = load_default_preset()

    oncotree = load_oncotree(config.patient_structuring["oncotree_resource"])
    ncit = load_ncit_drug_index(config.patient_structuring["ncit_resource"])

    assert oncotree.code == "TISSUE"
    assert len(oncotree.children) > 20
    assert ncit.search("pembrolizumab", limit=1)[0].code == "C106432"
    assert ncit.search("Keytruda", limit=1)[0].preferred_name == "Pembrolizumab"


def test_rejects_empty_summary() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        structure_patient_summary("   ")
