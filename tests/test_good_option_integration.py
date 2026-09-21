"""Guard existing artifacts while relocating GoodOption APIs and prompts."""

import hashlib
import importlib
import json

from matchminer_ai.trials import drug_catalog as catalog, DrugSummary, DrugIdentity
from matchminer_ai.matching import good_options as scoring
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.trials.drug_research import default_sources

drug = DrugIdentity(
    drug_id="D1",
    preferred_name="Synthetic Agent {A}",
    ncit_code="C123",
    definition="Synthetic marker inhibitor.",
)
summary = DrugSummary(
    drug_id="D1",
    preferred_name=drug.preferred_name,
    ncit_code="",
    research_status="complete",
    synthesis_status="ok",
    structured_facts={},
    good_option_summary="Drug: Synthetic Agent {A}\nSynthetic evidence.",
    help_me_choose_summary="",
    evidence_count=1,
)
patient = "Synthetic patient {marker}: α."
study = {
    "protocolSection": {
        "identificationModule": {"briefTitle": "Synthetic trial"},
        "armsInterventionsModule": {
            "armGroups": [
                {
                    "label": "Test",
                    "type": "EXPERIMENTAL",
                    "description": "Synthetic treatment",
                }
            ],
            "interventions": [
                {
                    "type": "DRUG",
                    "name": drug.preferred_name,
                    "armGroupLabels": ["Test"],
                }
            ],
        },
    }
}


def outputs(c, s):
    messages = s.build_good_option_messages(
        patient_summary=patient, drug_summaries=[summary]
    )
    return {
        "screen": c.build_intervention_screening_messages(
            "NCT12345678", c._extract_registry_interventions("NCT12345678", study)
        ),
        "synthesis": c.build_synthesis_messages(
            drug, c._ncit_definition_evidence(drug, ncit_version="test")
        ),
        "scoring": messages,
        "checker": s.build_good_option_checker_text(patient, summary),
        "retry": s._append_good_option_retry_feedback(
            messages,
            response="{}",
            parse_error="Missing {criterion}",
            finish_reason="length",
        ),
        "retry_without_finish": s._append_good_option_retry_feedback(
            messages, response="", parse_error="", finish_reason=""
        ),
        "compatibility_id": c._compatibility_id(ncit_version="26.07d"),
    }


def test_prompt_and_artifact_versions_match_pre_refactor_snapshot():
    # Fabricated data, including literal braces and Unicode. A text change
    # requires deliberate artifact-version review.
    #
    # Re-captured for the class- and indication-aware retrieval work: the
    # synthesis prompt was rewritten (claim -> text), the scoring prompt gained
    # the class-evidence legend, and the compatibility id gained the class
    # prompt version. Re-captured again for the screening-role fix, which
    # rewrote the screen prompt and bumped the role policy and prompt versions,
    # and for serial synthesis, which bumped the synthesis prompt version.
    expected = {
        "screen": "c44cb04dbc13dbf47bab65710a86380c19a09f41158d5e448f69a4701801e807",
        "synthesis": "13f0476492ca2d769502d9ee91d3c96225635834075c5bb225c0de7cf8290e58",
        "scoring": "21d1254fff62df863ebb413b15afbf97b82e09b2705f47abfc8a2c2539aa36ae",
        "checker": "1123fd83802f67d38aa57b351b74516e8dfc14f8c7f1122ccb33e21f03281fc9",
        "retry": "be4026558080f5c0bf5b5ff7242ae91dea7c69a4b1e50f4ee74d04a7e534b8ab",
        "retry_without_finish": "327f5fa6a423706f85250730b0311da316cd7f6b9b9c0edf3908dde816b27988",
        "compatibility_id": "c2ea8b6d776c3886c2df71e08f99de463997153f3c2dc9f8afd4c38e0c80fa5b",
    }
    actual = {
        key: hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        for key, value in outputs(catalog, scoring).items()
    }
    assert actual == expected


def test_catalog_retry_templates_match_pre_refactor_snapshot():
    expected = {
        "trial_drug_screen.retry.txt": "f4a6e1d0c3389fa8b0dcc21d330c200b23d87f20b936be2e57be5229d5cf55fa",
        "trial_drug_synthesis.retry.txt": "bd0cde2ac9b567c8066c78f581e91e8d3ae81ce25a26728cff6939f377c92f61",
    }
    for filename, digest in expected.items():
        rendered = load_prompt_text(filename).format(
            previous_content="Synthetic {data}",
            attempt=2,
            max_attempts=3,
            previous_error="Missing {field}",
        )
        assert hashlib.sha256(rendered.encode()).hexdigest() == digest


def test_legacy_exports_and_modules_retain_identity():
    legacy = importlib.import_module("matchminer_ai.good_options")
    trials = importlib.import_module("matchminer_ai.trials")
    matching = importlib.import_module("matchminer_ai.matching")
    for name in legacy.__all__:
        owner = matching if name in matching.__all__ else trials
        assert getattr(legacy, name) is getattr(owner, name), name
    for old, new in (
        ("catalog", "trials.drug_catalog"),
        ("research", "trials.drug_research"),
        ("models", "trials.drug_evidence"),
        ("scoring", "matching.good_options"),
    ):
        assert importlib.import_module(
            f"matchminer_ai.good_options.{old}"
        ) is importlib.import_module(f"matchminer_ai.{new}")


def test_checkpoint_source_identities_survive_module_move():
    for source in default_sources():
        assert catalog._callable_identity(source, default="") == (
            f"matchminer_ai.good_options.research.{type(source).__qualname__}"
        )

    # External providers must retain their own identity.
    class CustomSource:
        pass

    assert catalog._callable_identity(CustomSource(), default="") == (
        f"{__name__}.{CustomSource.__qualname__}"
    )
