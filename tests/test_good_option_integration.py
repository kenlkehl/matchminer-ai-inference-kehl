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
    # Captured from commit 5f9fac4 with fabricated data, including literal braces
    # and Unicode. A text change requires deliberate artifact-version review.
    expected = {
        "screen": "efbc763d49e9ff434f993a4a5d088c143eb351b38f161675c07ec26410000826",
        "synthesis": "1ea8a4230a74c06ce5ffe9ff9cc5707c81c039c7ca799ee9e9096a7fb9a55fa2",
        "scoring": "309b368a5bb519ae108ba1a0eb996964797549ae60f7174e04af77aa1253dd72",
        "checker": "1123fd83802f67d38aa57b351b74516e8dfc14f8c7f1122ccb33e21f03281fc9",
        "retry": "c61d618a3398f91e86441212b39cd9519125cd731d2df66aae7943c8a67f3145",
        "retry_without_finish": "7cfd9027c7372eec517f6dc59fdf31badb6cabcae5eb52cd8154d6d1ca6fa96e",
        "compatibility_id": "15b59bf69af525e7335ee62515e5c10ca7ffa07a31153113620dffccdc82cb55",
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
