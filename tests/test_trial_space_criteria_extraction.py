from __future__ import annotations

import hashlib
import json
from unittest.mock import patch

import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.trials import (
    TrialSpaceCriteriaExtractionError,
    extract_trial_space_eligibility_criteria,
)

TRIAL_SPACE = (
    "Age: 18+. Sex: Any. Cancer type allowed: Non-small cell lung cancer. "
    "Histology allowed: Non-squamous. Cancer burden allowed: Resected stage II "
    "to IIIB. Biomarkers required: Activating HER2 mutation."
)
INCLUSION_1 = "1. Histologically confirmed non-squamous NSCLC is required."
INCLUSION_2 = "2. An activating HER2 mutation must be documented."
EXCLUSION_1 = "1. Prior treatment with a HER2-directed tyrosine kinase inhibitor."
DOCUMENT = (
    "INCLUSION CRITERIA\n"
    f"{INCLUSION_1}\n"
    f"{INCLUSION_2}\n"
    "\f\n"
    "EXCLUSION CRITERIA\n"
    f"{EXCLUSION_1}\n"
)


class _FakeBackend:
    def __init__(
        self,
        responses: list[str | dict],
        *,
        finish_reasons: list[str] | None = None,
    ) -> None:
        self.responses = [
            response if isinstance(response, str) else json.dumps(response)
            for response in responses
        ]
        self.finish_reasons = list(finish_reasons or ["stop"] * len(responses))
        self.prompts = []
        self.runtime_configs = []

    def generate_llm_outputs(self, *, prompt_list, llm_config, **_kwargs):
        if not self.responses:
            raise AssertionError("Unexpected extra LLM request")
        self.prompts.append(prompt_list[0])
        self.runtime_configs.append(llm_config)
        return LLMGenerationResult(
            final_outputs=[self.responses.pop(0)],
            model_metadata={"model_name": "synthetic-criteria-model"},
            finish_reasons=[self.finish_reasons.pop(0)],
            reasoning_outputs=[""],
            raw_outputs=[],
        )


def _config(*, retry_limit: int = 0):
    config = load_default_preset()
    config.remote["enabled"] = True
    config.trial_space_criteria_extraction["response_retry_limit"] = retry_limit
    return config


def _valid_response() -> dict:
    return {
        "coverage_complete": True,
        "inclusion_criteria": [INCLUSION_1, INCLUSION_2],
        "exclusion_criteria": [EXCLUSION_1],
    }


def test_extracts_grounded_criteria_with_metadata_and_progress(tmp_path) -> None:
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(DOCUMENT, encoding="utf-8")
    backend = _FakeBackend([_valid_response()])
    progress = []

    with patch(
        "matchminer_ai.trials.eligibility.get_llm_backend",
        return_value=backend,
    ):
        result, metadata = extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(),
            return_metadata=True,
            progress_callback=lambda *args: progress.append(args),
        )

    assert result == {
        "trial_space": TRIAL_SPACE,
        "inclusion_criteria": [INCLUSION_1, INCLUSION_2],
        "exclusion_criteria": [EXCLUSION_1],
    }
    assert metadata["source_document"] == {
        "sha256": hashlib.sha256(DOCUMENT.encode()).hexdigest(),
        "character_count": len(DOCUMENT.strip()),
        "page_count": 2,
    }
    assert metadata["execution"] == {
        "llm_generation_calls": 1,
        "inclusion_criteria_count": 2,
        "exclusion_criteria_count": 1,
    }
    assert metadata["model_metadata"]["trial_space_criteria_extractor"] == {
        "model_name": "synthetic-criteria-model"
    }
    assert progress == [
        ("read", 0, 1, "Reading OCR eligibility checklist"),
        ("read", 1, 1, "OCR eligibility checklist loaded"),
        (
            "extract",
            0,
            1,
            "Extracting trial-space inclusion and exclusion criteria",
        ),
        ("extract", 1, 1, "Trial-space criteria extracted"),
        (
            "complete",
            1,
            1,
            "Trial-space eligibility criteria ready for human review",
        ),
    ]

    prompt = backend.prompts[0]
    assert prompt.messages is not None
    assert "untrusted data" in prompt.messages[0]["content"]
    prompt_payload = json.loads(prompt.messages[1]["content"])
    assert prompt_payload == {
        "trial_space": TRIAL_SPACE,
        "ocr_eligibility_checklist": DOCUMENT.strip(),
    }
    runtime_config = backend.runtime_configs[0]
    assert runtime_config["backend_mode"] == "remote"
    assert runtime_config["sampling_params"]["response_format"] == {
        "type": "json_object"
    }


def test_collapses_criterion_whitespace_without_changing_wording(tmp_path) -> None:
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(DOCUMENT, encoding="utf-8")
    response = _valid_response()
    response["inclusion_criteria"][0] = (
        "1. Histologically confirmed\nnon-squamous NSCLC is required."
    )
    backend = _FakeBackend([response])

    with patch(
        "matchminer_ai.trials.eligibility.get_llm_backend",
        return_value=backend,
    ):
        result = extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(),
        )

    assert result["inclusion_criteria"][0] == INCLUSION_1


def test_accepts_line_break_between_ordinal_number_and_suffix(tmp_path) -> None:
    document = "INCLUSION CRITERIA\n1. Stage is classified using the AJCC 9\nth edition."
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(document, encoding="utf-8")
    response = {
        "coverage_complete": True,
        "inclusion_criteria": [
            "1. Stage is classified using the AJCC 9th edition."
        ],
        "exclusion_criteria": [],
    }
    backend = _FakeBackend([response])

    with patch(
        "matchminer_ai.trials.eligibility.get_llm_backend",
        return_value=backend,
    ):
        result = extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(),
        )

    assert result["inclusion_criteria"] == response["inclusion_criteria"]


def test_grounds_criterion_across_repeated_page_header(tmp_path) -> None:
    header_page_1 = (
        "Sponsor Name 19 Jun 2025\n"
        "Trial No.: 1479-0032\n"
        "Clinical Trial Protocol Page 1 of 2\n"
        "Proprietary confidential information\n"
    )
    header_page_2 = header_page_1.replace("Page 1 of 2", "Page 2 of 2")
    document = (
        header_page_1
        + "INCLUSION CRITERIA\n"
        + "1. Adequate organ function requires neutrophils ≥1500/mm3 and\n"
        + "\f\n"
        + header_page_2
        + "platelets ≥100 x 103/mm3.\n"
    )
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(document, encoding="utf-8")
    criterion = (
        "1. Adequate organ function requires neutrophils ≥1500/mm3 and "
        "platelets ≥100 x 103/mm3."
    )
    response = {
        "coverage_complete": True,
        "inclusion_criteria": [criterion],
        "exclusion_criteria": [],
    }
    backend = _FakeBackend([response])

    with patch(
        "matchminer_ai.trials.eligibility.get_llm_backend",
        return_value=backend,
    ):
        result = extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(),
        )

    assert result["inclusion_criteria"] == [criterion]


def test_grounds_criterion_across_referenced_page_end_footnote(tmp_path) -> None:
    document = (
        "EXCLUSION CRITERIA\n"
        "1. History of another malignancy except effectively treated carcinoma in situ1\n"
        "1 Carcinoma in situ is defined by the protocol pathology manual.\n"
        "\f\n"
        "or localized prostate cancer on active surveillance.\n"
    )
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(document, encoding="utf-8")
    criterion = (
        "1. History of another malignancy except effectively treated carcinoma in situ1 "
        "or localized prostate cancer on active surveillance."
    )
    response = {
        "coverage_complete": True,
        "inclusion_criteria": [],
        "exclusion_criteria": [criterion],
    }
    backend = _FakeBackend([response])

    with patch(
        "matchminer_ai.trials.eligibility.get_llm_backend",
        return_value=backend,
    ):
        result = extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(),
        )

    assert result["exclusion_criteria"] == [criterion]


def test_retries_ungrounded_output_with_validation_feedback(tmp_path) -> None:
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(DOCUMENT, encoding="utf-8")
    invalid = _valid_response()
    invalid["exclusion_criteria"] = ["Invented cardiac exclusion criterion."]
    backend = _FakeBackend([invalid, _valid_response()])

    with patch(
        "matchminer_ai.trials.eligibility.get_llm_backend",
        return_value=backend,
    ):
        result = extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(retry_limit=1),
        )

    assert result["exclusion_criteria"] == [EXCLUSION_1]
    assert len(backend.prompts) == 2
    retry_prompt = backend.prompts[1]
    assert retry_prompt.messages is not None
    assert retry_prompt.messages[-2]["role"] == "assistant"
    assert "VALIDATION ERROR" in retry_prompt.messages[-1]["content"]
    assert "not a verbatim excerpt" in retry_prompt.messages[-1]["content"]


def test_rejects_token_limited_output_even_when_json_is_valid(tmp_path) -> None:
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(DOCUMENT, encoding="utf-8")
    backend = _FakeBackend([_valid_response()], finish_reasons=["length"])

    with (
        patch(
            "matchminer_ai.trials.eligibility.get_llm_backend",
            return_value=backend,
        ),
        pytest.raises(TrialSpaceCriteriaExtractionError, match="token limit"),
    ):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(),
        )


@pytest.mark.parametrize(
    "response",
    [
        {
            "coverage_complete": False,
            "inclusion_criteria": [INCLUSION_1],
            "exclusion_criteria": [EXCLUSION_1],
        },
        {
            "coverage_complete": True,
            "inclusion_criteria": [INCLUSION_1, INCLUSION_1],
            "exclusion_criteria": [EXCLUSION_1],
        },
        {
            "coverage_complete": True,
            "inclusion_criteria": [INCLUSION_1],
            "exclusion_criteria": [INCLUSION_1],
        },
        {
            "coverage_complete": True,
            "inclusion_criteria": [],
            "exclusion_criteria": [],
        },
        {
            "coverage_complete": True,
            "inclusion_criteria": "not a list",
            "exclusion_criteria": [EXCLUSION_1],
        },
    ],
)
def test_rejects_invalid_model_payloads(tmp_path, response) -> None:
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(DOCUMENT, encoding="utf-8")
    backend = _FakeBackend([response])

    with (
        patch(
            "matchminer_ai.trials.eligibility.get_llm_backend",
            return_value=backend,
        ),
        pytest.raises(TrialSpaceCriteriaExtractionError),
    ):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(),
        )


def test_rejects_non_json_model_output(tmp_path) -> None:
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(DOCUMENT, encoding="utf-8")
    backend = _FakeBackend(["not json"])

    with (
        patch(
            "matchminer_ai.trials.eligibility.get_llm_backend",
            return_value=backend,
        ),
        pytest.raises(TrialSpaceCriteriaExtractionError, match="after retries"),
    ):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=_config(),
        )


def test_validates_trial_space_and_document_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match="trial_space"):
        extract_trial_space_eligibility_criteria(" ", tmp_path / "missing.txt")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            tmp_path / "missing.txt",
            config=_config(),
        )

    wrong_extension = tmp_path / "eligibility.pdf"
    wrong_extension.write_text(DOCUMENT, encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.txt"):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            wrong_extension,
            config=_config(),
        )

    empty = tmp_path / "empty.txt"
    empty.write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            empty,
            config=_config(),
        )

    invalid_utf8 = tmp_path / "invalid.txt"
    invalid_utf8.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="UTF-8"):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            invalid_utf8,
            config=_config(),
        )


def test_enforces_configured_document_and_criteria_limits(tmp_path) -> None:
    document_path = tmp_path / "eligibility.txt"
    document_path.write_text(DOCUMENT, encoding="utf-8")
    config = _config()
    config.trial_space_criteria_extraction["max_document_characters"] = 10

    with pytest.raises(ValueError, match="max_document_characters"):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=config,
        )

    config = _config()
    config.trial_space_criteria_extraction["max_criteria_per_type"] = 1
    backend = _FakeBackend([_valid_response()])
    with (
        patch(
            "matchminer_ai.trials.eligibility.get_llm_backend",
            return_value=backend,
        ),
        pytest.raises(TrialSpaceCriteriaExtractionError, match="after retries"),
    ):
        extract_trial_space_eligibility_criteria(
            TRIAL_SPACE,
            document_path,
            config=config,
        )
