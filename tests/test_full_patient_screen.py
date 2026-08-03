from __future__ import annotations

import json
from concurrent.futures import Future
from unittest.mock import patch

import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.patients import full_patient_screen


CRITERIA = (
    "Inclusion: Histologically confirmed lung adenocarcinoma.\n"
    "Exclusion: Active brain metastases."
)


class _FakeBackend:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.prompts = []

    def generate_llm_outputs(self, *, prompt_list, **_kwargs):
        if not self.responses:
            raise AssertionError("Unexpected extra LLM request")
        self.prompts.append(prompt_list[0])
        return LLMGenerationResult(
            final_outputs=[self.responses.pop(0)],
            model_metadata={"model_name": "synthetic-full-screen-model"},
            finish_reasons=["stop"],
            reasoning_outputs=[""],
            raw_outputs=[],
        )


class _InlineProcessPool:
    instances: list["_InlineProcessPool"] = []

    def __init__(self, *, initializer, initargs, **kwargs) -> None:
        self.initializer = initializer
        self.initargs = initargs
        self.kwargs = kwargs
        self.submitted = []
        self.__class__.instances.append(self)

    def __enter__(self):
        self.initializer(*self.initargs)
        return self

    def __exit__(self, *_args) -> None:
        return None

    def submit(self, fn, *args) -> Future:
        self.submitted.append((fn, args))
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # pragma: no cover - mirrors executor behavior
            future.set_exception(exc)
        return future


def _config(*, remote: bool = True):
    config = load_default_preset()
    config.remote["enabled"] = remote
    config.full_patient_screen.update(
        {
            "max_workers": 2,
            "max_questions": 8,
            "response_retry_limit": 0,
        }
    )
    return config


def _decomposition() -> dict:
    return {
        "coverage_complete": True,
        "questions": [
            {
                "criterion_id": "inclusion_001",
                "criterion_type": "inclusion",
                "source_id": "source_0001",
                "question": "Do the notes document lung adenocarcinoma?",
            },
            {
                "criterion_id": "exclusion_001",
                "criterion_type": "exclusion",
                "source_id": "source_0002",
                "question": "Do the notes document active brain metastases?",
            },
        ],
    }


def _final_screen() -> dict:
    return {
        "overall_signal": "mixed",
        "summary": "The diagnosis is documented; brain status needs review.",
        "criteria_assessments": [
            {
                "criterion_id": "inclusion_001",
                "eligibility_signal": "supports_eligibility",
                "rationale": "The grounded answer documents the diagnosis.",
            },
            {
                "criterion_id": "exclusion_001",
                "eligibility_signal": "insufficient_information",
                "rationale": "The grounded answer does not establish current status.",
            },
        ],
        "limitations": ["Current imaging was unavailable."],
    }


def test_parallel_screen_uses_raw_note_qa_and_preserves_result_order() -> None:
    notes = pd.DataFrame(
        {
            "patient_id": ["synthetic-1"],
            "note_date": ["2026-05-01"],
            "note_text": ["Biopsy showed lung adenocarcinoma."],
        }
    )
    backend = _FakeBackend([_decomposition(), _final_screen()])
    calls: list[tuple[str, pd.DataFrame]] = []
    progress = []
    _InlineProcessPool.instances.clear()

    def fake_answer(question, patient_notes, **kwargs):
        assert kwargs["config"].remote["enabled"] is True
        assert kwargs["text_column"] == "note_text"
        assert kwargs["date_column"] == "note_date"
        calls.append((question, patient_notes.copy()))
        if "adenocarcinoma" in question:
            answer = "The biopsy documents lung adenocarcinoma."
            quote = "Biopsy showed lung adenocarcinoma."
        else:
            answer = "The available note does not document current brain imaging."
            quote = "Biopsy showed lung adenocarcinoma."
        return {
            "question": question,
            "answer": answer,
            "evidence": [
                {
                    "chunk_id": "chunk_0000",
                    "quote": quote,
                    "reason": "Synthetic exact quote.",
                    "note_date": "2026-05-01",
                }
            ],
            "limitations": [],
        }

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
        patch(
            "matchminer_ai.patients.full_screen.ProcessPoolExecutor",
            _InlineProcessPool,
        ),
        patch(
            "matchminer_ai.patients.full_screen."
            "answer_question_with_raw_patient_notes",
            side_effect=fake_answer,
        ),
    ):
        result, metadata = full_patient_screen(
            notes,
            CRITERIA,
            config=_config(),
            max_workers=2,
            return_metadata=True,
            progress_callback=lambda *args: progress.append(args),
        )

    assert len(_InlineProcessPool.instances) == 1
    pool = _InlineProcessPool.instances[0]
    assert pool.kwargs["max_workers"] == 2
    assert len(pool.submitted) == 2
    assert [call[0] for call in calls] == [
        "Do the notes document lung adenocarcinoma?",
        "Do the notes document active brain metastases?",
    ]
    assert [item["criterion_id"] for item in result["criteria"]] == [
        "inclusion_001",
        "exclusion_001",
    ]
    assert result["criteria"][0]["patient_note_response"]["evidence"][0][
        "note_date"
    ] == "2026-05-01"
    assert result["workflow"] == {
        "question_count": 2,
        "answered_count": 2,
        "failed_count": 0,
        "process_workers": 2,
        "process_start_method": "spawn",
    }
    assert "does not establish clinical-trial eligibility" in result[
        "research_use_notice"
    ]
    assert metadata["model_metadata"]["full_patient_screen_llm"]["model_name"] == (
        "synthetic-full-screen-model"
    )
    assert progress[-1][0] == "complete"
    assert backend.responses == []

    decomposition_prompt = backend.prompts[0]
    assert decomposition_prompt.messages is not None
    decomposition_payload = json.loads(
        decomposition_prompt.messages[-1]["content"]
    )
    assert decomposition_payload["criterion_sources"] == [
        {
            "source_id": "source_0001",
            "criterion_text": (
                "Inclusion: Histologically confirmed lung adenocarcinoma."
            ),
        },
        {
            "source_id": "source_0002",
            "criterion_text": "Exclusion: Active brain metastases.",
        },
    ]
    assert result["criteria"][0]["criterion_text"] == (
        "Inclusion: Histologically confirmed lung adenocarcinoma."
    )
    synthesis_prompt = backend.prompts[-1]
    assert synthesis_prompt.messages is not None
    synthesis_payload = json.loads(synthesis_prompt.messages[-1]["content"])
    assert len(synthesis_payload["grounded_question_results"]) == 2
    json.dumps(result)


def test_screen_accepts_structured_trial_space_criteria() -> None:
    structured_criteria = {
        "trial_space": "Synthetic lung cancer trial space.",
        "inclusion_criteria": [
            "Histologically confirmed lung adenocarcinoma."
        ],
        "exclusion_criteria": ["Active brain metastases."],
    }
    decomposition = _decomposition()
    decomposition["questions"][0]["criterion_type"] = "other"
    backend = _FakeBackend([decomposition, _final_screen()])

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
        patch(
            "matchminer_ai.patients.full_screen."
            "answer_question_with_raw_patient_notes",
            return_value={
                "question": "Synthetic question.",
                "answer": "The synthetic note was reviewed.",
                "evidence": [],
                "limitations": [],
            },
        ),
    ):
        result = full_patient_screen(
            "Synthetic note text.",
            structured_criteria,
            config=_config(),
            max_workers=1,
        )

    decomposition_prompt = backend.prompts[0]
    assert decomposition_prompt.messages is not None
    decomposition_payload = json.loads(
        decomposition_prompt.messages[-1]["content"]
    )
    assert decomposition_payload["criterion_sources"] == [
        {
            "source_id": "source_0001",
            "criterion_type": "inclusion",
            "criterion_text": "Histologically confirmed lung adenocarcinoma.",
        },
        {
            "source_id": "source_0002",
            "criterion_type": "exclusion",
            "criterion_text": "Active brain metastases.",
        },
    ]
    assert [item["criterion_type"] for item in result["criteria"]] == [
        "inclusion",
        "exclusion",
    ]


def test_question_failure_is_retained_and_cannot_produce_no_concern_signal() -> None:
    final = _final_screen()
    final["overall_signal"] = "no_concern_identified"
    backend = _FakeBackend([_decomposition(), final])

    def fake_answer(question, *_args, **_kwargs):
        if "brain" in question:
            raise RuntimeError("synthetic worker failure")
        return {
            "question": question,
            "answer": "The diagnosis is documented.",
            "evidence": [],
            "limitations": [],
        }

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
        patch(
            "matchminer_ai.patients.full_screen.ProcessPoolExecutor",
            _InlineProcessPool,
        ),
        patch(
            "matchminer_ai.patients.full_screen."
            "answer_question_with_raw_patient_notes",
            side_effect=fake_answer,
        ),
    ):
        result = full_patient_screen(
            "Synthetic note text.",
            CRITERIA,
            config=_config(),
            max_workers=2,
        )

    assert result["overall_signal"] == "mixed"
    assert result["workflow"]["failed_count"] == 1
    failed = result["criteria"][1]
    assert failed["status"] == "error"
    assert failed["eligibility_signal"] == "insufficient_information"
    assert any("1 of 2" in limitation for limitation in result["limitations"])


def test_parallel_mode_rejects_in_process_llm_before_patient_questions() -> None:
    backend = _FakeBackend([_decomposition()])
    with patch(
        "matchminer_ai.patients.raw_note_qa.get_llm_backend",
        return_value=backend,
    ):
        with pytest.raises(ValueError, match="remote/OpenAI-compatible"):
            full_patient_screen(
                "Synthetic note text.",
                CRITERIA,
                config=_config(remote=False),
                max_workers=2,
            )


def test_rejects_decomposition_criteria_not_present_in_source() -> None:
    invalid = _decomposition()
    invalid["questions"][0]["source_id"] = "invented_source"
    backend = _FakeBackend([invalid])
    config = _config()
    config.full_patient_screen["response_retry_limit"] = 0

    with patch(
        "matchminer_ai.patients.raw_note_qa.get_llm_backend",
        return_value=backend,
    ):
        with pytest.raises(
            ValueError,
            match="does not reference a supplied source_id",
        ):
            full_patient_screen(
                "Synthetic note text.",
                CRITERIA,
                config=config,
                max_workers=1,
            )
