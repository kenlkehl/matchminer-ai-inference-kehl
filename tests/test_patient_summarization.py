import asyncio
from unittest.mock import MagicMock

import pandas as pd

from matchminer_ai.config import MMAIConfig
from matchminer_ai.llm.backends import LLMGenerationResult, LocalBackend
from matchminer_ai.llm.prompt_rendering import Prompt
from matchminer_ai.llm.remote_inference import generate_remote_llm_outputs
from matchminer_ai.patients import summarize_patients
from matchminer_ai.patients.postprocess import (
    clean_bad_data,
    parse_boilerplate,
)
from matchminer_ai.patients.prompt_builder import (
    PromptWorkItem,
    _RESPONSE_TOKEN_MARGIN,
    build_prompt_worker,
    get_serial_patient_prompt,
)
from matchminer_ai.patients.summarize import (
    summarize_patient_notes,
    validate_existing_summaries,
)


class MockTokenResult:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class MockTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return MockTokenResult(list(text))

    def decode(self, input_ids, skip_special_tokens=True):
        return "".join(input_ids)


def _stub_patient_qc(monkeypatch):
    monkeypatch.setattr(
        "matchminer_ai._qc.patients.patient_summary_qc_report",
        lambda *args, **kwargs: pd.DataFrame(),
    )


def _patient_config() -> dict:
    return {
        "model_name": "model",
        "prompt_files": {
            "primer": "patient.serial.user.primer.txt",
            "question": "patient.serial.user.question.txt",
        },
        "boilerplate_marker": "Boilerplate conditions:",
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "max_tokens": 10,
            "repetition_penalty": 1.0,
        },
        "chunk_size": 50,
        "chunk_overlap": 5,
        "prompt_margin_tokens": 10,
    }


def _config(debug_mode: bool = False) -> MMAIConfig:
    return MMAIConfig(
        preset_name="default",
        debug_mode=debug_mode,
        trial={},
        patient=_patient_config(),
        local={
            "patient": {
                "max_model_len": 100,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            }
        },
        remote={},
        model_metadata_cache_dir=None,
        raw={"config": "snapshot"},
        embedding={
            "model_path": "mock-model",
            "device": "cpu",
            "prompt_file": "embedding.txt",
        },
    )


def _remote_config(debug_mode: bool = False) -> MMAIConfig:
    config = _config(debug_mode=debug_mode)
    config.remote = {
        "enabled": True,
        "server_urls": ["http://server-a/v1"],
        "max_concurrent_requests": 2,
        "request_timeout": 123,
        "max_retries": 2,
        "batch_size": 1000,
    }
    config.patient = {
        **config.patient,
        "prompt_build_workers": 2,
    }
    return config


def test_validate_existing_summaries_accepts_combined_column():
    """Normalize combined existing summaries to the canonical column."""
    existing = pd.DataFrame(
        [
            {
                "patient_id": 1,
                "last_note_date": "2024-01-03",
                "patient_summary_with_boilerplate": (
                    "Summary\n\nBoilerplate conditions:\nNone"
                ),
            }
        ]
    )

    normalized = validate_existing_summaries(existing)

    assert normalized.loc[0, "patient_id"] == "1"
    assert normalized.loc[0, "last_note_date"] == "2024-01-03"
    assert (
        normalized.loc[0, "patient_summary_with_boilerplate"]
        == "Summary\n\nBoilerplate conditions:\nNone"
    )


def test_validate_existing_summaries_concatenates_split_columns():
    """Build canonical prior-summary text from split output columns."""
    existing = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "last_note_date": "2024-01-03",
                "cancer_history_summary": "Cancer summary",
                "general_exclusion_criteria_evidence": "ECOG 1",
            }
        ]
    )

    normalized = validate_existing_summaries(existing)

    assert (
        normalized.loc[0, "patient_summary_with_boilerplate"]
        == "Cancer summary\n\nBoilerplate conditions:\nECOG 1"
    )


def test_validate_existing_summaries_concatenates_training_split_columns():
    """Accept training output columns as split summary and boilerplate text."""
    existing = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "last_note_date": "2024-01-03",
                "patient_summary": "Training summary",
                "patient_boilerplate_text": "Training boilerplate",
            }
        ]
    )

    normalized = validate_existing_summaries(existing)

    assert (
        normalized.loc[0, "patient_summary_with_boilerplate"]
        == "Training summary\n\nBoilerplate conditions:\nTraining boilerplate"
    )


def test_validate_existing_summaries_rejects_missing_date():
    """Existing summaries need a cutoff date for incremental note filtering."""
    existing = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "patient_summary_with_boilerplate": "Summary",
            }
        ]
    )

    try:
        validate_existing_summaries(existing)
    except ValueError as exc:
        assert "last_note_date" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing last_note_date")


def test_validate_existing_summaries_rejects_invalid_date():
    """Existing summary cutoff dates must be parseable dates."""
    existing = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "last_note_date": "not-a-date",
                "patient_summary_with_boilerplate": "Summary",
            }
        ]
    )

    try:
        validate_existing_summaries(existing)
    except ValueError as exc:
        assert "P1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid last_note_date")


def test_validate_existing_summaries_rejects_missing_summary_columns():
    """Existing summaries need either combined or split summary text."""
    existing = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "last_note_date": "2024-01-03",
            }
        ]
    )

    try:
        validate_existing_summaries(existing)
    except ValueError as exc:
        assert "patient_summary_with_boilerplate" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing summary text")


def test_parse_boilerplate_splits_summary_and_exclusions():
    """Split patient summaries into cancer history vs exclusion evidence."""
    df = pd.DataFrame(
        [
            {
                "original_patient_summary": (
                    "Cancer history here.\n" "Boilerplate conditions:\n" "No CNS mets."
                )
            },
            {"original_patient_summary": "Cancer only."},
        ]
    )

    parsed = parse_boilerplate(
        df,
        boilerplate_marker="Boilerplate conditions:",
    )

    assert parsed.loc[0, "cancer_history_summary"] == "Cancer history here."
    assert parsed.loc[0, "general_exclusion_criteria_evidence"] == "No CNS mets."
    assert parsed.loc[1, "general_exclusion_criteria_evidence"] == "Cancer only."


def test_parse_boilerplate_accepts_final_only_v22_output():
    """Parsed final content should not need a reasoning marker."""
    df = pd.DataFrame(
        [
            {
                "original_patient_summary": (
                    "Cancer history here.\n"
                    "\n"
                    "Boilerplate conditions:\n"
                    "Remote inactive prostate cancer."
                )
            }
        ]
    )

    parsed = parse_boilerplate(
        df,
        boilerplate_marker="Boilerplate conditions:",
    )

    assert parsed.loc[0, "cancer_history_summary"] == "Cancer history here."
    assert (
        parsed.loc[0, "general_exclusion_criteria_evidence"]
        == "Remote inactive prostate cancer."
    )


def test_parse_boilerplate_accepts_legacy_boilerplate_marker():
    """Keep compatibility with training's legacy marker aliases."""
    df = pd.DataFrame(
        [
            {
                "original_patient_summary": (
                    "Cancer history here.\n" "Boilerplate:\n" "Legacy marker text."
                )
            }
        ]
    )

    parsed = parse_boilerplate(
        df,
        boilerplate_marker="Boilerplate conditions:",
    )

    assert parsed.loc[0, "cancer_history_summary"] == "Cancer history here."
    assert (
        parsed.loc[0, "general_exclusion_criteria_evidence"]
        == "Legacy marker text."
    )


def test_clean_bad_data_filters_invalid_summaries():
    """Drop rows with empty or invalid patient summaries."""
    df = pd.DataFrame(
        [
            {"cancer_history_summary": "No evidence of malignancy"},
            {"cancer_history_summary": "No primary identified"},
            {"cancer_history_summary": "No information"},
            {"cancer_history_summary": "Valid summary"},
        ]
    )

    cleaned, qc_artifact = clean_bad_data(df)

    assert cleaned["cancer_history_summary"].tolist() == ["Valid summary"]
    assert qc_artifact["metric"] == "patients_dropped_noninformative_summary"


def test_local_backend_truncate_texts_splits_long_inputs(monkeypatch):
    """Truncate long patient text using the tokenizer without loading real models."""

    class MockTokenResult:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    class MockTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return MockTokenResult(list(text))

        def decode(self, input_ids):
            return "".join(input_ids)

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = MockTokenizer()
    monkeypatch.setitem(__import__("sys").modules, "transformers", mock_transformers)

    backend = LocalBackend()
    truncated = backend.truncate_texts(
        ["abcdefghij"],
        patient_config={
            "model_name": "mock",
            "text_token_threshold": 6,
        },
    )

    assert truncated == ["abc ... hij"]


def test_get_serial_patient_prompt_includes_prior_summary_and_chunk_text():
    """Build a serial prompt containing prior summary state and the next note chunk."""
    prompts = get_serial_patient_prompt(
        prior_summary="Age: 70",
        first_date="2024-01-01",
        last_date="2024-01-02",
        chunk_text="Clinical note text.",
        tokenizer=MockTokenizer(),
        max_model_len=100,
        primer_filename="patient.serial.user.primer.txt",
        question_filename="patient.serial.user.question.txt",
        margin_tokens=10,
        model_name="google/gemma-4-31B-it",
    )

    assert len(prompts) == 2
    assert prompts[0]["role"] == "system"
    assert prompts[1]["role"] == "user"
    assert "Age: 70" in prompts[1]["content"]
    assert "Clinical note text." in prompts[1]["content"]
    assert "Boilerplate conditions:" in prompts[1]["content"]
    assert "contradictory information across notes" in prompts[1]["content"]


def test_build_prompt_worker_leaves_response_token_margin(monkeypatch):
    """Leave a small generation margin for remote chat-template token drift."""

    class FixedPromptTokenizer(MockTokenizer):
        def apply_chat_template(
            self,
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        ):
            return "x" * 600

    monkeypatch.setattr(
        "matchminer_ai.patients.prompt_builder._worker_tokenizer",
        FixedPromptTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.prompt_builder._worker_config",
        {
            **_patient_config(),
            "model_name": "google/gemma-4-31B-it",
            "max_model_len": 1000,
            "sampling_params": {
                **_patient_config()["sampling_params"],
                "max_tokens": 900,
            },
        },
    )

    prompt = build_prompt_worker(
        PromptWorkItem(
            row_idx=0,
            prior_summary_text=None,
            first_date="2024-01-01",
            last_date="2024-01-02",
            chunk_text="Clinical note text.",
        )
    )

    assert prompt.max_tokens == 1000 - 600 - _RESPONSE_TOKEN_MARGIN


def test_summarize_patient_notes_updates_running_summary_across_rounds(monkeypatch):
    """Carry each round's summary forward as prior state for the next chunk."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: (
            pd.DataFrame([{"patient_id": "P1", "last_note_date": "2024-01-02"}]),
            pd.DataFrame(
                [
                    {
                        "patient_id": "P1",
                        "chunk_index": 0,
                        "first_date": "2024-01-01",
                        "last_date": "2024-01-01",
                        "chunk_text": "chunk one",
                    },
                    {
                        "patient_id": "P1",
                        "chunk_index": 1,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-02",
                        "chunk_text": "chunk two",
                    },
                ]
            ),
        ),
    )

    seen_prior_summaries = []

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
            seen_prior_summaries.extend(item.prior_summary_text for item in work_items)
            return [
                Prompt(row_idx=item.row_idx, prompt_text=item.chunk_text, max_tokens=7)
                for item in work_items
            ]

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prep_prompt_pool",
        lambda patient_config, n_workers: FakePromptPool(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.shutdown_prompt_pool",
        lambda prompt_pool: None,
    )

    class MockBackend:
        def __init__(self):
            self.calls = 0

        def generate_llm_outputs(
            self,
            *,
            prompt_list,
            llm_config,
            model_metadata_cache_dir=None,
        ):
            self.calls += 1
            if self.calls == 1:
                return LLMGenerationResult(
                    final_outputs=["Round 1\nBoilerplate conditions:\nNone"],
                    model_metadata={"model_name": "model", "model_sha": "sha"},
                    finish_reasons=["stop"],
                    reasoning_outputs=[""],
                    raw_outputs=[],
                )
            return LLMGenerationResult(
                final_outputs=["Round 2\nBoilerplate conditions:\nNone"],
                model_metadata={"model_name": "model", "model_sha": "sha"},
                finish_reasons=["stop"],
                reasoning_outputs=[""],
                raw_outputs=[],
            )

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        lambda config: MockBackend(),
    )

    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-01"}]
    )
    result, metadata = summarize_patient_notes(notes, config=_config())

    assert seen_prior_summaries == [None, "Round 1\nBoilerplate conditions:\nNone"]
    assert result.loc[result.index[0], "cancer_history_summary"] == "Round 2"
    assert metadata["model_metadata"]["model_sha"] == "sha"


def test_summarize_patient_notes_uses_existing_summary_in_first_round(monkeypatch):
    """Use a provided existing summary as the starting state for round 1."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: (
            pd.DataFrame([{"patient_id": "P1", "last_note_date": "2024-01-02"}]),
            pd.DataFrame(
                [
                    {
                        "patient_id": "P1",
                        "chunk_index": 0,
                        "first_date": "2024-01-02",
                        "last_date": "2024-01-02",
                        "chunk_text": "new chunk",
                    }
                ]
            ),
        ),
    )

    seen_prior_summaries = []

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
            seen_prior_summaries.extend(item.prior_summary_text for item in work_items)
            return [
                Prompt(row_idx=item.row_idx, prompt_text=item.chunk_text, max_tokens=7)
                for item in work_items
            ]

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prep_prompt_pool",
        lambda patient_config, n_workers: FakePromptPool(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.shutdown_prompt_pool",
        lambda prompt_pool: None,
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        lambda config: MagicMock(
            generate_llm_outputs=MagicMock(
                return_value=LLMGenerationResult(
                    final_outputs=["Updated\nBoilerplate conditions:\nNone"],
                    model_metadata={"model_name": "model", "model_sha": "sha"},
                    finish_reasons=["stop"],
                    reasoning_outputs=[""],
                    raw_outputs=[],
                )
            )
        ),
    )

    existing_summaries = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "last_note_date": "2024-01-01",
                "patient_summary": "Existing summary",
            }
        ]
    )
    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-02"}]
    )

    result, _ = summarize_patient_notes(
        notes,
        config=_config(),
        existing_summaries=existing_summaries,
    )

    assert seen_prior_summaries == ["Existing summary"]
    assert result.loc[result.index[0], "cancer_history_summary"] == "Updated"


def test_summarize_patient_notes_filters_existing_notes_by_last_note_date(monkeypatch):
    """Only notes newer than the prior summary date should reach chunking."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    captured_notes = {}

    def fake_prepare_patient_notes(notes, tokenizer, chunk_size, chunk_overlap):
        captured_notes["notes"] = notes.copy()
        return (
            pd.DataFrame([{"patient_id": "P1", "last_note_date": "2024-01-03"}]),
            pd.DataFrame(
                [
                    {
                        "patient_id": "P1",
                        "chunk_index": 0,
                        "first_date": "2024-01-03",
                        "last_date": "2024-01-03",
                        "chunk_text": "new chunk",
                    }
                ]
            ),
        )

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        fake_prepare_patient_notes,
    )

    seen_prior_summaries = []

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
            seen_prior_summaries.extend(item.prior_summary_text for item in work_items)
            return [
                Prompt(row_idx=item.row_idx, prompt_text=item.chunk_text, max_tokens=7)
                for item in work_items
            ]

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prep_prompt_pool",
        lambda patient_config, n_workers: FakePromptPool(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.shutdown_prompt_pool",
        lambda prompt_pool: None,
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        lambda config: MagicMock(
            generate_llm_outputs=MagicMock(
                return_value=LLMGenerationResult(
                    final_outputs=["Updated\nBoilerplate conditions:\nNone"],
                    model_metadata={"model_name": "model", "model_sha": "sha"},
                    finish_reasons=["stop"],
                    reasoning_outputs=[""],
                    raw_outputs=[],
                )
            )
        ),
    )

    existing_summaries = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "last_note_date": "2024-01-02",
                "patient_summary_with_boilerplate": "Existing summary",
            }
        ]
    )
    notes = pd.DataFrame(
        [
            {"patient_id": "P1", "note_text": "old", "note_date": "2024-01-01"},
            {"patient_id": "P1", "note_text": "same", "note_date": "2024-01-02"},
            {"patient_id": "P1", "note_text": "new", "note_date": "2024-01-03"},
        ]
    )

    result, _ = summarize_patient_notes(
        notes,
        config=_config(),
        existing_summaries=existing_summaries,
    )

    assert captured_notes["notes"]["note_text"].tolist() == ["new"]
    assert seen_prior_summaries == ["Existing summary"]
    assert result.loc[result.index[0], "last_note_date"] == "2024-01-03"
    assert result.loc[result.index[0], "cancer_history_summary"] == "Updated"


def test_summarize_patient_notes_passes_through_when_no_newer_notes(monkeypatch):
    """Existing rows with no newer notes should return without model setup."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        MagicMock(side_effect=AssertionError("tokenizer should not load")),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        MagicMock(side_effect=AssertionError("backend should not load")),
    )

    existing_summaries = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "last_note_date": "2024-01-02",
                "patient_summary_with_boilerplate": (
                    "No evidence of malignancy\n"
                    "Boilerplate conditions:\n"
                    "None"
                ),
            }
        ]
    )
    notes = pd.DataFrame(
        [
            {"patient_id": "P1", "note_text": "old", "note_date": "2024-01-01"},
            {"patient_id": "P1", "note_text": "same", "note_date": "2024-01-02"},
        ]
    )

    result, metadata = summarize_patient_notes(
        notes,
        config=_config(),
        existing_summaries=existing_summaries,
    )

    assert result["patient_id"].tolist() == ["P1"]
    assert result.loc[result.index[0], "last_note_date"] == "2024-01-02"
    assert (
        result.loc[result.index[0], "cancer_history_summary"]
        == "No evidence of malignancy"
    )
    assert result.loc[result.index[0], "general_exclusion_criteria_evidence"] == "None"
    assert metadata["model_metadata"] == {}


def test_summarize_patient_notes_updates_and_passes_through_mixed_cohort(
    monkeypatch,
):
    """Mix updated existing, unchanged existing, and new patients in one run."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    captured_notes = {}

    def fake_prepare_patient_notes(notes, tokenizer, chunk_size, chunk_overlap):
        captured_notes["patient_ids"] = notes["patient_id"].tolist()
        return (
            pd.DataFrame(
                [
                    {"patient_id": "P1", "last_note_date": "2024-01-03"},
                    {"patient_id": "P3", "last_note_date": "2024-01-01"},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "patient_id": "P1",
                        "chunk_index": 0,
                        "first_date": "2024-01-03",
                        "last_date": "2024-01-03",
                        "chunk_text": "p1 chunk",
                    },
                    {
                        "patient_id": "P3",
                        "chunk_index": 0,
                        "first_date": "2024-01-01",
                        "last_date": "2024-01-01",
                        "chunk_text": "p3 chunk",
                    },
                ]
            ),
        )

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        fake_prepare_patient_notes,
    )
    pool_calls = {}

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
            pool_calls["prior_summaries"] = [
                item.prior_summary_text for item in work_items
            ]
            return [
                Prompt(row_idx=item.row_idx, prompt_text=item.chunk_text, max_tokens=7)
                for item in work_items
            ]

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prep_prompt_pool",
        lambda patient_config, n_workers: FakePromptPool(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.shutdown_prompt_pool",
        lambda prompt_pool: None,
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        lambda config: MagicMock(
            generate_llm_outputs=MagicMock(
                return_value=LLMGenerationResult(
                    final_outputs=[
                        "Updated P1\nBoilerplate conditions:\nNone",
                        "New P3\nBoilerplate conditions:\nNone",
                    ],
                    model_metadata={"model_name": "model", "model_sha": "sha"},
                    finish_reasons=["stop", "stop"],
                    reasoning_outputs=["", ""],
                    raw_outputs=[],
                )
            )
        ),
    )

    existing_summaries = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "last_note_date": "2024-01-02",
                "cancer_history_summary": "Old P1",
                "general_exclusion_criteria_evidence": "Old boilerplate P1",
            },
            {
                "patient_id": "P2",
                "last_note_date": "2024-01-05",
                "cancer_history_summary": "Old P2",
                "general_exclusion_criteria_evidence": "Old boilerplate P2",
            },
        ]
    )
    notes = pd.DataFrame(
        [
            {"patient_id": "P1", "note_text": "new p1", "note_date": "2024-01-03"},
            {"patient_id": "P2", "note_text": "old p2", "note_date": "2024-01-04"},
            {"patient_id": "P3", "note_text": "new p3", "note_date": "2024-01-01"},
        ]
    )

    result, metadata = summarize_patient_notes(
        notes,
        config=_config(),
        existing_summaries=existing_summaries,
    )

    summaries_by_patient = dict(
        zip(
            result["patient_id"],
            result["cancer_history_summary"],
            strict=False,
        )
    )
    assert captured_notes["patient_ids"] == ["P1", "P3"]
    assert pool_calls["prior_summaries"] == [
        "Old P1\n\nBoilerplate conditions:\nOld boilerplate P1",
        None,
    ]
    assert summaries_by_patient == {
        "P1": "Updated P1",
        "P2": "Old P2",
        "P3": "New P3",
    }
    assert metadata["model_metadata"]["model_sha"] == "sha"


def test_remote_summarize_patient_notes_uses_parallel_prompt_workers(monkeypatch):
    """Remote patient summarization builds pre-rendered prompts via prompt pool."""
    _stub_patient_qc(monkeypatch)
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.AutoTokenizer.from_pretrained",
        lambda model_name, **kwargs: MockTokenizer(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prepare_patient_notes",
        lambda notes, tokenizer, chunk_size, chunk_overlap: (
            pd.DataFrame(
                [
                    {"patient_id": "P1", "last_note_date": "2024-01-01"},
                    {"patient_id": "P2", "last_note_date": "2024-01-01"},
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "patient_id": "P1",
                        "chunk_index": 0,
                        "first_date": "2024-01-01",
                        "last_date": "2024-01-01",
                        "chunk_text": "chunk one",
                    },
                    {
                        "patient_id": "P2",
                        "chunk_index": 0,
                        "first_date": "2024-01-01",
                        "last_date": "2024-01-01",
                        "chunk_text": "chunk two",
                    },
                ]
            ),
        ),
    )

    pool_calls = {}

    class FakePromptPool:
        def map(self, func, work_items, chunksize=1):
            pool_calls["chunksize"] = chunksize
            pool_calls["work_items"] = work_items
            return [
                Prompt(row_idx=item.row_idx, prompt_text=item.chunk_text, max_tokens=7)
                for item in work_items
            ]

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.prep_prompt_pool",
        lambda patient_config, n_workers: FakePromptPool(),
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.shutdown_prompt_pool",
        lambda prompt_pool: pool_calls.setdefault("shutdown", True),
    )

    captured = {}

    class MockBackend:
        def generate_llm_outputs(
            self,
            *,
            prompt_list,
            llm_config,
            model_metadata_cache_dir=None,
        ):
            captured["prompt_list"] = prompt_list
            return LLMGenerationResult(
                final_outputs=[
                    "Remote 1\nBoilerplate conditions:\nNone",
                    "Remote 2\nBoilerplate conditions:\nNone",
                ],
                model_metadata={"model_name": "model", "model_sha": "sha"},
                finish_reasons=["stop", "stop"],
                reasoning_outputs=["", ""],
                raw_outputs=[],
            )

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize.get_summarization_backend",
        lambda config: MockBackend(),
    )

    notes = pd.DataFrame(
        [{"patient_id": "P1", "note_text": "x", "note_date": "2024-01-01"}]
    )
    result, metadata = summarize_patient_notes(notes, config=_remote_config())

    assert [prompt.prompt_text for prompt in captured["prompt_list"]] == [
        "chunk one",
        "chunk two",
    ]
    assert [item.prior_summary_text for item in pool_calls["work_items"]] == [
        None,
        None,
    ]
    assert pool_calls["shutdown"] is True
    assert result["cancer_history_summary"].tolist() == ["Remote 1", "Remote 2"]
    assert metadata["model_metadata"]["model_sha"] == "sha"


def test_generate_remote_llm_outputs_handles_running_event_loop(monkeypatch):
    """Allow the sync remote wrapper to run from notebooks and async shells."""

    async def fake_generate_remote_llm_outputs_async(
        *,
        prompts,
        llm_config,
        server_urls,
        api_key,
    ):
        return ["Summary"], ["thinking"], ["stop"]

    monkeypatch.setattr(
        "matchminer_ai.llm.remote_inference.generate_remote_llm_outputs_async",
        fake_generate_remote_llm_outputs_async,
    )

    async def invoke_wrapper():
        return generate_remote_llm_outputs(
            prompts=[Prompt(row_idx=0, prompt_text="chunk", max_tokens=7)],
            llm_config={"model_name": "model"},
            server_urls=["http://server-a/v1"],
            api_key="not-needed",
        )

    texts, reasonings, finish_reasons = asyncio.run(invoke_wrapper())

    assert texts == ["Summary"]
    assert reasonings == ["thinking"]
    assert finish_reasons == ["stop"]


def test_summarize_patients_returns_metadata_and_qc(monkeypatch):
    """Return metadata and QC from the serial patient summarization entrypoint."""
    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Note text",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            }
        ]
    )
    summaries_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cancer_history_summary": "Summary",
                "general_exclusion_criteria_evidence": "None",
            }
        ]
    )
    qc_report = pd.DataFrame(
        [
            {
                "metric": "patients_dropped_noninformative_summary",
                "value": 0,
                "percent": 0.0,
                "ids": [],
            }
        ]
    )

    monkeypatch.setattr(
        "matchminer_ai.patients.summarize_patient_notes",
        MagicMock(
            return_value=(
                summaries_df,
                {"model_metadata": {"model_name": "summ", "model_sha": "sha"}},
                qc_report,
            )
        ),
    )

    result, metadata, returned_qc = summarize_patients(
        notes,
        config=_config(),
        return_metadata=True,
        return_qc=True,
    )

    assert result.equals(summaries_df)
    assert returned_qc.equals(qc_report)
    assert metadata["config_snapshot"]["config"] == "snapshot"
    assert metadata["config_snapshot"]["patient"] == _config().patient
    assert metadata["model_metadata"]["patient_summarizer"]["model_sha"] == "sha"


def test_summarize_patients_does_not_request_qc_by_default(monkeypatch):
    """Default patient summarization should skip QC-only embedding token counts."""
    notes = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "note_text": "Note text",
                "note_type": "clinical_note",
                "note_date": "2024-01-01",
            }
        ]
    )
    summaries_df = pd.DataFrame(
        [
            {
                "patient_id": "P1",
                "cancer_history_summary": "Summary",
                "general_exclusion_criteria_evidence": "None",
            }
        ]
    )
    summarize_mock = MagicMock(
        return_value=(
            summaries_df,
            {"model_metadata": {"model_name": "summ", "model_sha": "sha"}},
        )
    )
    monkeypatch.setattr(
        "matchminer_ai.patients.summarize_patient_notes",
        summarize_mock,
    )

    result = summarize_patients(notes, config=_config())

    assert result.equals(summaries_df)
    assert summarize_mock.call_args.kwargs["return_qc"] is False
