from __future__ import annotations

import json
import re
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from matchminer_ai.config import load_default_preset
from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.patients import answer_question_with_raw_patient_notes
from matchminer_ai.patients.raw_note_qa import (
    _NoteChunk,
    _PreparedRawNoteIndex,
    _QueryEmbeddingServer,
    _answer_question_with_prepared_raw_note_index,
    _request_query_embeddings,
)


class _WhitespaceTokenizer:
    model_max_length = 128

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del add_special_tokens
        self.calls.append(str(text))
        matches = list(re.finditer(r"\S+", str(text)))
        result = {"input_ids": list(range(1, len(matches) + 1))}
        if return_offsets_mapping:
            result["offset_mapping"] = [match.span() for match in matches]
        return result

    def num_special_tokens_to_add(self, *, pair=False):
        del pair
        return 2

    def decode(self, token_ids, **_kwargs):
        return " ".join(f"token-{token_id}" for token_id in token_ids)


class _FakeEmbeddingModel:
    max_seq_length = 128

    def __init__(self) -> None:
        self.tokenizer = _WhitespaceTokenizer()
        self.encoded_batches: list[list[str]] = []

    def encode(self, texts, **_kwargs):
        self.encoded_batches.append(list(texts))
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [
                    float("rash" in normalized or "toxicity" in normalized),
                    float("response" in normalized or "shrank" in normalized),
                    0.25,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class _FakeBackend:
    def __init__(
        self,
        responses: list[dict],
        *,
        finish_reasons: list[str] | None = None,
    ) -> None:
        self.responses = [json.dumps(response) for response in responses]
        self.finish_reasons = list(
            finish_reasons or ["stop"] * len(self.responses)
        )
        self.prompts = []

    def generate_llm_outputs(self, *, prompt_list, **_kwargs):
        if not self.responses:
            raise AssertionError("Unexpected extra LLM request")
        self.prompts.append(prompt_list[0])
        return LLMGenerationResult(
            final_outputs=[self.responses.pop(0)],
            model_metadata={"model_name": "synthetic-qa-model"},
            finish_reasons=[self.finish_reasons.pop(0)],
            reasoning_outputs=[""],
            raw_outputs=[],
        )


def _qa_config():
    config = load_default_preset()
    config.remote["enabled"] = True
    config.raw_patient_note_qa.update(
        {
            "embedding_model_name": "synthetic-embedding-model",
            "embedding_device": "cpu",
            "chunk_size": 100,
            "chunk_overlap": 5,
            "initial_top_k": 3,
            "tool_top_k": 3,
            "max_agent_steps": 5,
            "response_retry_limit": 1,
        }
    )
    return config


def test_dataframe_notes_are_sorted_and_concatenated_before_embedding() -> None:
    notes = pd.DataFrame(
        {
            "patient_id": ["synthetic-1", "synthetic-1"],
            "note_date": ["2026-03-02", "2026-01-05"],
            "note_text": ["Later finding.", "Earlier finding."],
        }
    )
    backend = _FakeBackend(
        [
            {
                "action": "final_answer",
                "answer": "The earlier finding was documented first.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "Earlier finding.",
                        "reason": "It appears under the earlier dated header.",
                    }
                ],
                "limitations": [],
            }
        ]
    )
    embedding_model = _FakeEmbeddingModel()
    progress = []

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa._get_embedding_model",
            return_value=embedding_model,
        ) as get_model,
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
    ):
        result, metadata = answer_question_with_raw_patient_notes(
            "Which finding came first?",
            notes,
            embedding_model_name="override/model",
            config=_qa_config(),
            return_metadata=True,
            progress_callback=lambda *args: progress.append(args),
        )

    embedded_notes = embedding_model.encoded_batches[0]
    assert len(embedded_notes) == 2
    document = "\n".join(embedded_notes)
    assert document.index("2026-01-05") < document.index("2026-03-02")
    assert document.index("Earlier finding.") < document.index("Later finding.")
    get_model.assert_called_once_with("override/model", "cpu")
    assert result["evidence"][0]["quote"] == "Earlier finding."
    assert result["evidence"][0]["note_date"] == "2026-01-05"
    initial_prompt = backend.prompts[0]
    assert initial_prompt.messages is not None
    initial_payload = json.loads(initial_prompt.messages[-1]["content"])
    retrieved = initial_payload["initial_pull_relevant_input_text_result"]
    assert [chunk["note_date"] for chunk in retrieved] == [
        "2026-01-05",
        "2026-03-02",
    ]
    assert metadata["retrieval"]["input_type"] == "dataframe"
    assert metadata["retrieval"]["source_note_count"] == 2
    assert metadata["retrieval"]["chunk_count"] == 2
    assert metadata["config_snapshot"]["raw_patient_note_qa"]
    assert progress[-1][0] == "complete"
    json.dumps(result)


def test_agent_can_retrieve_then_ask_related_question_before_answering() -> None:
    raw_notes = (
        "2026-01-01: The tumor shrank after therapy. "
        "2026-02-01: A grade 2 rash was documented."
    )
    backend = _FakeBackend(
        [
            {
                "action": "pull_relevant_input_text",
                "query": "treatment toxicity rash",
            },
            {
                "action": "ask_related_question",
                "question": "Was a rash documented after therapy?",
            },
            {
                "answer": "A grade 2 rash was documented after therapy.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "A grade 2 rash was documented.",
                        "reason": "This directly documents the toxicity.",
                    }
                ],
                "limitations": [],
            },
            {
                "action": "final_answer",
                "answer": "The notes document shrinkage followed by a grade 2 rash.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "The tumor shrank after therapy.",
                        "reason": "This documents response.",
                    },
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "A grade 2 rash was documented.",
                        "reason": "This documents later toxicity.",
                    },
                ],
                "limitations": [],
            },
        ]
    )

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa._get_embedding_model",
            return_value=_FakeEmbeddingModel(),
        ),
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
    ):
        result = answer_question_with_raw_patient_notes(
            "What response and toxicity were documented?",
            raw_notes,
            config=_qa_config(),
        )

    assert "shrinkage" in result["answer"]
    assert [item["quote"] for item in result["evidence"]] == [
        "The tumor shrank after therapy.",
        "A grade 2 rash was documented.",
    ]
    assert [item["note_date"] for item in result["evidence"]] == [None, None]
    assert len(backend.prompts) == 4
    related_prompt = backend.prompts[2]
    assert related_prompt.messages is not None
    assert (
        "Was a rash documented after therapy?" in related_prompt.messages[-1]["content"]
    )
    assert backend.responses == []


def test_prepared_cpu_index_is_reused_without_reembedding_note_chunks() -> None:
    prepared_index = _PreparedRawNoteIndex(
        chunks=(
            _NoteChunk(
                chunk_id="chunk_0000",
                text="A grade 2 rash was documented.",
                token_start=0,
                token_end=7,
            ),
        ),
        embeddings=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        input_type="string",
        source_note_count=0,
        embedding_model_name="synthetic-embedding-model",
        embedding_device="cuda",
        embedding_batch_size=8,
        query_prefix="query: ",
        min_similarity=-1.0,
        configured_chunk_size=100,
        effective_chunk_size=100,
        chunk_overlap=5,
    )
    query_batches: list[list[str]] = []

    def encode_queries(texts: list[str]) -> np.ndarray:
        query_batches.append(list(texts))
        return np.asarray([[1.0, 0.0, 0.0] for _text in texts], dtype=np.float32)

    backend = _FakeBackend(
        [
            {
                "action": "final_answer",
                "answer": "A grade 2 rash was documented.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "A grade 2 rash was documented.",
                        "reason": "Exact note text.",
                    }
                ],
                "limitations": [],
            }
        ]
    )

    with patch(
        "matchminer_ai.patients.raw_note_qa.get_llm_backend",
        return_value=backend,
    ):
        result, metadata = _answer_question_with_prepared_raw_note_index(
            "Was a rash documented?",
            prepared_index=prepared_index,
            query_encoder=encode_queries,
            config=_qa_config(),
            return_metadata=True,
        )

    assert result["answer"] == "A grade 2 rash was documented."
    assert query_batches == [["query: Was a rash documented?"]]
    assert metadata["retrieval"]["embedding_device"] == "cuda"


def test_query_embedding_server_keeps_the_model_in_its_owner_process() -> None:
    embedding_model = _FakeEmbeddingModel()

    with _QueryEmbeddingServer(embedding_model, batch_size=4) as endpoint:
        vectors = _request_query_embeddings(endpoint, ["rash", "response"])

    assert vectors.shape == (2, 3)
    assert embedding_model.encoded_batches == [["rash", "response"]]


def test_embedding_tokenizer_defines_overlapping_chunk_boundaries() -> None:
    embedding_model = _FakeEmbeddingModel()
    backend = _FakeBackend(
        [
            {
                "action": "final_answer",
                "answer": "The requested words are documented.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0001",
                        "quote": "five six",
                        "reason": "Exact words from the retrieved chunk.",
                    }
                ],
                "limitations": [],
            }
        ]
    )
    config = _qa_config()
    config.raw_patient_note_qa.update(
        {
            "chunk_size": 5,
            "chunk_overlap": 1,
            "initial_top_k": 3,
        }
    )

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa._get_embedding_model",
            return_value=embedding_model,
        ),
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
    ):
        result, metadata = answer_question_with_raw_patient_notes(
            "Where are five and six?",
            "zero one two three four five six seven eight nine",
            config=config,
            return_metadata=True,
        )

    embedded_chunks = embedding_model.encoded_batches[0]
    assert embedded_chunks == [
        "zero one two three four",
        "four five six seven eight",
        "eight nine",
    ]
    assert metadata["retrieval"]["chunk_count"] == 3
    assert metadata["retrieval"]["effective_chunk_size"] == 5
    assert result["evidence"][0]["chunk_id"] == "chunk_0001"
    assert result["evidence"][0]["note_date"] is None


def test_invented_evidence_quote_is_rejected_and_agent_can_correct_it() -> None:
    backend = _FakeBackend(
        [
            {
                "action": "final_answer",
                "answer": "A rash was documented.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "A severe rash was documented.",
                        "reason": "Purported evidence.",
                    }
                ],
                "limitations": [],
            },
            {
                "action": "final_answer",
                "answer": "A grade 2 rash was documented.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "A grade 2 rash was documented.",
                        "reason": "Exact note text.",
                    }
                ],
                "limitations": [],
            },
        ]
    )

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa._get_embedding_model",
            return_value=_FakeEmbeddingModel(),
        ),
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
    ):
        result = answer_question_with_raw_patient_notes(
            "Was a rash documented?",
            "A grade 2 rash was documented.",
            config=_qa_config(),
        )

    assert result["evidence"][0]["quote"] == "A grade 2 rash was documented."
    correction_prompt = backend.prompts[1]
    assert correction_prompt.messages is not None
    assert "FINAL ANSWER VALIDATION ERROR" in correction_prompt.messages[-1]["content"]


def test_tool_exhaustion_restarts_with_compact_final_only_prompt() -> None:
    backend = _FakeBackend(
        [
            {
                "action": "pull_relevant_input_text",
                "query": "rash severity",
            },
            {
                "action": "final_answer",
                "answer": "A severe rash was documented.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "A severe rash was documented.",
                        "reason": "Purported evidence.",
                    }
                ],
                "limitations": [],
            },
            {
                "action": "final_answer",
                "answer": "A grade 2 rash was documented.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "A grade 2 rash was documented.",
                        "reason": "Exact note text.",
                    }
                ],
                "limitations": [],
            },
        ]
    )
    config = _qa_config()
    config.raw_patient_note_qa.update(
        {"max_agent_steps": 1, "response_retry_limit": 1}
    )

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa._get_embedding_model",
            return_value=_FakeEmbeddingModel(),
        ),
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
    ):
        result = answer_question_with_raw_patient_notes(
            "Was a rash documented?",
            "A grade 2 rash was documented.",
            config=config,
        )

    assert result["evidence"][0]["quote"] == "A grade 2 rash was documented."
    assert len(backend.prompts) == 3
    compact_prompt = backend.prompts[1]
    assert compact_prompt.messages is not None
    assert len(compact_prompt.messages) == 2
    assert "Do not call a tool" in compact_prompt.messages[0]["content"]
    compact_payload = json.loads(compact_prompt.messages[1]["content"])
    assert compact_payload["original_question"] == "Was a rash documented?"
    assert compact_payload["retrieved_chunks"] == [
        {
            "chunk_id": "chunk_0000",
            "note_date": None,
            "text": "A grade 2 rash was documented.",
        }
    ]
    assert "LOCAL TOOL RESULT" not in compact_prompt.prompt_text
    correction_prompt = backend.prompts[2]
    assert correction_prompt.messages is not None
    assert "FINAL ANSWER VALIDATION ERROR" in correction_prompt.messages[-1]["content"]


def test_terminal_error_reports_safe_validation_and_finish_reason() -> None:
    backend = _FakeBackend(
        [
            {
                "action": "pull_relevant_input_text",
                "query": "rash severity",
            },
            {
                "action": "final_answer",
                "answer": "A severe rash was documented.",
                "evidence": [
                    {
                        "chunk_id": "chunk_0000",
                        "quote": "A severe rash was documented.",
                        "reason": "Purported evidence.",
                    }
                ],
                "limitations": [],
            },
        ]
    )
    config = _qa_config()
    config.raw_patient_note_qa.update(
        {"max_agent_steps": 1, "response_retry_limit": 0}
    )

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa._get_embedding_model",
            return_value=_FakeEmbeddingModel(),
        ),
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
        pytest.raises(
            ValueError,
            match=(
                "evidence quote is not an exact substring of chunk_0000.*"
                "finish_reason: stop"
            ),
        ),
    ):
        answer_question_with_raw_patient_notes(
            "Was a rash documented?",
            "A grade 2 rash was documented.",
            config=config,
        )


def test_token_limited_json_is_retried_instead_of_accepted() -> None:
    backend = _FakeBackend(
        [
            {
                "action": "final_answer",
                "answer": "A grade 2 rash was documented.",
                "evidence": [],
                "limitations": [],
            }
        ],
        finish_reasons=["length"],
    )
    config = _qa_config()
    config.raw_patient_note_qa["response_retry_limit"] = 0

    with (
        patch(
            "matchminer_ai.patients.raw_note_qa._get_embedding_model",
            return_value=_FakeEmbeddingModel(),
        ),
        patch(
            "matchminer_ai.patients.raw_note_qa.get_llm_backend",
            return_value=backend,
        ),
        pytest.raises(
            ValueError,
            match="output token limit.*finish_reason: length",
        ),
    ):
        answer_question_with_raw_patient_notes(
            "Was a rash documented?",
            "A grade 2 rash was documented.",
            config=config,
        )


@pytest.mark.parametrize(
    ("notes", "match"),
    [
        (
            pd.DataFrame({"note_text": ["text"]}),
            "missing required columns: note_date",
        ),
        (
            pd.DataFrame(
                {
                    "patient_id": ["one", "two"],
                    "note_text": ["a", "b"],
                    "note_date": ["2026-01-01", "2026-01-02"],
                }
            ),
            "exactly one patient",
        ),
        (
            pd.DataFrame(
                {
                    "note_text": ["text"],
                    "note_date": ["not-a-date"],
                }
            ),
            "parseable as datetimes",
        ),
    ],
)
def test_rejects_invalid_dataframe_inputs(notes, match) -> None:
    with pytest.raises(ValueError, match=match):
        answer_question_with_raw_patient_notes(
            "A question?",
            notes,
            config=_qa_config(),
        )


def test_rejects_empty_question_before_loading_models() -> None:
    with pytest.raises(ValueError, match="question must be a non-empty"):
        answer_question_with_raw_patient_notes(" ", "Synthetic note")
