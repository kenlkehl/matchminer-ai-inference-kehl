"""Full trial-eligibility screening over one patient's raw notes."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from copy import deepcopy
from importlib import resources
from multiprocessing import get_context
from typing import Any

import pandas as pd

from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config

from .raw_note_qa import (
    _JsonLLMRunner,
    _prepare_raw_text,
    answer_question_with_raw_patient_notes,
)

FullPatientScreenProgress = Callable[[str, int, int, str], None]

_CRITERION_TYPES = {"inclusion", "exclusion", "other"}
_ELIGIBILITY_SIGNALS = {
    "supports_eligibility",
    "raises_eligibility_concern",
    "insufficient_information",
    "not_assessable_from_notes",
}
_OVERALL_SIGNALS = {
    "no_concern_identified",
    "potential_concern_identified",
    "mixed",
    "insufficient_information",
}
_RESEARCH_USE_NOTICE = (
    "Research decision support only. This output does not establish clinical-trial "
    "eligibility, diagnose disease, or recommend treatment. Verify every criterion "
    "against the complete current protocol with the treating and trial teams."
)


class FullPatientScreenError(ValueError):
    """Raised when a complete raw-note eligibility screen cannot finish safely."""


def _load_prompt_text(filename: str) -> str:
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def _emit_progress(
    callback: FullPatientScreenProgress | None,
    stage: str,
    completed: int,
    total: int,
    detail: str,
) -> None:
    if callback is not None:
        callback(stage, completed, total, detail)


def _task_runtime_config(
    screen_config: dict[str, Any],
    *,
    config: MMAIConfig,
) -> dict[str, Any]:
    llm_only_config = {
        key: deepcopy(screen_config[key])
        for key in ("reasoning_parser", "local", "remote")
        if key in screen_config
    }
    return build_llm_runtime_config(
        "full_patient_screen",
        llm_only_config,
        config=config,
    )


def _normalize_for_source_check(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _build_criterion_sources(
    eligibility_criteria: str,
) -> list[dict[str, str]]:
    """Assign stable IDs to exact non-empty lines of the supplied criteria."""

    return [
        {
            "source_id": f"source_{index:04d}",
            "criterion_text": line.strip(),
        }
        for index, line in enumerate(
            (line for line in eligibility_criteria.splitlines() if line.strip()),
            start=1,
        )
    ]


def _prepare_eligibility_criteria(
    eligibility_criteria: str | Mapping[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    if isinstance(eligibility_criteria, str):
        criteria_text = eligibility_criteria.strip()
        if not criteria_text:
            raise ValueError("eligibility_criteria must be non-empty.")
        return criteria_text, _build_criterion_sources(criteria_text)

    if not isinstance(eligibility_criteria, Mapping):
        raise TypeError(
            "eligibility_criteria must be a string or a mapping containing "
            "inclusion_criteria and exclusion_criteria lists."
        )

    criterion_sources: list[dict[str, str]] = []
    criteria_by_type: dict[str, list[str]] = {}
    seen: set[str] = set()
    for criterion_type, field_name in (
        ("inclusion", "inclusion_criteria"),
        ("exclusion", "exclusion_criteria"),
    ):
        values = eligibility_criteria.get(field_name)
        if not isinstance(values, list):
            raise TypeError(f"eligibility_criteria[{field_name!r}] must be a list.")
        criteria_by_type[criterion_type] = []
        for position, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"eligibility_criteria[{field_name!r}][{position}] must be "
                    "a non-empty string."
                )
            criterion_text = value.strip()
            normalized = _normalize_for_source_check(criterion_text)
            if normalized in seen:
                raise ValueError(
                    "eligibility_criteria contains a duplicated inclusion or "
                    "exclusion criterion."
                )
            seen.add(normalized)
            criteria_by_type[criterion_type].append(criterion_text)
            criterion_sources.append(
                {
                    "source_id": f"source_{len(criterion_sources) + 1:04d}",
                    "criterion_type": criterion_type,
                    "criterion_text": criterion_text,
                }
            )

    if not criterion_sources:
        raise ValueError("eligibility_criteria must contain at least one criterion.")
    criteria_text = "\n".join(
        [
            "INCLUSION CRITERIA",
            *criteria_by_type["inclusion"],
            "EXCLUSION CRITERIA",
            *criteria_by_type["exclusion"],
        ]
    )
    return criteria_text, criterion_sources


def _normalize_limitations(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(text for item in value if (text := str(item).strip())))


def _validate_question_payload(
    payload: dict[str, Any],
    *,
    eligibility_criteria: str,
    criterion_sources: list[dict[str, str]] | None = None,
    max_questions: int,
) -> list[dict[str, str]]:
    if payload.get("coverage_complete") is not True:
        raise FullPatientScreenError(
            "criterion decomposition did not confirm complete criteria coverage."
        )
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise FullPatientScreenError("questions must be a non-empty JSON array.")
    if len(raw_questions) > max_questions:
        raise FullPatientScreenError(
            f"criterion decomposition produced {len(raw_questions)} questions, "
            f"exceeding full_patient_screen.max_questions={max_questions}."
        )

    normalized_source = _normalize_for_source_check(eligibility_criteria)
    source_lookup = {
        item["source_id"]: item
        for item in (criterion_sources or [])
    }
    questions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for index, raw_question in enumerate(raw_questions, start=1):
        if not isinstance(raw_question, dict):
            raise FullPatientScreenError(
                f"question {index} must be a JSON object."
            )
        criterion_id = str(raw_question.get("criterion_id") or "").strip()
        criterion_type = str(
            raw_question.get("criterion_type") or ""
        ).strip().casefold()
        source_id = str(raw_question.get("source_id") or "").strip()
        model_criterion_text = str(
            raw_question.get("criterion_text") or ""
        ).strip()
        source = source_lookup.get(source_id)
        criterion_text = (
            str(source["criterion_text"])
            if source is not None
            else model_criterion_text
        )
        question = str(raw_question.get("question") or "").strip()
        if not criterion_id or not question:
            raise FullPatientScreenError(
                f"question {index} is missing criterion_id or question."
            )
        if source_lookup and source is None:
            raise FullPatientScreenError(
                f"question {criterion_id!r} does not reference a supplied source_id."
            )
        if not criterion_text:
            raise FullPatientScreenError(
                f"question {criterion_id!r} is missing criterion_text."
            )
        if criterion_type not in _CRITERION_TYPES:
            raise FullPatientScreenError(
                f"question {criterion_id!r} has unsupported criterion_type "
                f"{criterion_type!r}."
            )
        if source is not None and source.get("criterion_type") in _CRITERION_TYPES:
            criterion_type = str(source["criterion_type"])
        if criterion_id in seen_ids:
            raise FullPatientScreenError(
                f"criterion_id {criterion_id!r} appears more than once."
            )
        normalized_question = _normalize_for_source_check(question)
        if normalized_question in seen_questions:
            raise FullPatientScreenError(
                "the same patient-note question was emitted more than once: "
                f"{question!r}."
            )
        if _normalize_for_source_check(criterion_text) not in normalized_source:
            raise FullPatientScreenError(
                f"criterion_text for {criterion_id!r} was not copied from the supplied "
                "eligibility criteria."
            )
        seen_ids.add(criterion_id)
        seen_questions.add(normalized_question)
        questions.append(
            {
                "criterion_id": criterion_id,
                "criterion_type": criterion_type,
                "criterion_text": criterion_text,
                "question": question,
                **({"source_id": source_id} if source_id else {}),
            }
        )
    return questions


def _validate_final_payload(
    payload: dict[str, Any],
    *,
    criterion_ids: list[str],
) -> dict[str, Any]:
    overall_signal = str(payload.get("overall_signal") or "").strip()
    summary = str(payload.get("summary") or "").strip()
    if overall_signal not in _OVERALL_SIGNALS:
        raise FullPatientScreenError(
            f"unsupported overall_signal {overall_signal!r}."
        )
    if not summary:
        raise FullPatientScreenError("the final summary must be non-empty.")
    raw_assessments = payload.get("criteria_assessments")
    if not isinstance(raw_assessments, list):
        raise FullPatientScreenError("criteria_assessments must be a JSON array.")

    expected_ids = set(criterion_ids)
    assessments: dict[str, dict[str, str]] = {}
    for raw_assessment in raw_assessments:
        if not isinstance(raw_assessment, dict):
            raise FullPatientScreenError(
                "every criteria_assessments item must be an object."
            )
        criterion_id = str(raw_assessment.get("criterion_id") or "").strip()
        signal = str(
            raw_assessment.get("eligibility_signal") or ""
        ).strip()
        rationale = str(raw_assessment.get("rationale") or "").strip()
        if criterion_id not in expected_ids:
            raise FullPatientScreenError(
                f"the final screen returned unknown criterion_id {criterion_id!r}."
            )
        if criterion_id in assessments:
            raise FullPatientScreenError(
                f"the final screen duplicated criterion_id {criterion_id!r}."
            )
        if signal not in _ELIGIBILITY_SIGNALS:
            raise FullPatientScreenError(
                f"criterion {criterion_id!r} has unsupported eligibility_signal "
                f"{signal!r}."
            )
        if not rationale:
            raise FullPatientScreenError(
                f"criterion {criterion_id!r} has an empty rationale."
            )
        assessments[criterion_id] = {
            "eligibility_signal": signal,
            "rationale": rationale,
        }
    missing = [
        criterion_id
        for criterion_id in criterion_ids
        if criterion_id not in assessments
    ]
    if missing:
        raise FullPatientScreenError(
            "the final screen omitted criterion IDs: " + ", ".join(missing)
        )
    return {
        "overall_signal": overall_signal,
        "summary": summary,
        "assessments": assessments,
        "limitations": _normalize_limitations(payload.get("limitations")),
    }


def _generate_validated_payload(
    runner: _JsonLLMRunner,
    messages: list[dict[str, str]],
    *,
    validator: Callable[[dict[str, Any]], Any],
    retry_limit: int,
    stage_name: str,
) -> Any:
    working_messages = [dict(message) for message in messages]
    last_error: Exception | None = None
    for _attempt in range(retry_limit + 1):
        payload, response_text = runner.generate(working_messages)
        try:
            return validator(payload)
        except FullPatientScreenError as exc:
            last_error = exc
            working_messages.extend(
                [
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": (
                            f"{stage_name.upper()} VALIDATION ERROR: {exc} Return one "
                            "corrected JSON object matching the requested schema."
                        ),
                    },
                ]
            )
    raise FullPatientScreenError(
        f"the LLM did not return a valid {stage_name} payload after retries. "
        f"Last validation error: {last_error}"
    ) from last_error


_WORKER_PATIENT_NOTES: str | pd.DataFrame | None = None
_WORKER_CONFIG: MMAIConfig | None = None
_WORKER_EMBEDDING_MODEL_NAME: str | None = None
_WORKER_TEXT_COLUMN = "note_text"
_WORKER_DATE_COLUMN = "note_date"


def _initialize_question_worker(
    patient_notes: str | pd.DataFrame,
    config: MMAIConfig,
    embedding_model_name: str | None,
    text_column: str,
    date_column: str,
) -> None:
    global _WORKER_PATIENT_NOTES
    global _WORKER_CONFIG
    global _WORKER_EMBEDDING_MODEL_NAME
    global _WORKER_TEXT_COLUMN
    global _WORKER_DATE_COLUMN

    _WORKER_PATIENT_NOTES = patient_notes
    _WORKER_CONFIG = config
    _WORKER_EMBEDDING_MODEL_NAME = embedding_model_name
    _WORKER_TEXT_COLUMN = text_column
    _WORKER_DATE_COLUMN = date_column


def _answer_one_question(
    question_spec: dict[str, str],
    *,
    patient_notes: str | pd.DataFrame,
    config: MMAIConfig,
    embedding_model_name: str | None,
    text_column: str,
    date_column: str,
) -> dict[str, Any]:
    answer = answer_question_with_raw_patient_notes(
        question_spec["question"],
        patient_notes,
        embedding_model_name=embedding_model_name,
        text_column=text_column,
        date_column=date_column,
        config=config,
    )
    return {
        **question_spec,
        "status": "answered",
        "patient_note_response": answer,
    }


def _question_worker(question_spec: dict[str, str]) -> dict[str, Any]:
    if _WORKER_PATIENT_NOTES is None or _WORKER_CONFIG is None:
        raise RuntimeError("full-patient-screen worker was not initialized.")
    return _answer_one_question(
        question_spec,
        patient_notes=_WORKER_PATIENT_NOTES,
        config=_WORKER_CONFIG,
        embedding_model_name=_WORKER_EMBEDDING_MODEL_NAME,
        text_column=_WORKER_TEXT_COLUMN,
        date_column=_WORKER_DATE_COLUMN,
    )


def _failed_question_result(
    question_spec: dict[str, str],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        **question_spec,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "patient_note_response": None,
    }


def _run_questions(
    questions: list[dict[str, str]],
    *,
    patient_notes: str | pd.DataFrame,
    config: MMAIConfig,
    embedding_model_name: str | None,
    text_column: str,
    date_column: str,
    max_workers: int,
    process_start_method: str,
    progress_callback: FullPatientScreenProgress | None,
) -> list[dict[str, Any]]:
    total = len(questions)
    _emit_progress(
        progress_callback,
        "questions",
        0,
        total,
        f"Starting {total} raw-note eligibility question(s)",
    )
    if max_workers <= 1 or total <= 1:
        results: list[dict[str, Any]] = []
        for completed, question_spec in enumerate(questions, start=1):
            try:
                result = _answer_one_question(
                    question_spec,
                    patient_notes=patient_notes,
                    config=config,
                    embedding_model_name=embedding_model_name,
                    text_column=text_column,
                    date_column=date_column,
                )
            except Exception as exc:  # noqa: BLE001 - retain partial screen results
                result = _failed_question_result(question_spec, exc)
            results.append(result)
            _emit_progress(
                progress_callback,
                "questions",
                completed,
                total,
                f"Completed raw-note question {completed}/{total}",
            )
        return results

    try:
        mp_context = get_context(process_start_method)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported full_patient_screen.process_start_method "
            f"{process_start_method!r}."
        ) from exc

    ordered_results: list[dict[str, Any] | None] = [None] * total
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=_initialize_question_worker,
        initargs=(
            patient_notes,
            config,
            embedding_model_name,
            text_column,
            date_column,
        ),
    ) as executor:
        future_indexes: dict[Future[dict[str, Any]], int] = {
            executor.submit(_question_worker, question_spec): index
            for index, question_spec in enumerate(questions)
        }
        for completed, future in enumerate(as_completed(future_indexes), start=1):
            index = future_indexes[future]
            try:
                ordered_results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - retain partial screen results
                ordered_results[index] = _failed_question_result(questions[index], exc)
            _emit_progress(
                progress_callback,
                "questions",
                completed,
                total,
                f"Completed raw-note question {completed}/{total}",
            )

    return [
        result
        if result is not None
        else _failed_question_result(
            questions[index],
            RuntimeError("worker returned no result"),
        )
        for index, result in enumerate(ordered_results)
    ]


def _synthesis_question_results(
    question_results: list[dict[str, Any]],
    *,
    evidence_limit: int,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for result in question_results:
        item: dict[str, Any] = {
            key: result[key]
            for key in (
                "criterion_id",
                "criterion_type",
                "criterion_text",
                "question",
                "status",
            )
        }
        if result["status"] == "answered":
            response = dict(result.get("patient_note_response") or {})
            item["answer"] = str(response.get("answer") or "").strip()
            item["limitations"] = _normalize_limitations(
                response.get("limitations")
            )
            evidence = response.get("evidence")
            item["validated_evidence"] = (
                evidence[:evidence_limit] if isinstance(evidence, list) else []
            )
        else:
            item["error"] = str(result.get("error") or "question failed")
        payload.append(item)
    return payload


def full_patient_screen(
    patient_notes: str | pd.DataFrame,
    eligibility_criteria: str | Mapping[str, Any],
    *,
    embedding_model_name: str | None = None,
    text_column: str = "note_text",
    date_column: str = "note_date",
    max_workers: int | None = None,
    config: MMAIConfig | None = None,
    return_metadata: bool = False,
    progress_callback: FullPatientScreenProgress | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """
    Screen complete trial eligibility criteria against one patient's raw notes.

    The configured LLM first converts every independently assessable criterion
    into a focused question. Each question is answered through
    :func:`answer_question_with_raw_patient_notes`, preserving its local
    embedding retrieval, exact-quote validation, and DataFrame date provenance.
    With a remote LLM backend and CPU raw-note embeddings, independent questions
    run concurrently in spawned CPU processes so multiple mini-agents can call
    the configured endpoint at the same time. A final LLM pass synthesizes the
    grounded answers into a JSON-compatible research screen.

    ``patient_notes`` accepts the same concatenated string or one-patient
    DataFrame contract as raw-note question answering. ``eligibility_criteria``
    can be the complete criteria text for one clinical trial or the structured
    mapping returned by
    :func:`matchminer_ai.trials.extract_trial_space_eligibility_criteria`.
    Retrieved patient excerpts and the final aggregated question results reach
    the configured LLM backend; no web search is used.

    Parallel process mode intentionally requires ``remote.enabled=True`` and
    ``raw_patient_note_qa.embedding_device=cpu``. Set ``max_workers=1`` for an
    in-process local vLLM backend. The result is not an eligibility
    determination and requires review against the current complete protocol.
    """
    eligibility_criteria_text, criterion_sources = _prepare_eligibility_criteria(
        eligibility_criteria
    )
    if not isinstance(patient_notes, (str, pd.DataFrame)):
        raise TypeError("patient_notes must be a string or pandas DataFrame.")
    _prepare_raw_text(
        patient_notes,
        text_column=text_column,
        date_column=date_column,
    )

    resolved_config = config or load_default_preset()
    if not isinstance(resolved_config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    screen_config = dict(resolved_config.full_patient_screen)
    if not screen_config:
        raise ValueError("Config is missing full_patient_screen settings.")
    max_questions = max(1, int(screen_config.get("max_questions", 128)))
    retry_limit = max(0, int(screen_config.get("response_retry_limit", 2)))
    requested_workers = (
        int(max_workers)
        if max_workers is not None
        else int(screen_config.get("max_workers", 4))
    )
    if requested_workers < 1:
        raise ValueError("max_workers must be a positive integer.")

    runtime_config = _task_runtime_config(screen_config, config=resolved_config)
    runner = _JsonLLMRunner(
        config=resolved_config,
        runtime_config=runtime_config,
        retry_limit=retry_limit,
    )

    _emit_progress(
        progress_callback,
        "decompose",
        0,
        1,
        "Converting complete eligibility criteria into patient-note questions",
    )
    decomposition_messages = [
        {
            "role": "system",
            "content": _load_prompt_text("full_patient_screen.questions.system.txt"),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "maximum_question_count": max_questions,
                    "criterion_sources": criterion_sources,
                },
                ensure_ascii=False,
            ),
        },
    ]
    questions = _generate_validated_payload(
        runner,
        decomposition_messages,
        validator=lambda payload: _validate_question_payload(
            payload,
            eligibility_criteria=eligibility_criteria_text,
            criterion_sources=criterion_sources,
            max_questions=max_questions,
        ),
        retry_limit=retry_limit,
        stage_name="criterion decomposition",
    )
    _emit_progress(
        progress_callback,
        "decompose",
        1,
        1,
        f"Created {len(questions)} patient-note question(s)",
    )

    effective_workers = min(
        requested_workers,
        len(questions),
        max(1, os.cpu_count() or 1),
    )
    if effective_workers > 1:
        if not bool(resolved_config.remote.get("enabled", False)):
            raise ValueError(
                "Parallel full_patient_screen requires a remote/OpenAI-compatible "
                "LLM backend. Set config.remote['enabled']=True or max_workers=1 "
                "for local in-process vLLM."
            )
        embedding_device = str(
            resolved_config.raw_patient_note_qa.get("embedding_device", "cpu")
        ).strip().casefold()
        if embedding_device != "cpu":
            raise ValueError(
                "Parallel full_patient_screen requires "
                "raw_patient_note_qa.embedding_device='cpu' to avoid duplicating "
                "a GPU embedding model across worker processes."
            )

    process_start_method = str(
        screen_config.get("process_start_method", "spawn")
    ).strip() or "spawn"
    question_results = _run_questions(
        questions,
        patient_notes=patient_notes,
        config=resolved_config,
        embedding_model_name=embedding_model_name,
        text_column=text_column,
        date_column=date_column,
        max_workers=effective_workers,
        process_start_method=process_start_method,
        progress_callback=progress_callback,
    )

    evidence_limit = max(
        0,
        int(screen_config.get("synthesis_evidence_limit_per_question", 6)),
    )
    synthesis_input = _synthesis_question_results(
        question_results,
        evidence_limit=evidence_limit,
    )
    _emit_progress(
        progress_callback,
        "synthesize",
        0,
        1,
        "Synthesizing grounded criterion results",
    )
    synthesis_messages = [
        {
            "role": "system",
            "content": _load_prompt_text("full_patient_screen.final.system.txt"),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "complete_eligibility_criteria": eligibility_criteria_text,
                    "grounded_question_results": synthesis_input,
                },
                ensure_ascii=False,
            ),
        },
    ]
    criterion_ids = [question["criterion_id"] for question in questions]
    final_payload = _generate_validated_payload(
        runner,
        synthesis_messages,
        validator=lambda payload: _validate_final_payload(
            payload,
            criterion_ids=criterion_ids,
        ),
        retry_limit=retry_limit,
        stage_name="final screen",
    )

    failed_count = sum(result["status"] != "answered" for result in question_results)
    criteria: list[dict[str, Any]] = []
    for result in question_results:
        criterion_id = result["criterion_id"]
        assessment = dict(final_payload["assessments"][criterion_id])
        if result["status"] != "answered":
            assessment = {
                "eligibility_signal": "insufficient_information",
                "rationale": "The raw-note question could not be completed.",
            }
        criteria.append({**result, **assessment})

    limitations = list(final_payload["limitations"])
    if failed_count:
        limitations.append(
            f"{failed_count} of {len(question_results)} raw-note questions failed; "
            "those criteria were forced to insufficient_information."
        )
    overall_signal = final_payload["overall_signal"]
    if failed_count == len(question_results):
        overall_signal = "insufficient_information"
    elif failed_count and overall_signal == "no_concern_identified":
        overall_signal = "mixed"
    result = {
        "overall_signal": overall_signal,
        "summary": final_payload["summary"],
        "criteria": criteria,
        "limitations": list(dict.fromkeys(limitations)),
        "workflow": {
            "question_count": len(question_results),
            "answered_count": len(question_results) - failed_count,
            "failed_count": failed_count,
            "process_workers": effective_workers,
            "process_start_method": (
                process_start_method if effective_workers > 1 else "not_used"
            ),
        },
        "research_use_notice": _RESEARCH_USE_NOTICE,
    }
    _emit_progress(
        progress_callback,
        "synthesize",
        1,
        1,
        "Grounded full-patient screen synthesized",
    )
    _emit_progress(
        progress_callback,
        "complete",
        1,
        1,
        "Full patient screen ready for human review",
    )

    if not return_metadata:
        return result
    return result, {
        "config_snapshot": config_snapshot(resolved_config),
        "model_metadata": {"full_patient_screen_llm": runner.model_metadata},
        "execution": dict(result["workflow"]),
    }


__all__ = [
    "FullPatientScreenError",
    "FullPatientScreenProgress",
    "full_patient_screen",
]
