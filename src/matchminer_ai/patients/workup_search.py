"""Workup documentation review using the isolated note-search question harness."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from importlib import resources

import pandas as pd

from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, remote_enabled
from matchminer_ai.llm.structured import resolve_structured_config

from ._note_record import history_from_notes, prepare_record, source_excerpts
from ._workup_backup import WorkupBackup
from .note_search_qa import (
    NoteSearchLimits,
    NoteSearchLLMConfig,
    _AnswerFormat,
    _object,
    _resolve_llm,
    _run_question_batch,
)
from .workup import APPLICABILITY, STATUSES
from .workup import _notes as _notes
from .workup_search_review import WorkupSearchReviewConfig, review_answers


def _history(source):
    """Preserve original note text and map exact spans to structured provenance."""
    return history_from_notes(source)


def _source_note(evidence, spans):
    return next(
        (
            n
            for n in spans
            if n["start"] <= evidence["start"] < evidence["end"] <= n["end"]
        ),
        None,
    )


def _answer_format(spans, *, request_review=False):
    def prepare_evidence(excerpts):
        # A search context may cross headers/notes. Split in code, preserving
        # only original text and assigning each fragment its own note date.
        return source_excerpts(excerpts, spans)

    def validate(value):
        answer = value["answer"]
        if not isinstance(answer, dict) or set(answer) != {
            "applicability",
            "status",
            "bottom_line",
        }:
            raise ValueError("Return the structured workup answer fields.")
        if (
            answer["applicability"] not in APPLICABILITY
            or answer["status"] not in STATUSES
        ):
            raise ValueError("Invalid workup status or applicability.")
        if (
            not isinstance(answer["bottom_line"], str)
            or not answer["bottom_line"].strip()
            or len(answer["bottom_line"]) > 6000
        ):
            raise ValueError("Supply a bounded workup explanation.")
        if any(_source_note(e, spans) is None for e in value["evidence"]):
            raise ValueError(
                "Each evidence quote must stay inside one original note, excluding headers."
            )
        if not value["evidence"] and (
            answer["status"] not in {"not_documented", "unclear"}
            or answer["applicability"] != "uncertain"
        ):
            raise ValueError(
                "Documented status and applicability require patient-note evidence."
            )
        if value["status"] == "unknown" and answer["status"] not in {
            "not_documented",
            "unclear",
        }:
            raise ValueError(
                "Unknown answers must retain an uncertain documentation status."
            )
        if (
            answer["status"] in {"not_documented", "unclear"}
            and value["status"] != "unknown"
        ):
            raise ValueError(
                "Missing or unclear documentation must be an unknown answer."
            )

    instructions = (
        resources.files("matchminer_ai.prompts")
        .joinpath("patient.workup_assessment.txt")
        .read_text(encoding="utf-8")
    )
    search_instructions = (
        resources.files("matchminer_ai.prompts")
        .joinpath("patient.workup_search.system.txt")
        .read_text(encoding="utf-8")
        .replace("{assessment_contract}", instructions)
    )
    return _AnswerFormat(
        schema=_object(
            {
                "applicability": {"type": "string", "enum": list(APPLICABILITY)},
                "status": {"type": "string", "enum": list(STATUSES)},
                "bottom_line": {"type": "string", "maxLength": 6000},
            }
        ),
        instructions=instructions,
        validate=validate,
        prepare_evidence=prepare_evidence,
        request_review=request_review,
        search_instructions=search_instructions,
        python_answer={
            "applicability": "uncertain",
            "status": "unclear",
            "bottom_line": "",
        },
        example_answer={
            "applicability": "uncertain",
            "status": "completed",
            "bottom_line": "The reviewed note documents a completed chest CT; "
            "guideline applicability remains uncertain.",
        },
        example_limitations=("The guideline indication remains uncertain.",),
        unknown_answer={
            "applicability": "uncertain",
            "status": "not_documented",
            "bottom_line": "Chest CT documentation was not located in the searched passages.",
        },
    )


def review_patient_workup_with_note_search(
    notes: str | pd.DataFrame | None = None,
    recommendations: list[dict] | None = None,
    *,
    history: str | None = None,
    patient_summary: str | None = None,
    config: MMAIConfig | None = None,
    backup_config: MMAIConfig | None = None,
    max_consecutive_failures: int = 3,
    population_context: str | None = None,
    limits: NoteSearchLimits | None = None,
    max_parallel_questions: int | None = None,
    thinking: str | None = "off",
    search_reasoning_effort: str | None = "low",
    review: WorkupSearchReviewConfig | None = None,
    progress_callback=None,
) -> dict:
    """Search for each workup item in parallel, retaining the workup review schema.

    Supply a notes DataFrame, concatenated text, or both (using history=). The
    DataFrame is authoritative when both exist; note_text is required and
    note_date/note_type are optional. An existing patient_summary guides pandas
    and regex navigation but is never a source of patient-note evidence.

    Uses the configured patient endpoint and sampling profile. Each item has an
    isolated Python REPL and a provisional structured assessment. By default,
    shared search vocabulary and up to two focused evidence reviews follow it;
    unresolved/coverage-limited items can use bounded serial full-record review.
    Pass WorkupSearchReviewConfig(max_review_passes=0) to disable follow-up.
    Thinking is off by default. Pass thinking="on" to enable it or None to inherit
    the configured patient endpoint switch. Providers without an off switch retain
    their supported effort setting. Reasoning traces are never replayed.
    Structured input dates are assigned by code after
    automatic capture of original pandas/search/read excerpts. The model does not emit
    quotes, offsets or citation IDs. Excerpts are reviewed context, not individually
    selected supporting citations. String input has unavailable dates. Per-item failures
    are status ``error``, never missing-documentation findings.
    Defaults to twelve Python cells and sixteen initial request attempts per item,
    including its answer and retries, plus at most twelve follow-up calls per item
    including vocabulary and retries. Up to five items can use full-record fallback, at most eight
    chunks each. Failed follow-up preserves the last validated assessment with an
    explicit limitation and audit status. All items may run concurrently. The configured
    endpoint request cap still applies. Pass explicit limits/concurrency to override.
    The first search-only request uses low effort where supported; subsequent
    assessment/follow-up turns retain the configured effort. Pass None to inherit
    that effort for every request instead.
    An explicit backup_config retries only failed items after repeated Python
    errors or failed request/validation attempts. Each initial attempt gets a
    fresh isolated REPL and the same bounded budget; combined costs are reported.
    Follow-up retains validated findings and can switch once within its existing
    call budget. Backup sampling and reasoning use backup_config independently.
    """
    started = time.monotonic()
    review = review or WorkupSearchReviewConfig()
    if not isinstance(review, WorkupSearchReviewConfig):
        raise TypeError("review must be a WorkupSearchReviewConfig instance.")
    config = config or load_default_preset()
    if not isinstance(config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance.")
    if thinking is not None and thinking not in ("on", "off"):
        raise ValueError("thinking must be on, off, or None.")
    if not remote_enabled(config):
        raise ValueError("Workup review requires a configured remote LLM endpoint.")
    backup = WorkupBackup(backup_config, max_consecutive_failures)
    limits = limits or NoteSearchLimits()
    if not isinstance(limits, NoteSearchLimits):
        raise TypeError("limits must be a NoteSearchLimits instance.")
    record = prepare_record(
        history,
        notes,
        patient_summary,
        max_bytes=limits.max_history_bytes,
        max_summary_chars=limits.max_summary_chars,
    )
    source = record.spans
    if not isinstance(recommendations, list) or not recommendations:
        raise ValueError("Supply at least one diagnostic/workup recommendation.")
    if population_context is not None and not isinstance(population_context, str):
        raise TypeError("population_context must be guideline population text or None.")
    for item in recommendations:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"].strip()
            or not isinstance(item.get("conditions", ""), str)
        ):
            raise ValueError("Each recommendation requires a name and text conditions.")
    if max_parallel_questions is None:
        max_parallel_questions = len(recommendations)
    if record.structured:
        history, spans = record.history, record.spans
    else:
        history, spans = _history(source)
        record.history, record.spans = history, spans
    questions = [
        json.dumps(
            {
                "task": "Assess documentation and applicability of this workup item.",
                "recommendation": {
                    "name": r["name"],
                    "conditions": r.get("conditions", ""),
                },
                "guideline_population": population_context,
            },
            ensure_ascii=False,
        )
        for r in recommendations
    ]
    progress = progress_callback or (lambda _: None)
    progress(f"Preparing agentic note search for {len(questions)} workup items")
    patient = copy.deepcopy(config.patient)
    if thinking is not None:
        patient.setdefault("local", {}).setdefault("chat_template_kwargs", {})[
            "enable_thinking"
        ] = thinking == "on"
        patient.setdefault("remote", {}).setdefault("extra_body", {}).setdefault(
            "chat_template_kwargs", {}
        )["enable_thinking"] = thinking == "on"
    runtime = build_llm_runtime_config("patient", patient, config=config)
    runtime["max_retries"] = backup.threshold if backup.enabled else 3
    resolved, _ = resolve_structured_config(runtime, cache_dir=None)
    llm = NoteSearchLLMConfig(
        **asdict(resolved),
        sampling_profile=runtime.get("sampling_profile", "auto"),
        reasoning_effort=runtime.get("reasoning_effort", "xhigh"),
        search_reasoning_effort=search_reasoning_effort,
    )

    def resolve_backup(backup_config):
        progress("Primary agent model failed repeatedly; trying the configured backup model")
        runtime = build_llm_runtime_config(
            "patient", backup_config.patient, config=backup_config
        )
        runtime["max_retries"] = 3
        resolved, _ = resolve_structured_config(runtime, cache_dir=None)
        return NoteSearchLLMConfig(
            **asdict(resolved), sampling_profile=runtime.get("sampling_profile", "auto"),
            reasoning_effort=runtime.get("reasoning_effort", "xhigh"),
            search_reasoning_effort=search_reasoning_effort,
        )

    def backup_factory():
        return backup.get(resolve_backup)
    completed = 0

    def on_progress(event):
        nonlocal completed
        completed += 1
        progress(
            f"Agentic note search: {completed}/{len(questions)} workup items finished "
            f"(item {event['question_index'] + 1}: {event['status']})"
        )

    contract = _answer_format(spans, request_review=review.max_review_passes > 0)
    result = _run_question_batch(
        [{"patient_id": "patient", "history": history, "questions": questions}],
        llm=llm,
        limits=limits,
        max_parallel_patients=1,
        max_parallel_questions=max_parallel_questions,
        max_active_questions=max_parallel_questions,
        progress_callback=on_progress,
        answer_format=contract,
        prepared_records=[record],
        backup_llm_factory=backup_factory if backup.enabled else None,
        failure_threshold=backup.threshold,
    )
    answers = result["patients"][0]["answers"]
    if review.max_review_passes:
        progress(
            "Checking search coverage and reviewing provisional workup assessments"
        )
        answers = review_answers(
            answers,
            history=history,
            spans=spans,
            llm=_resolve_llm(llm),
            config=review,
            contract=contract,
            concurrency=max_parallel_questions,
            progress=progress,
            patient_summary=record.patient_summary,
            backup_llm_factory=backup_factory if backup.enabled else None,
            failure_threshold=backup.threshold,
        )
    assessments = []
    for i, (answer, recommendation) in enumerate(
        zip(answers, recommendations, strict=True)
    ):
        finding = answer["answer"]
        if not isinstance(finding, dict):
            finding = {
                "status": "error" if answer["status"] == "error" else "unclear",
                "applicability": "uncertain",
                "bottom_line": finding
                or "The search failed; no documentation finding was made.",
            }
        evidence = []
        for item in answer["evidence"]:
            note = _source_note(item, spans)
            evidence.append(
                {
                    **item,
                    "note_number": note["note_number"],
                    "note_date": note["note_date"],
                }
            )
        assessments.append(
            {
                **finding,
                "name": recommendation["name"],
                "recommendation_index": i,
                "recommendation": copy.deepcopy(recommendation),
                "evidence": evidence,
                "review_status": answer["status"],
                "limitations": answer["limitations"],
                "metadata": answer["metadata"],
            }
        )
    progress(
        "Agentic workup documentation review finished; inspect findings and limitations"
    )
    return {
        "assessments": assessments,
        "notice": result["notice"],
        "metadata": {
            **result["metadata"],
            "method": "agentic",
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "followup_review": asdict(review),
            "incomplete_review_items": sum(
                a["metadata"].get("followup_review", {}).get("status") == "incomplete"
                for a in answers
            ),
            "scope": "searched_excerpts",
            "evidence_selection": "automatic_reviewed_excerpts",
            "note_count": len(source),
            "patient_summary_used": record.patient_summary is not None,
            "patient_summary_role": "navigation_context_only",
            "structured_notes": record.structured,
            "text_supplied": record.text_supplied,
            "input_precedence": "dataframe" if record.structured else "history",
            "requests": sum(a["metadata"]["requests"] for a in answers),
            "failed_items": sum(a["status"] == "error" for a in answers),
            "backup_used": any(a["metadata"].get("backup_used") for a in answers),
            "backup_items": sum(bool(a["metadata"].get("backup_used")) for a in answers),
            "undated_notes": sum(n["note_date"] is None for n in source),
        },
    }
