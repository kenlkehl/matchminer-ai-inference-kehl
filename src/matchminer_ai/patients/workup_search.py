"""Workup documentation review using the isolated note-search question harness."""

from __future__ import annotations

import copy
from dataclasses import asdict
from importlib import resources
import json

import pandas as pd

from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, remote_enabled
from matchminer_ai.llm.structured import resolve_structured_config

from .note_search_qa import (
    NoteSearchLLMConfig,
    NoteSearchLimits,
    _AnswerFormat,
    _object,
    _run_question_batch,
)
from .workup import APPLICABILITY, STATUSES, _notes


def _history(source):
    """Preserve original note text and map exact spans to structured provenance."""
    parts, spans, cursor = [], [], 0
    for note in source:
        header = f"\n[Note {note['note_number']} | {note['note_date'] or 'date unavailable'}]\n"
        parts.extend([header, note["text"]])
        start = cursor + len(header)
        cursor = start + len(note["text"])
        spans.append({**note, "start": start, "end": cursor})
    return "".join(parts), spans


def _source_note(evidence, spans):
    return next(
        (
            n
            for n in spans
            if n["start"] <= evidence["start"] < evidence["end"] <= n["end"]
        ),
        None,
    )


def _answer_format(spans):
    def prepare_evidence(excerpts):
        # A search context may cross headers/notes. Split in code, preserving
        # only original text and assigning each fragment its own note date.
        result = []
        for item in excerpts:
            for note in spans:
                start, end = (
                    max(item["start"], note["start"]),
                    min(item["end"], note["end"]),
                )
                if start < end:
                    result.append(
                        {
                            "start": start,
                            "end": end,
                            "quote": item["quote"][
                                start - item["start"] : end - item["start"]
                            ],
                            "note_number": note["note_number"],
                            "note_date": note["note_date"],
                        }
                    )
        return result

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
        .joinpath("patient.workup_search.system.txt")
        .read_text(encoding="utf-8")
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
    )


def review_patient_workup_with_note_search(
    notes: str | pd.DataFrame,
    recommendations: list[dict],
    *,
    config: MMAIConfig | None = None,
    population_context: str | None = None,
    limits: NoteSearchLimits | None = None,
    max_parallel_questions: int | None = None,
    search_reasoning_effort: str | None = "low",
    progress_callback=None,
) -> dict:
    """Search for each workup item in parallel, retaining the workup review schema.

    Uses the configured patient endpoint and sampling profile. Each item has an
    isolated Python REPL and a structured final assessment within the same loop;
    there is no second synthesis agent. Thinking is on by default and reasoning
    traces are never replayed. Structured input dates are assigned by code after
    automatic capture of original search/read excerpts. The model does not emit
    quotes, offsets or citation IDs. Excerpts are reviewed context, not individually
    selected supporting citations. String input has unavailable dates. Per-item failures
    are status ``error``, never missing-documentation findings.
    Defaults to three total generation calls per item (including final answer
    and retries), and all items eligible to run concurrently. The configured
    endpoint request cap still applies. Pass explicit limits/concurrency to override.
    The first search-only request uses low effort where supported; subsequent
    assessment/follow-up turns retain the configured effort. Pass None to inherit
    that effort for every request instead.
    """
    config = config or load_default_preset()
    if not isinstance(config, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance.")
    if not remote_enabled(config):
        raise ValueError("Workup review requires a configured remote LLM endpoint.")
    source = _notes(notes)
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
    limits = limits or NoteSearchLimits(max_cells=2, max_calls=3)
    if max_parallel_questions is None:
        max_parallel_questions = len(recommendations)
    history, spans = _history(source)
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
    runtime = build_llm_runtime_config("patient", config.patient, config=config)
    runtime["max_retries"] = 3
    resolved, _ = resolve_structured_config(runtime, cache_dir=None)
    llm = NoteSearchLLMConfig(
        **asdict(resolved),
        sampling_profile=runtime.get("sampling_profile", "auto"),
        reasoning_effort=runtime.get("reasoning_effort", "xhigh"),
        search_reasoning_effort=search_reasoning_effort,
    )
    completed = 0

    def on_progress(event):
        nonlocal completed
        completed += 1
        progress(
            f"Agentic note search: {completed}/{len(questions)} workup items finished "
            f"(item {event['question_index'] + 1}: {event['status']})"
        )

    result = _run_question_batch(
        [{"patient_id": "patient", "history": history, "questions": questions}],
        llm=llm,
        limits=limits,
        max_parallel_patients=1,
        max_parallel_questions=max_parallel_questions,
        max_active_questions=max_parallel_questions,
        progress_callback=on_progress,
        answer_format=_answer_format(spans),
    )
    answers = result["patients"][0]["answers"]
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
            "scope": "searched_excerpts",
            "evidence_selection": "automatic_reviewed_excerpts",
            "note_count": len(source),
            "patient_summary_used": False,
            "requests": sum(a["metadata"]["requests"] for a in answers),
            "failed_items": sum(a["status"] == "error" for a in answers),
            "undated_notes": sum(n["note_date"] is None for n in source),
        },
    }
