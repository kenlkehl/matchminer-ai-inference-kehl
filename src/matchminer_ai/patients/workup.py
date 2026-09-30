"""Serial, quote-grounded review of workup documentation in one patient's notes."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib import resources
from typing import Callable

import pandas as pd

from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config, remote_enabled
from matchminer_ai.llm.structured import StructuredClient, resolve_structured_config

STATUSES = (
    "completed",
    "partially_completed",
    "planned",
    "not_done",
    "not_documented",
    "unclear",
)
APPLICABILITY = ("applies", "not_applicable", "uncertain")
NOTICE = (
    "Documentation review of the supplied notes only, for human review. Not documented "
    "does not mean not done. This does not establish overall guideline concordance; "
    "clinical applicability, timing, and source completeness require review."
)


def _schema(size):
    def obj(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    return obj(
        {
            "assessments": {
                "type": "array",
                "minItems": size,
                "maxItems": size,
                "items": obj(
                    {
                        "name": {"type": "string"},
                        "applicability": {
                            "type": "string",
                            "enum": list(APPLICABILITY),
                        },
                        "status": {"type": "string", "enum": list(STATUSES)},
                        "bottom_line": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "maxItems": 8,
                            "items": obj(
                                {
                                    "note_number": {"type": "integer", "minimum": 1},
                                    "quote": {"type": "string", "minLength": 1},
                                }
                            ),
                        },
                    }
                ),
            }
        }
    )


def _notes(notes):
    if isinstance(notes, str):
        frame = pd.DataFrame([{"note_text": notes, "note_date": None}])
    elif isinstance(notes, pd.DataFrame):
        frame = notes.copy()
    else:
        raise TypeError("notes must be raw text or a single-patient DataFrame.")
    if "note_text" not in frame or frame.empty:
        raise ValueError("Supply nonempty raw notes with a note_text column.")
    if "patient_id" in frame and (
        frame.patient_id.isna().any() or frame.patient_id.nunique() != 1
    ):
        raise ValueError("Workup review requires exactly one patient.")
    if any(not isinstance(t, str) or not t.strip() for t in frame.note_text):
        raise ValueError("Every note must contain nonempty text.")
    if "note_date" not in frame:
        frame["note_date"] = None
    if "note_type" not in frame:
        frame["note_type"] = None
    if any(not pd.isna(t) and not isinstance(t, str) for t in frame.note_type):
        raise ValueError("note_type must contain strings or missing values.")
    dates = pd.to_datetime(frame.note_date, errors="coerce", format="mixed", utc=True)
    supplied = frame.note_date.notna() & frame.note_date.astype(str).str.strip().ne("")
    if (dates.isna() & supplied).any():
        raise ValueError("Invalid note date; leave unavailable dates empty.")
    frame["note_date"] = dates
    frame = frame.sort_values("note_date", kind="stable", na_position="last")
    return [
        dict(
            note_number=i,
            note_date=None if pd.isna(row.note_date) else row.note_date.isoformat(),
            note_type=None
            if pd.isna(row.note_type) or not row.note_type.strip()
            else row.note_type,
            text=row.note_text,
        )
        for i, row in enumerate(frame.itertuples(), 1)
    ]


def _fragments(notes, tokenizer, size, overlap):
    """Split using tokenizer character offsets so quotes remain exact original text."""
    fragments = []
    for note in notes:
        offsets = tokenizer(
            note["text"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        if len(offsets) <= size:
            fragments.append({**note, "tokens": len(offsets)})
            continue
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            left = 0 if start == 0 else offsets[start][0]
            right = len(note["text"]) if end == len(offsets) else offsets[end][0]
            fragments.append(
                {**note, "text": note["text"][left:right], "tokens": end - start}
            )
            if end == len(offsets):
                break
    return fragments


def _pack(fragments, size):
    packets, packet, count = [], [], 0
    for fragment in fragments:
        if packet and count + fragment["tokens"] > size:
            packets.append(packet)
            packet, count = [], 0
        packet.append(fragment)
        count += fragment["tokens"]
    if packet:
        packets.append(packet)
    return packets


def _validate(value, recommendations, previous, packet):
    if not isinstance(value, dict) or set(value) != {"assessments"}:
        raise ValueError("Expected assessments object.")
    rows = value["assessments"]
    if not isinstance(rows, list) or len(rows) != len(recommendations):
        raise ValueError("Return every recommendation exactly once in order.")
    for row, recommendation, prior in zip(rows, recommendations, previous, strict=True):
        if not isinstance(row, dict) or set(row) != {
            "name",
            "applicability",
            "status",
            "bottom_line",
            "evidence",
        }:
            raise ValueError("Invalid assessment fields.")
        if row["name"] != recommendation["name"]:
            raise ValueError("Recommendation names/order changed.")
        if row["status"] not in STATUSES or row["applicability"] not in APPLICABILITY:
            raise ValueError("Invalid assessment status.")
        if not isinstance(row["bottom_line"], str) or not row["bottom_line"].strip():
            raise ValueError("Provide a bottom line.")
        if not isinstance(row["evidence"], list) or len(row["evidence"]) > 8:
            raise ValueError("Invalid evidence list.")
        allowed = [(n["note_number"], n["text"]) for n in packet]
        allowed += [(e["note_number"], e["quote"]) for e in prior["evidence"]]
        for evidence in row["evidence"]:
            if not isinstance(evidence, dict) or set(evidence) != {
                "note_number",
                "quote",
            }:
                raise ValueError("Invalid evidence fields.")
            if (
                type(evidence["note_number"]) is not int
                or not isinstance(evidence["quote"], str)
                or not evidence["quote"].strip()
                or not any(
                    evidence["note_number"] == number and evidence["quote"] in text
                    for number, text in allowed
                )
            ):
                raise ValueError("Quote is not verbatim in the cited supplied note.")
        if (
            row["status"] != "not_documented"
            or row["applicability"] == "not_applicable"
        ) and not (row["evidence"] or prior["evidence"]):
            raise ValueError("Documented status/applicability requires evidence.")
        if row["status"] == "not_documented" and prior["status"] != "not_documented":
            raise ValueError(
                "Later silence cannot erase a previously documented finding."
            )


def review_patient_workup(
    notes: str | pd.DataFrame,
    recommendations: list[dict],
    *,
    config: MMAIConfig | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    recommendation_batch_size: int = 6,
    population_context: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Assess each workup item against every raw-note chunk, with exact quotes.

    ``recommendations`` accepts the catalog's diagnostic_workup list (name,
    conditions, category, evidence). One JSON-compatible assessment is returned
    per input item, in order, retaining that entire original recommendation.
    Uses the patient LLM's remote endpoint, vendor sampling and reasoning settings.
    Context is discovered from the endpoint unless patient.context_window is set;
    the configured output reserve is never reduced to squeeze in notes.
    No patient prompts, responses or checkpoints are written to disk.
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
    size = config.patient["chunk_size"] if chunk_size is None else chunk_size
    overlap = (
        config.patient["chunk_overlap"] if chunk_overlap is None else chunk_overlap
    )
    if (
        type(size) is not int
        or size < 1
        or type(overlap) is not int
        or not 0 <= overlap < size
        or type(recommendation_batch_size) is not int
        or not 1 <= recommendation_batch_size <= 20
    ):
        raise ValueError("Invalid chunk size, overlap or recommendation batch size.")
    progress = progress_callback or (lambda _: None)
    progress("Preparing raw notes for workup review")
    runtime = build_llm_runtime_config("patient", config.patient, config=config)
    runtime["max_retries"] = 3
    client_config, _ = resolve_structured_config(runtime, cache_dir=None)
    client = StructuredClient(client_config, None)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        runtime.get("tokenizer_name", runtime["model_name"]), use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Workup review needs a fast tokenizer for exact note offsets.")
    system = (
        resources.files("matchminer_ai.prompts")
        .joinpath("patient.workup_review.system.txt")
        .read_text()
    )
    packets = _pack(_fragments(source, tokenizer, size, overlap), size)
    assessments, calls = [], 0
    for batch_start in range(0, len(recommendations), recommendation_batch_size):
        batch = recommendations[batch_start : batch_start + recommendation_batch_size]
        previous = [
            dict(
                name=r["name"],
                applicability="uncertain",
                status="not_documented",
                bottom_line="No notes reviewed yet.",
                evidence=[],
            )
            for r in batch
        ]
        pending = list(packets)
        processed = 0
        while pending:
            packet = pending.pop(0)
            payload = {
                "guideline_population": population_context,
                "recommendations": [
                    dict(name=r["name"], conditions=r.get("conditions", ""))
                    for r in batch
                ],
                "prior_assessments": previous,
                "raw_note_fragments": [
                    {k: v for k, v in n.items() if k != "tokens"} for n in packet
                ],
            }
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            if not client.fits(messages):
                if len(packet) > 1:
                    middle = len(packet) // 2
                    pending[0:0] = [packet[:middle], packet[middle:]]
                elif packet[0]["tokens"] > 32:
                    smaller = max(16, packet[0]["tokens"] // 2)
                    pieces = _fragments(
                        packet, tokenizer, smaller, min(overlap, smaller // 4)
                    )
                    pending[0:0] = [[piece] for piece in pieces]
                else:
                    raise ValueError(
                        "Workup state exceeds input capacity; reduce recommendation_batch_size."
                    )
                continue
            progress(
                f"Reviewing raw-note chunk {processed + 1}/{processed + 1 + len(pending)} "
                f"for workup items {batch_start + 1}–{batch_start + len(batch)} "
                f"of {len(recommendations)}"
            )
            value = client.complete(
                "patient-workup",
                messages,
                _schema(len(batch)),
                lambda v: _validate(v, batch, previous, packet),
            )
            for row, prior in zip(value["assessments"], previous, strict=True):
                merged = list(prior["evidence"])
                for evidence in row["evidence"]:
                    if evidence not in merged:
                        merged.append(evidence)
                row["evidence"] = merged
            previous = value["assessments"]
            calls += 1
            processed += 1
        for offset, (row, recommendation) in enumerate(
            zip(previous, batch, strict=True)
        ):
            row["recommendation_index"] = batch_start + offset
            row["recommendation"] = copy.deepcopy(recommendation)
            for evidence in row["evidence"]:
                evidence["note_date"] = source[evidence["note_number"] - 1]["note_date"]
            assessments.append(row)
    progress("Workup documentation review complete")
    return {
        "assessments": assessments,
        "notice": NOTICE,
        "metadata": {
            "method": "full_notes",
            "note_count": len(source),
            "initial_chunk_count": len(packets),
            "requests": calls,
            "model": client_config.model,
            "context_window": client_config.context_window,
            "reserved_output_tokens": client_config.max_tokens,
            "chunk_size": size,
            "chunk_overlap": overlap,
            "undated_notes": sum(n["note_date"] is None for n in source),
            "scope": "all_supplied_raw_notes",
            "patient_summary_used": False,
            "reasoning_effort": runtime.get("reasoning_effort", "xhigh"),
            "sampling_profile": runtime.get("sampling_profile", "auto"),
            "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "note_date_range": [
                min((n["note_date"] for n in source if n["note_date"]), default=None),
                max((n["note_date"] for n in source if n["note_date"]), default=None),
            ],
        },
    }
