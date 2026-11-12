"""Patient-level structured extraction: one record per patient from all of their notes.

Like serial summarization, a patient's notes are read in date order, in chunks
that fit the context, and each chunk's request sees the record built from the
earlier chunks. Unlike it (and unlike the lung abstraction this generalizes,
which asked for the whole running JSON again on every chunk, at a cost that
grew with the square of the record's length), the model returns only the new
and changed items of each chunk, under ids it keeps, and code merges them. So
each item's evidence is quoted from, and located in, the chunk that documents
it, and an item is never lost because a later answer leaves it out.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from matchminer_ai._metadata import package_metadata
from matchminer_ai._storage import atomic_json, digest, read_json
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.backends import get_llm_backend
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.llm.request_queue import RequestQueue

from .common import (
    attempt_entry,
    attempt_group,
    fill,
    ncit_agent_codes,
    parse_answer,
    qc_rows,
    retry_messages,
    section_settings,
)
from .evidence import evidence_entry, locate_quote
from .extensions import extension_guidance, extension_item_types, resolve_extensions
from .notes import normalize_notes, note_fragments, pack_fragments, patient_note_list, render_chunk
from .schema import (
    SCHEMA_VERSION,
    Problem,
    ResponseCheck,
    check_items,
    item_types_for,
    patient_response_schema,
    render_field_guide,
    to_record_item,
)

logger = logging.getLogger(__name__)

SECTION = "patient_extraction"
PROMPT_FILE = "patient.extraction.patient.txt"
CHECKPOINT_SUBDIR = "patient_records"
PATIENT_COLUMNS = [
    "patient_id",
    "extraction_status",
    "error",
    "note_count",
    "chunk_count",
    "failed_chunks",
    "item_count",
    "record",
]


@lru_cache(maxsize=4)
def load_tokenizer(name: str):
    """The fast tokenizer that sizes chunks (its character offsets keep exact text)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("Patient extraction needs a fast tokenizer for exact note offsets.")
    return tokenizer


class PatientState:
    """The record built so far: items by list key and id, in the order added."""

    def __init__(self, item_types):
        self.item_types = tuple(item_types)
        self.items: dict[str, dict[str, dict]] = {t.key: {} for t in self.item_types}

    def ids(self, key: str) -> set[str]:
        return set(self.items[key])

    def render(self) -> str:
        """The prompt's view: every item's fields with its supporting note numbers."""
        lines = []
        for item_type in self.item_types:
            entries = self.items[item_type.key]
            if not entries:
                continue
            lines.append(f"{item_type.key}:")
            for item_id, entry in entries.items():
                notes = sorted({e["note_number"] for e in entry["evidence"]})
                shown = {"item_id": item_id, **entry["fields"], "notes": notes}
                lines.append("  " + json.dumps(shown, ensure_ascii=False))
        return "\n".join(lines) if lines else "(empty: this is the first part)"

    def apply(self, check: ResponseCheck) -> None:
        for key, entries in check.items.items():
            for entry, evidence in entries:
                fields = {k: v for k, v in entry.items() if k not in {"item_id", "evidence"}}
                current = self.items[key].get(entry["item_id"])
                if current is None:
                    self.items[key][entry["item_id"]] = {"fields": fields, "evidence": list(evidence)}
                    continue
                current["fields"] = fields
                known = {(e["note_id"], e["start"], e["end"]) for e in current["evidence"]}
                current["evidence"] += [
                    e for e in evidence if (e["note_id"], e["start"], e["end"]) not in known
                ]

    def to_json(self) -> dict:
        return {
            key: [{"item_id": i, **entry} for i, entry in entries.items()]
            for key, entries in self.items.items()
        }

    @classmethod
    def from_json(cls, item_types, data: dict) -> PatientState:
        state = cls(item_types)
        for key, entries in data.items():
            if key not in state.items:
                raise ValueError(f"Checkpoint has unknown item list {key!r}")
            for entry in entries:
                state.items[key][entry["item_id"]] = {
                    "fields": entry["fields"], "evidence": entry["evidence"],
                }
        return state


def check_patient_answer(value: dict, chunk: list[dict], state: PatientState) -> ResponseCheck:
    """Validate one chunk's answer: fields, ids, cancer links and evidence quotes."""
    item_types = state.item_types
    expected = {t.key for t in item_types}
    if set(value) != expected:
        reason = f"the answer must have exactly the fields {sorted(expected)}"
        return ResponseCheck(problems=[Problem("", reason, reason)])
    by_number: dict[int, list[dict]] = {}
    for fragment in chunk:
        by_number.setdefault(fragment["note_number"], []).append(fragment)

    def locate(quote, path):
        fragments = by_number.get(quote["note"])
        if not fragments:
            return None, f"note {quote['note']} is not in this part"
        for fragment in fragments:
            span = locate_quote(fragment["text"], quote["quote"])
            if span is not None:
                found = evidence_entry(fragment, fragment["text"], fragment["offset"], span)
                # The number is the prompt's handle on the note; records keep the id.
                return {**found, "note_number": fragment["note_number"]}, None
        return None, f"quote not found in note {quote['note']}"

    result = check_items(value, item_types, "patient", locate)
    new_diagnoses = {
        e.get("item_id") for e in value.get("diagnoses", []) if isinstance(e, dict)
    }
    known_cancers = state.ids("diagnoses") | new_diagnoses
    for item_type in item_types:
        seen = set()
        pattern = re.compile(rf"{item_type.id_prefix}[1-9]\d*")
        for index, entry in enumerate(value[item_type.key]):
            if not isinstance(entry, dict):
                continue
            path = f"{item_type.key}[{index}]"
            item_id = entry.get("item_id")
            if not isinstance(item_id, str) or not pattern.fullmatch(item_id):
                reason = f"item_id must be {item_type.id_prefix} followed by a number"
                result.problems.append(Problem(path, f"{path}: {reason}", f"{path}: {reason}"))
                continue
            if item_id in seen:
                reason = f"item_id {item_id} is returned twice"
                result.problems.append(Problem(path, f"{path}: {reason}", f"{path}: repeated id"))
            seen.add(item_id)
            unknown = sorted(set(entry.get("cancer_ids") or []) - known_cancers)
            if isinstance(entry.get("cancer_ids"), list) and unknown:
                reason = f"cancer_ids {unknown} are not diagnosis ids of this record"
                result.problems.append(Problem(path, f"{path}: {reason}", f"{path}: unknown cancer ids"))
    return result


def build_patient_record(
    patient_id: str, state: PatientState, notes: list[dict], *, chunk_count: int,
    failed_chunks: list[int], extensions,
) -> dict:
    """The returned record of one patient's state."""
    items = {}
    for item_type in state.item_types:
        built = []
        for item_id, entry in state.items[item_type.key].items():
            evidence = [{k: e[k] for k in ("note_id", "note_date", "quote", "start", "end")}
                        for e in entry["evidence"]]
            fallback = None
            if item_type.occurrence and entry["fields"]["status"] == "performed":
                dates = sorted(e["note_date"] for e in evidence if e["note_date"])
                fallback = dates[0] if dates else None
            built.append(to_record_item(
                item_type, "patient", {"item_id": item_id, **entry["fields"]}, evidence,
                date_fallback=fallback, date_fallback_source="evidence_note_date",
            ))
        items[item_type.key] = built
    dates = [n["note_date"] for n in notes if n["note_date"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "level": "patient",
        "patient_id": patient_id,
        "extensions": [e.name for e in extensions],
        "note_count": len(notes),
        "first_note_date": min(dates) if dates else None,
        "last_note_date": max(dates) if dates else None,
        "chunk_count": chunk_count,
        "failed_chunks": list(failed_chunks),
        "items": items,
    }


def extract_patient_records(
    notes: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    checkpoint_dir: str | Path | None = None,
    return_metadata: bool = False,
    return_qc: bool = False,
) -> pd.DataFrame | tuple:
    """Extract one structured record per patient from all of their notes.

    A research abstraction aid: records need human review and are not clinical
    documentation. Each patient's notes are read in date order in chunks of
    ``patient_extraction.chunk_tokens`` tokens (a longer note is split, with
    ``chunk_overlap_tokens`` overlap); each chunk is one LLM request that sees
    the record so far and returns the chunk's new and changed items (decoded
    under the JSON schema with remote ``structured_outputs``). Answers are
    validated and their evidence quotes located in the chunk's notes; a failed
    answer is retried with its problems named (``max_attempts``, then once with
    thinking off when ``reasoning_off_fallback`` is on). On the last attempt,
    items whose quotes are all missing from the notes are dropped; a chunk with
    no valid answer is skipped and listed in ``failed_chunks``. A request that
    fails in transport stops that patient (``failed``); a rerun with the same
    ``checkpoint_dir`` resumes it at that chunk.

    Parameters
    ----------
    notes : pd.DataFrame
        One row per note: ``patient_id``, ``note_text``, ``note_date``, and
        optionally ``note_id`` and ``note_type``, as for
        :func:`extract_document_records`.
    checkpoint_dir : path, optional
        Each patient's progress is saved after every chunk (one JSON file per
        patient under ``patient_records/``, named by a hash of the patient id);
        a rerun with unchanged notes and settings resumes where it stopped. The
        files hold patient-derived text.
    return_metadata, return_qc : bool
        Also return a metadata dict and a QC report.

    Returns
    -------
    pd.DataFrame
        One row per patient: ``patient_id``, ``extraction_status``
        (``complete``, ``partial`` when a chunk failed or ungrounded items were
        dropped, or ``failed``), ``error``, ``note_count``, ``chunk_count``,
        ``failed_chunks``, ``item_count`` and ``record`` (None when failed).
    """
    resolved = config or load_default_preset()
    if not isinstance(resolved, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    frame, counts = normalize_notes(notes)
    extensions = resolve_extensions((getattr(resolved, SECTION, None) or {}).get("extensions") or [])
    extra_types = extension_item_types(extensions)
    item_types = item_types_for("patient", extra_types)
    schema = patient_response_schema(extra_types)
    settings = section_settings(resolved, SECTION, "clinical_patient_record_update", schema)
    values = settings.values
    chunk_tokens = values.get("chunk_tokens", 40000)
    overlap = values.get("chunk_overlap_tokens", 500)
    if type(chunk_tokens) is not int or chunk_tokens < 256:
        raise ValueError(f"{SECTION}.chunk_tokens must be an integer of at least 256")
    if type(overlap) is not int or not 0 <= overlap < chunk_tokens // 2:
        raise ValueError(f"{SECTION}.chunk_overlap_tokens must be from 0 to half of chunk_tokens")
    context = values.get("context_tokens")
    if context is None:
        context = values.get("local", {}).get("engine", {}).get("max_model_len")
    max_tokens = int(settings.runtime["sampling_params"].get("max_tokens", 0))
    tokenizer_name = settings.runtime.get("tokenizer_name", settings.runtime["model_name"])
    template = load_prompt_text(PROMPT_FILE)
    guide = render_field_guide("patient", extra_types)
    guidance = extension_guidance(extensions)
    ncit_resource = values.get(
        "ncit_resource", resolved.good_option_catalog.get("ncit_resource", "Thesaurus_26.07d.FLAT.zip")
    )
    contract = digest({
        "schema_version": SCHEMA_VERSION, "template": template, "guide": guide,
        "guidance": guidance, "schema": schema,
    })
    debug = resolved.debug_mode
    folder = Path(checkpoint_dir) / CHECKPOINT_SUBDIR if checkpoint_dir is not None else None

    patients = {
        str(pid): patient_note_list(group)
        for pid, group in frame.groupby("patient_id", sort=False)
    }
    inputs = {
        pid: digest({
            "notes": [[n["note_id"], n["note_date"], n["note_type"], digest(n["text"])] for n in ns],
            "chunk_tokens": chunk_tokens, "overlap": overlap, "tokenizer": tokenizer_name,
        })
        for pid, ns in patients.items()
    }
    states: dict[str, dict[str, Any]] = {}
    reused = 0
    for pid in patients:
        saved = None
        path = folder / f"{digest(pid)[:32]}.json" if folder is not None else None
        if path is not None and path.exists():
            candidate = read_json(path)
            if (
                candidate.get("patient_id") == pid
                and candidate.get("contract_sha256") == contract
                and candidate.get("input_sha256") == inputs[pid]
            ):
                saved = candidate
        if saved is not None:
            reused += saved["status"] in {"complete", "partial"}
            states[pid] = {
                "state": PatientState.from_json(item_types, saved["state"]),
                "next_chunk": saved["next_chunk"], "chunk_count": saved["chunk_count"],
                "failed_chunks": saved["failed_chunks"], "dropped": saved["dropped"],
                "attempts": saved["attempts"],
                "status": "running" if saved["status"] == "failed" else saved["status"],
                "error": None, "path": path,
            }
        else:
            states[pid] = {
                "state": PatientState(item_types), "next_chunk": 0, "chunk_count": None,
                "failed_chunks": [], "dropped": 0, "attempts": {}, "status": "running",
                "error": None, "path": path,
            }
    pending = [pid for pid, s in states.items() if s["status"] == "running"]
    chunks: dict[str, list[list[dict]]] = {}
    if pending:
        tokenizer = load_tokenizer(tokenizer_name)
        for pid in pending:
            chunks[pid] = pack_fragments(
                note_fragments(patients[pid], tokenizer, chunk_tokens - 64, overlap), chunk_tokens
            )
            states[pid]["chunk_count"] = len(chunks[pid])

    def save(pid):
        s = states[pid]
        if s["path"] is None:
            return
        atomic_json(s["path"], {
            "patient_id": pid, "schema_version": SCHEMA_VERSION, "contract_sha256": contract,
            "input_sha256": inputs[pid], "chunk_count": s["chunk_count"],
            "next_chunk": s["next_chunk"], "failed_chunks": s["failed_chunks"],
            "dropped": s["dropped"], "status": s["status"], "error": s["error"],
            "attempts": s["attempts"], "state": s["state"].to_json(),
        })

    def messages(pid):
        s = states[pid]
        index = s["next_chunk"]
        content = fill(template, {
            "field_guide": guide,
            "extension_guidance": f"\n{guidance}\n" if guidance else "",
            "record": s["state"].render(),
            "part": str(index + 1),
            "parts": str(s["chunk_count"]),
            "notes": render_chunk(chunks[pid][index]),
        })
        if context and max_tokens:
            prompt_tokens = len(tokenizer(content, add_special_tokens=False)["input_ids"])
            if prompt_tokens + max_tokens + 1024 > context:
                return None
        return [{"role": "user", "content": content}]

    def finish(pid):
        s = states[pid]
        s["status"] = "partial" if s["failed_chunks"] or s["dropped"] else "complete"
        save(pid)

    model_metadata: dict[str, Any] = {}
    if pending:
        queue = RequestQueue(
            get_llm_backend(resolved), settings.runtimes, chunk_size=settings.batch_size,
            chunks_in_flight=settings.batches_in_flight,
            model_metadata_cache_dir=resolved.model_metadata_cache_dir,
        )

        def start(pid):
            """Queue the patient's next chunk, or finish the patient."""
            s = states[pid]
            if s["next_chunk"] >= s["chunk_count"]:
                finish(pid)
                return
            original = messages(pid)
            if original is None:
                s["status"] = "failed"
                s["error"] = (
                    f"part {s['next_chunk'] + 1}: the record so far and the notes exceed "
                    f"{SECTION}.context_tokens with the output reserve"
                )
                save(pid)
                return
            queue.add((pid, s["next_chunk"], 1, original), original, "main")

        def record(item, answer, reasoning, finish_reason, provenance):
            pid, index, attempt, original = item
            s = states[pid]
            group = attempt_group(attempt, settings)
            entry = attempt_entry(attempt, group, finish_reason, provenance, debug=debug,
                                  answer=answer, reasoning=reasoning)
            s["attempts"].setdefault(str(index), []).append(entry)
            if finish_reason == "error":
                s["status"] = "failed"
                s["error"] = f"part {index + 1}: the LLM request failed after its transport retries"
                save(pid)
                return
            final = attempt >= settings.total_attempts
            try:
                value = parse_answer(answer, finish_reason)
            except ValueError as exc:
                check, error = None, str(exc)
            else:
                check, error = check_patient_answer(value, chunks[pid][index], s["state"]), None
            problems = [] if check is None else check.problems
            accept = error is None and (not problems or (final and not check.structural))
            if not accept:
                entry["problems"] = [error] if error else [p.code for p in problems]
                if not final:
                    queue.add((pid, index, attempt + 1, original),
                              retry_messages(original, answer, check is not None, problems, error),
                              attempt_group(attempt + 1, settings))
                    return
                s["failed_chunks"].append(index)
            else:
                if problems:
                    entry["problems"] = [p.code for p in check.grounding]
                    s["dropped"] += len(check.dropped)
                s["state"].apply(check)
            s["next_chunk"] = index + 1
            save(pid)
            start(pid)

        for pid in pending:
            start(pid)
        logger.info("Extracting patient records for %d patients.", len(pending))
        model_metadata = queue.run(record)

    rows = []
    for pid, s in states.items():
        built = None
        if s["status"] != "failed":
            built = build_patient_record(
                pid, s["state"], patients[pid], chunk_count=s["chunk_count"],
                failed_chunks=s["failed_chunks"], extensions=extensions,
            )
            if settings.ncit_agent_codes:
                for item in built["items"].get("treatments", []):
                    item["agents_ncit"] = ncit_agent_codes(item["agents"], ncit_resource)
        rows.append({
            "patient_id": pid, "extraction_status": s["status"], "error": s["error"],
            "note_count": len(patients[pid]), "chunk_count": s["chunk_count"],
            "failed_chunks": list(s["failed_chunks"]),
            "item_count": sum(map(len, built["items"].values())) if built else 0,
            "record": built,
        })
    result = pd.DataFrame(rows, columns=PATIENT_COLUMNS)
    statuses = {pid: s["status"] for pid, s in states.items()}
    metadata = {
        "package": package_metadata(),
        "config_snapshot": config_snapshot(resolved),
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract,
        "prompt_sha256": digest(template),
        "response_schema_sha256": digest(schema),
        "extensions": [e.name for e in extensions],
        "structured_outputs": settings.structured,
        "model_metadata": model_metadata,
        "counts": {
            "patients": len(states), "notes": int(len(frame)), "reused": reused,
            "chunks": sum(s["chunk_count"] or 0 for s in states.values()),
            **{k: sum(v == k for v in statuses.values()) for k in ("complete", "partial", "failed")},
            **counts,
        },
        "patients": {
            pid: {"extraction_status": s["status"], "attempts": s["attempts"],
                  "failed_chunks": s["failed_chunks"], "dropped_items": s["dropped"]}
            for pid, s in states.items()
        },
    }
    outputs: list[Any] = [result]
    if return_metadata:
        outputs.append(metadata)
    if return_qc:
        truncated = [
            pid for pid, s in states.items()
            if any(e["finish_reason"] == "length" for a in s["attempts"].values() for e in a)
        ]
        outputs.append(pd.DataFrame(qc_rows([
            ("patients_failed_extraction", [p for p, v in statuses.items() if v == "failed"]),
            ("patients_partial_extraction", [p for p, v in statuses.items() if v == "partial"]),
            ("patients_truncated_llm_response", truncated),
        ], len(states))))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


__all__ = [
    "CHECKPOINT_SUBDIR",
    "PATIENT_COLUMNS",
    "PatientState",
    "build_patient_record",
    "check_patient_answer",
    "extract_patient_records",
    "load_tokenizer",
]
