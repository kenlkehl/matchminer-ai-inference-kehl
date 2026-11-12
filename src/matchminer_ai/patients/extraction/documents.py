"""Document-level structured extraction: one record per note."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from matchminer_ai._metadata import package_metadata
from matchminer_ai._storage import digest
from matchminer_ai.config import MMAIConfig, config_snapshot, load_default_preset
from matchminer_ai.llm.backends import get_llm_backend
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.llm.request_queue import RequestQueue

from . import vocabulary as v
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
from .notes import normalize_notes
from .schema import (
    SCHEMA_VERSION,
    Problem,
    ResponseCheck,
    check_items,
    document_response_schema,
    item_types_for,
    render_field_guide,
    to_record_item,
)

logger = logging.getLogger(__name__)

SECTION = "document_extraction"
PROMPT_FILE = "patient.extraction.document.txt"
CHECKPOINT_FILE = "document_records.jsonl"
DOCUMENT_COLUMNS = [
    "patient_id",
    "note_id",
    "note_date",
    "note_type",
    "document_type",
    "extraction_status",
    "error",
    "item_count",
    "record",
]
#: Report types whose record must include the study or procedure behind them.
_SOURCE_REQUIRED = {"imaging_report": "imaging_studies", "pathology_report": "procedures"}


def known_cancers(patient_records: pd.DataFrame) -> dict[str, list[dict]]:
    """Each patient's diagnoses (id, site, histology, date) from patient records."""
    if not isinstance(patient_records, pd.DataFrame) or not {"patient_id", "record"} <= set(
        patient_records
    ):
        raise ValueError("patient_records must be extract_patient_records output")
    known: dict[str, list[dict]] = {}
    for patient_id, record in zip(patient_records.patient_id, patient_records.record, strict=True):
        if not isinstance(record, dict):
            continue
        known[str(patient_id)] = [
            {
                "item_id": d["item_id"],
                "primary_site": d.get("primary_site", ""),
                "histology": d.get("histology", ""),
                "diagnosis_date": (d.get("diagnosis_date") or {}).get("value"),
            }
            for d in record.get("items", {}).get("diagnoses", [])
        ]
    return known


def linking_block(cancers: list[dict] | None) -> str:
    """The prompt's patient context for attributing items to known cancers."""
    if cancers is None:
        return ""
    lines = ["LINKING ITEMS TO CANCERS"]
    if cancers:
        lines.append("This patient's known cancer diagnoses, from the patient-level record:")
        for cancer in cancers:
            bits = [cancer["primary_site"] or "cancer"]
            if cancer["histology"]:
                bits.append(cancer["histology"])
            if cancer["diagnosis_date"]:
                bits.append(f"diagnosed {cancer['diagnosis_date']}")
            lines.append(f"  {cancer['item_id']}: {', '.join(bits)}")
        lines.append(
            "Each item's cancer_ids lists the ids above that the item concerns: one id "
            "when it clearly concerns one cancer, several when it genuinely concerns "
            "several, and [] when it is not clear. A diagnosis's cancer_ids is the known "
            "cancer it is, or [] for a cancer not listed."
        )
    else:
        lines.append(
            "The patient-level record lists no cancer diagnoses for this patient; use [] "
            "for every cancer_ids."
        )
    return "\n".join(lines) + "\n"


def check_document_answer(
    value: dict,
    note: dict,
    item_types,
    *,
    linking: bool,
    known_ids: set[str],
) -> ResponseCheck:
    """Validate one document answer and resolve its evidence against the note."""
    expected = {"document_type", "document_type_text", *(t.key for t in item_types)}
    if set(value) != expected:
        reason = f"the answer must have exactly the fields {sorted(expected)}"
        return ResponseCheck(problems=[Problem("", reason, reason)])
    problems = []
    if value["document_type"] not in v.DOCUMENT_TYPES.codes:
        reason = f"document_type must be one of {list(v.DOCUMENT_TYPES.codes)}"
        problems.append(Problem("document_type", reason, reason))
    elif not isinstance(value["document_type_text"], str):
        problems.append(Problem("document_type_text", "document_type_text must be a string",
                                "document_type_text: not a string"))
    elif value["document_type"] == "other" and not value["document_type_text"].strip():
        reason = "document_type is other, so document_type_text must name it"
        problems.append(Problem("document_type_text", reason, reason))
    text = note["text"]

    def locate(quote, path):
        span = locate_quote(text, quote["quote"])
        if span is None:
            return None, "quote not found in the document"
        return evidence_entry(note, text, 0, span), None

    result = check_items(value, item_types, "document", locate, linking=linking)
    result.problems = problems + result.problems
    if linking:
        for key, entries in result.items.items():
            for index, (entry, _) in enumerate(entries):
                unknown = sorted(set(entry.get("cancer_ids", [])) - known_ids)
                if unknown:
                    reason = f"{key}: cancer_ids {unknown} are not known cancer ids"
                    result.problems.append(Problem(key, reason, f"{key}: unknown cancer ids"))
    required = _SOURCE_REQUIRED.get(value.get("document_type"))
    if required and not any(
        isinstance(e, dict) and e.get("relation") == "this_report" for e in value.get(required, [])
    ):
        noun = "imaging study" if required == "imaging_studies" else "procedure"
        reason = (
            f"a {value['document_type'].replace('_', ' ')} must list the {noun} that "
            f"produced it in {required} with relation this_report (use not_stated for "
            "values it does not give)"
        )
        result.problems.append(Problem(required, reason, reason))
    return result


def build_document_record(
    note: dict, value: dict, check: ResponseCheck, item_types, extensions
) -> dict:
    """The returned record of one accepted document answer."""
    items = {}
    for item_type in item_types:
        built = []
        for entry, evidence in check.items.get(item_type.key, []):
            source = item_type.occurrence and entry["relation"] == "this_report"
            built.append(
                to_record_item(
                    item_type, "document", entry, evidence,
                    date_fallback=note["note_date"] if source else None,
                )
            )
        items[item_type.key] = built
    return {
        "schema_version": SCHEMA_VERSION,
        "level": "document",
        "patient_id": note["patient_id"],
        "note_id": note["note_id"],
        "note_date": note["note_date"],
        "note_type": note["note_type"],
        "document_type": value["document_type"],
        "document_type_text": value["document_type_text"].strip(),
        "extensions": [e.name for e in extensions],
        "items": items,
    }


def add_ncit_codes(record: dict, resource: str) -> None:
    for item in record["items"].get("treatments", []):
        item["agents_ncit"] = ncit_agent_codes(item["agents"], resource)


def extract_document_records(
    notes: pd.DataFrame,
    *,
    config: MMAIConfig | None = None,
    patient_records: pd.DataFrame | None = None,
    checkpoint_dir: str | Path | None = None,
    return_metadata: bool = False,
    return_qc: bool = False,
) -> pd.DataFrame | tuple:
    """Extract one structured record from each clinical note.

    A research abstraction aid: records need human review and are not clinical
    documentation. Each note is one LLM request with the settings of
    ``config.document_extraction``; the answer is decoded under the record's JSON
    schema (remote backends with ``structured_outputs``), validated, and every
    evidence quote is located in the note's own text. An answer that fails is
    retried with the problems named (``max_attempts``, then once with thinking
    off when ``reasoning_off_fallback`` is on). When the last attempt's only
    problems are quotes not found in the note, the items left without any found
    quote are dropped and the record is ``partial``.

    Parameters
    ----------
    notes : pd.DataFrame
        One row per note: ``patient_id``, ``note_text``, ``note_date`` (missing
        dates stay unknown) and optionally ``note_id`` (unique) and
        ``note_type`` (the record system's label, shown to the model as a hint).
        Without ``note_id`` each note gets a stable id from a hash of its
        patient, date, type and text.
    patient_records : pd.DataFrame, optional
        Output of :func:`extract_patient_records`. Its diagnoses are listed in
        each prompt so that items can name the cancers they concern
        (``cancer_ids``), as the lung abstraction did after its patient pass.
    checkpoint_dir : path, optional
        Each finished document is appended to ``document_records.jsonl`` there;
        a rerun reuses complete and partial records whose note, context and
        extraction contract are unchanged and retries failed ones. The file
        holds patient-derived text: keep it where the notes may be kept.
    return_metadata, return_qc : bool
        Also return a metadata dict (schema version, prompt and schema hashes,
        model metadata, per-document attempts) and a QC report.

    Returns
    -------
    pd.DataFrame
        One row per note with text, in patient and date order:
        ``patient_id``, ``note_id``, ``note_date``, ``note_type``,
        ``document_type``, ``extraction_status`` (``complete``, ``partial`` or
        ``failed``), ``error``, ``item_count`` and ``record`` (the record dict;
        None when failed). :func:`extraction_items` flattens the records to one
        row per item.
    """
    resolved = config or load_default_preset()
    if not isinstance(resolved, MMAIConfig):
        raise TypeError("config must be an MMAIConfig instance or None.")
    frame, counts = normalize_notes(notes)
    extensions = resolve_extensions((getattr(resolved, SECTION, None) or {}).get("extensions") or [])
    extra_types = extension_item_types(extensions)
    item_types = item_types_for("document", extra_types)
    linking = patient_records is not None
    cancers = known_cancers(patient_records) if linking else {}
    schema = document_response_schema(extra_types, linking=linking)
    settings = section_settings(resolved, SECTION, "clinical_document_record", schema)
    template = load_prompt_text(PROMPT_FILE)
    guide = render_field_guide("document", extra_types, linking=linking)
    guidance = extension_guidance(extensions)
    limit = settings.values.get("max_document_characters", 200000)
    if type(limit) is not int or limit < 1:
        raise ValueError(f"{SECTION}.max_document_characters must be a positive integer")
    ncit_resource = settings.values.get(
        "ncit_resource", resolved.good_option_catalog.get("ncit_resource", "Thesaurus_26.07d.FLAT.zip")
    )
    contract = digest({
        "schema_version": SCHEMA_VERSION, "template": template, "guide": guide,
        "guidance": guidance, "schema": schema, "ncit": settings.ncit_agent_codes,
    })
    debug = resolved.debug_mode

    notes_list = [
        {"patient_id": r.patient_id, "note_id": r.note_id, "note_date": r.note_date,
         "note_type": r.note_type, "text": r.note_text}
        for r in frame.itertuples(index=False)
    ]
    contexts = {
        n["note_id"]: linking_block(cancers.get(n["patient_id"], [])) if linking else ""
        for n in notes_list
    }
    inputs = {
        n["note_id"]: digest([n["patient_id"], n["note_date"], n["note_type"], n["text"],
                              contexts[n["note_id"]]])
        for n in notes_list
    }
    path = Path(checkpoint_dir) / CHECKPOINT_FILE if checkpoint_dir is not None else None
    saved: dict[str, dict] = {}
    if path is not None and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if (
                entry.get("contract_sha256") == contract
                and entry.get("input_sha256") == inputs.get(entry.get("note_id"))
                and entry.get("extraction_status") in {"complete", "partial"}
            ):
                saved[entry["note_id"]] = entry

    outcomes: dict[str, dict] = {}
    attempts: dict[str, list] = {}
    pending = []
    for note in notes_list:
        note_id = note["note_id"]
        if note_id in saved:
            outcomes[note_id] = saved[note_id]
            attempts[note_id] = saved[note_id].get("attempts", [])
        elif len(note["text"]) > limit:
            outcomes[note_id] = {"extraction_status": "failed", "record": None,
                                 "error": f"document exceeds {SECTION}.max_document_characters"}
        else:
            pending.append(note)
    reused = len(saved)
    by_id = {n["note_id"]: n for n in notes_list}

    def messages(note):
        content = fill(template, {
            "field_guide": guide,
            "extension_guidance": f"\n{guidance}\n" if guidance else "",
            "linking": f"\n{contexts[note['note_id']]}" if contexts[note["note_id"]] else "",
            "note_date": note["note_date"] or "unknown",
            "note_type": note["note_type"] or "none",
            "note_text": note["text"],
        })
        return [{"role": "user", "content": content}]

    finished: list[str] = []

    def write(note_ids):
        if path is None or not note_ids:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as handle:
            for note_id in note_ids:
                outcome = outcomes[note_id]
                handle.write(json.dumps({
                    "note_id": note_id, "patient_id": by_id[note_id]["patient_id"],
                    "schema_version": SCHEMA_VERSION, "contract_sha256": contract,
                    "input_sha256": inputs[note_id], **outcome,
                    "attempts": attempts.get(note_id, []), "completed_at": now,
                }, ensure_ascii=False) + "\n")

    model_metadata: dict[str, Any] = {}
    if pending:
        queue = RequestQueue(
            get_llm_backend(resolved), settings.runtimes, chunk_size=settings.batch_size,
            chunks_in_flight=settings.batches_in_flight,
            model_metadata_cache_dir=resolved.model_metadata_cache_dir,
        )
        # Prompts are built as requests are sent, so a large run never holds
        # every rendered prompt at once.
        waiting = iter(pending)

        def feed():
            note = next(waiting, None)
            if note is not None:
                original = messages(note)
                queue.add((note["note_id"], 1, original), original, "main")

        for _ in range(2 * settings.batch_size * settings.batches_in_flight):
            feed()

        def record(item, answer, reasoning, finish_reason, provenance):
            note_id, attempt, original = item
            note = by_id[note_id]
            group = attempt_group(attempt, settings)
            entry = attempt_entry(attempt, group, finish_reason, provenance, debug=debug,
                                  answer=answer, reasoning=reasoning)
            attempts.setdefault(note_id, []).append(entry)
            if finish_reason == "error":
                entry["error"] = "request failed"
                outcomes[note_id] = {"extraction_status": "failed", "record": None,
                                     "error": "the LLM request failed after its transport retries"}
                finished.append(note_id)
                feed()
                return
            final = attempt >= settings.total_attempts
            try:
                value = parse_answer(answer, finish_reason)
            except ValueError as exc:
                check, error = None, str(exc)
            else:
                check, error = check_document_answer(
                    value, note, item_types, linking=linking,
                    known_ids={c["item_id"] for c in cancers.get(note["patient_id"], [])},
                ), None
            problems = [] if check is None else check.problems
            if error is None and not problems:
                status = "complete"
            elif error is None and final and not check.structural:
                status = "partial"
            else:
                entry["problems"] = [error] if error else [p.code for p in problems]
                if not final:
                    queue.add((note_id, attempt + 1, original),
                              retry_messages(original, answer, check is not None, problems, error),
                              attempt_group(attempt + 1, settings))
                    return
                outcomes[note_id] = {"extraction_status": "failed", "record": None,
                                     "error": "no valid answer after every attempt: "
                                     + "; ".join(entry["problems"][:5])}
                finished.append(note_id)
                feed()
                return
            built = build_document_record(note, value, check, item_types, extensions)
            if settings.ncit_agent_codes:
                add_ncit_codes(built, ncit_resource)
            outcome = {"extraction_status": status, "record": built, "error": None}
            if status == "partial":
                entry["problems"] = [p.code for p in check.grounding]
                outcome["dropped_items"] = check.dropped
                outcome["dropped_quotes"] = len(check.grounding)
            outcomes[note_id] = outcome
            finished.append(note_id)
            feed()

        def flush():
            write(finished)
            finished.clear()

        logger.info("Extracting document records for %d notes.", len(pending))
        model_metadata = queue.run(record, progress=flush)

    rows = []
    for note in notes_list:
        outcome = outcomes[note["note_id"]]
        built = outcome.get("record")
        rows.append({
            "patient_id": note["patient_id"], "note_id": note["note_id"],
            "note_date": note["note_date"], "note_type": note["note_type"],
            "document_type": built["document_type"] if built else None,
            "extraction_status": outcome["extraction_status"], "error": outcome.get("error"),
            "item_count": sum(map(len, built["items"].values())) if built else 0,
            "record": built,
        })
    result = pd.DataFrame(rows, columns=DOCUMENT_COLUMNS)
    statuses = result.set_index("note_id").extraction_status if len(result) else pd.Series(dtype=str)
    metadata = {
        "package": package_metadata(),
        "config_snapshot": config_snapshot(resolved),
        "schema_version": SCHEMA_VERSION,
        "contract_sha256": contract,
        "prompt_sha256": digest(template),
        "response_schema_sha256": digest(schema),
        "extensions": [e.name for e in extensions],
        "linking": linking,
        "structured_outputs": settings.structured,
        "model_metadata": model_metadata,
        "counts": {
            "documents": len(notes_list), "reused": reused,
            **{s: int((statuses == s).sum()) for s in ("complete", "partial", "failed")},
            **counts,
        },
        "documents": {
            n["note_id"]: {
                "extraction_status": outcomes[n["note_id"]]["extraction_status"],
                "attempts": attempts.get(n["note_id"], []),
                **{k: outcomes[n["note_id"]][k] for k in ("dropped_items", "dropped_quotes")
                   if k in outcomes[n["note_id"]]},
            }
            for n in notes_list
        },
    }
    outputs: list[Any] = [result]
    if return_metadata:
        outputs.append(metadata)
    if return_qc:
        truncated = [i for i, a in attempts.items() if any(e["finish_reason"] == "length" for e in a)]
        outputs.append(pd.DataFrame(qc_rows([
            ("documents_failed_extraction", list(statuses[statuses == "failed"].index)),
            ("documents_partial_extraction", list(statuses[statuses == "partial"].index)),
            ("documents_truncated_llm_response", truncated),
        ], len(notes_list))))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


__all__ = [
    "CHECKPOINT_FILE",
    "DOCUMENT_COLUMNS",
    "build_document_record",
    "check_document_answer",
    "extract_document_records",
    "known_cancers",
    "linking_block",
]
