"""Run accepted workup SQL queries over a cohort and report like the compiled-query method.

Each requirement's query runs ONCE over the cohort's whole extraction database
(one row per patient), in the isolated worker, with no model calls. Results have
the shape of :func:`~matchminer_ai.patients.workup_queries.check_workup_with_extractions`
(statuses met, not_met, unknown, not_applicable; applicability; reasons;
evidence quotes; limitations), so dashboards, drilldowns and exports read both.
Code keeps the same overrides: no records gives unknown, an incomplete
extraction turns not_met into unknown, and an unexpressible requirement or a
failed query gives unknown. A patient whose records were extracted without an
item type the query reads (an older extraction schema, such as one without lab
tests or counseling) also gets unknown instead of not_met. A requirement that
advises against its step reports a documented step as done_not_recommended.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence

from matchminer_ai.cancellation import check_cancelled
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.patients.workup_queries.evaluate import (
    MAX_EVIDENCE,
    PatientData,
    _date_text,
    _incomplete,
    aggregate_extraction_workup,
    describe_item,
)

from .agent import WorkupSQLLimits, agent_limits, run_final_query
from .database import ExtractionDatabase, build_extraction_database
from .sandbox import SQLSandbox, SQLSandboxError

METHOD = "extraction_sql"
NOTICE = (
    "Structured-extraction workup check for human review, not a determination. An LLM agent "
    "wrote each SQL query after exploring extracted items, and code ran it over the cohort's "
    "items that an LLM extracted from the notes; check every item against its source note. "
    "Not documented means no extracted item matched, not that the step was not done. This "
    "does not establish overall guideline concordance."
)
_LEADS = {"met": "Met", "not_met": "Not documented in the extraction records",
          "unknown": "Unknown", "not_applicable": "Not applicable",
          "done_not_recommended": "Done though the guideline advises against it"}
_STATES = {"met": "yes", "unknown": "unknown", "not_applicable": "yes",
           "done_not_recommended": "yes"}


def _result(status, reason, applicability, bottom_line, *, entry=None, evidence=None,
            limitations=None, reference_date=None) -> dict:
    return {
        "status": status,
        "reason": reason,
        "applicability": applicability,
        "bottom_line": bottom_line,
        "criteria": [],
        "evidence": evidence or [],
        "limitations": limitations if limitations is not None else list(
            (entry or {}).get("approximations") or []),
        "reference_date": reference_date,
        "query_summary": (entry or {}).get("summary") or None,
    }


def _evidence(views, state) -> list[dict]:
    evidence, seen = [], set()
    for key, view in views:
        for quote in view.item.get("evidence") or []:
            mark = (quote.get("note_id"), quote.get("start"), quote.get("end"))
            if mark in seen:
                continue
            seen.add(mark)
            evidence.append({**quote, "state": state, "date_text": _date_text(view),
                             "item": describe_item(view), "level": view.level, "item_key": key})
    return evidence[:MAX_EVIDENCE]


def assess(row: dict | None, entry: dict, patient: PatientData, database: ExtractionDatabase,
           *, reference_date=None, failure: str | None = None) -> dict:
    """One patient's assessment of one requirement from its query's result row."""
    if entry.get("status") != "compiled":
        return _result("unknown", "query_unavailable", "uncertain",
                       "Unknown: no SQL query could be accepted for this requirement.",
                       entry=entry, reference_date=reference_date)
    if entry.get("expressible") is False:
        reason = (entry.get("not_expressible_reason") or "").strip()
        return _result("unknown", "not_expressible", "uncertain",
                       "Unknown: not checkable from extracted items" + (
                           f" ({reason})" if reason else "") + ".",
                       entry=entry, reference_date=reference_date)
    if failure is not None:
        return _result("unknown", "query_unavailable", "uncertain", f"Unknown: {failure}",
                       entry=entry, reference_date=reference_date)
    if not patient.has_records:
        return _result("unknown", "no_extraction", "uncertain",
                       "Unknown: no extraction records are available for this patient.",
                       entry=entry, reference_date=reference_date)
    status, reason, applicability = row["status"], row["reason"], row["applicability"]
    views = [(k, database.views[k]) for k in row["evidence_item_keys"] if k in database.views]
    state = "planned" if reason == "planned_only" else _STATES.get(status, "yes")
    evidence = _evidence(views, state)
    lead = ("Not documented as performed (planned or ordered only)" if reason == "planned_only"
            else _LEADS[status])
    detail = row["detail"].rstrip(".")
    text = f"{lead}: {detail}." if detail else f"{lead}."
    if views:
        text += " Items: " + "; ".join(f"{describe_item(v)}, {_date_text(v)}"
                                       for _, v in views[:3]) + "."
    limitations = list(entry.get("approximations") or [])
    missing = database.uncovered(patient.patient_id, entry.get("sql") or "")
    if status == "not_met" and missing:
        status, reason = "unknown", "incomplete_extraction"
        text = ("Unknown: not found, but this patient's records were extracted without "
                f"{', '.join(missing)} (an older extraction schema).")
    elif status == "not_met" and patient.incomplete:
        status, reason = "unknown", "incomplete_extraction"
        text = f"Unknown: not found in the extraction records, but {_incomplete(patient)}."
    elif reason == "not_documented":
        text += " A missing item does not mean the step was not done."
    if patient.incomplete and status != "unknown":
        limitations.append(_incomplete(patient).capitalize() + ".")
    return _result(status, reason, applicability, text, entry=entry, evidence=evidence,
                   limitations=limitations, reference_date=reference_date)


def check_workup_with_sql(
    records,
    compiled: Sequence[dict],
    *,
    patient_ids: Sequence[str] | None = None,
    reference_dates: Mapping[str, object] | str | None = None,
    coverage: Mapping[str, Mapping] | None = None,
    config: MMAIConfig | None = None,
    limits: WorkupSQLLimits | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Run every accepted query once over the cohort's records; one assessment per patient × item.

    ``records``, ``patient_ids``, ``reference_dates`` and ``coverage`` are as for
    :func:`~matchminer_ai.patients.workup_queries.check_workup_with_extractions`;
    ``compiled`` is :func:`compile_workup_sql` output, one entry per requirement
    (each with its ``requirement``). Returns ``{"patients", "aggregate", "notice",
    "queries"}``; ``queries`` reports how each query ran on this cohort.
    """
    limits = limits or agent_limits(config or load_default_preset())
    database = build_extraction_database(records, coverage=coverage, patient_ids=patient_ids,
                                         reference_dates=reference_dates)
    ids = database.patient_ids
    progress = progress_callback or (lambda done, total: None)
    runnable = [n for n, e in enumerate(compiled)
                if e.get("status") == "compiled" and e.get("expressible") is not False
                and isinstance(e.get("sql"), str)]
    rows: dict[int, dict] = {}
    failures: dict[int, str] = {}
    runs = [{"status": "not_run"} for _ in compiled]
    progress(0, len(compiled))
    if runnable and ids:
        with SQLSandbox(database=database, limits=limits.sandbox()) as box:
            for done, number in enumerate(runnable, 1):
                check_cancelled()
                try:
                    check = run_final_query(
                        box, compiled[number]["sql"], len(ids), limits,
                        recommends_against=bool(compiled[number].get("recommends_against")))
                except SQLSandboxError:
                    failures[number] = "the isolated SQL worker failed while running this query."
                    runs[number] = {"status": "failed", "problem": "worker_failure"}
                    break
                if check.problems:
                    failed = check.problems[0].startswith("The query failed")
                    failures[number] = ("the SQL query failed on this cohort's records."
                                        if failed else "the SQL query's output broke its "
                                        "contract on this cohort's records.")
                    runs[number] = {"status": "failed",
                                    "problem": "error" if failed else "contract"}
                else:
                    rows[number] = check.rows
                    runs[number] = {"status": "ran", "distribution": check.distribution}
                progress(done, len(compiled))
    for number in runnable:
        if runs[number]["status"] == "not_run":
            failures[number] = "the isolated SQL worker failed before this query ran."
    reference = {pid: row["reference_date"] for pid, row in
                 zip(ids, database.tables["patients"].to_pylist(), strict=True)}
    patients = []
    for key, pid in enumerate(ids, 1):
        patient = database.data.get(pid) or PatientData(pid)
        day = reference[pid].isoformat() if reference[pid] else None
        assessments = []
        for number, entry in enumerate(compiled):
            result = assess((rows.get(number) or {}).get(key), entry, patient, database,
                            reference_date=day, failure=failures.get(number))
            assessments.append({
                "recommendation_index": number,
                "recommendation": copy.deepcopy(entry["requirement"]),
                "name": entry["requirement"]["name"],
                **result,
                "query_key": entry.get("cache_key"),
                "sql": entry.get("sql"),
                "recommends_against": bool(entry.get("recommends_against")),
            })
        patients.append({
            "patient_id": pid,
            "review": {"assessments": assessments, "notice": NOTICE,
                       "metadata": {"method": METHOD, "coverage": patient.coverage()}},
        })
    requirements = [entry["requirement"] for entry in compiled]
    progress(len(compiled), len(compiled))
    return {"patients": patients, "aggregate": aggregate_extraction_workup(patients, requirements),
            "notice": NOTICE, "queries": runs}


__all__ = ["METHOD", "NOTICE", "assess", "check_workup_with_sql"]
