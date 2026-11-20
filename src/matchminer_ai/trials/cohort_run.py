"""Reading a cohort extraction run's results while it runs and after: one trial's per-trial files, or the run's
final ``cohorts.parquet``.

``process_trials`` (with ``output_dir``) writes each trial's step results as they complete, atomically:

- ``<trial>/extraction.json`` (protocols): the cancer-treatment screen and the extracted cohorts (``decision``);
- ``<trial>/merge.json``: which extracted cohorts each merged cohort stands for (``groups``);
- ``<trial>/deboilerplate.json``: the merged cohorts with key and boilerplate criteria and every criteria line
  tagged inclusion/exclusion and key/boilerplate, written once all of the trial's lines are answered, so the file is
  a complete result.

``read_trial_results`` turns them into one trial's state and cohorts, falling back to the run's
``cohorts.parquet`` rows (``read_run_cohorts``) for a trial without per-trial files.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable

RUN_COLUMNS = ("trial_id", "cohort_number", "cohort_name", "merged_from", "cohort_key_criteria",
               "cohort_boilerplate_criteria", "cohort_criteria_items")


def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _list(value):
    """``merged_from`` as a list (the run stores a list, or its repr or JSON text)."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str) and value:
        for parse in (ast.literal_eval, json.loads):
            try:
                parsed = parse(value)
                return list(parsed) if isinstance(parsed, (list, tuple)) else None
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                continue
    return None


def cohort_items(entry: dict) -> list[dict]:
    """A ``deboilerplate.json`` cohort's lines in the ``cohort_criteria_items`` form the decision model reads."""
    return [{"number": int(i["number"]), "heading": i.get("label") or "", "text": i.get("text") or "",
             "criterion_type": i.get("criterion_type"), "boilerplate": bool(i.get("boilerplate"))}
            for i in entry.get("items") or [] if "text" in i]


def criteria_from_items(items: list[dict], boilerplate: bool | None = None) -> str:
    """The lines (all, or the key or boilerplate ones) under their headings, as the package lays criteria out."""
    lines, heading = [], None
    for item in items:
        if boilerplate is not None and bool(item["boilerplate"]) != boilerplate:
            continue
        if item["heading"] and item["heading"] != heading:
            lines.append(item["heading"])
        heading = item["heading"]
        lines.append(item["text"])
    return "\n".join(lines)


def read_run_cohorts(run_dir: str | Path) -> dict[str, list[dict]]:
    """The run's final ``cohorts.parquet`` rows by trial ID (``RUN_COLUMNS``); empty when the run has none yet."""
    path = Path(run_dir) / "cohorts.parquet"
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    rows: dict[str, list[dict]] = {}
    for row in pq.read_table(path, columns=list(RUN_COLUMNS)).to_pylist():
        rows.setdefault(row["trial_id"], []).append(row)
    return rows


def read_trial_results(trial_dir: str | Path, *, screened: bool = True, fallback_rows: list[dict] | None = None,
                       read_json: Callable[[Path], object] | None = None) -> dict:
    """One trial's results in a run: ``{"state", "cohorts", "completed_at", "screen", "extracted_cohorts"}``.

    ``state`` is ``complete`` (its ``deboilerplate.json``, or else its ``fallback_rows`` from the run's
    ``cohorts.parquet``), ``stale`` (a newer extraction is on its way), ``not_treatment`` (the screen said no),
    ``extracting`` (extracted, not yet finished) or ``pending``. Each cohort is ``{"number", "name", "key",
    "boilerplate", "items", "merged_from"}``. ``screened``: the run's trials have ``extraction.json`` (protocols).
    ``read_json`` reads a file, or returns None for a missing or half-written one (default: read it now).
    """
    read = read_json or _read_json
    folder = Path(trial_dir)
    extraction = read(folder / "extraction.json") if screened else None
    merge = read(folder / "merge.json")
    final = read(folder / "deboilerplate.json")
    out: dict = {"state": "pending", "cohorts": [], "completed_at": None, "screen": None, "extracted_cohorts": None}
    if isinstance(extraction, dict) and extraction:
        decision = extraction.get("decision") or {}
        out["screen"] = decision.get("cancer_treatment_trial")
        out["extracted_cohorts"] = len(decision.get("cohorts") or [])
        if decision.get("cancer_treatment_trial") == "no":
            out.update(state="not_treatment", completed_at=extraction.get("completed_at"))
            return out
    groups = (merge or {}).get("groups") or [] if isinstance(merge, dict) else []
    if isinstance(final, dict) and final.get("cohorts") is not None:
        stale = bool(extraction and (extraction.get("completed_at") or "") > (final.get("completed_at") or ""))
        out.update(state="stale" if stale else "complete", completed_at=final.get("completed_at"))
        for entry in final["cohorts"]:
            items = cohort_items(entry)
            number = int(entry["cohort_number"])
            if "key_criteria" in entry:
                key, common = entry.get("key_criteria") or "", entry.get("boilerplate_criteria") or ""
            else:  # every line was classified as boilerplate: the criteria are kept whole, as the package does
                key, common = criteria_from_items(items), ""
            out["cohorts"].append({
                "number": number, "name": entry.get("cohort_name") or f"Cohort {number + 1}",
                "key": key, "boilerplate": common, "items": items,
                "merged_from": len(groups[number]) if number < len(groups) else 1})
        return out
    if fallback_rows:
        out.update(state="complete", completed_at=None)
        for row in fallback_rows:
            items = json.loads(row["cohort_criteria_items"]) if row.get("cohort_criteria_items") else []
            merged = _list(row.get("merged_from"))
            out["cohorts"].append({"number": int(row["cohort_number"]), "name": row.get("cohort_name") or "",
                                   "key": row.get("cohort_key_criteria") or "",
                                   "boilerplate": row.get("cohort_boilerplate_criteria") or "", "items": items,
                                   "merged_from": len(merged) if merged else 1})
    elif extraction:
        out["state"] = "extracting"
    return out
