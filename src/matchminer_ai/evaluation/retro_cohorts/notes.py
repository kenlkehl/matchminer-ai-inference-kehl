"""Reports available before each index date, and v22's note-level dataset.

v22 prepare_data.py (both cohorts) built ``note_level_dataset.parquet``: for each index
event, every imaging report, clinical note and pathology report of the patient dated
before ``trial_start_dt + days_buffer`` (default 5 days). Index events with no such report
produced no rows, so they never reached summarization or evaluation: an implicit
exclusion. ``count_prior_reports`` reproduces that cutoff on dates alone (no note text is
read) so the denominators can carry ``n_prior_reports``; ``build_note_level_dataset``
writes the note text itself and is a separate, explicit step.
"""

from __future__ import annotations

import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .trial_masking import RESIDUAL_CHECKS, TrialIdentifiers, event_trial_identifiers, mask_trial_identifiers, residual_flags

REPORT_FILES = {
    "imaging": "all_imaging_reports.parquet",
    "clinical": "all_clinical_notes.parquet",
    "path": "all_path_reports.parquet",
    # New in pan_dfci_2026; not used by v22 and off by default.
    "radonc": "all_rad_completion.parquet",
}
DEFAULT_SOURCES = ("imaging", "clinical", "path")

_EPOCH = np.datetime64("1800-01-01", "D")
_DAY_SPAN = 100_000  # days after _EPOCH representable in a key (through 2073)


def _day_numbers(dates) -> np.ndarray:
    days = np.asarray(dates, dtype="datetime64[ns]").astype("datetime64[D]")
    return (days - _EPOCH).astype(np.int64)


def load_report_dates(derived_data_dir, sources=DEFAULT_SOURCES, patients=None) -> tuple[pd.DataFrame, dict]:
    """(dfci_mrn, report day) for every report in ``sources``; reads two columns only."""
    frames, stats = [], {}
    for source in sources:
        path = Path(derived_data_dir) / REPORT_FILES[source]
        table = pq.read_table(path, columns=["dfci_mrn", "date"])
        frame = table.to_pandas()
        n = len(frame)
        if patients is not None:
            frame = frame[frame.dfci_mrn.isin(patients)]
        frame = frame[frame.date.notna()]
        stats[source] = {"file": str(path), "reports": int(n), "reports_for_cohort_patients": int(len(frame))}
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), stats


def count_prior_reports(events: pd.DataFrame, report_dates: pd.DataFrame, days_buffer: int = 5) -> np.ndarray:
    """Reports per event dated before ``trial_start_dt + days_buffer`` (v22's cutoff, exclusive).

    v22 compared report timestamps with the midnight cutoff (``date < trial_start + 5 days``);
    comparing calendar days is equivalent. Reports dated outside 1800-2073 are ignored.
    """
    report_day = _day_numbers(report_dates.date.values)
    ok = (report_day >= 0) & (report_day < _DAY_SPAN)
    keys = np.sort(report_dates.dfci_mrn.values[ok].astype(np.int64) * _DAY_SPAN + report_day[ok])
    cutoff = _day_numbers((events.trial_start_dt + pd.Timedelta(days=days_buffer)).values)
    if (cutoff < 0).any() or (cutoff >= _DAY_SPAN).any():
        raise ValueError("Index date outside the supported range")
    base = events.dfci_mrn.values.astype(np.int64) * _DAY_SPAN
    return np.searchsorted(keys, base + cutoff, side="left") - np.searchsorted(keys, base, side="left")


_REPORT_COLUMNS = ["dfci_mrn", "date", "note_type", "RPT_ID", "text"]
_OUTPUT_COLUMNS = ["pseudo_mrn", "dfci_mrn", "trial_start_dt", "note_cutoff_dt", "date", "note_type", "RPT_ID", "text"]
MASK_COLUMNS = {"protocol": "n_masked_protocol", "nct": "n_masked_nct", "code": "n_masked_code"}
_NO_IDS = TrialIdentifiers()
_STATE: dict = {}


def _output_schema(mask: bool) -> pa.Schema:
    fields = [("pseudo_mrn", pa.int64()), ("dfci_mrn", pa.int64()), ("trial_start_dt", pa.timestamp("ns")),
              ("note_cutoff_dt", pa.timestamp("ns")), ("date", pa.timestamp("ns")), ("note_type", pa.string()),
              ("RPT_ID", pa.string()), ("text", pa.string())]
    return pa.schema(fields + ([(c, pa.int32()) for c in MASK_COLUMNS.values()] if mask else []))


def _init_worker(state: dict) -> None:
    pa.set_cpu_count(1)  # one process per row group: no extra Arrow threads per worker
    _STATE.clear()
    _STATE.update(state)


def _select_and_mask(table: pa.Table, state: dict) -> tuple[pd.DataFrame | None, Counter, Counter]:
    """Rows of one report table kept for each index event (date < cutoff), with that event's identifiers masked."""
    meta = table.select(["dfci_mrn", "date"]).to_pandas()
    meta["_row"] = np.arange(len(meta))
    meta = meta[meta.dfci_mrn.isin(state["wanted"]) & meta.date.notna()]
    forms, codes = Counter(), Counter()
    if meta.empty:
        return None, forms, codes
    merged = meta.merge(state["events"], on="dfci_mrn", how="inner")
    merged = merged[merged.date < merged.cutoff].reset_index(drop=True)
    if merged.empty:
        return None, forms, codes
    # Only the kept rows' text leaves Arrow (a note shared by several events of one patient is taken once per event).
    rest = table.select(["note_type", "RPT_ID", "text"]).take(pa.array(merged["_row"].to_numpy()))
    out = merged.rename(columns={"cutoff": "note_cutoff_dt"})[_OUTPUT_COLUMNS[:5]].assign(
        note_type=rest["note_type"].to_pylist(), RPT_ID=rest["RPT_ID"].to_pylist())
    texts = rest["text"].to_pylist()
    ids = state["ids"]
    if ids is None:
        out["text"] = texts
    else:
        masked = [mask_trial_identifiers(t, ids.get(int(p), _NO_IDS), forms, codes)
                  for t, p in zip(texts, out.pseudo_mrn.tolist())]
        out["text"] = [m[0] for m in masked]
        for i, column in enumerate(MASK_COLUMNS.values(), 1):
            out[column] = np.array([m[i] for m in masked], dtype=np.int32)
    return out, forms, codes


def _row_group_task(task):
    source, path, row_group = task
    table = pq.ParquetFile(path).read_row_group(row_group, columns=_REPORT_COLUMNS, use_threads=False)
    return (source, *_select_and_mask(table, _STATE))


def _pool(workers: int, state: dict) -> ProcessPoolExecutor:
    # spawn, not fork: the parent has live Arrow threads, and fork after threads can deadlock.
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker,
                               initargs=(state,))


def build_note_level_dataset(events: pd.DataFrame, derived_data_dir, output_parquet, days_offset: int,
                             sources=DEFAULT_SOURCES, batch_rows: int = 200_000, anchor_col: str = "trial_start_dt",
                             mask: bool = False, titles=None, workers: int = 1, progress=None) -> dict:
    """Write the note-level dataset: one row per (index event, report dated before its cutoff).

    The cutoff is ``anchor + days_offset`` days, exclusive (``date < cutoff``). v22 used the index date as anchor and
    +5 days (its enrollment index was the OnCore consent date, its SOC index the plan start). For the research-plan
    trial cohort the anchor should be ``note_anchor_dt``, the earliest known registration, because registration and
    screening notes name the trial. ``events`` needs pseudo_mrn, dfci_mrn, trial_start_dt and ``anchor_col``. An
    event with several rows (several protocols starting the same day) takes its earliest anchor.

    ``mask=True`` (trial cohort; needs protocol_number and nct_id in ``events`` and ``titles``, NCT ID -> ctgov
    title) masks, in each kept note, the identifiers of all of that event's own trials (see trial_masking.py) and
    adds ``n_masked_protocol``, ``n_masked_nct`` and ``n_masked_code``. Other NN-NNN patterns are left alone.

    ``workers > 1`` reads one report row group per process (spawned); ``workers == 1`` streams ``batch_rows`` rows
    at a time in this process. Output rows are in source, then file order either way.

    Differences from v22, which looped over every enrollment row: each index event gets one copy of its reports
    (v22 repeated them for each row sharing a pseudo_mrn), trial or plan details are not copied onto every note
    (join on pseudo_mrn instead), only the columns shared by all report files are kept, and rows are not globally
    sorted (the summarizer sorts by patient and date). Streams each file once. Returns counts only.
    """
    if type(days_offset) is not int:
        raise ValueError("days_offset must be an explicit integer (negative = before the anchor)")
    ids, id_counts = None, None
    if mask:
        missing = {"protocol_number", "nct_id"} - set(events.columns)
        if missing:
            raise ValueError(f"Masking needs {sorted(missing)} in the denominator (the trial cohort); pass mask=False")
        if titles is None:
            raise ValueError("Masking needs trial titles (NCT ID -> ctgov title) for the title codes")
        ids, id_counts = event_trial_identifiers(events, titles)
    anchors = events.groupby("pseudo_mrn")[anchor_col].min()
    events = events[["pseudo_mrn", "dfci_mrn", "trial_start_dt"]].drop_duplicates("pseudo_mrn").copy()
    events["cutoff"] = events.pseudo_mrn.map(anchors) + pd.Timedelta(days=days_offset)
    if events.cutoff.isna().any():
        raise ValueError(f"{int(events.cutoff.isna().sum())} index events have no {anchor_col}")
    events["dfci_mrn"] = events.dfci_mrn.astype("int64")
    state = {"events": events, "wanted": np.unique(events.dfci_mrn.to_numpy()), "ids": ids}
    schema = _output_schema(mask)
    paths = {source: Path(derived_data_dir) / REPORT_FILES[source] for source in sources}

    def results():
        if workers <= 1:
            for source, path in paths.items():
                for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=_REPORT_COLUMNS):
                    yield (source, *_select_and_mask(pa.Table.from_batches([batch]), state))
            return
        tasks = [(source, str(path), g) for source, path in paths.items()
                 for g in range(pq.ParquetFile(path).num_row_groups)]
        with _pool(workers, state) as pool:
            for n, result in enumerate(pool.map(_row_group_task, tasks, chunksize=1), 1):
                if progress and (n % 25 == 0 or n == len(tasks)):
                    progress(f"{n}/{len(tasks)} report row groups done")
                yield result

    output_parquet = Path(output_parquet)
    tmp = output_parquet.with_name(output_parquet.name + ".tmp")
    per_source = dict.fromkeys(sources, 0)
    seen = set()
    totals = {k: {"replacements": 0, "notes": 0, "index_events": set()} for k in MASK_COLUMNS}
    forms, codes = Counter(), Counter()
    try:
        with pq.ParquetWriter(tmp, schema) as writer:
            os.chmod(tmp, 0o600)
            for source, out, task_forms, task_codes in results():
                if out is None or out.empty:
                    continue
                writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
                per_source[source] += len(out)
                seen.update(out.pseudo_mrn.tolist())
                if mask:
                    forms.update(task_forms)
                    codes.update(task_codes)
                    for kind, column in MASK_COLUMNS.items():
                        hit = out[column].to_numpy() > 0
                        totals[kind]["replacements"] += int(out[column].sum())
                        totals[kind]["notes"] += int(hit.sum())
                        totals[kind]["index_events"].update(out.pseudo_mrn.to_numpy()[hit].tolist())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.chmod(0o600)
    tmp.replace(output_parquet)
    stats = {"rows": int(sum(per_source.values())), "rows_by_source": {k: int(v) for k, v in per_source.items()},
             "index_events": int(len(events)), "index_events_with_notes": int(len(seen)),
             "index_events_without_notes": int(len(events) - len(seen)), "anchor": anchor_col,
             "days_offset": days_offset, "mask": bool(mask)}
    if mask:
        any_events = set().union(*(t["index_events"] for t in totals.values()))
        stats["masking"] = {
            "identifiers": id_counts,
            "replacements": {k: {"replacements": t["replacements"], "notes": t["notes"],
                                 "index_events": len(t["index_events"])} for k, t in totals.items()},
            "index_events_with_any_replacement": len(any_events),
            "protocol_spellings_masked": dict(forms.most_common()),
            "code_replacements_by_code": dict(codes.most_common()),
        }
    return stats


def _residual_task(task):
    path, row_group = task
    table = pq.ParquetFile(path).read_row_group(row_group, columns=["pseudo_mrn", "text"], use_threads=False)
    ids = _STATE["ids"]
    notes = dict.fromkeys(RESIDUAL_CHECKS, 0)
    events = {k: set() for k in RESIDUAL_CHECKS}
    pseudo = table["pseudo_mrn"].to_pylist()
    for p, text in zip(pseudo, table["text"].to_pylist()):
        for check, hit in residual_flags(text, ids.get(p, _NO_IDS)).items():
            if hit:
                notes[check] += 1
                events[check].add(p)
    return len(pseudo), notes, events


def scan_residual_identifiers(parquet_path, ids: dict, workers: int = 1, progress=None) -> dict:
    """Re-run the leakage checks over a written note-level dataset: notes and index events still containing one of
    their own trial identifiers (``trial_masking.residual_flags``; counts only)."""
    path = str(parquet_path)
    tasks = [(path, g) for g in range(pq.ParquetFile(path).num_row_groups)]
    rows, notes, events = 0, dict.fromkeys(RESIDUAL_CHECKS, 0), {k: set() for k in RESIDUAL_CHECKS}

    def merge(result):
        nonlocal rows
        n, task_notes, task_events = result
        rows += n
        for k in RESIDUAL_CHECKS:
            notes[k] += task_notes[k]
            events[k] |= task_events[k]

    if workers <= 1:
        _STATE.clear()
        _STATE["ids"] = ids
        for task in tasks:
            merge(_residual_task(task))
    else:
        with _pool(workers, {"ids": ids}) as pool:
            for n, result in enumerate(pool.map(_residual_task, tasks, chunksize=1), 1):
                merge(result)
                if progress and (n % 50 == 0 or n == len(tasks)):
                    progress(f"{n}/{len(tasks)} output row groups checked")
    return {"rows_scanned": rows,
            "checks": {k: {"notes": notes[k], "index_events": len(events[k])} for k in RESIDUAL_CHECKS}}
