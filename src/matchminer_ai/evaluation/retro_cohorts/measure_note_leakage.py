#!/usr/bin/env python3
"""How often do pre-index notes name the patient's own trial? Choose the note cutoff and masking from this.

User, 2026-11-14: "measure. i'm leaning towards -14 but also masking the protocol number at least via regex, but
measure first."

For each trial start in processed_trial_enrollments.csv (research-plan cohort), every imaging report, clinical note
and pathology report of the patient dated before the index (trial_start_dt) is checked for:
- the trial's own protocol number, in any written form NN-NNN (hyphen or dash, optional spaces, an optional DFCI or
  protocol prefix): ``protocol``;
- the trial's own NCT ID: ``nct``;
- an investigational code from the trial's own ctgov title (for example ABC-1234 or ABC 1234), a proxy for
  drug-name leakage that protocol masking does not catch: ``title_code``.

Each mention is placed by days from the note anchor (note_anchor_dt, the earliest known registration). The summary
reports, for cutoffs anchor + c (c = 0, -7, -14, -30, -60, -90) and for "index" (v22-like: every note before the
index):
- the share of starts with any mention in the notes that cutoff keeps;
- how many kept notes contain one (the masking workload).

It also counts kept notes with some other NN-NNN pattern, to size the collateral damage of masking every such
pattern. Counts only: no note text, identifiers or dates are printed or written. Writes note_leakage_summary.json
next to the denominator CSV (mode 600). CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from matchminer_ai.evaluation.retro_cohorts.io_utils import write_json_private
from matchminer_ai.evaluation.retro_cohorts.notes import REPORT_FILES
# The leakage regexes live in the package so build_note_level_dataset.py masks exactly what this script measures.
from matchminer_ai.evaluation.retro_cohorts.trial_masking import CODE, NCT, PROTOCOL, title_codes  # noqa: F401

CUTOFFS = [0, -7, -14, -30, -60, -90]
_EVENTS = None


def _init(events_path):
    global _EVENTS
    _EVENTS = pd.read_pickle(events_path)


def _scan(task):
    path, row_group, source = task
    events = _EVENTS
    table = pq.ParquetFile(path).read_row_group(row_group, columns=["dfci_mrn", "date", "text"])
    notes = table.to_pandas()
    notes = notes[notes.dfci_mrn.isin(events.dfci_mrn.unique()) & notes.date.notna()]
    if notes.empty:
        return [], {}
    notes = notes.assign(day=notes.date.dt.normalize()).merge(events, on="dfci_mrn")
    notes = notes[notes.day < notes.trial_start_dt]
    if notes.empty:
        return [], {}
    notes["rel"] = (notes.day - notes.note_anchor_dt).dt.days
    text = notes.text.fillna("")
    keys = text.map(lambda t: {int(a + b) for a, b in PROTOCOL.findall(t)})
    ncts = text.map(lambda t: {re.sub(r"\D", "", m).lstrip("0") for m in NCT.findall(t)})
    codes = text.map(lambda t: {(a + b).upper() for a, b in CODE.findall(t)})
    own_protocol = [k in s for k, s in zip(notes.protocol_key, keys)]
    own_nct = [isinstance(n, str) and n in s for n, s in zip(notes.nct_digits, ncts)]
    own_code = [bool(c) and bool(set(c) & s) for c, s in zip(notes.codes, codes)]
    other_pattern = [bool(s - {k}) for k, s in zip(notes.protocol_key, keys)]
    notes = notes.assign(own_protocol=own_protocol, own_nct=own_nct, own_code=own_code, other_pattern=other_pattern)
    mentions = notes[notes.own_protocol | notes.own_nct | notes.own_code][
        ["pseudo_mrn", "rel", "own_protocol", "own_nct", "own_code"]].assign(source=source)
    # Per cutoff, kept notes (and kept notes with another NN-NNN pattern), summed over events.
    counts = {}
    for c in CUTOFFS:
        kept = notes.rel < c
        counts[str(c)] = [int(kept.sum()), int((kept & notes.other_pattern).sum())]
    counts["index"] = [int(len(notes)), int(notes.other_pattern.sum())]
    return mentions.to_dict("records"), counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--denominator-csv", type=Path, required=True)
    p.add_argument("--derived-data-dir", type=Path,
                   help="Default: derived_data under the site config's paths.retro_source_root")
    p.add_argument("--ctgov-csv", type=Path, help="Default: <data root>/no_phi/ctgov_trials.csv")
    p.add_argument("--sources", default="clinical,imaging,path")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args(argv)
    from matchminer_ai.site import load_site_config
    from matchminer_ai.training.paths import no_phi
    args.derived_data_dir = args.derived_data_dir or Path(load_site_config().require("paths.retro_source_root")) / "derived_data"
    args.ctgov_csv = args.ctgov_csv or no_phi("ctgov_trials.csv")
    os.umask(0o077)
    events = pd.read_csv(args.denominator_csv, usecols=["pseudo_mrn", "dfci_mrn", "protocol_number", "nct_id",
                                                        "trial_start_dt", "note_anchor_dt"],
                         parse_dates=["trial_start_dt", "note_anchor_dt"])
    events["protocol_key"] = events.protocol_number.str.replace("-", "", regex=False).astype("int64")
    events["nct_digits"] = events.nct_id.str.replace("NCT", "", regex=False).str.lstrip("0")
    titles = pd.read_csv(args.ctgov_csv, usecols=["nct_id", "title"]).set_index("nct_id").title
    events["codes"] = events.nct_id.map(titles).map(title_codes)
    out_dir = args.denominator_csv.parent
    events_path = out_dir / ".leakage_events.pkl"
    events[["pseudo_mrn", "dfci_mrn", "protocol_key", "nct_digits", "codes", "trial_start_dt", "note_anchor_dt"]].to_pickle(events_path)
    events_path.chmod(0o600)
    tasks = []
    for source in [s.strip() for s in args.sources.split(",") if s.strip()]:
        path = args.derived_data_dir / REPORT_FILES[source]
        tasks += [(str(path), g, source) for g in range(pq.ParquetFile(path).num_row_groups)]
    mentions, kept = [], {}
    try:
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(events_path,)) as pool:
            for n, (rows, counts) in enumerate(pool.map(_scan, tasks, chunksize=1), 1):
                mentions += rows
                for c, (a, b) in counts.items():
                    kept.setdefault(c, [0, 0])
                    kept[c][0] += a
                    kept[c][1] += b
                if n % 50 == 0:
                    print(f"{n}/{len(tasks)} row groups scanned", flush=True)
    finally:
        events_path.unlink(missing_ok=True)

    m = pd.DataFrame(mentions, columns=["pseudo_mrn", "rel", "own_protocol", "own_nct", "own_code", "source"])
    n_events = int(events.pseudo_mrn.nunique())
    summary = {"events": n_events, "events_with_title_codes": int(events.codes.map(bool).sum()),
               "cutoff_definition": "notes dated before note_anchor_dt + c days are kept; 'index' keeps every note before trial_start_dt",
               "by_cutoff": {}}
    for c in [str(x) for x in CUTOFFS] + ["index"]:
        sel = m if c == "index" else m[m.rel < int(c)]
        row = {"kept_notes": kept.get(c, [0, 0])[0], "kept_notes_with_other_nn_nnn_pattern": kept.get(c, [0, 0])[1]}
        for kind in ("own_protocol", "own_nct", "own_code"):
            s = sel[sel[kind]]
            row[f"events_with_{kind}"] = int(s.pseudo_mrn.nunique())
            row[f"events_with_{kind}_pct"] = round(100 * s.pseudo_mrn.nunique() / n_events, 2)
            row[f"kept_notes_with_{kind}"] = int(len(s))
        any_ = sel[sel.own_protocol | sel.own_nct | sel.own_code]
        row["events_with_any_pct"] = round(100 * any_.pseudo_mrn.nunique() / n_events, 2)
        row["events_with_code_but_no_protocol_or_nct_pct"] = round(100 * len(
            set(sel[sel.own_code].pseudo_mrn) - set(sel[sel.own_protocol | sel.own_nct].pseudo_mrn)) / n_events, 2)
        row["own_protocol_notes_by_source"] = {k: int(v) for k, v in sel[sel.own_protocol].source.value_counts().items()}
        summary["by_cutoff"][c] = row
    # Where mentions fall relative to the anchor (days before registration), for own-protocol mentions.
    bins = [-10**6, -90, -60, -30, -14, -7, 0, 10**6]
    labels = ["<-90", "-90..-61", "-60..-31", "-30..-15", "-14..-8", "-7..-1", ">=0 (after registration, before index)"]
    own = m[m.own_protocol]
    summary["own_protocol_mentions_by_days_from_anchor"] = {
        str(k): int(v) for k, v in pd.cut(own.rel, bins=bins, labels=labels, right=False).value_counts().sort_index().items()}
    summary["events_first_own_protocol_mention_by_days_from_anchor"] = {
        str(k): int(v) for k, v in pd.cut(own.groupby("pseudo_mrn").rel.min(), bins=bins, labels=labels,
                                         right=False).value_counts().sort_index().items()}
    write_json_private(summary, out_dir / "note_leakage_summary.json")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
