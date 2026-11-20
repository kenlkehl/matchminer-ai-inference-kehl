#!/usr/bin/env python3
"""Build v22's note-level dataset (report text before each index date) for one denominator.

The second half of v22 prepare_data.py (create_note_level_dataset), kept as its own step
because it copies note text: run it only when summarization is planned, and write only to
a PHI directory. Reads processed_trial_enrollments.csv or processed_soc_treatments.csv from
pull_denominators.py and writes a parquet of (pseudo_mrn, dfci_mrn, trial_start_dt, date,
note_type, RPT_ID, text), mode 600. See mmai_real_eval/notes.py for differences from v22.

The cutoff is explicit: ``--days-offset`` (no default; negative = before the anchor) from ``--anchor``:
  index        the index date (v22: index + 5 days, with its consent-date enrollment index and plan-start SOC index)
  note_anchor  note_anchor_dt, the earliest known trial registration (research-plan trial cohort; default when the
               column is present). Registration and screening notes name the trial, so cut before them.

Trial identifiers are masked per index event by default when the CSV has protocol_number and nct_id (the trial
cohort): each kept note loses that event's own protocol number(s) -> [TRIAL PROTOCOL], NCT ID(s) -> [NCT ID] and
investigational codes from its trial title(s) -> [STUDY DRUG CODE] (mmai_real_eval/trial_masking.py), and gets
n_masked_protocol / n_masked_nct / n_masked_code. Other NN-NNN numbers are untouched. --no-mask turns it off; SOC
has neither column, so masking is off there. After a masked build the leakage checks are re-run on the output.

Every run writes <output stem>.summary.json (mode 600, counts only) beside the parquet.

Example (research-plan trial cohort, notes dated before registration minus 14 days, masked):
  python build_note_level_dataset.py --anchor note_anchor --days-offset -14 \
      --denominator-csv ../data/phi/real_eval_denominators_20261114/processed_trial_enrollments.csv \
      --output-parquet ../data/phi/real_eval_denominators_20261114/enrollments_note_level_dataset_minus14_masked.parquet
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa

from matchminer_ai.evaluation.retro_cohorts.io_utils import file_stamp, utc_now, write_json_private
from matchminer_ai.evaluation.retro_cohorts.notes import DEFAULT_SOURCES, REPORT_FILES, build_note_level_dataset, scan_residual_identifiers
from matchminer_ai.evaluation.retro_cohorts.trial_masking import event_trial_identifiers
from matchminer_ai.evaluation.retro_cohorts.pull_denominators import git_state
from matchminer_ai.site import load_site_config
from matchminer_ai.training.paths import no_phi, phi


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--denominator-csv", type=Path, required=True)
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--derived-data-dir", type=Path,
                   help="Default: derived_data under the site config's paths.retro_source_root")
    p.add_argument("--days-offset", type=int, required=True,
                   help="Cutoff = anchor + this many days, exclusive; negative = before the anchor (v22: +5 from index)")
    p.add_argument("--anchor", choices=["index", "note_anchor"], default=None,
                   help="index = trial_start_dt; note_anchor = note_anchor_dt (default when the CSV has it)")
    p.add_argument("--report-sources", default=",".join(DEFAULT_SOURCES), help=", ".join(REPORT_FILES))
    p.add_argument("--require-prior-reports", action="store_true",
                   help="Skip index events whose has_prior_reports is False (they produce no rows anyway)")
    p.add_argument("--no-mask", action="store_true",
                   help="Do not mask the event's own trial identifiers (masking is on whenever the CSV has "
                        "protocol_number and nct_id, i.e. the trial cohort)")
    p.add_argument("--ctgov-csv", type=Path,
                   help="nct_id,title: the trial titles investigational codes are taken from (masking only; default "
                        "<data root>/no_phi/ctgov_trials.csv)")
    p.add_argument("--workers", type=int, default=16,
                   help="Processes, one report row group each; 1 streams in this process")
    args = p.parse_args(argv)
    site = load_site_config()
    args.derived_data_dir = args.derived_data_dir or Path(site.require("paths.retro_source_root")) / "derived_data"
    args.ctgov_csv = args.ctgov_csv or no_phi("ctgov_trials.csv")
    os.umask(0o077)
    # Note text goes only to the data root's phi tree or a PHI directory the site config names (paths.phi_roots).
    allowed = [phi(), *map(Path, site.get("paths.phi_roots") or [])]
    if not any(args.output_parquet.resolve().is_relative_to(root.resolve()) for root in allowed):
        raise SystemExit("Refusing to write note text outside a PHI directory (<data root>/phi or paths.phi_roots)")
    t0 = time.time()
    events = pd.read_csv(args.denominator_csv, usecols=lambda c: c in {
        "pseudo_mrn", "dfci_mrn", "trial_start_dt", "note_anchor_dt", "has_prior_reports", "protocol_number", "nct_id"},
        dtype={"protocol_number": "string", "nct_id": "string"})
    events["trial_start_dt"] = pd.to_datetime(events.trial_start_dt)
    anchor = args.anchor or ("note_anchor" if "note_anchor_dt" in events else "index")
    if anchor == "note_anchor" and "note_anchor_dt" not in events:
        raise SystemExit("--anchor note_anchor needs a note_anchor_dt column (research-plan trial cohort)")
    anchor_col = "note_anchor_dt" if anchor == "note_anchor" else "trial_start_dt"
    events[anchor_col] = pd.to_datetime(events[anchor_col])
    if args.require_prior_reports and "has_prior_reports" in events:
        events = events[events.has_prior_reports.astype(str).str.lower().eq("true")]
    has_ids = {"protocol_number", "nct_id"} <= set(events.columns)
    mask = has_ids and not args.no_mask
    titles = None
    if mask:
        ctgov = pd.read_csv(args.ctgov_csv, usecols=["nct_id", "title"], dtype="string")
        ctgov = ctgov[ctgov.nct_id.isin(set(events.nct_id.dropna()))].drop_duplicates("nct_id")
        titles = {n: (t if isinstance(t, str) else None) for n, t in zip(ctgov.nct_id, ctgov.title.astype(object))}
        print("Masking each event's own protocol numbers, NCT IDs and title codes")
    else:
        print("Masking off: " + ("--no-mask" if has_ids else "no protocol_number/nct_id columns (SOC cohort)"))
    sources = [s.strip() for s in args.report_sources.split(",") if s.strip()]
    stats = build_note_level_dataset(events, args.derived_data_dir, args.output_parquet, args.days_offset, sources,
                                     anchor_col=anchor_col, mask=mask, titles=titles, workers=args.workers,
                                     progress=lambda m: print(m, flush=True))
    t_build = time.time()
    print(f"Wrote {stats['rows']} report rows for {stats['index_events']} index events "
          f"({stats['rows_by_source']}; cutoff = {anchor_col} {args.days_offset:+d} days; "
          f"{stats['index_events_without_notes']} events without notes) to {args.output_parquet}")
    summary = {
        "created_utc": utc_now(), "script": "build_note_level_dataset.py", "code": git_state(),
        "python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pa.__version__,
        "parameters": {"anchor": anchor_col, "days_offset": args.days_offset, "cutoff": f"date < {anchor_col} "
                       f"{args.days_offset:+d} days", "sources": sources, "mask": mask, "workers": args.workers,
                       "require_prior_reports": args.require_prior_reports},
        "inputs": {"denominator_csv": file_stamp(args.denominator_csv),
                   "reports": {s: file_stamp(args.derived_data_dir / REPORT_FILES[s]) for s in sources},
                   **({"ctgov_csv": file_stamp(args.ctgov_csv)} if mask else {})},
        "output": file_stamp(args.output_parquet),
        "counts": {k: stats[k] for k in ("rows", "index_events", "index_events_with_notes",
                                         "index_events_without_notes", "rows_by_source")},
    }
    if mask:
        summary["masking"] = stats["masking"]
        for kind, r in stats["masking"]["replacements"].items():
            print(f"Masked {kind}: {r['replacements']} replacements in {r['notes']} notes, {r['index_events']} events")
        ids, _ = event_trial_identifiers(events, titles)
        residual = scan_residual_identifiers(args.output_parquet, ids, args.workers,
                                             progress=lambda m: print(m, flush=True))
        leaks = {k: residual["checks"][k] for k in ("own_protocol", "own_nct", "own_code", "own_code_any_case")}
        residual["rows_match_build"] = residual["rows_scanned"] == stats["rows"]
        residual["passed"] = residual["rows_match_build"] and all(v["notes"] == 0 for v in leaks.values())
        summary["residual_check"] = residual
        print("Residual own identifiers after masking (notes / events): "
              + "; ".join(f"{k} {v['notes']}/{v['index_events']}" for k, v in residual["checks"].items()))
    summary["runtime_seconds"] = {"build": round(t_build - t0, 1), "total": round(time.time() - t0, 1)}
    summary_path = args.output_parquet.with_suffix(".summary.json")
    write_json_private(summary, summary_path)
    print(f"Summary (counts only): {summary_path}")
    if mask and not summary["residual_check"]["passed"]:
        print("RESIDUAL CHECK FAILED: own identifiers survive masking (see the summary)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
