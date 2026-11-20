#!/usr/bin/env python3
"""Trial line items for the real-data evaluations, built from cohorts.

The evaluations index one row per trial cohort. This script runs training's ctgov cohort
extraction, ````training.trials.extract_ctgov_cohorts````, on a table of ctgov-style listings. That is
the inference package's cohort workflow:
- a cancer-treatment screen;
- the cohort names, then each cohort's criteria copied verbatim;
- merging of equivalent cohorts;
- one request per criteria line, tagging it key or boilerplate and inclusion or exclusion.

It then writes one row per cohort, under the column names the evaluations read:

  id                       cohort_trial_id (<nct_id>-cohort-<n>)
  nct_id, title
  space_number             cohort_number (0-based)
  this_space               cohort_key_criteria: what retrieval and the match questions read
  trial_boilerplate_text   cohort_boilerplate_criteria: what the boilerplate question reads
  cohort_name, cohort_eligibility_criteria (complete), cohort_criteria_items (per-line tags, JSON)
  trial_text               the listing the extraction read: title, brief summary, eligibility criteria

Input (--trials): CSV, parquet or JSONL with nct_id, title, brief_summary and
eligibility_criteria. One row is kept per NCT ID, and rows without a valid NCT ID are
dropped and counted. Trials the screen rejects as not cancer treatment get no rows.
Cohorts with no key criteria (every line boilerplate) are dropped and counted.

Extraction is remote only: pass --server_urls or --server_urls_file, and --model with the
served model's id. Other unrecognized options (--tokenizer-model, --allow-failures,
--max-tokens, --workers, ...) go to ``training.trials.extract_ctgov_cohorts``. Its checkpoints live in
--work-dir, so a rerun resumes. Without --allow-failures, a trial that fails every attempt
stops the export (0d saves partial files); with it, the trial gets no rows and is listed in
<work-dir>/extraction/errors.json.

This replaced the legacy pre-cohort trial-space extractor (removed 2026-11-06).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from matchminer_ai.training.paths import child_env

EXTRACTOR = "matchminer_ai.training.trials.extract_ctgov_cohorts"
LISTING_COLUMNS = ["nct_id", "title", "brief_summary", "eligibility_criteria"]
SERVER_FLAGS = {"--server_urls", "--server-urls", "--server_urls_file", "--server-urls-file"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=Path, required=True,
                   help="Listings table (CSV, parquet or JSONL) with " + ", ".join(LISTING_COLUMNS))
    p.add_argument("--output-spaces", type=Path, required=True, help="Cohort line-item CSV to write")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="Extraction inputs, exports and resumable checkpoints")
    p.add_argument("--python", default=sys.executable, help="Python for the ctgov cohort extractor")
    return p.parse_known_args(argv)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, dtype=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def listings(frame: pd.DataFrame) -> pd.DataFrame:
    """One listing per valid NCT ID, in the shape ``training.trials.extract_ctgov_cohorts`` reads."""
    missing = [c for c in LISTING_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit(f"--trials is missing columns {missing}")
    frame = frame[LISTING_COLUMNS].astype(object)
    frame = frame.where(frame.notna(), "").astype(str).apply(lambda column: column.str.strip())
    frame["nct_id"] = frame.nct_id.str.upper()
    valid = frame.nct_id.str.fullmatch(r"NCT\d{8}")
    if not valid.all():
        print(f"Dropped {int((~valid).sum())} rows without a valid NCT ID", flush=True)
    frame = frame[valid].drop_duplicates("nct_id", keep="first")
    if frame.empty:
        raise SystemExit("No listings with a valid NCT ID")
    return frame.sort_values("nct_id").reset_index(drop=True)


def extraction_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "listings": work_dir / "listings.jsonl",
        "trials": work_dir / "trials_with_cohorts.csv",
        "cohorts": work_dir / "trial_cohort_lineitems.csv",
        "json": work_dir / "trial_cohorts.json",
        "deboilerplated": work_dir / "trial_cohort_lineitems_deboilerplated.csv",
        "extraction": work_dir / "extraction",
    }


def extract_command(python: str, paths: dict[str, Path], forwarded: list[str]) -> list[str]:
    return [
        python, "-m", EXTRACTOR,
        "--trials-jsonl", str(paths["listings"]),
        "--output-trials", str(paths["trials"]),
        "--output-cohorts", str(paths["cohorts"]),
        "--output-json", str(paths["json"]),
        "--output-deboilerplated-cohorts", str(paths["deboilerplated"]),
        "--work-dir", str(paths["extraction"]),
        *forwarded,
    ]


def lineitems(cohorts: pd.DataFrame) -> pd.DataFrame:
    """0d's de-boilerplated cohort rows under the evaluations' column names."""
    def listing_text(row):
        return "\n\n".join(part for part in (row.title, row.brief_summary, row.eligibility_criteria) if part)

    rows = pd.DataFrame({
        "id": cohorts.cohort_trial_id,
        "nct_id": cohorts.nct_id,
        "title": cohorts.title,
        "space_number": cohorts.cohort_number,
        "this_space": cohorts.cohort_key_criteria.str.strip(),
        "trial_boilerplate_text": cohorts.cohort_boilerplate_criteria.str.strip(),
        "cohort_name": cohorts.cohort_name,
        "cohort_eligibility_criteria": cohorts.cohort_eligibility_criteria,
        "cohort_criteria_items": cohorts.cohort_criteria_items,
        "trial_text": cohorts.apply(listing_text, axis=1) if len(cohorts) else pd.Series(dtype=str),
    })
    empty = rows.this_space == ""
    if empty.any():
        print(f"Dropped {int(empty.sum())} cohorts with no key criteria (every line boilerplate)", flush=True)
    return rows[~empty].reset_index(drop=True)


def main(argv=None) -> None:
    args, forwarded = parse_args(argv)
    if not any(option.split("=", 1)[0] in SERVER_FLAGS for option in forwarded):
        raise SystemExit("Cohort extraction is remote only: pass --server_urls or --server_urls_file")
    paths = extraction_paths(args.work_dir.expanduser().resolve())
    frame = listings(read_table(args.trials.expanduser()))
    paths["listings"].parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(paths["listings"], orient="records", lines=True, force_ascii=False)
    command = extract_command(args.python, paths, forwarded)
    print("$ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True, env=child_env())

    rows = lineitems(pd.read_csv(paths["deboilerplated"], dtype=str, keep_default_na=False))
    output = args.output_spaces.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    rows.to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"Wrote {len(rows)} cohort line items for {rows.nct_id.nunique()} of {len(frame)} trials to {output}")


if __name__ == "__main__":
    main()
