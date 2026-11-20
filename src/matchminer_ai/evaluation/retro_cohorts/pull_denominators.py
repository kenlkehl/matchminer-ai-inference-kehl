#!/usr/bin/env python3
"""Pull the v23 real-patient denominators: DFCI trial enrollments and standard-of-care starts.

Ports v22 real_eval_code/eval_phi_enrollments/prepare_data.py and
eval_phi_soc/prepare_data.py (cohort steps only) to pan_dfci_2026. See README.md.

The trial cohort comes from research chemo plans by default (--trial-source research_plans, since 2026-11-14:
first research plan per patient and OnCore 'Treatment' protocol, index date = that plan's start). The registration
pull from PROTOCOL_ENROLLMENT (--trial-source enrollment_table) remains for cross-checks; it cannot identify the 22%
of registrations whose protocol was still active at extraction.

Writes, under --output-dir (mode 700; files mode 600):
  processed_trial_enrollments.csv   one row per qualifying trial start (research plans) or registration
  enrolled_trials.csv               one row per NCT ID among them (dates, flags, listing text)
  processed_soc_treatments.csv      one row per qualifying standard-of-care plan
  denominator_summary.json          parameters, input file stamps, attrition and counts

Prints counts only. Needs no GPU, LLM or network.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

from matchminer_ai.evaluation.retro_cohorts import enrollments as enr
from matchminer_ai.evaluation.retro_cohorts import research_plans as rpl
from matchminer_ai.evaluation.retro_cohorts import soc
from matchminer_ai.evaluation.retro_cohorts import trials as tr
from matchminer_ai.evaluation.retro_cohorts.io_utils import (cohort_counts, ensure_private_dir, file_stamp, utc_now,
                                     write_csv_private, write_json_private)
from matchminer_ai.evaluation.retro_cohorts.notes import DEFAULT_SOURCES, REPORT_FILES, count_prior_reports, load_report_dates

from matchminer_ai.site import load_site_config
from matchminer_ai.training.paths import no_phi

REPO = Path(__file__).resolve().parent
# Default inputs: the institution's structured EHR extract (the site config's paths.retro_source_root) and the
# data root's no_phi tree. Each option overrides its default.
SOURCE_DEFAULTS = {
    "protocol_enrollment_csv": ("source", "structured_data/PROTOCOL_ENROLLMENT.csv"),
    "treatment_plan_csv": ("source", "structured_data/TREATMENT_PLAN.csv"),
    "split_csv": ("source", "derived_data/split_8-2026_with_text_but_not_structured_data_yet.csv"),
    "derived_data_dir": ("source", "derived_data"),
    "oncore_status_csv": ("no_phi", "oncore_nct_status_10-17-26.csv"),
    "dfci_trials_csv": ("no_phi", "dfci_trials_summary_10-18-26.csv"),
    "ctgov_csv": ("no_phi", "ctgov_trials.csv"),
    "training_sample_csv": ("no_phi", "training_run_screened_sample5000_20261104/registry_sample_trials.csv"),
}

ENROLLMENT_COLUMNS = [
    "pseudo_mrn", "dfci_mrn", "protocol_number", "nct_id", "trial_start_dt", "index_date_rule",
    "reg_dt", "protocol_on_dt", "treatment_start_dt", "patient_study_seq", "split",
    "dfci_accrual_open_dt", "dfci_accrual_close_dt", "index_in_dfci_accrual_window",
    "index_in_v22_style_window", "index_in_ctgov_window", "nct_source", "first_protocol_for_nct",
    "n_prior_reports", "has_prior_reports",
]
RESEARCH_PLAN_COLUMNS = [
    "pseudo_mrn", "dfci_mrn", "protocol_number", "nct_id", "trial_start_dt", "index_date_rule",
    "first_cycle_dt", "epic_tplan_id", "research_plan", "tplan_goal", "n_research_plans_on_protocol", "split",
    "oncore_status", "oncore_phase", "nct_source", "enrollment_table_match", "registration_dt",
    "days_registration_to_index", "masked_registration_dt", "note_anchor_dt", "dfci_accrual_open_dt",
    "dfci_accrual_close_dt", "index_in_dfci_accrual_window",
    "index_in_v22_style_window", "index_in_ctgov_window", "n_prior_reports", "has_prior_reports",
]
TRIAL_COLUMNS = [
    "nct_id", "protocol_numbers", "oncore_status", "oncore_phase", "n_rows", "n_index_events", "n_patients",
    "first_dfci_treatment_start", "last_dfci_treatment_start", "dfci_accrual_open_dt", "dfci_accrual_close_dt",
    "ctgov_start_date", "ctgov_start_date_type", "ctgov_completion_date", "ctgov_completion_date_type",
    "in_v23_training_sample", "v23_dfci_holdout_2025_2026",
    "title", "brief_summary", "eligibility_criteria", "trial_text",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True,
                   help="PHI output directory, e.g. ../data/phi/real_eval_denominators_YYYYMMDD")
    p.add_argument("--cohorts", default="enrollments,soc", help="Comma list: enrollments, soc")
    p.add_argument("--start-date", default="2016-01-01", help="First index date (inclusive); v22: 2016-01-01")
    p.add_argument("--end-date", default="2025-12-31",
                   help="Last index date (inclusive). v22: SOC 2024-01-01, enrollments unbounded (data ended 2024-03)")
    p.add_argument("--trial-source", default="research_plans", choices=["research_plans", "enrollment_table"],
                   help="Trial cohort source: first research chemo plan per patient-protocol (default) or "
                        "PROTOCOL_ENROLLMENT registrations (masked active protocols excluded)")
    p.add_argument("--index-date-rule", default="earliest", choices=sorted(enr.INDEX_DATE_RULES),
                   help="Enrollment-table index date (consent proxy); --trial-source enrollment_table only")
    p.add_argument("--no-summary-nct-fill", action="store_true",
                   help="Map protocols to NCT IDs with the OnCore status file only (v22-like)")
    p.add_argument("--no-require-ctgov-listing", dest="require_ctgov_listing", action="store_false",
                   help="Keep enrollments whose NCT ID has no listing in the local ctgov export")
    p.add_argument("--soc-split-filter", default="test", help="Patient split for SOC ('' keeps every split); v22: test")
    p.add_argument("--days-buffer", type=int, default=5, help="Report cutoff = index date + N days (v22: 5)")
    p.add_argument("--report-sources", default=",".join(DEFAULT_SOURCES),
                   help=f"Report files counted for n_prior_reports: {', '.join(REPORT_FILES)}")
    p.add_argument("--skip-report-counts", action="store_true", help="Do not compute n_prior_reports")
    for option in SOURCE_DEFAULTS:
        root, name = SOURCE_DEFAULTS[option]
        where = "the site config's paths.retro_source_root" if root == "source" else "<data root>/no_phi"
        p.add_argument("--" + option.replace("_", "-"), type=Path, help=f"Default: {where}/{name}")
    args = p.parse_args(argv)
    for option, (root, name) in SOURCE_DEFAULTS.items():
        if getattr(args, option) is None:
            base = Path(load_site_config().require("paths.retro_source_root")) if root == "source" else no_phi()
            setattr(args, option, base / name)
    return args


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001 - provenance only
            return ""
    return {"commit": run("rev-parse", "--verify", "-q", "HEAD") or None,
            "dirty_files": run("status", "--porcelain").splitlines()}


def value_counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def show(title: str, counts: dict) -> None:
    print(f"\n{title}: {counts['rows']} rows, {counts['index_events']} index events, "
          f"{counts['patients']} patients ({counts['first_index_date']} to {counts['last_index_date']})")
    for year, c in counts["by_year"].items():
        print(f"  {year}: {c['rows']} rows, {c['index_events']} index events, {c['patients']} patients")


def show_attrition(attrition: list) -> None:
    for step in attrition:
        print(f"  {step['rows']:>9} rows {step['patients']:>8} patients  {step['step']}")


def main(argv=None) -> int:
    args = parse_args(argv)
    os.umask(0o077)
    cohorts = {c.strip() for c in args.cohorts.split(",") if c.strip()}
    if not cohorts <= {"enrollments", "soc"}:
        raise SystemExit(f"Unknown cohort in --cohorts: {sorted(cohorts - {'enrollments', 'soc'})}")
    previous = Path(args.output_dir) / "denominator_summary.json"
    if "soc" in cohorts and previous.exists():
        earlier = json.loads(previous.read_text()).get("parameters", {}).get("soc_split_filter")
        if earlier != args.soc_split_filter:
            raise SystemExit(f"{args.output_dir} holds a SOC pull for split {earlier!r}, not {args.soc_split_filter!r}; "
                             "use a separate --output-dir per split (e.g. <run>/soc_validation)")
    out_dir = ensure_private_dir(args.output_dir)
    sources = [s.strip() for s in args.report_sources.split(",") if s.strip()]

    inputs = {"split_csv": args.split_csv}
    if "enrollments" in cohorts:
        if args.trial_source == "research_plans":
            inputs["treatment_plan_csv"] = args.treatment_plan_csv
        inputs.update(protocol_enrollment_csv=args.protocol_enrollment_csv, oncore_status_csv=args.oncore_status_csv,
                      dfci_trials_csv=args.dfci_trials_csv, ctgov_csv=args.ctgov_csv,
                      training_sample_csv=args.training_sample_csv)
    if "soc" in cohorts:
        inputs["treatment_plan_csv"] = args.treatment_plan_csv
    if not args.skip_report_counts:
        for s in sources:
            inputs[f"reports_{s}"] = Path(args.derived_data_dir) / REPORT_FILES[s]
    summary = {
        "created_utc": utc_now(), "host": platform.node(), "python": sys.version.split()[0],
        "pandas": pd.__version__, "code": git_state(),
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "inputs": {k: file_stamp(v) for k, v in inputs.items()},
    }
    split = soc.load_split(args.split_csv)
    results = {}

    if "enrollments" in cohorts:
        print("Enrollments: loading trial references...")
        oncore, oncore_counts = tr.load_oncore_treatment_protocols(
            args.oncore_status_csv, None if args.no_summary_nct_fill else args.dfci_trials_csv)
        listings_all = tr.load_ctgov_listings(args.ctgov_csv, set(oncore.nct_id))
        ctgov_ids = set(listings_all.nct_id) if args.require_ctgov_listing else None
        if args.trial_source == "research_plans":
            enrollments, qualifying, attrition, diagnostics = rpl.pull_trial_starts(
                args.treatment_plan_csv, oncore, ctgov_ids, args.start_date, args.end_date,
                args.protocol_enrollment_csv)
        else:
            enrollments, qualifying, attrition, diagnostics = enr.pull_enrollments(
                args.protocol_enrollment_csv, oncore, ctgov_ids, args.start_date, args.end_date, args.index_date_rule)
        trials = enr.trial_table(enrollments, qualifying, tr.load_dfci_trial_dates(args.dfci_trials_csv),
                                 listings_all, tr.load_training_nct_ids(args.training_sample_csv),
                                 tr.dfci_holdout_nct_ids(args.dfci_trials_csv), args.start_date, args.end_date)
        enrollments = enr.add_trial_window_flags(enrollments, trials)
        enrollments = enrollments.merge(split, on="dfci_mrn", how="left", validate="many_to_one")
        results["enrollments"] = enrollments
        summary["enrollments"] = {
            "trial_source": args.trial_source,
            "oncore_reference": oncore_counts,
            "oncore_treatment_ncts_with_local_ctgov_listing": int(listings_all.nct_id.nunique()),
            "attrition": attrition,
            "diagnostics": diagnostics,
            "trials": int(trials.nct_id.nunique()),
            "trials_in_v23_training_sample": int(trials.in_v23_training_sample.sum()),
            "trials_in_v23_dfci_holdout_2025_2026": int(trials.v23_dfci_holdout_2025_2026.sum()),
            "rows_v22_oncore_list_would_drop": {
                "non_first_protocol_for_nct": int(enrollments.first_protocol_for_nct.eq(False).sum()),
                "protocol_or_nct_only_in_dfci_summary": int(enrollments.nct_source.eq("dfci_trials_summary").sum()),
            },
            "patients_without_split": int(enrollments.split.isna().sum()),
            "index_in_dfci_accrual_window": value_counts(enrollments.index_in_dfci_accrual_window),
            "index_in_v22_style_window": value_counts(enrollments.index_in_v22_style_window),
            "index_in_ctgov_window": value_counts(enrollments.index_in_ctgov_window),
        }
        print("Enrollment attrition:")
        show_attrition(attrition)
        if args.trial_source == "enrollment_table":
            print(f"  Masked active-protocol registrations with an index date in the window (excluded): "
                  f"{diagnostics['masked_active_protocol_rows_in_window']} rows, by year "
                  f"{diagnostics['masked_active_protocol_rows_in_window_by_year']}")
        else:
            print(f"  Cross-check against PROTOCOL_ENROLLMENT: {diagnostics['enrollment_table_match']}")
        results["trials"] = trials

    if "soc" in cohorts:
        print("\nSOC: loading treatment plans...")
        soc_rows, attrition, all_splits = soc.pull_soc(args.treatment_plan_csv, split, args.start_date,
                                                       args.end_date, args.soc_split_filter or None)
        results["soc"] = soc_rows
        summary["soc"] = {"attrition": attrition, "before_split_filter_by_split": all_splits}
        print("SOC attrition:")
        show_attrition(attrition)

    if not args.skip_report_counts:
        patients = set()
        for key in ("enrollments", "soc"):
            if key in results:
                patients |= set(results[key].dfci_mrn)
        print(f"\nCounting reports before index + {args.days_buffer} days ({', '.join(sources)})...")
        report_dates, report_stats = load_report_dates(args.derived_data_dir, sources, patients)
        summary["report_counts"] = {"sources": report_stats, "days_buffer": args.days_buffer}
        for key in ("enrollments", "soc"):
            if key in results:
                frame = results[key]
                frame["n_prior_reports"] = count_prior_reports(frame, report_dates, args.days_buffer)
                frame["has_prior_reports"] = frame.n_prior_reports > 0
        del report_dates
    else:
        for key in ("enrollments", "soc"):
            if key in results:
                results[key]["n_prior_reports"] = pd.NA
                results[key]["has_prior_reports"] = pd.NA

    outputs = {}
    if "enrollments" in results:
        columns = RESEARCH_PLAN_COLUMNS if args.trial_source == "research_plans" else ENROLLMENT_COLUMNS
        frame = results["enrollments"].sort_values(["pseudo_mrn", "protocol_number"])[columns]
        outputs["processed_trial_enrollments"] = write_csv_private(frame, out_dir / "processed_trial_enrollments.csv")
        outputs["enrolled_trials"] = write_csv_private(
            results["trials"].sort_values("nct_id")[TRIAL_COLUMNS], out_dir / "enrolled_trials.csv")
        summary["enrollments"]["counts"] = cohort_counts(frame)
        if not args.skip_report_counts:
            summary["enrollments"]["counts_with_prior_reports"] = cohort_counts(frame[frame.has_prior_reports])
        show("Enrollments", summary["enrollments"]["counts"])
    if "soc" in results:
        frame = results["soc"].sort_values("pseudo_mrn")
        frame = frame[[c for c in soc.OUTPUT_COLUMNS] + ["n_prior_reports", "has_prior_reports"]]
        outputs["processed_soc_treatments"] = write_csv_private(frame, out_dir / "processed_soc_treatments.csv")
        summary["soc"]["counts"] = cohort_counts(frame)
        if not args.skip_report_counts:
            summary["soc"]["counts_with_prior_reports"] = cohort_counts(frame[frame.has_prior_reports])
        show("SOC", summary["soc"]["counts"])

    summary["outputs"] = {k: file_stamp(v) for k, v in outputs.items()}
    write_json_private(summary, out_dir / "denominator_summary.json")
    print(f"\nWrote {len(outputs)} tables and denominator_summary.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
