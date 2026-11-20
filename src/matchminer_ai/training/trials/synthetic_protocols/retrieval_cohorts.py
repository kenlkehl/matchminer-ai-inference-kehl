"""Combine each sampled trial's protocol and ctgov cohorts into the retrieval cohort table.

Users may hold some trials' full protocols and only the ClinicalTrials.gov text
of others, so retrieval and labeling see both representations of a trial: the
cohorts extracted from its synthetic protocol (extract_cohorts.py, via
patient_cohorts.py) and the cohorts extracted from the ctgov record that
protocol was written from (``trials.extract_ctgov_cohorts`` --trials-jsonl on the
batch's seeds.jsonl). Both use the same prompt rules and response contract, and
the same training steps after extraction: merged, then key and boilerplate criteria separated
(patient_cohorts.py --retrieval-output and 0d's --output-deboilerplated-cohorts); retrieval and
match labels read the key criteria, boilerplate labels the boilerplate criteria, per-criterion labels the tagged lines.
Patients are generated from the protocol cohorts only, with all their criteria.

The output keeps every protocol row unchanged (cohort_criteria_source
"protocol", IDs <NCT>-cohort-<n>) and adds the ctgov rows (cohort_criteria_source
"ctgov", IDs <NCT>-ctgov-cohort-<n>, ctgov fields renamed ctgov_*,
cohort_index continuing after the protocol rows), each trial's split taken
from the ctgov sample. (Before 2026-11-05 ctgov was called "registry": tables then have
"registry" sources, <NCT>-registry-cohort-<n> IDs and registry_* fields, and still combine.)
A trial accepted in only one version keeps that one.
Mining then represents each trial by one version, chosen at random per
patient-trial pair and per trial for cohort queries
(trialspace/retrieval.py). Writes <output> and <output>.provenance.json.
See synthetic_protocols/README.md.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from matchminer_ai.training.cohort_format import file_stamp, read_clinical_csv, training_rows
from matchminer_ai.training.paths import commit as git_commit, no_phi
CTGOV_FIELDS = ("title", "brief_summary", "detailed_description", "eligibility_criteria",
                   "last_update_post_date", "retrieved_at")


def build(protocol, ctgov, splits):
    """Protocol rows unchanged plus source-qualified ctgov rows, validated together."""
    if not protocol.cohort_criteria_source.eq("protocol").all():
        raise ValueError("Every protocol-table row must come from protocol extraction")
    if not ctgov.cohort_criteria_source.isin(["ctgov", "registry"]).all():  # "registry" before 2026-11-05
        raise ValueError("Every ctgov row must come from current ctgov extraction (trial-registry-cohorts-v1)")
    split = splits.drop_duplicates()
    if split.nct_id.duplicated().any():
        raise ValueError("A sampled trial has more than one source split")
    split = split.set_index("nct_id").split
    known = protocol.drop_duplicates("trial_id").set_index("trial_id").split
    if not known.eq(split.reindex(known.index)).all():
        raise ValueError("Protocol cohort splits disagree with the ctgov sample")
    ctgov = ctgov.rename(columns={key: f"ctgov_{key}" for key in CTGOV_FIELDS if key in ctgov})
    ctgov = ctgov.drop(columns=["split"], errors="ignore").assign(
        cohort_trial_id=ctgov.trial_id + "-ctgov-cohort-" + ctgov.cohort_number.astype(int).astype(str),
        nct_id=ctgov.trial_id, split=ctgov.trial_id.map(split))
    if ctgov.split.isna().any():
        raise ValueError(f"No source split for {sorted(ctgov.loc[ctgov.split.isna(), 'trial_id'].unique())[:5]}")
    ctgov = ctgov.sort_values(["trial_id", "cohort_number"]).reset_index(drop=True)
    ctgov["cohort_index"] = range(int(protocol.cohort_index.max()) + 1, int(protocol.cohort_index.max()) + 1 + len(ctgov))
    frame = pd.concat([protocol, ctgov], ignore_index=True)
    rebuilt = training_rows(frame)
    if not rebuilt.cohort_eligibility_criteria.eq(frame.cohort_eligibility_criteria).all():
        raise ValueError("Rebuilt cohort criteria differ from the extracted criteria")
    return frame


def main(args):
    protocol = read_clinical_csv(args.protocol_cohorts)
    ctgov = read_clinical_csv(args.ctgov_cohorts)
    audit = read_clinical_csv(args.ctgov_trials)
    splits = read_clinical_csv(args.splits_from, usecols=["nct_id", "split"])
    frame = build(protocol, ctgov, splits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(args.output)
    protocol_trials, ctgov_trials = set(protocol.trial_id), set(ctgov.trial_id)
    commit = git_commit()
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(args.output.resolve()), "output_stamp": file_stamp(args.output), "commit": commit,
        "sources": {name: {"path": str(path.resolve()), "stamp": file_stamp(path)} for name, path in (
            ("protocol_cohorts", args.protocol_cohorts), ("ctgov_cohorts", args.ctgov_cohorts),
            ("ctgov_trials", args.ctgov_trials), ("splits_from", args.splits_from))},
        "cohorts": {"protocol": len(protocol), "ctgov": len(ctgov), "total": len(frame)},
        "trials": {"both_versions": len(protocol_trials & ctgov_trials),
                   "protocol_only": len(protocol_trials - ctgov_trials),
                   "ctgov_only": len(ctgov_trials - protocol_trials),
                   "total": len(protocol_trials | ctgov_trials)},
        "ctgov_extraction": {
            "records": len(audit), "not_cancer_treatment": int(audit.cancer_treatment_trial.eq("no").sum()),
            "failed": sorted(audit.loc[audit.extraction_error.notna(), "nct_id"])},
        "retrieval_policy": "one version per trial: random per patient-trial pair, and per trial for cohort queries",
    }
    Path(f"{args.output}.provenance.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(json.dumps({key: receipt[key] for key in ("cohorts", "trials")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol-cohorts", type=Path, required=True,
                        help="patient_cohorts.py --retrieval-output: the cohorts patients were generated from, with key criteria")
    parser.add_argument("--ctgov-cohorts", "--registry-cohorts", type=Path, required=True,
                        help="trials.extract_ctgov_cohorts --trials-jsonl cohort rows for the same batch's seeds")
    parser.add_argument("--ctgov-trials", "--registry-trials", type=Path, required=True,
                        help="The same run's per-trial audit table (--output-trials)")
    parser.add_argument("--output", type=Path, required=True, help="Retrieval cohort CSV to write")
    parser.add_argument("--splits-from", type=Path,
                        help="ctgov sample whose nct_id/split columns give each trial's source split "
                             "(default: <data root>/no_phi/sample_trial_cohort_lineitems.csv)")
    args = parser.parse_args()
    args.splits_from = args.splits_from or no_phi("sample_trial_cohort_lineitems.csv")
    main(args)
