"""Turn a synthetic protocol batch's extracted cohorts into the patient-generation cohort table.

Reads cohorts.parquet (the merged cohorts with all their criteria, boilerplate included;
--cohorts-file; user, 2026-11-04) and trials.parquet from an extract_cohorts.py output
directory (--extraction-dir, default <batch-dir>/cohort_extraction)
and writes a CSV that ``synthetic.generate`` (``training.pipeline`` --sampled-cohorts)
accepts: the accepted cancer-treatment cohorts in seed order with a contiguous
cohort_index, each trial's source split copied from the ctgov sample it was
seeded from, and every provenance column kept. Cohort criteria come from the
synthetic protocol; the original ctgov criteria stay in
ctgov_eligibility_criteria (registry_eligibility_criteria before 2026-11-05) and are never used as cohort criteria. --retrieval-output
also writes the same rows from cohorts_deboilerplated.parquet (--deboilerplated-file: the
merged cohorts with their key criteria in cohort_key_criteria and boilerplate criteria in
cohort_boilerplate_criteria), the protocol version that retrieval_cohorts.py reads. A <output>.provenance.json receipt beside each output records
its sources. See synthetic_protocols/README.md.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pandas as pd

from matchminer_ai.training.cohort_format import (
    REPRESENTATION, file_stamp, read_clinical_csv, representation_for, training_rows,
)
from matchminer_ai.training.paths import commit as git_commit, no_phi


def build(cohorts, trials, splits):
    """Generation-ready rows, validated by the same adapter the generator uses."""
    if "eligibility_criteria" in cohorts:
        raise ValueError("eligibility_criteria would replace protocol criteria for single cohorts")
    if not cohorts.cohort_criteria_source.eq("protocol").all():
        raise ValueError("Every cohort must come from protocol extraction")
    order = {trial: rank for rank, trial in enumerate(trials.trial_id)}
    frame = cohorts.assign(_rank=cohorts.trial_id.map(order))
    if frame._rank.isna().any():
        raise ValueError("Cohorts include trials missing from the trial table")
    frame = frame.sort_values(["_rank", "cohort_number"]).drop(columns="_rank").reset_index(drop=True)
    split = splits.drop_duplicates()
    if split.nct_id.duplicated().any():
        raise ValueError("A seed trial has more than one source split")
    frame = frame.merge(split, left_on="trial_id", right_on="nct_id", how="left", validate="many_to_one")
    if frame.split.isna().any():
        raise ValueError(f"No source split for {sorted(frame.loc[frame.split.isna(), 'trial_id'].unique())[:5]}")
    frame["cohort_index"] = range(len(frame))
    frame["trial_representation"] = representation_for(frame)
    if frame.trial_representation.iloc[0] != REPRESENTATION:
        raise ValueError("Protocol cohorts must use the named-cohort representation")
    # The generator rebuilds criteria from these rows; protocol text must survive unchanged.
    rebuilt = training_rows(frame)
    if not rebuilt.cohort_eligibility_criteria.eq(frame.cohort_eligibility_criteria).all():
        raise ValueError("Rebuilt cohort criteria differ from the extracted protocol criteria")
    return frame


def main(args):
    extraction = args.extraction_dir or args.batch_dir / "cohort_extraction"
    metadata = json.loads((extraction / "run_metadata.json").read_text())
    if metadata["trial_errors"]:
        print(f"note: {len(metadata['trial_errors'])} trials failed extraction and have no cohorts", flush=True)
    trials = pd.read_parquet(extraction / "trials.parquet")
    splits = read_clinical_csv(args.splits_from, usecols=["nct_id", "split"])
    frame = build(pd.read_parquet(extraction / args.cohorts_file), trials, splits)
    outputs = [(args.output, args.cohorts_file, frame)]
    if args.retrieval_output:
        clean = build(pd.read_parquet(extraction / args.deboilerplated_file), trials, splits)
        if not clean.cohort_trial_id.equals(frame.cohort_trial_id):
            raise ValueError("The de-boilerplated cohorts are not the generation cohorts")
        outputs.append((args.retrieval_output, args.deboilerplated_file, clean))
    commit = git_commit()
    for output, cohorts_file, table in outputs:
        write(output, cohorts_file, table, extraction, metadata, commit, args)


def write(output, cohorts_file, frame, extraction, metadata, commit, args):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": str(output.resolve()), "output_stamp": file_stamp(output),
        "batch_dir": str(args.batch_dir.resolve()), "cohorts_file": cohorts_file,
        "sources": {name: {"path": str((extraction / name).resolve()), "stamp": file_stamp(extraction / name)}
                    for name in (cohorts_file, "trials.parquet", "run_metadata.json")},
        "splits_from": {"path": str(args.splits_from.resolve()), "stamp": file_stamp(args.splits_from)},
        "commit": commit, "trial_representation": REPRESENTATION,
        "cohort_criteria_source": "protocol",
        "trials": int(frame.trial_id.nunique()), "cohorts": len(frame),
        "multiple_cohort_trials": int(frame.loc[frame.multiple_cohorts.eq("yes"), "trial_id"].nunique()),
        "split_counts": frame.drop_duplicates("trial_id").split.value_counts().to_dict(),
        "extraction_status_counts": metadata["status_counts"],
        "extraction_failures": sorted(metadata["trial_errors"]),
        "not_cancer_treatment": metadata["trial_rejections"],
        "synthetic_prompt_sha256": metadata["synthetic_prompt_sha256"],
        "extraction_prompt_sha256": metadata["prompt_sha256"], "extraction_schema_version": metadata["schema_version"],
    }
    Path(f"{output}.provenance.json").write_text(json.dumps(receipt, indent=2, default=str))
    print(f"wrote {len(frame)} cohorts from {receipt['trials']} trials ({cohorts_file}) to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch-dir", type=Path, required=True, help="A generate_batch.py output directory")
    parser.add_argument("--extraction-dir", type=Path, help="extract_cohorts.py output; default <batch-dir>/cohort_extraction")
    parser.add_argument("--cohorts-file", default="cohorts.parquet",
                        help="Cohort table that seeds patients; default the merged cohorts, boilerplate included")
    parser.add_argument("--deboilerplated-file", default="cohorts_deboilerplated.parquet",
                        help="Cohort table for --retrieval-output; default the merged cohorts with boilerplate removed")
    parser.add_argument("--output", type=Path, required=True, help="Generation input CSV to write")
    parser.add_argument("--retrieval-output", type=Path,
                        help="Also write the same cohorts from --deboilerplated-file (retrieval's protocol version)")
    parser.add_argument("--splits-from", type=Path,
                        help="ctgov sample whose nct_id/split columns give each trial's source split "
                             "(default: <data root>/no_phi/sample_trial_cohort_lineitems.csv)")
    args = parser.parse_args()
    args.splits_from = args.splits_from or no_phi("sample_trial_cohort_lineitems.csv")
    main(args)
