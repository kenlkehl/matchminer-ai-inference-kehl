#!/usr/bin/env python3
"""Sample the trial corpus for the SOC evaluation (port of v22 eval_phi_soc/prepare_trials.py).

v22 took every trial in v20_public_data/trial_space_lineitems.csv (its spacified ctgov
corpus), removed the trials in the synthetic-training sample
(sample_trial_space_lineitems.csv), drew 500 unique NCT IDs with seed 42 and kept all
spaces of those trials. SOC patients have no "true" trial, so this corpus is what their
retrieved candidates come from. No patient data is read or written.

v23 has no spacified corpus (the real-data evaluations now build cohorts with
matchminer-ai-training-kehl real_eval_code/eval_cohort_lineitems.py, an LLM step outside
this repository). The defaults use the v23 analog of v22's pool: the trials the v23
training path screened as cancer treatment trials (screened_trials.csv,
cancer_treatment_trial == 'yes'), minus the 5,000 v23 synthetic-training trials. The
output has one row per sampled trial with the listing columns eval_cohort_lineitems.py
reads (nct_id, title, brief_summary, eligibility_criteria) plus trial_text.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from matchminer_ai.evaluation.retro_cohorts.io_utils import file_stamp, utc_now, write_csv_private, write_json_private
from matchminer_ai.evaluation.retro_cohorts.trials import load_ctgov_listings, load_training_nct_ids, sample_trials

RUN = "training_run_screened_sample5000_20261104"  # under <data root>/no_phi


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--screened-trials", type=Path,
                   help=f"Trial pool: CSV with nct_id and cancer_treatment_trial (default: <data root>/no_phi/{RUN}/"
                        "screened_trials.csv)")
    p.add_argument("--training-trials", type=Path,
                   help=f"CSV with the nct_id of every synthetic-training trial to exclude (default: <data root>/no_phi/"
                        f"{RUN}/registry_sample_trials.csv)")
    p.add_argument("--ctgov-csv", type=Path, help="Default: <data root>/no_phi/ctgov_trials.csv")
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    args = p.parse_args(argv)
    from matchminer_ai.training.paths import no_phi
    args.screened_trials = args.screened_trials or no_phi(RUN, "screened_trials.csv")
    args.training_trials = args.training_trials or no_phi(RUN, "registry_sample_trials.csv")
    args.ctgov_csv = args.ctgov_csv or no_phi("ctgov_trials.csv")
    os.umask(0o077)

    pool = pd.read_csv(args.screened_trials, dtype=str, usecols=["nct_id", "cancer_treatment_trial"])
    pool_ids = set(pool.nct_id[pool.cancer_treatment_trial.eq("yes")].str.strip().str.upper())
    training = load_training_nct_ids(args.training_trials)
    listings = load_ctgov_listings(args.ctgov_csv, pool_ids)
    eligible = set(listings.nct_id)  # a trial needs listing text to be extracted into cohorts
    sampled = sample_trials(eligible, training, args.n_trials, args.random_seed)
    out = listings.set_index("nct_id").loc[sampled].reset_index()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    write_csv_private(out, args.output_file)
    receipt = {
        "created_utc": utc_now(),
        "inputs": {"screened_trials": file_stamp(args.screened_trials),
                   "training_trials": file_stamp(args.training_trials), "ctgov_csv": file_stamp(args.ctgov_csv)},
        "counts": {"pool_cancer_treatment_trials": len(pool_ids), "pool_with_listing": len(eligible),
                   "training_trials": len(training), "pool_minus_training": len(eligible - training),
                   "sampled": len(sampled)},
        "n_trials": args.n_trials, "random_seed": args.random_seed,
    }
    write_json_private(receipt, args.output_file.with_name(args.output_file.stem + ".sampling.json"))
    print(f"Pool {len(pool_ids)} cancer treatment trials ({len(eligible)} with listings); "
          f"{len(eligible - training)} after removing {len(training)} training trials; sampled {len(sampled)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
