"""Denominator 1: real patients registered to, and starting treatment on, a DFCI therapeutic trial.

v22 built this in two places:

1. pan_dfci_2024/code/1.5_pull_eligibility_criteria.ipynb wrote useful_trial_enrollments.csv:
   PROTOCOL_ENROLLMENT_DFHCC rows with TREATMENT_TYPE_CD == 'Tre', a TREATMENT_START_DT and a
   STUDY_NBR, joined by numeric protocol number to OnCore 'Treatment' protocols that have an
   NCT ID and a ClinicalTrials.gov listing with a non-empty brief summary.
2. eval_phi_enrollments/prepare_data.py (parse_enrollments, load_and_process_trial_enrollments)
   kept one row per (dfci_mrn, protocol), inner-joined it to OnCore subject records
   (forken.json, consents at the main DFCI site only) and used the OnCore CONSENT date as the
   index date ``trial_start_dt``; kept index dates >= 2016-01-01; one pseudo_mrn per unique
   (dfci_mrn, trial_start_dt).

pan_dfci_2026 has no OnCore consent date and no forken.json equivalent, so the index date
here is a consent proxy taken from PROTOCOL_ENROLLMENT itself (README.md, "Index date").
"""

from __future__ import annotations

import pandas as pd

from .io_utils import assign_pseudo_mrn, protocol_number_to_int, to_day
from .trials import index_in_ctgov_window

PE_COLUMNS = [
    "DFCI_MRN", "PATIENT_STUDY_SEQ", "STUDY_NBR", "TREATMENT_TYPE_CD", "REG_DT", "PROTOCOL_ON_DT",
    "TREATMENT_START_DT", "STUDY_ACCRUAL_OPEN_DT", "STUDY_ACCRUAL_CLOSE_DT",
]

INDEX_DATE_RULES = {
    # Earliest OnCore date of the registration. Consent precedes registration, on-study and
    # treatment start, so the earliest of the three is the closest available proxy for v22's
    # consent date (validated against v22's consent dates; README.md).
    "earliest": ["reg_dt", "protocol_on_dt", "treatment_start_dt"],
    "reg_dt": ["reg_dt"],
    "protocol_on_dt": ["protocol_on_dt"],
    "treatment_start_dt": ["treatment_start_dt"],
}


def _step(attrition: list, name: str, frame: pd.DataFrame, mrn_col: str = "dfci_mrn") -> None:
    attrition.append({"step": name, "rows": int(len(frame)), "patients": int(frame[mrn_col].nunique())})


def pull_enrollments(
    protocol_enrollment_csv,
    oncore_protocols: pd.DataFrame,
    ctgov_nct_ids: set | None,
    start_date: str,
    end_date: str,
    index_date_rule: str = "earliest",
) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    """Return (enrollments, qualifying registrations of any date, attrition, diagnostics).

    ``oncore_protocols`` comes from trials.load_oncore_treatment_protocols. ``ctgov_nct_ids``
    is the set of NCT IDs with a local ctgov listing (None skips that requirement).
    The second frame (every qualifying registration regardless of index date) is used to
    derive v22-style first/last DFCI treatment-start dates per trial.
    """
    if index_date_rule not in INDEX_DATE_RULES:
        raise ValueError(f"index_date_rule must be one of {sorted(INDEX_DATE_RULES)}")
    attrition: list = []
    pe = pd.read_csv(protocol_enrollment_csv, dtype=str, usecols=PE_COLUMNS)
    pe["dfci_mrn"] = pd.to_numeric(pe.DFCI_MRN, errors="raise").astype("int64")
    _step(attrition, "PROTOCOL_ENROLLMENT rows (all protocol types)", pe)

    pe = pe[pe.TREATMENT_TYPE_CD.eq("Tre")]
    _step(attrition, "TREATMENT_TYPE_CD == 'Tre' (therapeutic protocols)", pe)
    pe = pe[pe.TREATMENT_START_DT.notna()]
    _step(attrition, "TREATMENT_START_DT present (patient started protocol treatment)", pe)
    pe = pe[pe.STUDY_NBR.notna()].copy()  # v22 guard against masked protocols; no-op in 2026 data
    _step(attrition, "STUDY_NBR present", pe)

    pe = pe.assign(
        reg_dt=to_day(pe.REG_DT),
        protocol_on_dt=to_day(pe.PROTOCOL_ON_DT),
        treatment_start_dt=to_day(pe.TREATMENT_START_DT),
        dfci_accrual_open_dt=to_day(pe.STUDY_ACCRUAL_OPEN_DT),
        dfci_accrual_close_dt=to_day(pe.STUDY_ACCRUAL_CLOSE_DT),
        patient_study_seq=pe.PATIENT_STUDY_SEQ,
    )
    pe["trial_start_dt"] = pe[INDEX_DATE_RULES[index_date_rule]].min(axis=1)
    pe["index_date_rule"] = index_date_rule
    lo, hi = pd.Timestamp(start_date), pd.Timestamp(end_date)
    in_window = (pe.trial_start_dt >= lo) & (pe.trial_start_dt <= hi)

    # The extract replaces the protocol of every registration on a protocol that was still
    # active at extraction with STUDY_NBR '0000' ("Unspecified Active Therapeutic Protocol").
    # Those registrations cannot be tied to a trial (v22 lost them the same way).
    masked = pe.STUDY_NBR.str.strip().eq("0000")
    masked_in_window = pe[masked & in_window]
    diagnostics = {
        "masked_active_protocol_rows_in_window": int(len(masked_in_window)),
        "masked_active_protocol_patients_in_window": int(masked_in_window.dfci_mrn.nunique()),
        "masked_active_protocol_rows_in_window_by_year": {
            str(int(y)): int(n) for y, n in masked_in_window.trial_start_dt.dt.year.value_counts().sort_index().items()},
    }
    pe = pe[~masked]
    _step(attrition, "STUDY_NBR is not '0000' (masked: protocol still active at extraction)", pe)

    pe["protocol_nbr"] = protocol_number_to_int(pe.STUDY_NBR)
    pe = pe.merge(oncore_protocols, on="protocol_nbr", how="inner", validate="many_to_one")
    _step(attrition, "protocol is an OnCore therapeutic protocol with an NCT ID", pe)

    if ctgov_nct_ids is not None:
        pe = pe[pe.nct_id.isin(ctgov_nct_ids)]
        _step(attrition, "NCT ID has a ClinicalTrials.gov listing with a brief summary (local export)", pe)

    pe = pe[pe.trial_start_dt.notna()]
    qualifying = pe.copy()

    pe = pe[(pe.trial_start_dt >= lo) & (pe.trial_start_dt <= hi)]
    _step(attrition, f"index date within {start_date}..{end_date} (inclusive)", pe)

    before = len(pe)
    pe = pe.drop_duplicates(["dfci_mrn", "protocol_number", "trial_start_dt"])
    _step(attrition, "drop exact duplicate (patient, protocol, index date) rows", pe)
    if before - len(pe):
        attrition[-1]["dropped_rows"] = int(before - len(pe))

    pe = assign_pseudo_mrn(pe, "trial_start_dt")

    # Index date relative to the trial's DFCI accrual window (OnCore, from the same table).
    accrual_known = pe.dfci_accrual_open_dt.notna()
    in_accrual = (pe.trial_start_dt >= pe.dfci_accrual_open_dt) & (
        pe.dfci_accrual_close_dt.isna() | (pe.trial_start_dt <= pe.dfci_accrual_close_dt))
    pe["index_in_dfci_accrual_window"] = in_accrual.astype("boolean").mask(~accrual_known)
    diagnostics["rows_by_nct_source"] = {k: int(v) for k, v in pe.nct_source.value_counts().items()}
    return pe, qualifying, attrition, diagnostics


def trial_table(enrollments: pd.DataFrame, qualifying: pd.DataFrame, trial_dates: pd.DataFrame,
                listings: pd.DataFrame, training_nct_ids: set, holdout_nct_ids: set,
                start_date: str, end_date: str) -> pd.DataFrame:
    """One row per enrolled NCT ID: v22's trials.parquet dates plus other trial windows.

    ``first/last_dfci_treatment_start``: v22's trials.parquet open_date/close_date were
    (verified on v22's files, 99% exact) the earliest and latest TREATMENT_START_DT among the
    trial's useful_trial_enrollments rows on or after 2016-01-01; retrieval then searched only
    trials whose [open_date, close_date] contained the patient's index date. Reproduced over
    all qualifying registrations with a treatment start within start_date..end_date.
    """
    lo, hi = pd.Timestamp(start_date), pd.Timestamp(end_date)
    starts = qualifying[(qualifying.treatment_start_dt >= lo) & (qualifying.treatment_start_dt <= hi)]
    v22_window = (starts.groupby("nct_id").treatment_start_dt.agg(["min", "max"])
                  .rename(columns={"min": "first_dfci_treatment_start", "max": "last_dfci_treatment_start"}))

    accrual = qualifying.groupby("nct_id").agg(
        dfci_accrual_open_dt=("dfci_accrual_open_dt", "min"),
        dfci_accrual_close_dt=("dfci_accrual_close_dt", "max"),
        _any_open=("dfci_accrual_close_dt", lambda s: bool(s.isna().any())),
    )
    accrual.loc[accrual._any_open, "dfci_accrual_close_dt"] = pd.NaT
    accrual = accrual.drop(columns="_any_open")

    per_trial = enrollments.groupby("nct_id").agg(
        protocol_numbers=("protocol_number", lambda s: ";".join(sorted(set(s)))),
        oncore_status=("oncore_status", lambda s: ";".join(sorted(set(s.dropna())))),
        oncore_phase=("oncore_phase", lambda s: ";".join(sorted(set(s.dropna())))),
        n_rows=("pseudo_mrn", "size"),
        n_index_events=("pseudo_mrn", "nunique"),
        n_patients=("dfci_mrn", "nunique"),
    )
    out = (per_trial.join(v22_window, how="left").join(accrual, how="left").reset_index()
           .merge(trial_dates, on="nct_id", how="left"))
    out["in_v23_training_sample"] = out.nct_id.isin(training_nct_ids)
    out["v23_dfci_holdout_2025_2026"] = out.nct_id.isin(holdout_nct_ids)
    out = out.merge(listings, on="nct_id", how="left")
    return out


def add_trial_window_flags(enrollments: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Per-row flags: is the index date inside the trial's v22-style and ctgov windows?"""
    cols = ["nct_id", "first_dfci_treatment_start", "last_dfci_treatment_start",
            "ctgov_start_date", "ctgov_completion_date"]
    merged = enrollments.merge(trials[cols], on="nct_id", how="left", validate="many_to_one")
    v22_known = merged.first_dfci_treatment_start.notna()
    in_v22 = ((merged.trial_start_dt >= merged.first_dfci_treatment_start)
              & (merged.trial_start_dt <= merged.last_dfci_treatment_start))
    merged["index_in_v22_style_window"] = in_v22.astype("boolean").mask(~v22_known)
    merged["index_in_ctgov_window"] = index_in_ctgov_window(
        merged.trial_start_dt, merged.ctgov_start_date, merged.ctgov_completion_date)
    return merged.drop(columns=cols[1:])
