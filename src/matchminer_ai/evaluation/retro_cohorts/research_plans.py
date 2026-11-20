"""Denominator 1 (default since 2026-11-14): starts of research treatment on DFCI therapeutic trials.

User, 2026-11-14: "refactor to use the research chemo plans as the protocol number source, with index date as you
note, and filter for 'Treatment' protocols."

Why: PROTOCOL_ENROLLMENT replaces the protocol of every registration on a protocol still active at extraction with
STUDY_NBR '0000'. That hides 22% of 2016-2025 therapeutic registrations and 71% of 2025's. TREATMENT_PLAN's research
chemo plans keep their protocol number (RESEARCH_CHEMO_PLAN_NBR), currently open protocols included.
- Checked on unmasked registrations: the same patient's research plan starting 14 days before to 90 days after
  registration named the true protocol 99% of the time.
- 98% of research-plan starts map to an NCT ID in every year, 2024-2025 included.

Definition:
- One index event per (patient, protocol): the patient's FIRST research chemo plan on that protocol, on any date. It
  is kept when that first start falls within start_date..end_date, so a later re-plan is not a new start.
- The index date is that plan's TPLAN_START_DT: the start of research treatment. That is typically days to weeks after
  registration, and it is the same kind of event as the SOC cohort's standard-chemo plan start.
- The protocol must be an OnCore protocol_type 'Treatment' protocol with an NCT ID. Summary-only protocols have no
  type and are excluded.
- By default the NCT ID must also have a local ClinicalTrials.gov listing with a brief summary (v22's requirement).

Missed by construction: trials without a research chemo plan (oral-only, radiation or surgical protocols). About 21%
of unmasked registrations have no nearby research plan.

Cross-check against PROTOCOL_ENROLLMENT (``enrollment_table_match``):
- 'same_protocol': the patient has an unmasked therapeutic registration on that protocol; ``registration_dt`` is its
  earliest date.
- 'masked_registration_nearby': no unmasked registration on the protocol, but a masked one 90 days before to 14 days
  after the index date.
- 'none': neither.
DFCI accrual open/close dates come from the enrollment table and are known only for protocols it names.
"""

from __future__ import annotations

import pandas as pd

from .io_utils import assign_pseudo_mrn, protocol_number_to_int, to_day

RESEARCH_CATEGORY = "ONCOLOGY RESEARCH CHEMO PLAN"
TP_COLUMNS = ["DFCI_MRN", "TREATMENT_PLAN_CATEGORY", "RESEARCH_CHEMO_PLAN", "RESEARCH_CHEMO_PLAN_NBR",
              "TPLAN_START_DT", "FIRST_CYCLE_DT", "TPLAN_GOAL", "EPIC_TPLAN_ID"]
PE_COLUMNS = ["DFCI_MRN", "STUDY_NBR", "TREATMENT_TYPE_CD", "REG_DT", "PROTOCOL_ON_DT", "TREATMENT_START_DT",
              "STUDY_ACCRUAL_OPEN_DT", "STUDY_ACCRUAL_CLOSE_DT"]
INDEX_DATE_RULE = "first_research_plan_start"
# A masked registration "explains" a research-plan start when the plan starts 14 days before to 90 days after it.
MASKED_BEFORE_DAYS, MASKED_AFTER_DAYS = 90, 14


def _step(attrition: list, name: str, frame: pd.DataFrame) -> None:
    attrition.append({"step": name, "rows": int(len(frame)), "patients": int(frame["dfci_mrn"].nunique())})


def load_enrollment_reference(protocol_enrollment_csv) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(unmasked registrations per patient-protocol, accrual window per protocol, masked registration dates)."""
    pe = pd.read_csv(protocol_enrollment_csv, dtype=str, usecols=PE_COLUMNS)
    pe = pe[pe.TREATMENT_TYPE_CD.eq("Tre")].copy()
    pe["dfci_mrn"] = pd.to_numeric(pe.DFCI_MRN, errors="raise").astype("int64")
    pe["registration_dt"] = pd.concat([to_day(pe.REG_DT), to_day(pe.PROTOCOL_ON_DT), to_day(pe.TREATMENT_START_DT)],
                                      axis=1).min(axis=1)
    masked = pe.STUDY_NBR.astype("string").str.strip().eq("0000").fillna(False)
    named = pe[~masked].assign(protocol_nbr=lambda x: protocol_number_to_int(x.STUDY_NBR)).dropna(subset=["protocol_nbr"])
    registrations = (named.groupby(["dfci_mrn", "protocol_nbr"]).registration_dt.min().reset_index())
    named = named.assign(open_dt=to_day(named.STUDY_ACCRUAL_OPEN_DT), close_dt=to_day(named.STUDY_ACCRUAL_CLOSE_DT))
    accrual = named.groupby("protocol_nbr").agg(dfci_accrual_open_dt=("open_dt", "min"),
                                                dfci_accrual_close_dt=("close_dt", "max"),
                                                _any_open=("close_dt", lambda s: bool(s.isna().any())))
    accrual.loc[accrual._any_open, "dfci_accrual_close_dt"] = pd.NaT
    accrual = accrual.drop(columns="_any_open").reset_index()
    masked_dates = pe.loc[masked, ["dfci_mrn", "registration_dt"]].dropna().drop_duplicates()
    return registrations, accrual, masked_dates


def pull_trial_starts(
    treatment_plan_csv,
    oncore_protocols: pd.DataFrame,
    ctgov_nct_ids: set | None,
    start_date: str,
    end_date: str,
    protocol_enrollment_csv=None,
) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    """Return (trial starts in the window, first starts of any date, attrition, diagnostics).

    ``oncore_protocols`` comes from trials.load_oncore_treatment_protocols; only rows with oncore_protocol_type
    'Treatment' are used. The second frame (first starts of any date, with trial columns) feeds trial_table's
    per-trial windows.
    """
    attrition: list = []
    tp = pd.read_csv(treatment_plan_csv, dtype=str, usecols=TP_COLUMNS)
    tp["dfci_mrn"] = pd.to_numeric(tp.DFCI_MRN, errors="raise").astype("int64")
    _step(attrition, "TREATMENT_PLAN rows", tp)
    tp = tp[tp.TREATMENT_PLAN_CATEGORY.eq(RESEARCH_CATEGORY)].copy()
    _step(attrition, f"TREATMENT_PLAN_CATEGORY == '{RESEARCH_CATEGORY}'", tp)
    tp["protocol_nbr"] = protocol_number_to_int(tp.RESEARCH_CHEMO_PLAN_NBR)
    tp = tp[tp.protocol_nbr.notna()]
    _step(attrition, "RESEARCH_CHEMO_PLAN_NBR is a protocol number", tp)
    tp = tp.assign(tplan_start_dt=to_day(tp.TPLAN_START_DT), first_cycle_dt=to_day(tp.FIRST_CYCLE_DT))
    tp = tp[tp.tplan_start_dt.notna()]
    _step(attrition, "TPLAN_START_DT present", tp)

    treatment = oncore_protocols[oncore_protocols.oncore_protocol_type.eq("Treatment")]
    tp = tp.merge(treatment, on="protocol_nbr", how="inner", validate="many_to_one")
    _step(attrition, "protocol is an OnCore protocol_type 'Treatment' protocol with an NCT ID", tp)
    if ctgov_nct_ids is not None:
        tp = tp[tp.nct_id.isin(ctgov_nct_ids)]
        _step(attrition, "NCT ID has a ClinicalTrials.gov listing with a brief summary (local export)", tp)

    # The first research plan per (patient, protocol) is the start; later plans are re-plans of the same start.
    tp = tp.sort_values(["dfci_mrn", "protocol_nbr", "tplan_start_dt", "EPIC_TPLAN_ID"])
    plans = tp.groupby(["dfci_mrn", "protocol_nbr"]).size().rename("n_research_plans_on_protocol")
    first = tp.drop_duplicates(["dfci_mrn", "protocol_nbr"]).join(plans, on=["dfci_mrn", "protocol_nbr"])
    _step(attrition, "first research plan per (patient, protocol) (any date)", first)
    first = first.rename(columns={"RESEARCH_CHEMO_PLAN": "research_plan", "TPLAN_GOAL": "tplan_goal",
                                  "EPIC_TPLAN_ID": "epic_tplan_id"})
    first["trial_start_dt"] = first.tplan_start_dt
    first["treatment_start_dt"] = first.tplan_start_dt  # trial_table's per-trial window reads this column
    first["index_date_rule"] = INDEX_DATE_RULE

    diagnostics: dict = {}
    if protocol_enrollment_csv is not None:
        registrations, accrual, masked_dates = load_enrollment_reference(protocol_enrollment_csv)
        first = first.merge(registrations, on=["dfci_mrn", "protocol_nbr"], how="left", validate="one_to_one")
        first = first.merge(accrual, on="protocol_nbr", how="left", validate="many_to_one")
        near = first[first.registration_dt.isna()][["dfci_mrn", "protocol_nbr", "trial_start_dt"]].merge(
            masked_dates, on="dfci_mrn")
        lag = (near.trial_start_dt - near.registration_dt).dt.days
        near = near[(lag >= -MASKED_AFTER_DAYS) & (lag <= MASKED_BEFORE_DAYS)].assign(_gap=lag.abs())
        near = near.sort_values("_gap").drop_duplicates(["dfci_mrn", "protocol_nbr"])  # nearest masked registration
        first = first.merge(near[["dfci_mrn", "protocol_nbr", "registration_dt"]]
                            .rename(columns={"registration_dt": "masked_registration_dt"}).assign(_masked_near=True),
                            on=["dfci_mrn", "protocol_nbr"], how="left")
        first["enrollment_table_match"] = "none"
        first.loc[first._masked_near.eq(True), "enrollment_table_match"] = "masked_registration_nearby"
        first.loc[first.registration_dt.notna(), "enrollment_table_match"] = "same_protocol"
        first = first.drop(columns="_masked_near")
        first["days_registration_to_index"] = (first.trial_start_dt - first.registration_dt).dt.days
    else:
        for col in ("registration_dt", "masked_registration_dt", "dfci_accrual_open_dt", "dfci_accrual_close_dt"):
            first[col] = pd.NaT
        first["enrollment_table_match"] = pd.NA
        first["days_registration_to_index"] = pd.NA

    # Notes for summarization stop before the patient's trial registration, not before the research plan start:
    # registration and screening notes name the trial. The anchor is the earliest of the registration on the same
    # protocol, the nearest masked registration, and the index date (build_note_level_dataset.py, --anchor note_anchor).
    first["note_anchor_dt"] = first[["registration_dt", "masked_registration_dt", "trial_start_dt"]].min(axis=1)
    lo, hi = pd.Timestamp(start_date), pd.Timestamp(end_date)
    starts = first[(first.trial_start_dt >= lo) & (first.trial_start_dt <= hi)].copy()
    _step(attrition, f"first research plan start within {start_date}..{end_date} (inclusive)", starts)
    starts = assign_pseudo_mrn(starts, "trial_start_dt")

    accrual_known = starts.dfci_accrual_open_dt.notna()
    in_accrual = (starts.trial_start_dt >= starts.dfci_accrual_open_dt) & (
        starts.dfci_accrual_close_dt.isna() | (starts.trial_start_dt <= starts.dfci_accrual_close_dt))
    starts["index_in_dfci_accrual_window"] = in_accrual.astype("boolean").mask(~accrual_known)
    diagnostics.update(
        rows_by_nct_source={str(k): int(v) for k, v in starts.nct_source.value_counts().items()},
        enrollment_table_match={str(k): int(v) for k, v in starts.enrollment_table_match.value_counts(dropna=False).items()},
        enrollment_table_match_by_year={str(int(y)): {str(k): int(v) for k, v in g.enrollment_table_match.value_counts(dropna=False).items()}
                                        for y, g in starts.groupby(starts.trial_start_dt.dt.year)},
        days_registration_to_index_quantiles={str(q): float(v) for q, v in
                                              starts.days_registration_to_index.dropna().astype(float).quantile([.1, .25, .5, .75, .9]).items()},
    )
    return starts.reset_index(drop=True), first.reset_index(drop=True), attrition, diagnostics
