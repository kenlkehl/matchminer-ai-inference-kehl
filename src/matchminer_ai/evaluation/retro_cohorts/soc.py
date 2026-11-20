"""Denominator 2: real patients starting a standard-of-care oncology treatment plan.

Port of v22 eval_phi_soc/prepare_data.py (load_soc_treatments, create_pseudo_mrn), which
pull_soc_treatments.py duplicates without the pseudo_mrn step. v22 rules, in order:

1. TREATMENT_PLAN rows with a standard or research chemo plan name;
   ``plan`` = standard plan name, else research plan name; ``is_trial`` = no standard name.
2. TPLAN_GOAL present; ``is_palliative`` = goal contains PALLIATIVE or CONTROL.
3. TREATMENT_PLAN_CATEGORY == 'ONCOLOGY STANDARD CHEMO PLAN'.
4. start_date <= TPLAN_START_DT <= end_date (inclusive). v22 run_pipeline.py passed
   2016-01-01 and 2024-01-01 (prepare_data.py's own default end was 2023-01-01).
5. Inner join to the patient-level train/validation/test split; keep split 'test'.
6. Index date ``trial_start_dt`` = TPLAN_START_DT; one pseudo_mrn per (dfci_mrn, date).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import assign_pseudo_mrn, to_day

SOC_CATEGORY = "ONCOLOGY STANDARD CHEMO PLAN"

TP_COLUMNS = [
    "DFCI_MRN", "TREATMENT_PLAN_CATEGORY", "STD_CHEMO_PLAN", "RESEARCH_CHEMO_PLAN",
    "RESEARCH_CHEMO_PLAN_NBR", "TPLAN_START_DT", "TPLAN_ICD_DX_CODES", "TPLAN_DX_NAME", "TPLAN_GOAL",
    "TPLAN_DC_REASON", "FIRST_CYCLE_DT", "EPIC_TPLAN_ID", "TPLAN_SOURCE_NM",
]

OUTPUT_COLUMNS = [
    "pseudo_mrn", "dfci_mrn", "split", "trial_start_dt", "tplan_start_dt", "epic_tplan_id", "plan",
    "is_trial", "treatment_plan_category", "tplan_goal", "is_palliative", "dx", "tplan_icd_dx_codes",
    "tplan_dx_name", "protocol_nbr", "first_cycle_dt", "tplan_dc_reason", "tplan_source_nm",
]


def _step(attrition: list, name: str, frame: pd.DataFrame) -> None:
    attrition.append({"step": name, "rows": int(len(frame)), "patients": int(frame["dfci_mrn"].nunique())})


def load_split(split_csv) -> pd.DataFrame:
    split = pd.read_csv(split_csv, usecols=["dfci_mrn", "split"])
    if split.dfci_mrn.isna().any() or (split.dfci_mrn % 1 != 0).any():
        raise ValueError("Non-integer dfci_mrn in the split file")
    split["dfci_mrn"] = split.dfci_mrn.astype("int64")
    if split.dfci_mrn.duplicated().any():
        raise ValueError("Duplicate dfci_mrn in the split file")
    return split


def pull_soc(treatment_plan_csv, split: pd.DataFrame, start_date: str, end_date: str,
             split_filter: str | None = "test") -> tuple[pd.DataFrame, list, dict]:
    """Return (SOC treatments after the split filter, attrition, counts across all splits)."""
    attrition: list = []
    tp = pd.read_csv(treatment_plan_csv, dtype=str, usecols=TP_COLUMNS)
    tp["dfci_mrn"] = pd.to_numeric(tp.DFCI_MRN, errors="raise").astype("int64")
    _step(attrition, "TREATMENT_PLAN rows", tp)

    tp["tplan_start_dt"] = to_day(tp.TPLAN_START_DT)
    tp = tp[~(tp.STD_CHEMO_PLAN.isna() & tp.RESEARCH_CHEMO_PLAN.isna())].copy()
    _step(attrition, "standard or research chemo plan name present", tp)
    tp["plan"] = np.where(tp.STD_CHEMO_PLAN.isna(), tp.RESEARCH_CHEMO_PLAN, tp.STD_CHEMO_PLAN)
    tp["is_trial"] = np.where(tp.STD_CHEMO_PLAN.isna(), 1, 0)
    tp["dx"] = tp.TPLAN_ICD_DX_CODES.str[:3]

    tp = tp[tp.TPLAN_GOAL.notna()].copy()
    _step(attrition, "TPLAN_GOAL present", tp)
    tp["is_palliative"] = np.where(tp.TPLAN_GOAL.str.contains("PALLIATIVE|CONTROL"), 1, 0)

    tp = tp[tp.TREATMENT_PLAN_CATEGORY.eq(SOC_CATEGORY)]
    _step(attrition, f"TREATMENT_PLAN_CATEGORY == '{SOC_CATEGORY}'", tp)

    lo, hi = pd.Timestamp(start_date), pd.Timestamp(end_date)
    tp = tp[(tp.tplan_start_dt >= lo) & (tp.tplan_start_dt <= hi)]
    _step(attrition, f"TPLAN_START_DT within {start_date}..{end_date} (inclusive)", tp)

    tp = tp.merge(split, on="dfci_mrn", how="inner", validate="many_to_one")
    _step(attrition, "patient has a train/validation/test split assignment", tp)

    tp = tp.rename(columns={
        "TREATMENT_PLAN_CATEGORY": "treatment_plan_category", "TPLAN_GOAL": "tplan_goal",
        "RESEARCH_CHEMO_PLAN_NBR": "protocol_nbr", "TPLAN_ICD_DX_CODES": "tplan_icd_dx_codes",
        "TPLAN_DX_NAME": "tplan_dx_name", "TPLAN_DC_REASON": "tplan_dc_reason",
        "EPIC_TPLAN_ID": "epic_tplan_id", "TPLAN_SOURCE_NM": "tplan_source_nm",
    })
    tp["first_cycle_dt"] = to_day(tp.FIRST_CYCLE_DT)
    tp["trial_start_dt"] = tp.tplan_start_dt  # v22 alias, for code shared with enrollments

    all_splits = {}
    for name, group in tp.groupby("split"):
        all_splits[name] = {"rows": int(len(group)), "patients": int(group.dfci_mrn.nunique()),
                            "index_events": int(group[["dfci_mrn", "trial_start_dt"]].drop_duplicates().shape[0])}

    if split_filter:
        tp = tp[tp.split.eq(split_filter)]
        _step(attrition, f"split == '{split_filter}'", tp)

    tp = assign_pseudo_mrn(tp, "trial_start_dt")
    return tp[OUTPUT_COLUMNS].reset_index(drop=True), attrition, all_splits
