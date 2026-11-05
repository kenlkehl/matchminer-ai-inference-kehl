"""Which text each stage reads: key text for retrieval and TrialChecker, boilerplate for BoilerplateChecker.

A cohort's key criteria are its criteria that are not boilerplate (``cohort_key_criteria`` from
``remove_boilerplate_criteria``); a patient summary's key section is its cancer history
(``cancer_history_summary``, the text before ``Other relevant history:``). The boilerplate
counterparts are ``cohort_boilerplate_criteria`` and ``general_exclusion_criteria_evidence``.
Historical space tables (``clinical_space_summary`` / ``general_exclusion_criteria``) are accepted.
"""

from __future__ import annotations

import pandas as pd

TRIAL_ID_COLUMNS = ("cohort_trial_id", "space_trial_id")
TRIAL_KEY_COLUMNS = ("cohort_key_criteria", "cohort_eligibility_criteria", "clinical_space_summary")
TRIAL_BOILERPLATE_COLUMNS = ("cohort_boilerplate_criteria", "general_exclusion_criteria")


def _first(df: pd.DataFrame, columns: tuple[str, ...], what: str) -> str:
    column = next((c for c in columns if c in df.columns), None)
    if column is None:
        raise ValueError(f"{what} needs one of the columns {list(columns)}")
    return column


def trial_id_column(df: pd.DataFrame) -> str:
    return _first(df, TRIAL_ID_COLUMNS, "trial rows")


def trial_key_texts(df: pd.DataFrame) -> list[str]:
    """Key criteria, or all the criteria when the boilerplate step did not run."""
    return df[_first(df, TRIAL_KEY_COLUMNS, "trial rows")].fillna("").astype(str).tolist()


def trial_boilerplate_texts(df: pd.DataFrame) -> list[str]:
    return df[_first(df, TRIAL_BOILERPLATE_COLUMNS, "trial rows")].fillna("").astype(str).tolist()


def patient_key_texts(df: pd.DataFrame) -> list[str]:
    if "cancer_history_summary" in df.columns:
        return df["cancer_history_summary"].fillna("").astype(str).tolist()
    if "patient_summary" in df.columns:
        from matchminer_ai.patients.postprocess import split_patient_summary
        return [split_patient_summary(text)[0] for text in df["patient_summary"].fillna("").astype(str)]
    raise ValueError("patient rows need cancer_history_summary or patient_summary")


def patient_boilerplate_texts(df: pd.DataFrame) -> list[str]:
    if "general_exclusion_criteria_evidence" in df.columns:
        return df["general_exclusion_criteria_evidence"].fillna("").astype(str).tolist()
    if "patient_summary" in df.columns:
        from matchminer_ai.patients.postprocess import split_patient_summary
        return [split_patient_summary(text)[1] for text in df["patient_summary"].fillna("").astype(str)]
    raise ValueError("patient rows need general_exclusion_criteria_evidence or patient_summary")
