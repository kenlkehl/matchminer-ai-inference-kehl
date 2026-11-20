"""Trial reference tables (no PHI): OnCore protocols, DFCI trial dates, ctgov listings.

v22 sources and their v23 replacements (README.md, "Data source mapping"):

* OnCore protocol list. v22: pan_dfci_2024/structured_data/oncore_all.clean.csv
  (protocol_no, nct_id, protocol_type). v23: data/no_phi/oncore_nct_status_10-17-26.csv,
  the same OnCore fields from the 2026-10-16 OnCore dump.
* ctgov listing text. v22 fetched title/summary/criteria live from the ClinicalTrials.gov
  API in 2024. v23 reads the local 10-15-26 export (data/no_phi/ctgov_trials.csv) and
  makes no network calls.
* DFCI trial dates. data/no_phi/dfci_trials_summary_10-18-26.csv. Its open_date and
  close_date are ClinicalTrials.gov study start and completion dates (month or day
  precision) joined to OnCore protocols by NCT ID, not DFCI site accrual dates.
"""

from __future__ import annotations

import pandas as pd

from .io_utils import ctgov_date_bounds, protocol_number_to_int

LISTING_COLUMNS = ["nct_id", "title", "brief_summary", "eligibility_criteria", "trial_text"]


def _clean_nct(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.upper()
    return text.where(text.str.fullmatch(r"NCT\d{8}").fillna(False))


def load_oncore_treatment_protocols(oncore_status_csv, dfci_trials_csv=None) -> tuple[pd.DataFrame, dict]:
    """OnCore therapeutic protocols with an NCT ID, one row per protocol.

    v22 used every protocol_type == 'Treatment' protocol with an NCT ID in its OnCore list.
    The v23 status file is the same kind of list. With ``dfci_trials_csv`` (the full
    2026-10-16 OnCore dump, 11,277 protocols, with NCT IDs but no protocol type) two gaps
    are filled, and ``nct_source`` records which file gave the NCT ID:

    * a 'Treatment' protocol in the status file without an NCT ID takes the summary's;
    * a protocol absent from the status file (mostly older protocols) takes the summary's
      NCT ID, and its therapeutic type then rests on PROTOCOL_ENROLLMENT.TREATMENT_TYPE_CD
      == 'Tre', which the enrollment pull requires anyway. (Where both exist, 'Tre' and
      OnCore 'Treatment' agree for every protocol with enrollments.)

    v22 also kept only the first protocol per NCT ID (``oncore.groupby('nct_id').first()``),
    silently dropping enrollments on the other protocols sharing that NCT ID. That is not
    reproduced: every protocol is kept and ``first_protocol_for_nct`` marks the protocol
    v22's rule would have kept (status-file order; <NA> for summary-only protocols), so the
    effect can be counted.
    """
    status = pd.read_csv(oncore_status_csv, dtype=str)
    status["protocol_nbr"] = protocol_number_to_int(status.protocol_no)
    status["nct_id"] = _clean_nct(status.nct_id)
    counts = {"status_file_protocols": int(len(status))}
    treatment = status[status.protocol_type.eq("Treatment") & status.protocol_nbr.notna()].copy()
    counts["status_file_treatment_protocols"] = int(len(treatment))
    treatment["nct_source"] = pd.Series("oncore_status", index=treatment.index).where(treatment.nct_id.notna())
    with_nct = treatment[treatment.nct_id.notna()]
    treatment["first_protocol_for_nct"] = pd.Series(pd.NA, index=treatment.index, dtype="boolean")
    treatment.loc[with_nct.index, "first_protocol_for_nct"] = ~with_nct.nct_id.duplicated(keep="first")
    counts["status_file_treatment_protocols_with_nct"] = int(len(with_nct))
    treatment = treatment.rename(columns={"protocol_no": "protocol_number", "status": "oncore_status",
                                          "phase": "oncore_phase", "protocol_type": "oncore_protocol_type"})
    keep = ["protocol_number", "protocol_nbr", "nct_id", "nct_source", "oncore_status", "oncore_phase",
            "oncore_protocol_type", "first_protocol_for_nct"]
    out = treatment[keep]

    if dfci_trials_csv is not None:
        summary = pd.read_csv(dfci_trials_csv, dtype=str)
        summary["protocol_nbr"] = protocol_number_to_int(summary.protocol_no)
        summary["nct_id"] = _clean_nct(summary.nct_id)
        summary = summary[summary.protocol_nbr.notna() & summary.nct_id.notna()]
        summary_nct = summary.set_index("protocol_nbr").nct_id
        fill = out.nct_id.isna() & out.protocol_nbr.isin(summary_nct.index)
        out.loc[fill, "nct_id"] = out.loc[fill, "protocol_nbr"].map(summary_nct)
        out.loc[fill, "nct_source"] = "dfci_trials_summary"
        counts["status_file_treatment_protocols_nct_from_summary"] = int(fill.sum())
        extra = summary[~summary.protocol_nbr.isin(status.protocol_nbr)].rename(
            columns={"protocol_no": "protocol_number", "status": "oncore_status"})
        # Summary-only protocols carry no OnCore protocol type (the research-plan cohort excludes them).
        extra = extra.assign(oncore_phase=pd.NA, oncore_protocol_type=pd.NA, nct_source="dfci_trials_summary",
                             first_protocol_for_nct=pd.Series(pd.NA, index=extra.index, dtype="boolean"))
        counts["summary_only_protocols_with_nct_any_type"] = int(len(extra))
        out = pd.concat([out, extra[keep]], ignore_index=True)

    out = out[out.nct_id.notna()].reset_index(drop=True)
    if out.protocol_nbr.duplicated().any():
        raise ValueError("Two OnCore protocol numbers map to the same numeric key")
    counts["protocols_with_nct"] = int(len(out))
    counts["nct_ids"] = int(out.nct_id.nunique())
    return out, counts


def load_dfci_trial_dates(dfci_trials_csv) -> pd.DataFrame:
    """One row per NCT ID: ctgov study start/completion dates from the DFCI trial summary.

    The file has one row per OnCore protocol; protocols that share an NCT ID share its
    ctgov dates, so the first non-null value per NCT ID is kept.
    """
    dates = pd.read_csv(dfci_trials_csv, dtype=str)
    dates["nct_id"] = _clean_nct(dates.nct_id)
    dates = dates[dates.nct_id.notna()].copy()
    for column in ["open_date", "close_date"]:
        values = dates[column].dropna()
        if not values.str.fullmatch(r"\d{4}-\d{2}(-\d{2})?").all():
            raise ValueError(f"Unexpected {column} format in {dfci_trials_csv}")
    out = (dates.groupby("nct_id")[["open_date", "open_date_type", "close_date", "close_date_type"]]
           .first().reset_index()
           .rename(columns={"open_date": "ctgov_start_date", "open_date_type": "ctgov_start_date_type",
                            "close_date": "ctgov_completion_date",
                            "close_date_type": "ctgov_completion_date_type"}))
    return out


def dfci_holdout_nct_ids(dfci_trials_csv, first_year: int = 2025, last_year: int = 2026) -> set:
    """NCT IDs v23 training held out as 'open at DFCI 2025-2026'.

    Same rule as matchminer_ai.training.trials.sample_trials.dfci_nct_ids_open_during
    (re-implemented here so this repository stays freestanding): a trial counts unless its
    ctgov dates place it wholly outside the years; a missing date leaves that side open;
    a protocol still 'OPEN TO ACCRUAL' always counts.
    """
    trials = pd.read_csv(dfci_trials_csv, dtype=str).dropna(subset=["nct_id"])
    open_year = pd.to_numeric(trials.open_date.fillna("").str[:4], errors="coerce")
    close_year = pd.to_numeric(trials.close_date.fillna("").str[:4], errors="coerce")
    outside = (close_year < first_year) | (open_year > last_year)
    keep = ~outside | trials.status.eq("OPEN TO ACCRUAL")
    return set(trials.nct_id[keep].str.strip().str.upper())


def load_training_nct_ids(training_sample_csv) -> set:
    sample = pd.read_csv(training_sample_csv, dtype=str, usecols=["nct_id"])
    return set(sample.nct_id.str.strip().str.upper())


def load_ctgov_listings(ctgov_csv, nct_ids=None) -> pd.DataFrame:
    """Listing text (title, brief summary, eligibility criteria, trial_text) by NCT ID."""
    listings = pd.read_csv(ctgov_csv, dtype=str, usecols=LISTING_COLUMNS)
    listings["nct_id"] = listings.nct_id.str.strip().str.upper()
    if nct_ids is not None:
        listings = listings[listings.nct_id.isin(set(nct_ids))]
    # v22 dropped trials whose brief summary came back empty from the API.
    listings = listings[listings.brief_summary.fillna("").str.strip().ne("")]
    return listings.drop_duplicates("nct_id").reset_index(drop=True)


def index_in_ctgov_window(index_dates: pd.Series, start: pd.Series, completion: pd.Series) -> pd.Series:
    """Whether each index date falls within the trial's ctgov start..completion span.

    Month-precision dates are widened to the whole month. <NA> when either date is missing.
    """
    first, _ = ctgov_date_bounds(start)
    _, last = ctgov_date_bounds(completion)
    inside = (index_dates >= first) & (index_dates <= last)
    return inside.astype("boolean").mask(first.isna() | last.isna())


def sample_trials(universe_nct_ids, exclude_nct_ids, n_trials: int = 500, seed: int = 42) -> list:
    """v22 prepare_trials.py: drop training trials, then sample N unique NCT IDs.

    v22 did ``remaining.groupby('nct_id').first().reset_index()[['nct_id']].sample(n, random_state)``;
    groupby sorts the IDs, so sorting before sampling reproduces its draw for a given pool.
    """
    remaining = sorted(set(universe_nct_ids) - set(exclude_nct_ids))
    frame = pd.DataFrame({"nct_id": remaining})
    return frame.sample(n=min(n_trials, len(frame)), random_state=seed).nct_id.tolist()
