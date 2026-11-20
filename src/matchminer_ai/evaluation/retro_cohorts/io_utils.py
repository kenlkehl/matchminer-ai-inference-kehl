"""Shared helpers: dates, protocol numbers, file stamps and PHI-safe output writing.

Nothing here prints row-level values. Callers log counts only.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DIR_MODE = 0o700
FILE_MODE = 0o600


def to_day(series: pd.Series) -> pd.Series:
    """Parse a date/datetime column and floor it to the calendar day (NaT stays NaT)."""
    return pd.to_datetime(series, errors="coerce", format="mixed").dt.normalize()


def protocol_number_to_int(series: pd.Series) -> pd.Series:
    """Numeric protocol key shared by OnCore ('17-123') and the enrollment table (17123).

    v22 (pan_dfci_2024/code/1.5_pull_eligibility_criteria.ipynb) joined OnCore protocol
    numbers to PROTOCOL_ENROLLMENT.STUDY_NBR exactly this way: drop the dash and compare
    as numbers, so '05-123' matches STUDY_NBR 5123. Non-numeric protocol numbers (for
    example 'TP1234' or '12-345B') become <NA> and cannot join.
    """
    text = series.astype("string").str.strip().str.replace("-", "", regex=False)
    return pd.to_numeric(text, errors="coerce").astype("Int64")


def ctgov_date_bounds(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """First and last day covered by a ClinicalTrials.gov date at month or day precision.

    '2021-05' covers 2021-05-01 through 2021-05-31; '2021-05-17' covers one day.
    Anything else becomes NaT on both sides.
    """
    text = series.astype("string").str.strip()
    day = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False)
    month = text.str.fullmatch(r"\d{4}-\d{2}").fillna(False)
    first = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    last = first.copy()
    first[day] = pd.to_datetime(text[day], format="%Y-%m-%d", errors="coerce")
    last[day] = first[day]
    month_start = pd.to_datetime(text[month], format="%Y-%m", errors="coerce")
    first[month] = month_start
    last[month] = month_start + pd.offsets.MonthEnd(0)
    return first, last


def file_stamp(path: str | Path) -> dict:
    """Size plus whole-second mtime: change detection without hashing large files."""
    path = Path(path)
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime": int(info.st_mtime)}


def ensure_private_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) and set it to mode 700."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, DIR_MODE)
    return path


def _private_file(path: Path) -> None:
    os.chmod(path, FILE_MODE)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != FILE_MODE:
        raise PermissionError(f"Could not set mode 600 on {path} (mode {oct(mode)})")


def write_csv_private(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV through a temporary name, then set mode 600 and move it into place."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, date_format="%Y-%m-%d")
    _private_file(tmp)
    tmp.replace(path)
    return path


def write_json_private(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")
    _private_file(tmp)
    tmp.replace(path)
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def assign_pseudo_mrn(frame: pd.DataFrame, date_col: str = "trial_start_dt") -> pd.DataFrame:
    """One pseudo_mrn per unique (dfci_mrn, index date), as in v22's prepare_data.py.

    v22 numbered combinations in order of appearance; here they are numbered from 1 in
    (index date, dfci_mrn) order so reruns on the same inputs give the same numbers.
    """
    combos = (frame[["dfci_mrn", date_col]].drop_duplicates()
              .sort_values([date_col, "dfci_mrn"]).reset_index(drop=True))
    combos["pseudo_mrn"] = range(1, len(combos) + 1)
    return frame.merge(combos, on=["dfci_mrn", date_col], how="left", validate="many_to_one")


def counts_by_year(frame: pd.DataFrame, date_col: str = "trial_start_dt") -> dict:
    """Per calendar year of the index date: rows, unique index events and unique patients."""
    out = {}
    years = frame[date_col].dt.year
    for year, group in frame.groupby(years):
        out[str(int(year))] = {
            "rows": int(len(group)),
            "index_events": int(group["pseudo_mrn"].nunique()),
            "patients": int(group["dfci_mrn"].nunique()),
        }
    return out


def cohort_counts(frame: pd.DataFrame, date_col: str = "trial_start_dt") -> dict:
    return {
        "rows": int(len(frame)),
        "index_events": int(frame["pseudo_mrn"].nunique()),
        "patients": int(frame["dfci_mrn"].nunique()),
        "first_index_date": None if frame.empty else str(frame[date_col].min().date()),
        "last_index_date": None if frame.empty else str(frame[date_col].max().date()),
        "by_year": counts_by_year(frame, date_col),
    }
