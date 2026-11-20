"""The dashboard's trial data: OA trials, their cohorts and each cohort's criteria, assembled as extraction runs.

Sources (all read-only):

- ``trial_data/inputs.csv`` (``process_oa_trials.py``): one row per OA trial, with why a version was not run.
- ``trial_data/<version>/trials/<NCT>/deboilerplate.json`` for ``protocol`` and ``ctgov``: a trial's merged cohorts
  with key and boilerplate ("common") criteria and every criteria line tagged inclusion/exclusion and
  key/boilerplate. The package writes it atomically once all of the trial's lines are answered, so each file is a
  complete result; a trial appears here the moment its file does, without waiting for the run's final
  ``cohorts.parquet``. ``extraction.json`` (protocols) gives the cancer-treatment screen and whether a newer
  extraction is still on its way; ``merge.json`` how many extracted cohorts each merged cohort stands for.
- ``<version>/cohorts.parquet``: the last complete run, used only for a trial that has no per-trial file.
- The OA DuckDB copy's trial table only (titles, management group, PIs, contact, sites, ct.gov summary): the site
  config's ``dashboard.trial_table`` names the table and maps the dashboard's fields to its columns
  (``trial_table``). No patient table is read.
- The GoodOption catalog (``drugs.py``): each trial's drugs, their roles and NCIt mechanism classes. Without it the
  trials have no drugs.

Each trial shows one version on its disease center's dashboard: its protocol cohorts when it has them, else its
ct.gov cohorts; the other version stays in the trial's details. Cohorts are grouped by ``grouping.py``.

The output (``dashboard.json``) holds confidential protocol-derived text: it is written with owner-only
permissions next to the trial data, never into the app's repository.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time

from matchminer_ai.site import load_site_config

from matchminer_ai.trials.cohort_run import read_run_cohorts, read_trial_results

from .core.cache import file_stamp, text_key

from . import drugs as drug_catalog
from . import grouping

FORMAT = "mmai-dash-store-v2"  # v2: site_pi (a site-neutral name), center_short_name
VERSIONS = ("protocol", "ctgov")
VERSION_LABELS = {"protocol": "Protocol", "ctgov": "CT.gov"}
# The trial table's fields the dashboard reads, by its own names. A site maps them to its database's columns
# (dashboard.trial_table.columns; a field without an entry is a column of the same name) and names the table
# (dashboard.trial_table.name, default "trials").
TRIAL_FIELDS = ("nct_id", "protocol_no", "short_title", "long_title", "management_group", "contact_name",
                "contact_email", "contact_role", "site_pi", "overall_pi", "status", "phase", "brief_summary",
                "site_list", "ct_response")


def trial_table() -> tuple[str, dict[str, str]]:
    """(table, {field: column}) of the OA DuckDB copy's trial table."""
    site = load_site_config()
    columns = site.section("dashboard.trial_table.columns")
    return site.get("dashboard.trial_table.name") or "trials", {f: columns.get(f, f) for f in TRIAL_FIELDS}


def site_name_map() -> dict[str, str]:
    """The trial table's site names and the short names shown (dashboard.site_names). Without a map every site
    is shown by its own name; with one, only the mapped sites."""
    return {str(k): str(v) for k, v in load_site_config().section("dashboard.site_names").items()}


def branding() -> dict:
    """What the frontend shows of the site (dashboard.branding): its name, the confidentiality notice, the label of
    the site PI, institution groups of the disease centers (labels and name prefixes) and links from a trial's
    protocol number (URL templates with ``{protocol_no}``). Without a site config: none of them."""
    value = load_site_config().section("dashboard.branding")
    institutions = [{"label": str(i.get("label", "")), "prefixes": [str(p) for p in i.get("prefixes") or []]}
                    for i in value.get("institutions") or [] if isinstance(i, dict)]
    links = [{"label": str(i.get("label", "")), "url": str(i.get("url", ""))}
             for i in value.get("trial_links") or [] if isinstance(i, dict) and "{protocol_no}" in str(i.get("url"))]
    return {"institution": str(value.get("institution") or ""),
            "confidentiality": str(value.get("confidentiality") or ""),
            "pi_label": str(value.get("pi_label") or "Site PI"),
            "higher_threshold_label": str(value.get("higher_threshold_label") or "Exclude higher-threshold programs"),
            "institutions": institutions, "trial_links": links}


def higher_threshold_groups() -> list[str]:
    """Coordinating centers left out of the AI tab unless included (dashboard.higher_threshold_groups)."""
    return [str(g) for g in load_site_config().get("dashboard.higher_threshold_groups") or []]


def cohort_id(trial_id: str, version: str, number: int) -> str:
    """The training corpus's convention: ``<NCT>-cohort-<n>`` (protocol), ``<NCT>-ctgov-cohort-<n>``."""
    return f"{trial_id}-cohort-{number}" if version == "protocol" else f"{trial_id}-ctgov-cohort-{number}"


def center_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "unassigned"


def center_short_name(name: str) -> str:
    """The center's name without the site config's ``dashboard.center_prefix`` (a regular expression, such as the
    home institution's prefix); other centers keep their prefix."""
    prefix = load_site_config().get("dashboard.center_prefix")
    return (re.sub(prefix, "", name).strip() or name) if prefix else name


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("none", "nan") else text


def _literal(value):
    """OA stores ct_response and site_list as Python reprs (sometimes JSON)."""
    if not value or not isinstance(value, str):
        return value or None
    for parse in (ast.literal_eval, json.loads):
        try:
            parsed = parse(value)
            return _literal(parsed) if isinstance(parsed, str) else parsed
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return None


def _ct_text(value) -> str:
    """ct_response strings are quoted once more ("'text'")."""
    text = _clean(value)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    return text.replace("\\n", "\n").replace("\\>", ">").replace("\\<", "<")


def site_names(site_list) -> list[str]:
    sites = _literal(site_list) or {}
    mapping = site_name_map()
    listed = [s.get("site_name") for s in (sites.get("site") or []) if isinstance(s, dict) and s.get("site_name")]
    names = {mapping[s] for s in listed if s in mapping} if mapping else set(map(str, listed))
    return sorted(names)


def phase_label(phase: str) -> str:
    phase = _clean(phase)
    return phase if not phase or phase.lower().startswith(("phase", "expanded")) else f"Phase {phase}"


def status_label(status: str) -> str:
    status = _clean(status)
    return {"OPEN TO ACCRUAL": "Open to accrual", "inactive_trial": "Inactive"}.get(status, status.title())


# ---------------------------------------------------------------------------------------------- reading files


class FileCache:
    """Parsed JSON per path, re-read only when the file's stamp changes (the builder runs every poll)."""

    def __init__(self):
        self.values: dict[Path, tuple[str | None, object]] = {}
        self.lock = threading.Lock()

    def json(self, path: Path):
        stamp = file_stamp(path)
        if stamp is None:
            return None
        with self.lock:
            cached = self.values.get(path)
        if cached and cached[0] == stamp:
            return cached[1]
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):  # being written (non-atomic writers): tried again next poll
            return cached[1] if cached else None
        with self.lock:
            self.values[path] = (stamp, value)
        return value


def _version_trial(directory: Path, version: str, trial_id: str, files: FileCache, fallback: dict) -> dict:
    """One version of one trial: its state and merged cohorts, from the per-trial files (else the run's parquet;
    ``matchminer_ai.trials.cohort_run``)."""
    return {"version": version, **read_trial_results(
        directory / version / "trials" / trial_id, screened=version == "protocol",
        fallback_rows=fallback.get((version, trial_id)), read_json=files.json)}


def _parquet_rows(directory: Path, version: str) -> dict:
    return {(version, trial_id): rows for trial_id, rows in read_run_cohorts(directory / version).items()}


def read_oa_trials(db_path: Path) -> dict[str, dict]:
    """The trial table (``trial_table``), by NCT ID, with the dashboard's field names (trial-level columns only)."""
    import duckdb

    def quoted(name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    table, columns = trial_table()
    select = ", ".join(quoted(columns[f]) for f in TRIAL_FIELDS)
    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute(f"SELECT {select} FROM {quoted(table)}").fetchall()
    return {row[0]: dict(zip(TRIAL_FIELDS, row)) for row in rows if row[0]}


def read_inputs(path: Path) -> list[dict]:
    import pandas as pd

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return frame.to_dict("records")


# ---------------------------------------------------------------------------------------------- building


@dataclass
class Sources:
    trial_data: Path
    oa_db: Path  # a file, or a glob pattern (e.g. oa_db_*.duckdb) meaning its newest match
    grouping: Path
    good_options: Path | None = None  # the GoodOption catalog directory (drugs.py); None or missing: no drugs

    @property
    def oa_db_file(self) -> Path:
        if not any(c in self.oa_db.name for c in "*?["):
            return self.oa_db
        matches = sorted(self.oa_db.parent.glob(self.oa_db.name), key=lambda p: p.stat().st_mtime)
        if not matches:
            raise FileNotFoundError(f"no OA database matches {self.oa_db}")
        return matches[-1]

    def signature(self) -> dict:
        """What the store was built from: file stamps (size and mtime), never content hashes of data files."""
        db = self.oa_db_file
        stamps = {"inputs": file_stamp(self.trial_data / "inputs.csv"), "oa_db": f"{db.name}:{file_stamp(db)}",
                  "grouping": file_stamp(self.grouping), "pipeline": PIPELINE_VERSION}
        site = load_site_config()
        if site.path is not None:  # the trial table, site names and center prefix come from it
            stamps["site_config"] = file_stamp(site.path)
        for version in VERSIONS:
            stamps[f"{version}/cohorts.parquet"] = file_stamp(self.trial_data / version / "cohorts.parquet")
            folder = self.trial_data / version / "trials"
            entries = []
            if folder.is_dir():
                for trial in sorted(os.scandir(folder), key=lambda e: e.name):
                    if trial.is_dir():
                        for name in ("extraction.json", "merge.json", "deboilerplate.json"):
                            stamp = file_stamp(Path(trial.path) / name)
                            if stamp:
                                entries.append(f"{trial.name}/{name}:{stamp}")
            stamps[f"{version}/trials"] = text_key(entries)
        if self.good_options is not None:
            stamps["good_options"] = str(self.good_options)
            for name, stamp in drug_catalog.catalog_stamps(self.good_options).items():
                stamps[f"good_options/{name}"] = stamp
        return stamps


@dataclass
class Builder:
    sources: Sources
    files: FileCache = field(default_factory=FileCache)
    _oa: tuple[str | None, dict] | None = None
    _parquet: dict = field(default_factory=dict)
    _catalog: tuple | None = None

    def oa_trials(self) -> dict:
        db = self.sources.oa_db_file
        stamp = f"{db}:{file_stamp(db)}"
        if self._oa is None or self._oa[0] != stamp:
            self._oa = (stamp, read_oa_trials(db))
        return self._oa[1]

    def parquet_rows(self) -> dict:
        rows = {}
        for version in VERSIONS:
            stamp = file_stamp(self.sources.trial_data / version / "cohorts.parquet")
            if version not in self._parquet or self._parquet[version][0] != stamp:
                self._parquet[version] = (stamp, _parquet_rows(self.sources.trial_data, version))
            rows.update(self._parquet[version][1])
        return rows

    def catalog(self) -> tuple[dict[str, list[dict]], dict]:
        """Each trial's drugs from the GoodOption catalog, re-read only when its files' stamps change."""
        directory = self.sources.good_options
        key = (str(directory), drug_catalog.catalog_stamps(directory) if directory is not None else None)
        if self._catalog is None or self._catalog[0] != key:
            self._catalog = (key, drug_catalog.read_catalog(directory))
        return self._catalog[1]

    def build(self) -> dict:
        started = time.monotonic()
        signature = self.sources.signature()
        rules = grouping.load_rules(self.sources.grouping)
        inputs = read_inputs(self.sources.trial_data / "inputs.csv")
        oa = self.oa_trials()
        fallback = self.parquet_rows()
        trial_drugs, catalog = self.catalog()
        trials, cohorts, centers = {}, {}, {}
        for row in inputs:
            trial_id = row["trial_id"]
            info = oa.get(trial_id, {})
            center = _clean(info.get("management_group")) or "Unassigned"
            ct = (_literal(info.get("ct_response")) or {}).get("protocolSection", {}) if info.get("ct_response") else {}
            versions = {}
            for version in VERSIONS:
                excluded = _clean(row.get(f"{version}_excluded_reason"))
                state = _version_trial(self.sources.trial_data, version, trial_id, self.files, fallback)
                if excluded and not state["cohorts"]:
                    state["state"] = "not_run"
                state["excluded_reason"] = excluded or None
                versions[version] = state
            primary = next((v for v in VERSIONS if versions[v]["cohorts"]), None)
            trial = {
                "id": trial_id, "nct_id": trial_id, "protocol_no": _clean(row.get("protocol_no")) or _clean(info.get("protocol_no")),
                "short_title": _clean(info.get("short_title")) or _clean(row.get("short_title")),
                "long_title": _clean(info.get("long_title")),
                "official_title": _ct_text((ct.get("identificationModule") or {}).get("officialTitle")),
                "phase": phase_label(info.get("phase") or row.get("phase")),
                "status": status_label(info.get("status") or row.get("oa_status")),
                "is_open": (_clean(info.get("status") or row.get("oa_status")) == "OPEN TO ACCRUAL"),
                "center": center, "center_id": center_id(center), "center_short_name": center_short_name(center),
                "site_pi": _clean(info.get("site_pi")),
                "overall_pi": _clean(info.get("overall_pi")),
                "contact": {"name": _clean(info.get("contact_name")),
                            "email": _clean(info.get("contact_email")),
                            "role": _clean(info.get("contact_role"))},
                "sites": site_names(info.get("site_list")),
                "brief_summary": _clean(info.get("brief_summary")) or _ct_text(
                    (ct.get("descriptionModule") or {}).get("briefSummary")),
                "detailed_description": _ct_text((ct.get("descriptionModule") or {}).get("detailedDescription")),
                "primary_version": primary, "versions": {}, "cohort_ids": [],
                "drugs": trial_drugs.get(trial_id, []),
            }
            for version in VERSIONS:
                state = versions[version]
                ids = []
                for c in sorted(state["cohorts"], key=lambda c: c["number"]):
                    cid = cohort_id(trial_id, version, c["number"])
                    ids.append(cid)
                    cohorts[cid] = {"id": cid, "trial_id": trial_id, "version": version, "primary": version == primary,
                                    **c, "cancer_types": grouping.cohort_cancer_types(c, trial)}
                trial["versions"][version] = {k: state[k] for k in ("state", "completed_at", "screen",
                                                                    "extracted_cohorts", "excluded_reason")}
                trial["versions"][version]["cohort_ids"] = ids
            trial["cohort_ids"] = trial["versions"][primary]["cohort_ids"] if primary else []
            trial["state"] = dashboard_state(trial)
            trials[trial_id] = trial
            centers.setdefault(trial["center_id"], {"id": trial["center_id"], "name": center,
                                                    "short_name": center_short_name(center), "trial_ids": []})
            centers[trial["center_id"]]["trial_ids"].append(trial_id)
        for center in centers.values():
            grouping.assign_groups(center, trials, cohorts, rules)
        ordered = sorted(centers.values(), key=lambda c: (-len(c["trial_ids"]), c["name"]))
        return {"format": FORMAT, "built_at": time.time(), "build_seconds": round(time.monotonic() - started, 2),
                "signature": signature, "oa_db": str(self.sources.oa_db_file), "rules": {"path": str(self.sources.grouping), "centers": rules.describe()},
                "centers": ordered, "trials": trials, "cohorts": cohorts,
                "higher_threshold_groups": higher_threshold_groups(),
                "good_options": {**catalog, "shown": drug_catalog.shown_counts(trials)},
                "counts": counts(trials, cohorts)}


PIPELINE_VERSION = "2026-11-20.1"  # site config (trial table, site names, center prefix); site_pi  # trials' drugs (GoodOption catalog), with their classes' IDs


def dashboard_state(trial: dict) -> str:
    """Why a trial has (or lacks) cohorts on the dashboard."""
    if trial["primary_version"]:
        return "stale" if trial["versions"][trial["primary_version"]]["state"] == "stale" else "ready"
    states = {v["state"] for v in trial["versions"].values()}
    if "not_treatment" in states:
        return "not_treatment"
    if states <= {"not_run"}:
        return "not_run"
    return "pending"


def counts(trials: dict, cohorts: dict) -> dict:
    by_state: dict[str, int] = {}
    for trial in trials.values():
        by_state[trial["state"]] = by_state.get(trial["state"], 0) + 1
    by_version = {v: sum(1 for c in cohorts.values() if c["version"] == v) for v in VERSIONS}
    return {"trials": len(trials), "trials_by_state": by_state, "cohorts": len(cohorts),
            "cohorts_by_version": by_version, "dashboard_cohorts": sum(1 for c in cohorts.values() if c["primary"]),
            "trials_by_version": {v: sum(1 for t in trials.values() if t["versions"][v]["cohort_ids"]) for v in VERSIONS}}


def write_store(store: dict, path: Path) -> None:
    """Atomically, owner-only (the trial data's own permissions)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(store, stream, ensure_ascii=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_store(path: Path) -> dict | None:
    try:
        store = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return store if store.get("format") == FORMAT else None
