"""The TrialSpace corpus: the exact patient and cohort texts the models read, in vector-row order.

``trialspace_corpus/patients.jsonl`` holds each synthetic patient's key summary section (``text``) and
boilerplate section; ``cohorts.jsonl`` each merged cohort's key criteria (``text``), boilerplate criteria and
tagged lines (``items``: number, heading, text, criterion_type, boilerplate). Rows are loaded with the
inference package's own ``load_corpus`` (as training's mining does), so row *i* is the row the mining
stage encoded as vector *i*. The rows' digests are checked against the mining manifest (see vectors.py);
they are computed once per file stamp and cached.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np

from .cache import JSONCache, file_stamp
from .textsearch import TextIndex, parse_terms

HEADER = {"age": re.compile(r"^Age:\s*(.+)$", re.I), "sex": re.compile(r"^Sex:\s*(.+)$", re.I),
          "cancer_type": re.compile(r"^Cancer type:\s*(.+)$", re.I),
          "histology": re.compile(r"^Histology:\s*(.+)$", re.I)}
SEARCH_SCOPES = ("text", "fields")  # the list search: whole texts, or the IDs and short identifying fields


def parse_header(text: str) -> dict:
    """Age, sex, cancer type and histology from the first lines of a key summary section (when present)."""
    found = {}
    for line in text.splitlines()[:12]:
        line = line.strip()
        for key, pattern in HEADER.items():
            match = pattern.match(line)
            if match and key not in found:
                found[key] = match.group(1).strip()
    return found


def cohort_number(cohort_id: str) -> int | None:
    match = re.search(r"-(?:ctgov-)?cohort-(\d+)$", cohort_id)
    return int(match.group(1)) if match else None


class Corpus:
    def __init__(self, directory: Path, metadata_path: Path | None, cache: JSONCache, *, verify: bool = True):
        from matchminer_ai.labeling.corpus import load_corpus

        started = time.monotonic()
        self.directory = Path(directory)
        self.receipt = _read_json(self.directory / "stage_receipt.json")
        self.info = _read_json(self.directory / "corpus.json")
        self.patients, self.cohorts = load_corpus(self.directory)
        self.patient_index = {row["id"]: i for i, row in enumerate(self.patients)}
        self.cohort_index = {row["id"]: i for i, row in enumerate(self.cohorts)}
        self.headers = [parse_header(row["text"]) for row in self.patients]
        self.meta = load_patient_metadata(metadata_path) if metadata_path else {}
        self.trial_cohorts: dict[tuple[str, str], list[str]] = {}
        for row in self.cohorts:
            self.trial_cohorts.setdefault((row["trial_id"], row["source"]), []).append(row["id"])
        self.from_cohort: dict[str, list[str]] = {}
        for pid, meta in self.meta.items():
            if meta.get("source_cohort"):
                self.from_cohort.setdefault(meta["source_cohort"], []).append(pid)
        self.digests = self._digests(cache) if verify else {}
        self.load_seconds = round(time.monotonic() - started, 1)
        self._search_lock = threading.Lock()
        self._text_indexes: dict[tuple[str, str], TextIndex] = {}
        self._patient_columns: dict[str, np.ndarray] | None = None

    def _digests(self, cache: JSONCache) -> dict:
        """The patients' and cohorts' row digests (as the mining manifest records them), cached per file stamp."""
        from matchminer_ai.labeling.files import digest

        key = {name: file_stamp(self.directory / name) for name in ("patients.jsonl", "cohorts.jsonl")}
        cached = cache.get("corpus_digests")
        if cached and cached.get("key") == key:
            return cached["value"]
        value = {"patients": digest(self.patients), "cohorts": digest(self.cohorts)}
        cache.put("corpus_digests", {"key": key, "value": value})
        return value

    @property
    def receipt_id(self) -> str | None:
        return (self.receipt or {}).get("receipt_id")

    # ------------------------------------------------------------------ patients

    def patient(self, pid: str) -> dict | None:
        i = self.patient_index.get(pid)
        return None if i is None else self.patients[i]

    def patient_summary(self, pid: str) -> dict:
        i = self.patient_index[pid]
        row, meta = self.patients[i], self.meta.get(pid, {})
        return {"id": pid, **self.headers[i], "cls": meta.get("cls"), "source_trial": meta.get("source_trial")
                or row.get("source_trial_id"), "source_cohort": meta.get("source_cohort")}

    def _patient_text(self, i: int, scope: str) -> str:
        row, h, meta = self.patients[i], self.headers[i], self.meta.get(self.patients[i]["id"], {})
        ids = [row["id"], meta.get("source_trial") or row.get("source_trial_id") or "", meta.get("source_cohort") or ""]
        if scope == "fields":
            return "\n".join(ids + [h.get("cancer_type", ""), h.get("histology", "")])
        return "\n".join(ids + [row.get("text") or "", row.get("boilerplate") or ""])

    def _cohort_text(self, i: int, scope: str, titles: dict) -> str:
        row = self.cohorts[i]
        fields = [row["id"], row.get("cohort_name") or "", titles.get(row["trial_id"], "")]
        return "\n".join(fields if scope == "fields" else fields + [row.get("text") or "", row.get("boilerplate") or ""])

    def _text_index(self, side: str, scope: str, titles: dict | None = None) -> TextIndex:
        """The search texts of every patient or cohort, built on first use (a second or two for the full corpus)."""
        if scope not in SEARCH_SCOPES:
            raise ValueError(f"scope must be one of {SEARCH_SCOPES}")
        with self._search_lock:
            if (side, scope) not in self._text_indexes:
                if side == "patients":
                    texts = (self._patient_text(i, scope) for i in range(len(self.patients)))
                else:
                    texts = (self._cohort_text(i, scope, titles or {}) for i in range(len(self.cohorts)))
                self._text_indexes[side, scope] = TextIndex(texts)
            return self._text_indexes[side, scope]

    def warm_search(self, titles: dict | None = None) -> None:
        """Build the whole-text indexes now (a few seconds, at startup) so the first search does not wait."""
        self._text_index("patients", "text")
        self._text_index("cohorts", "text", titles)

    def search_patients(self, q: str = "", cls: str = "", sex: str = "", offset: int = 0, limit: int = 200,
                        allowed=None, scope: str = "text") -> dict:
        """Every patient matching the query (regex terms, all must match; textsearch.py) and the filters, in corpus
        order. ``scope`` "text" searches the IDs and the whole summary (key section and other relevant history),
        "fields" the IDs, cancer type and histology. ``allowed`` masks rows (e.g. a cancer type)."""
        started = time.monotonic()
        keep = np.ones(len(self.patients), dtype=bool) if allowed is None else np.array(allowed, dtype=bool)
        if cls or sex:
            with self._search_lock:
                if self._patient_columns is None:
                    self._patient_columns = {
                        "cls": np.array([self.meta.get(row["id"], {}).get("cls") or "" for row in self.patients]),
                        "sex": np.array([(h.get("sex") or "").lower() for h in self.headers])}
            if cls:
                keep &= self._patient_columns["cls"] == cls
            if sex:
                keep &= self._patient_columns["sex"] == sex.lower()
        if parse_terms(q):
            keep &= self._text_index("patients", scope).match(q)
        hits = np.flatnonzero(keep)
        rows = [self.patient_summary(self.patients[i]["id"]) for i in hits[offset:offset + limit]]
        return {"total": int(len(hits)), "of": len(self.patients), "offset": offset, "rows": rows,
                "search_ms": round((time.monotonic() - started) * 1000, 1)}

    # ------------------------------------------------------------------ cohorts

    def cohort(self, cid: str) -> dict | None:
        i = self.cohort_index.get(cid)
        return None if i is None else self.cohorts[i]

    def cohort_summary(self, cid: str, titles: dict | None = None) -> dict:
        row = self.cohorts[self.cohort_index[cid]]
        return {"id": cid, "trial_id": row["trial_id"], "source": row["source"], "name": row.get("cohort_name") or "",
                "number": cohort_number(cid), "title": (titles or {}).get(row["trial_id"], ""),
                "lines": len(row.get("items") or []), "patients_generated": len(self.from_cohort.get(cid, []))}

    def search_cohorts(self, q: str = "", source: str = "", titles: dict | None = None, offset: int = 0,
                       limit: int = 200, allowed=None, scope: str = "text") -> dict:
        """Every cohort matching the query (regex terms, all must match; textsearch.py) and the filters, in corpus
        order. ``scope`` "text" searches the ID, name, trial title and the key and boilerplate criteria, "fields"
        the ID, name and trial title. ``allowed`` masks rows (e.g. a cancer type)."""
        started = time.monotonic()
        keep = np.ones(len(self.cohorts), dtype=bool) if allowed is None else np.array(allowed, dtype=bool)
        if source:
            keep &= np.array([row["source"] == source for row in self.cohorts], dtype=bool)
        if parse_terms(q):
            keep &= self._text_index("cohorts", scope, titles).match(q)
        hits = np.flatnonzero(keep)
        return {"total": int(len(hits)), "of": len(self.cohorts), "offset": offset,
                "rows": [self.cohort_summary(self.cohorts[i]["id"], titles) for i in hits[offset:offset + limit]],
                "search_ms": round((time.monotonic() - started) * 1000, 1)}

    def describe(self) -> dict:
        trials, cohorts = {}, {}  # per source: the trials with at least one cohort in the corpus, and the cohorts
        for (trial_id, source), ids in self.trial_cohorts.items():
            trials.setdefault(source, set()).add(trial_id)
            cohorts[source] = cohorts.get(source, 0) + len(ids)
        return {"directory": str(self.directory), "patients": len(self.patients), "cohorts": len(self.cohorts),
                "trials": len(set().union(*trials.values())) if trials else 0,
                "trials_by_source": {k: len(v) for k, v in sorted(trials.items())},
                "cohorts_by_source": dict(sorted(cohorts.items())), "receipt_id": self.receipt_id, "digests": self.digests, "load_seconds": self.load_seconds,
                "patients_with_metadata": len(self.meta), "info": self.info}


def load_patient_metadata(path: Path) -> dict:
    """Each synthetic patient's generation label: its class (positive/negative) and source trial and cohort."""
    path = Path(path)
    if not path.exists():
        return {}
    import pyarrow.parquet as pq

    columns = ["patient_id", "patient_class", "nct_id", "cohort_trial_id"]
    table = pq.read_table(path, columns=columns).to_pydict()
    return {pid: {"cls": cls, "source_trial": nct, "source_cohort": cohort}
            for pid, cls, nct, cohort in zip(*(table[c] for c in columns))}


def _read_json(path: Path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None
