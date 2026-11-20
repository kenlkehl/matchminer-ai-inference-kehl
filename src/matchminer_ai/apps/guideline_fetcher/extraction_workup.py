"""Extraction-based workup checks: stored extraction records, compiled-query cache, jobs.

The inference package owns the query language, the compile step and the
evaluator (``matchminer_ai.patients.workup_queries``); this module owns where
records and compiled queries live for the app:

* **Extraction record stores** are directories built offline by
  the ``preprocess_extractions`` module (``extract_document_records`` and optionally
  ``extract_patient_records`` for a population) and read here for the matched
  patients only. PHI-mode stores sit beside the population cache under
  ``data/phi`` (created 0700, files 0600); the non-PHI app's hold synthetic
  notes under ``data/no_phi``. A store's mode side is checked whenever it is
  built or opened.
* **Compiled queries** contain guideline text only, never patient data. They are
  cached as JSON files in the mode's own app cache, keyed by the package's
  ``workup_query_cache_key`` (requirement text, population context, extraction
  schema and query language versions, prompt and response-schema hashes).
* **On-demand extraction** for the patient view's one patient keeps its records
  in server memory for the browser session, like saved note compressions.
* **SQL agent** (``matchminer_ai.patients.workup_sql``): each store gets an
  exploration database, a DuckDB file built once from the whole store into the
  mode's own app cache (``extraction_databases/``, private in PHI mode). It holds
  extracted values but no patient ids, note text or quotes. The agent explores it
  with read-only SQL in an isolated worker, and the exploration results reach the
  concordance connection. The accepted query of each workup item (SQL, summary,
  aggregate counts; never results) is cached as JSON in ``workup_sql/``, keyed by
  the requirement, the space's key criteria and that database. Each query then runs
  once over the cohort being checked.

Store layout::

  manifest.json             format, version, mode, source, levels, counts, contract
  document_records.parquet  patient_id, note_id, note_date, note_type,
                            extraction_status, error, item_count, record_json
  patient_records.parquet   patient_id, extraction_status, error, failed_chunks_json,
                            item_count, note_ids_json, record_json
"""

import copy
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .services import InputError
from .settings import data_path, mode_side_error

FORMAT = "guideline-fetcher extraction records"
VERSION = 1
DOCUMENT_FILE = "document_records.parquet"
PATIENT_FILE = "patient_records.parquet"
ROW_GROUP = 2000
LEVELS = {"document": ("document",), "document_and_patient": ("document", "patient")}
SQL_METHODS = ("extraction_queries", "extraction_sql")
SQL_EXPLORATION_NOTICE = (
    " The SQL agent explored the extraction records of {store} ({patients} patients) with "
    "read-only SQL; its query results (extracted values, without patient ids, note text or "
    "quotes) were sent to the concordance connection."
)
EXTRACTION_NOTICE = (
    " On-demand extraction sent each note to the selected LLM connection; its records are "
    "kept in server memory for this browser session only."
)


# ---------------------------------------------------------------------------
# Compiled-query cache (guideline text only)
# ---------------------------------------------------------------------------


class QueryCache:
    """Compiled workup queries as one JSON file per cache key, written atomically."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def _path(self, key):
        if not re.fullmatch(r"[0-9a-f]{64}", str(key)):
            raise KeyError(key)
        return self.folder / f"{key}.json"

    def get(self, key, default=None):
        try:
            return json.loads(self._path(key).read_text())
        except (OSError, ValueError, KeyError):
            return default

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        return self.get(key) is not None

    def __setitem__(self, key, value):
        path = self._path(key)
        self.folder.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".query-", dir=self.folder)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(value, handle, ensure_ascii=False)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


def query_cache(settings):
    return QueryCache(Path(settings.cache) / "workup_queries")


def population_context(space):
    return space.get("cohort_key_criteria") or space.get("clinical_space_summary") or ""


def cached_query_count(space, settings, config):
    """How many of a space's workup items already have a cached compiled query."""
    from matchminer_ai.patients.workup_queries import (
        compile_contract,
        default_extensions,
        workup_query_cache_key,
    )

    cache = query_cache(settings)
    contract = compile_contract(default_extensions(config))
    keys = {
        workup_query_cache_key(item, population_context(space), contract=contract)
        for item in space["diagnostic_workup"]
    }
    return len(keys), sum(key in cache for key in keys)


def compile_space_queries(space, settings, config, progress):
    """One compiled query per workup item of ``space`` (cached across sessions)."""
    import matchminer_ai.patients as patients

    items = space.get("diagnostic_workup") or []
    if not items:
        raise InputError("This space has no cataloged diagnostic/workup items.")
    compiled, metadata = patients.compile_workup_queries(
        items, config=config, population_context=population_context(space),
        cache=query_cache(settings), return_metadata=True,
        progress_callback=lambda done, total: progress(
            f"Compiling workup queries · {done}/{total} ready"),
    )
    return compiled, metadata


def query_summary(compiled, metadata):
    return {
        "model": ", ".join(metadata.get("compiled_by") or []) or metadata.get("model"),
        "connection_model": metadata.get("model"),
        "structured_outputs": metadata.get("structured_outputs"),
        "requests": metadata.get("requests", 0),
        "cached": metadata.get("cached", 0),
        "compiled": metadata.get("compiled", 0),
        "distinct": metadata.get("distinct", 0),
        "first_attempt_valid": metadata.get("first_attempt_valid", 0),
        "contract": metadata.get("contract"),
        "queries": [
            {k: entry.get(k) for k in ("cache_key", "status", "query", "error", "attempt_count",
                                       "first_attempt_valid", "cached", "model")}
            for entry in compiled
        ],
    }


# ---------------------------------------------------------------------------
# Extraction record stores
# ---------------------------------------------------------------------------


def store_roots(settings):
    """Folders whose stores the app may open: the default's folder and, in PHI mode,
    the DuckDB browse roots; the non-PHI app also opens the data root's no_phi."""
    default = Path(settings.extraction_records).expanduser().resolve().parent
    roots = [default]
    if settings.phi:
        roots += [Path(r).expanduser().resolve() for r in settings.duckdb_roots]
    else:
        roots.append(data_path("no_phi").resolve())
    return list(dict.fromkeys(roots))


def _inside(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


def resolve_store(settings, value=None, *, must_exist=True):
    path = Path(value).expanduser() if value else Path(settings.extraction_records)
    if not path.is_absolute():
        raise InputError("Enter the absolute path of an extraction record folder.")
    path = path.resolve()
    error = mode_side_error(settings.phi, path)
    if error:
        raise InputError(error)
    if not _inside(path, store_roots(settings)):
        raise InputError("Choose extraction records within the configured data folders.")
    if must_exist and not (path / "manifest.json").is_file():
        raise InputError(
            "No extraction records here. Build them offline with the preprocess_extractions command.")
    return path


def _manifest(path, settings):
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError):
        raise InputError("This extraction record manifest cannot be read.") from None
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise InputError("Unsupported extraction record format. Rebuild the records.")
    if manifest.get("mode") != settings.mode:
        raise InputError("These extraction records were built for the other app mode.")
    for name, size in manifest["files"].items():
        if not (path / name).is_file() or (path / name).stat().st_size != size:
            raise InputError("These extraction records are incomplete or changed. Rebuild them.")
    return manifest


def inspect_store(settings, value=None):
    """Manifest facts for the UI; no patient rows are read."""
    path = resolve_store(settings, value)
    manifest = _manifest(path, settings)
    return {
        "path": str(path),
        "name": path.name,
        "created_utc": manifest["created_utc"],
        "levels": manifest["levels"],
        "counts": manifest["counts"],
        "source": {k: v for k, v in manifest["source"].items() if k != "path"},
        "schema_version": manifest["extraction"]["schema_version"],
        "model": manifest["extraction"].get("model"),
    }


@dataclass(frozen=True)
class ExtractionStore:
    path: Path
    manifest: dict

    def _read(self, name, patient_ids):
        import pyarrow.dataset as ds

        file = self.path / name
        if not file.is_file():
            return None
        table = ds.dataset(file, format="parquet").to_table(
            filter=ds.field("patient_id").isin([str(p) for p in patient_ids]))
        return table.to_pandas()

    def all_records(self):
        """(document records, patient records) of every patient in the store."""
        import pyarrow.parquet as pq

        frames = []
        for name in (DOCUMENT_FILE, PATIENT_FILE):
            file = self.path / name
            if not file.is_file():
                frames.append(None)
                continue
            frame = pq.read_table(file).to_pandas()
            frame["record"] = [json.loads(r) if isinstance(r, str) else None
                               for r in frame.pop("record_json")]
            if "failed_chunks_json" in frame:
                frame["failed_chunks"] = [json.loads(v) for v in frame.pop("failed_chunks_json")]
                frame.pop("note_ids_json")
            frames.append(frame)
        return frames[0], frames[1]

    @property
    def fingerprint(self):
        """Identifies this store's content: its manifest (file sizes, counts, contract)."""
        return hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()

    def records(self, patient_ids):
        """(document records, patient records) of these patients, with record dicts."""
        documents = self._read(DOCUMENT_FILE, patient_ids)
        if documents is not None:
            documents["record"] = [json.loads(r) if isinstance(r, str) else None
                                   for r in documents.pop("record_json")]
        patients = self._read(PATIENT_FILE, patient_ids)
        if patients is not None:
            patients["record"] = [json.loads(r) if isinstance(r, str) else None
                                  for r in patients.pop("record_json")]
            patients["failed_chunks"] = [json.loads(v) for v in patients.pop("failed_chunks_json")]
            patients["note_ids"] = [json.loads(v) for v in patients.pop("note_ids_json")]
        return documents, patients


def open_store(settings, value=None):
    path = resolve_store(settings, value)
    return ExtractionStore(path, _manifest(path, settings))


def _private(path):
    os.chmod(path, 0o600)


def _clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def write_store(settings, output, documents=None, patients=None, *, source, extraction,
                overwrite=False):
    """Write a store atomically from extraction output DataFrames.

    ``patients`` may carry ``note_ids`` (the ids of the notes each patient record
    was built from). PHI-mode stores are private (0700 directory, 0600 files).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output).expanduser()
    if not output.is_absolute():
        raise InputError("Use an absolute output folder.")
    output = output.resolve()
    error = mode_side_error(settings.phi, output)
    if error:
        raise InputError(error)
    if output.exists() and not overwrite:
        raise InputError(f"{output} exists. Pass --overwrite to replace it.")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = output.parent / f".{output.name}.partial-{secrets.token_hex(4)}"
    partial.mkdir(mode=0o700)
    try:
        files, counts, levels = {}, {}, []
        if documents is not None:
            levels.append("document")
            frame = documents.sort_values(["patient_id", "note_date", "note_id"],
                                          na_position="last", kind="stable")
            table = pa.table({
                "patient_id": [str(p) for p in frame.patient_id],
                "note_id": [str(n) for n in frame.note_id],
                "note_date": [_clean(d) for d in frame.note_date],
                "note_type": [_clean(t) for t in frame.note_type],
                "extraction_status": list(frame.extraction_status),
                "error": [_clean(e) for e in frame.error],
                "item_count": [int(c) for c in frame.item_count],
                "record_json": [json.dumps(r, ensure_ascii=False) if isinstance(r, dict) else None
                                for r in frame.record],
            })
            pq.write_table(table, partial / DOCUMENT_FILE, row_group_size=ROW_GROUP,
                           compression="zstd")
            counts["documents"] = {s: int((frame.extraction_status == s).sum())
                                   for s in ("complete", "partial", "failed")}
        if patients is not None:
            levels.append("patient")
            frame = patients.sort_values("patient_id", kind="stable")
            note_ids = frame["note_ids"] if "note_ids" in frame else [[]] * len(frame)
            table = pa.table({
                "patient_id": [str(p) for p in frame.patient_id],
                "extraction_status": list(frame.extraction_status),
                "error": [_clean(e) for e in frame.error],
                "failed_chunks_json": [json.dumps(list(c or [])) for c in frame.failed_chunks],
                "item_count": [int(c) for c in frame.item_count],
                "note_ids_json": [json.dumps(list(n)) for n in note_ids],
                "record_json": [json.dumps(r, ensure_ascii=False) if isinstance(r, dict) else None
                                for r in frame.record],
            })
            pq.write_table(table, partial / PATIENT_FILE, row_group_size=ROW_GROUP,
                           compression="zstd")
            counts["patient_records"] = {s: int((frame.extraction_status == s).sum())
                                         for s in ("complete", "partial", "failed")}
        if not levels:
            raise InputError("No extraction records to write.")
        ids = set()
        for frame in (documents, patients):
            if frame is not None:
                ids |= set(map(str, frame.patient_id))
        counts["patients"] = len(ids)
        for name in (DOCUMENT_FILE, PATIENT_FILE):
            if (partial / name).exists():
                files[name] = (partial / name).stat().st_size
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "mode": settings.mode,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "levels": levels,
            "counts": counts,
            "extraction": extraction,
            "files": files,
        }
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        if settings.phi:
            for name in (*files, "manifest.json"):
                _private(partial / name)
        if output.exists():
            retired = output.parent / f".{output.name}.retired-{secrets.token_hex(4)}"
            output.rename(retired)
            partial.rename(output)
            shutil.rmtree(retired)
        else:
            partial.rename(output)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest


def extraction_metadata(document_metadata=None, patient_metadata=None, model=None, *,
                        servers=None):
    """The contract facts of a build, without per-note attempts or patient text.

    ``servers``: the build's servers and the model each served, when it used several.
    """
    def part(metadata):
        if metadata is None:
            return None
        return {k: metadata.get(k) for k in ("schema_version", "contract_sha256", "prompt_sha256",
                                             "response_schema_sha256", "extensions",
                                             "structured_outputs", "linking", "counts")}

    first = document_metadata or patient_metadata or {}
    return {
        "schema_version": first.get("schema_version"),
        "model": model,
        "document": part(document_metadata),
        "patient": part(patient_metadata),
        "package": first.get("package"),
        **({"servers": [dict(s) for s in servers]} if servers else {}),
    }


# ---------------------------------------------------------------------------
# SQL agent: exploration databases and accepted queries
# ---------------------------------------------------------------------------


def sql_cache(settings):
    """Accepted SQL queries: SQL, summary and aggregate counts, never query results."""
    return QueryCache(Path(settings.cache) / "workup_sql")


def _database_path(settings, store):
    from matchminer_ai.patients.workup_sql import DATABASE_VERSION

    key = hashlib.sha256(f"{DATABASE_VERSION}:{store.fingerprint}".encode()).hexdigest()
    return Path(settings.cache) / "extraction_databases" / f"{key}.duckdb"


_BUILD_LOCK = threading.Lock()


def exploration_database(settings, store, progress=lambda _: None):
    """The store's exploration database for the SQL agent, built once into the app cache."""
    from matchminer_ai.patients.workup_sql import (
        DATABASE_VERSION, build_extraction_database, read_database_info,
        write_extraction_database,
    )

    path = _database_path(settings, store)
    error = mode_side_error(settings.phi, path)
    if error:
        raise InputError(error)
    with _BUILD_LOCK:
        if path.is_file():
            try:
                if read_database_info(path).get("database_version") == DATABASE_VERSION:
                    return path
            except Exception:  # noqa: BLE001 -- an unreadable file is rebuilt
                pass
        patients = store.manifest["counts"].get("patients", 0)
        progress(f"Building the SQL agent's extraction database · {patients:,} patients")
        documents, records = store.all_records()
        database = build_extraction_database(
            [frame for frame in (documents, records) if frame is not None])
        if settings.phi:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        write_extraction_database(path, database, private=settings.phi)
    return path


def existing_database(settings, store):
    """The store's exploration database when already built (None otherwise)."""
    path = _database_path(settings, store)
    return path if path.is_file() else None


def cached_sql_count(space, settings, config, store):
    """(distinct items, items with an accepted cached query, built) for a space and store."""
    from matchminer_ai.patients.workup_sql import (
        agent_contract, read_database_info, reusable_sql_entry, workup_sql_cache_key,
    )

    items = space["diagnostic_workup"]
    path = existing_database(settings, store)
    if path is None:
        distinct = {json.dumps([i.get("name"), i.get("conditions"), i.get("category")])
                    for i in items}
        return len(distinct), 0, False
    info = read_database_info(path)
    contract = agent_contract(config, json.loads(info.get("extensions") or "[]"))
    cache = sql_cache(settings)
    keys = {workup_sql_cache_key(item, population_context(space),
                                 database_fingerprint=info["fingerprint"], contract=contract)
            for item in items}
    # Failed runs are saved for diagnosis but explored again: not cached queries.
    return len(keys), sum(reusable_sql_entry(cache.get(key), contract) for key in keys), True


def _short(text, limit=36):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit - 1]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(" ,;:-") + "…"


def sql_progress_message(done, total, exploring, seconds):
    """One progress line: queries ready, the items being explored and their turns, time."""
    minutes, rest = divmod(int(seconds), 60)
    text = f"SQL agent · {done}/{total} workup queries ready"
    if exploring:
        waiting = max(0, total - done - len(exploring))
        text += f" · exploring {len(exploring)}: " + ", ".join(
            f"{_short(item['name'])} (turn {item['action']}/{item['max_actions']})"
            for item in exploring)
        if waiting:
            text += f" · {waiting} waiting"
    return text + f" · {minutes}:{rest:02d} elapsed"


def compile_space_sql(space, settings, config, store, progress):
    """One accepted cohort-level SQL query per workup item of ``space`` (cached).

    Progress names each item being explored and the turn its agent is on, so a long
    exploration (each item takes several minutes) visibly moves.
    """
    from matchminer_ai.patients.workup_sql import compile_workup_sql

    items = space.get("diagnostic_workup") or []
    if not items:
        raise InputError("This space has no cataloged diagnostic/workup items.")
    database = exploration_database(settings, store, progress)
    progress(f"SQL agent exploring {store.path.name} for {len(items)} workup items")
    started, state = time.monotonic(), {"done": 0, "total": len(items), "exploring": []}

    def show():
        progress(sql_progress_message(state["done"], state["total"], state["exploring"],
                                      time.monotonic() - started))

    def counts(done, total):
        state.update(done=done, total=total)
        show()

    def activity(exploring):
        state["exploring"] = exploring
        show()

    compiled, metadata = compile_workup_sql(
        items, database=database, config=config, population_context=population_context(space),
        cache=sql_cache(settings), return_metadata=True, progress_callback=counts,
        activity_callback=activity,
    )
    metadata["store"] = {"name": store.path.name, "created_utc": store.manifest["created_utc"],
                         "patients": store.manifest["counts"].get("patients")}
    return compiled, metadata


SQL_ENTRY_FIELDS = ("cache_key", "status", "expressible", "not_expressible_reason", "sql",
                    "summary", "approximations", "recommends_against", "accepted_by",
                    "termination", "error",
                    "action_count", "submit_count", "distribution", "cached", "model")


def sql_summary(compiled, metadata):
    return {
        "model": ", ".join(metadata.get("compiled_by") or []) or metadata.get("model"),
        "connection_model": metadata.get("model"),
        "requests": metadata.get("requests", 0),
        "prompt_tokens": metadata.get("prompt_tokens", 0),
        "completion_tokens": metadata.get("completion_tokens", 0),
        "cached": metadata.get("cached", 0),
        "compiled": metadata.get("compiled", 0),
        "failed": metadata.get("failed", 0),
        "distinct": metadata.get("distinct", 0),
        "exploration": {**(metadata.get("exploration") or {}), **(metadata.get("store") or {})},
        "contract": metadata.get("contract"),
        "queries": [{k: copy.deepcopy(entry.get(k)) for k in SQL_ENTRY_FIELDS}
                    for entry in compiled],
    }


# ---------------------------------------------------------------------------
# On-demand extraction for the patient view
# ---------------------------------------------------------------------------


def notes_frame(text, notes, patient_id="workspace-patient"):
    """The workspace's notes as an extraction note table, in the supplied order."""
    rows = notes or [{"note_text": text, "note_date": None}]
    frame = pd.DataFrame([
        {"patient_id": patient_id, "note_text": n["note_text"], "note_date": n.get("note_date"),
         "note_type": n.get("note_type")}
        for n in rows
    ])
    return frame


def estimate_cost(frame, levels, *, compile_requests, cached_queries, items):
    """What on-demand extraction and query compilation would send (no model calls)."""
    characters = int(frame.note_text.str.len().sum())
    tokens = characters // 4
    patient_chunks = math.ceil(tokens / 40000) if "patient" in LEVELS[levels] else 0
    return {
        "notes": len(frame),
        "characters": characters,
        "approximate_note_tokens": tokens,
        "document_requests": len(frame),
        "patient_requests": patient_chunks,
        "compile_requests": compile_requests,
        "cached_queries": cached_queries,
        "workup_items": items,
        "requests": len(frame) + patient_chunks + compile_requests,
        "output_reserve_per_request": 50000,
        "typical_reasoning_tokens": "10,000-15,000 per request with Nemotron 3.5 Super",
        "levels": list(LEVELS[levels]),
    }


class SessionExtractions:
    """On-demand extraction records per browser session, in memory only."""

    def __init__(self, ttl):
        self.ttl, self.lock, self.entries = ttl, threading.Lock(), {}

    @staticmethod
    def key(frame, levels, config):
        payload = {
            "notes": frame[["note_text", "note_date", "note_type"]].astype(object).where(
                frame[["note_text", "note_date", "note_type"]].notna(), None).values.tolist(),
            "levels": levels,
            "sections": {name: getattr(config, name) for name in
                         ("document_extraction", "patient_extraction")},
            "endpoint": config.remote.get("server_urls"),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, owner, key):
        with self.lock:
            entry = self.entries.get(owner)
            if entry and entry["key"] == key and time.monotonic() - entry["updated"] < self.ttl:
                entry["updated"] = time.monotonic()
                return copy.deepcopy(entry["value"])
            return None

    def save(self, owner, key, value):
        with self.lock:
            self.entries[owner] = {"key": key, "value": copy.deepcopy(value),
                                   "updated": time.monotonic()}

    def clear(self, owner):
        with self.lock:
            self.entries.pop(owner, None)

    def clean(self):
        with self.lock:
            for owner, entry in list(self.entries.items()):
                if time.monotonic() - entry["updated"] > self.ttl:
                    del self.entries[owner]


def extract_on_demand(frame, levels, config, progress):
    """Document records (and a patient record) for one patient's notes."""
    import matchminer_ai.patients as patients

    documents_needed = LEVELS[levels]
    patient_records = None
    requests = 0
    if "patient" in documents_needed:
        progress("Extracting a patient-level record from all notes")
        patient_records, patient_meta = patients.extract_patient_records(
            frame, config=config, return_metadata=True,
            progress_callback=lambda done, total: progress(
                f"Extracting the patient-level record · {done}/{total} chunks"),
        )
        requests += sum(len(a) for p in patient_meta["patients"].values()
                        for a in p["attempts"].values())
    progress(f"Extracting structured records from {len(frame)} notes")
    documents, document_meta = patients.extract_document_records(
        frame, config=config, patient_records=patient_records, return_metadata=True,
        progress_callback=lambda done, total: progress(
            f"Extracting structured records · {done}/{total} notes"),
    )
    requests += sum(len(d["attempts"]) for d in document_meta["documents"].values())
    return {
        "documents": documents.to_dict("records"),
        "patients": None if patient_records is None else patient_records.to_dict("records"),
        "metadata": {
            "levels": list(documents_needed),
            "requests": requests,
            "counts": document_meta["counts"],
            "schema_version": document_meta["schema_version"],
            "model": config.document_extraction["remote"]["model_name"],
        },
    }


def _note_numbers(frame):
    from matchminer_ai.patients.extraction.notes import note_ids

    return {note: number for number, note in enumerate(note_ids(frame), 1)}


def exploration_store(settings):
    """The SQL agent's exploration store in the patient view: the mode's default store."""
    try:
        return open_store(settings)
    except InputError:
        raise InputError(
            "The SQL agent explores an extraction record store before checking this patient, "
            "and the default store (GUIDELINE_EXTRACTION_RECORDS) is unavailable. Build one "
            "with the preprocess_extractions command.") from None


def patient_review(text, notes, space, *, settings, config, sessions, owner, levels, progress,
                   method="extraction_queries"):
    """The patient view's extraction-based workup review of one patient's notes.

    ``method``: ``extraction_queries`` (compiled JSON queries, no patient data reach
    an LLM after extraction) or ``extraction_sql`` (the SQL agent's queries, explored
    on the mode's default store and run once over this patient's records).
    """
    from matchminer_ai.patients import check_workup_with_extractions, check_workup_with_sql
    from matchminer_ai.patients.workup_queries import NOTICE
    from matchminer_ai.patients.workup_sql import NOTICE as SQL_NOTICE

    if levels not in LEVELS:
        raise InputError("Choose document-level or document and patient-level extraction.")
    if method not in SQL_METHODS:
        raise InputError("Choose extraction queries or the SQL agent.")
    store = exploration_store(settings) if method == "extraction_sql" else None
    frame = notes_frame(text, notes)
    key = SessionExtractions.key(frame, levels, config)
    saved = sessions.get(owner, key) if sessions is not None else None
    if saved is not None:
        progress(f"Reusing this session's extraction records · {len(frame)} notes")
        extraction, reused = saved, True
    else:
        extraction, reused = extract_on_demand(frame, levels, config, progress), False
        if sessions is not None:
            sessions.save(owner, key, extraction)
    records = [pd.DataFrame(extraction["documents"])]
    if extraction["patients"]:
        records.append(pd.DataFrame(extraction["patients"]))
    if method == "extraction_sql":
        compiled, metadata = compile_space_sql(space, settings, config, store, progress)
        summary = sql_summary(compiled, metadata)
    else:
        compiled, metadata = compile_space_queries(space, settings, config, progress)
        summary = query_summary(compiled, metadata)
    linked = [{**entry, "requirement": copy.deepcopy(item)}
              for entry, item in zip(compiled, space["diagnostic_workup"], strict=True)]
    if method == "extraction_sql":
        progress("Running the workup SQL queries over this patient's extracted items")
        report = check_workup_with_sql(records, linked, patient_ids=["workspace-patient"],
                                       config=config)
        notice = SQL_NOTICE + SQL_EXPLORATION_NOTICE.format(
            store=store.path.name, patients=store.manifest["counts"].get("patients", 0))
    else:
        progress("Checking workup queries against the extracted items")
        report = check_workup_with_extractions(records, linked,
                                               patient_ids=["workspace-patient"])
        notice = NOTICE
    review = report["patients"][0]["review"]
    numbers = _note_numbers(frame)
    for assessment in review["assessments"]:
        for evidence in assessment["evidence"]:
            evidence["note_number"] = numbers.get(evidence.get("note_id"))
    counts = extraction["metadata"]["counts"]
    review["notice"] = notice + ("" if reused else EXTRACTION_NOTICE)
    review["metadata"].update(
        method=method,
        note_count=len(frame),
        undated_notes=int(frame.note_date.isna().sum()),
        requests=(0 if reused else extraction["metadata"]["requests"]) + metadata["requests"],
        model=metadata.get("model"),
        extraction={**extraction["metadata"], "reused": reused,
                    "failed": counts.get("failed", 0), "partial": counts.get("partial", 0)},
        queries=summary,
        evidence_selection="extracted_item_quotes",
        scope="structured_extraction_records",
    )
    return review


__all__ = [
    "ExtractionStore",
    "QueryCache",
    "SessionExtractions",
    "compile_space_queries",
    "compile_space_sql",
    "estimate_cost",
    "exploration_database",
    "extraction_metadata",
    "inspect_store",
    "open_store",
    "patient_review",
    "query_cache",
    "resolve_store",
    "sql_cache",
    "sql_summary",
    "write_store",
]
