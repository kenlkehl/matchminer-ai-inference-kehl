"""Population UI adapters; reusable matching/review/archive behavior is in the package."""

import io
import json
import os
import secrets
import threading
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .services import InputError, without_history_and_physical
from matchminer_ai.cancellation import check_cancelled
from matchminer_ai.schema import SYNTHETIC_NOTES_TO_RUNTIME, TRAINING_TO_RUNTIME, note_columns


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


def adapt_tables(summaries, notes):
    """Explicit training-to-runtime boundary, never join on the synthetic prompt ID."""
    if "pseudo_mrn" in summaries:
        summary_columns = ("pseudo_mrn", "patient_summary", "patient_boilerplate_text")
        summaries = summaries.rename(columns={k: TRAINING_TO_RUNTIME[k] for k in summary_columns})
    if "pseudo_mrn" in notes:
        notes = notes.drop(columns=["patient_id"], errors="ignore").rename(columns=SYNTHETIC_NOTES_TO_RUNTIME)
    if not summaries.columns.is_unique or not notes.columns.is_unique:
        raise InputError("Ambiguous cohort columns. Supply the documented runtime schema.")
    return summaries, notes


def load_synthetic_sample(settings, size, seed, progress):
    """Non-PHI mode: a seeded sample of the configured synthetic summary and note Parquets."""
    from matchminer_ai.population import create_patient_cohort
    import pyarrow.dataset as ds

    progress("Reading synthetic summaries and matching note identifiers")
    summaries = pd.read_parquet(settings.population_summaries)
    dataset = ds.dataset(settings.population_notes, format="parquet")
    note_key, note_map = note_columns(dataset.schema.names)
    summary_key = "pseudo_mrn" if "pseudo_mrn" in summaries else "patient_id"
    summary_text = "patient_summary" if "patient_summary" in summaries else "cancer_history_summary"
    note_ids = dataset.to_table(columns=[note_key]).column(note_key).unique().to_pylist()
    eligible = summaries.loc[
        summaries[summary_key].astype(str).isin({str(x) for x in note_ids})
        & summaries[summary_text].map(lambda s: isinstance(s, str) and bool(s.strip()))
    ]
    if eligible[summary_key].duplicated().any():
        raise InputError("Population summaries must contain one row per patient.")
    if size > len(eligible):
        raise InputError(f"Only {len(eligible)} patients have both summaries and notes.")
    sample = eligible.sample(n=size, random_state=seed).reset_index(drop=True)
    keys = set(sample[summary_key].astype(str))
    typed_ids = [key for key in note_ids if str(key) in keys]
    columns = [c for c in note_map if c in dataset.schema.names]
    check_cancelled()
    progress(f"Loading original notes for {size} sampled patients")
    import pyarrow as pa

    batches = []
    note_count = 0
    scanner = dataset.scanner(
        columns=columns,
        filter=ds.field(note_key).isin(typed_ids),
        batch_size=8192,
        batch_readahead=2,
        fragment_readahead=1,
    )
    for batch in scanner.to_batches():
        check_cancelled()
        if len(batch):
            batches.append(batch)
            note_count += len(batch)
            progress(f"Loading original cohort notes · {note_count:,} notes read")
    if not batches:
        raise InputError("The selected patients have no available source notes.")
    notes = pa.Table.from_batches(batches).to_pandas()
    sample, notes = adapt_tables(sample, notes)
    provenance = {
        "source": "operator_configured_population",
        "sample_size": size,
        "random_seed": seed,
        "eligible_population_count": len(eligible),
        "summary_file": Path(settings.population_summaries).name,
        "notes_file": Path(settings.population_notes).name,
        "source_files": [
            {"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in (settings.population_summaries, settings.population_notes)
        ],
    }
    result = create_patient_cohort(sample, notes, provenance=provenance)
    progress(f"Loaded {len(result.summaries)} patients and {len(result.notes)} notes")
    return result


# PHI mode: MatchMiner-AI real-time DuckDB populations (see population_cache.py).
DUCKDB_SUFFIXES = (".duckdb", ".ddb", ".db")
# The MatchMiner-AI real-time database schema; patients.first/last names are never read.
DUCKDB_COLUMNS = {
    "patient_summary": {"mrn", "patient_summary", "patient_boilerplate", "is_current"},
    "patient_notes": {"id", "mrn", "rpt_text", "narrative_text", "event_date", "note_source"},
}
# The real-time summarizer's rule: rpt_text, or narrative_text when rpt_text is blank.
# DuckDB's one-argument trim removes only spaces, so name the common whitespace.
_BLANKS = "' ' || chr(9) || chr(10) || chr(11) || chr(12) || chr(13) || chr(160)"
NOTE_TEXT = (
    f"COALESCE(NULLIF(trim(rpt_text, {_BLANKS}), ''), "
    f"NULLIF(trim(narrative_text, {_BLANKS}), ''))"
)


def _text(values):
    """Python's strip, matching the inference package's nonempty-text validation."""
    return values.map(lambda t: t.strip() if isinstance(t, str) else "")


def clean_summaries(summaries):
    """Strip summaries and drop blank ones (blank by Python's strip, like the package)."""
    summaries = summaries.assign(cancer_history_summary=_text(summaries.cancer_history_summary))
    return summaries.loc[summaries.cancer_history_summary.ne("")]


def clean_notes(notes):
    """Order, strip, drop blank and deduplicate notes read with NOTE_TEXT.

    Order is the pipeline's: date, then database ID for same-time notes. Repeated
    copies of one note (same date, source and text) add nothing to review.
    Returns (notes without the id column, number of duplicates removed).
    """
    notes = notes.assign(note_text=_text(notes.note_text))
    notes = notes.loc[notes.note_text.ne("")]
    notes = notes.sort_values(["patient_id", "note_date", "id"], kind="stable",
                              na_position="last").drop(columns="id")
    count = len(notes)
    notes = notes.drop_duplicates(["patient_id", "note_date", "note_type", "note_text"])
    return notes, count - len(notes)


def require_phi(settings):
    """Real-patient databases and caches are available only in PHI mode (mmai-guideline-fetcher --phi)."""
    if not settings.phi:
        raise InputError("DuckDB populations and population caches need the PHI app (mmai-guideline-fetcher --phi).")


def _inside(path, roots):
    return any(path == root or path.is_relative_to(root) for root in roots)


def duckdb_roots(settings):
    roots = list(dict.fromkeys(Path(r).expanduser().resolve() for r in settings.duckdb_roots))
    # The configured default is always selectable, even outside the browse roots.
    default = Path(settings.population_duckdb).expanduser().resolve().parent
    return roots if _inside(default, roots) else [*roots, default]


def browse_databases(settings, path=None, limit=5000):
    """Folder names, DuckDB files and population caches; contents are never read here."""
    require_phi(settings)
    roots = duckdb_roots(settings)
    target = Path(path).expanduser() if path else Path(settings.population_duckdb).parent
    if not target.is_absolute():
        raise InputError("Enter an absolute folder path.")
    target = target.resolve()
    if target.is_file():
        target = target.parent
    if not _inside(target, roots):
        raise InputError("Browse within the configured database folders.")
    if not target.is_dir():
        raise InputError("This folder does not exist.")
    directories, files, caches = [], [], []
    try:
        with os.scandir(target) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if len(directories) + len(files) + len(caches) >= limit:
                    break
                try:
                    if entry.is_dir() and os.path.isfile(os.path.join(entry.path, "manifest.json")):
                        caches.append({"name": entry.name, "path": str(target / entry.name)})
                    elif entry.is_dir():
                        directories.append(entry.name)
                    elif entry.is_file() and entry.name.lower().endswith(DUCKDB_SUFFIXES):
                        info = entry.stat()
                        files.append({"name": entry.name, "path": str(target / entry.name),
                                      "size": info.st_size, "modified": info.st_mtime})
                except OSError:
                    continue
    except OSError:
        raise InputError("This folder cannot be read.") from None
    return {
        "path": str(target),
        "parent": str(target.parent) if _inside(target.parent, roots) and target not in roots
        else None,
        "roots": [str(r) for r in roots],
        "directories": sorted(directories, key=str.lower),
        "files": sorted(files, key=lambda f: f["name"].lower()),
        "caches": sorted(caches, key=lambda c: c["name"].lower()),
        "truncated": len(directories) + len(files) + len(caches) >= limit,
    }


def resolve_database(settings, value=None):
    require_phi(settings)
    path = Path(value).expanduser() if value else Path(settings.population_duckdb)
    if not path.is_absolute():
        raise InputError("Enter the absolute path of a DuckDB database file.")
    path = path.resolve()
    if not _inside(path, duckdb_roots(settings)):
        raise InputError("Choose a database within the configured database folders.")
    if path.suffix.lower() not in DUCKDB_SUFFIXES or not path.is_file():
        raise InputError("Choose an existing .duckdb database file.")
    return path


def _connect(path):
    import duckdb

    try:
        return duckdb.connect(str(path), read_only=True)
    except duckdb.Error:
        raise InputError(
            "Could not open this file as a read-only DuckDB database. It may be another "
            "format, newer than the installed DuckDB, or locked by a writing process."
        ) from None


def _check_schema(con):
    found = {}
    for table, column in con.execute(
        "SELECT table_name, column_name FROM duckdb_columns() "
        "WHERE schema_name = 'main' AND table_name IN ('patient_summary', 'patient_notes')"
    ).fetchall():
        found.setdefault(table, set()).add(column)
    missing = [
        f"{table}.{column}"
        for table, columns in DUCKDB_COLUMNS.items()
        for column in sorted(columns - found.get(table, set()))
    ]
    if missing:
        raise InputError(
            "This database lacks the MatchMiner-AI patient tables or columns: "
            + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
        )


def _file_info(path):
    info = path.stat()
    return {"path": str(path), "name": path.name, "size": info.st_size,
            "mtime_ns": info.st_mtime_ns}


def inspect_database(settings, value=None):
    """Validate a selection from table metadata only; no patient rows are read."""
    path = resolve_database(settings, value)
    con = _connect(path)
    try:
        _check_schema(con)
    finally:
        con.close()
    return _file_info(path)


def _interruptible(con, query, progress, message):
    """Run one DuckDB query off-thread so Stop interrupts long network-file scans."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    import duckdb

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="population-duckdb") as pool:
        future = pool.submit(query)
        while True:
            try:
                return future.result(timeout=1)
            except TimeoutError:
                try:
                    check_cancelled()
                    # DuckDB's query_progress is stale/flat for these joins; report time.
                    seconds = int(time.monotonic() - started)
                    progress(f"{message} · {seconds // 60}:{seconds % 60:02d} elapsed")
                except BaseException:
                    con.interrupt()
                    raise
            except duckdb.Error:
                raise InputError(
                    "Could not read patient records from this database. Check that the file "
                    "is complete and readable, then retry."
                ) from None


def load_population_sample(settings, size, seed, progress, database=None):
    """The mode's sample: DuckDB in PHI mode, else the synthetic Parquets."""
    if settings.phi:
        return load_duckdb_sample(settings, size, seed, progress, database)
    if database is not None:
        raise InputError("DuckDB populations need the PHI app (mmai-guideline-fetcher --phi).")
    return load_synthetic_sample(settings, size, seed, progress)


def load_duckdb_sample(settings, size, seed, progress, database=None):
    """PHI mode: random sample of current summaries plus every source note for those patients."""
    from matchminer_ai.population import create_patient_cohort

    path = resolve_database(settings, database)
    progress(f"Opening {path.name} read-only")
    con = _connect(path)
    try:
        _check_schema(con)
        progress("Finding patients with a current summary and source notes")
        eligible = _interruptible(con, lambda: con.execute(
            "SELECT s.mrn FROM patient_summary s WHERE s.is_current "
            "AND s.patient_summary IS NOT NULL AND s.mrn IS NOT NULL "
            "AND EXISTS (SELECT 1 FROM patient_notes n WHERE n.mrn = s.mrn) ORDER BY s.mrn"
        ).df(), progress, "Finding patients with a current summary and source notes")
        if eligible.mrn.duplicated().any():
            raise InputError("The database has more than one current summary for a patient.")
        if size > len(eligible):
            raise InputError(f"Only {len(eligible)} patients have both summaries and notes.")
        sample = eligible.sample(n=size, random_state=seed).reset_index(drop=True)
        con.register("population_sample", sample)
        check_cancelled()
        progress(f"Reading current summaries for {size} sampled patients")
        summaries = _interruptible(con, lambda: con.execute(
            "SELECT CAST(mrn AS VARCHAR) AS patient_id, "
            "patient_summary AS cancer_history_summary, "
            "COALESCE(patient_boilerplate, '') AS general_exclusion_criteria_evidence "
            "FROM patient_summary WHERE is_current "
            "AND mrn IN (SELECT mrn FROM population_sample)"
        ).df(), progress, f"Reading current summaries for {size} sampled patients")
        message = (f"Reading source notes for {len(summaries)} sampled patients; "
                   "an uncached network database can take several minutes")
        progress(message)
        # Blank text is filtered below, after the join, not across the whole table.
        notes = _interruptible(con, lambda: con.execute(
            f"SELECT CAST(mrn AS VARCHAR) AS patient_id, {NOTE_TEXT} AS note_text, "
            "event_date AS note_date, note_source AS note_type, id "
            "FROM patient_notes WHERE mrn IN (SELECT mrn FROM population_sample)"
        ).fetch_arrow_table(), progress, message).to_pandas()
    finally:
        con.close()
    check_cancelled()
    summaries = clean_summaries(summaries)
    notes, duplicates = clean_notes(notes)
    summaries = summaries.loc[summaries.patient_id.isin(set(notes.patient_id))]
    notes = notes.loc[notes.patient_id.isin(set(summaries.patient_id))]
    if summaries.empty:
        raise InputError("The selected patients have no available source notes.")
    excluded = size - len(summaries)
    # Preserve the reproducible sample order rather than database row order.
    order = {str(mrn): i for i, mrn in enumerate(sample.mrn)}
    summaries = summaries.sort_values("patient_id", key=lambda ids: ids.map(order))
    provenance = {
        "source": "duckdb_population",
        "database": _file_info(path),
        "tables": {"summaries": "patient_summary WHERE is_current", "notes": "patient_notes"},
        "sample_size": size,
        "random_seed": seed,
        "eligible_population_count": len(eligible),
        "excluded_without_text_count": excluded,
        "duplicate_notes_removed": duplicates,
    }
    result = create_patient_cohort(summaries.reset_index(drop=True), notes, provenance=provenance)
    progress(f"Loaded {len(result.summaries)} patients and {len(result.notes)} notes")
    return result


def parse_table(name, content):
    suffix = Path(name or "").suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(content))
    raise InputError("Cohort tables must be CSV or Parquet files.")


class PopulationInference:
    def __init__(self, settings):
        self.settings = settings

    def retrieve(self, source, paths, space_id, candidate_k, min_score, progress):
        import torch

        torch.set_num_threads(8)
        from matchminer_ai.matching import retrieve_guideline_patients
        from matchminer_ai.population import create_patient_cohort, subset_patient_cohort

        cache = None
        if source.provenance.get("source") == "population_cache":
            from .population_cache import open_cache

            progress("Opening the preprocessed population cache")
            info = source.provenance["cache"]
            cache = open_cache(self.settings, info["path"], expected=info["manifest_sha256"])
        selected, metadata = retrieve_guideline_patients(
            source.summaries,
            paths,
            space_trial_id=space_id,
            candidate_k=candidate_k,
            min_match_score=min_score,
            config=self.settings.inference_config(),
            progress_callback=progress,
            patient_embeddings=cache and cache.embeddings,
        )
        ids = selected.patient_id.tolist()
        active = subset_patient_cohort(
            source,
            ids,
            # History-and-physical items are always done; workup checks leave them out.
            space=without_history_and_physical(metadata.pop("space")),
            matches=records(selected.drop(columns="cancer_history_summary")),
            provenance={"retrieval": metadata},
        )
        if cache is None or not ids:
            return active
        check_cancelled()
        progress(f"Loading cached source notes for {len(ids)} matched patients")
        # Validate exactly as for any cohort: every matched patient needs notes.
        notes = create_patient_cohort(active.summaries, cache.notes(ids)).notes
        return replace(active, notes=notes)

    def compress(self, cohort, progress):
        from matchminer_ai.population import compress_patient_cohort

        selected = self.settings.for_task("compress")
        result = compress_patient_cohort(
            cohort,
            config=selected.inference_config(),
            thinking="on" if selected.reasoning else "off",
            progress_callback=progress,
        )
        return replace(
            result, compression={**result.compression, "connection": self.connection(selected)}
        )

    @staticmethod
    def connection(settings):
        from .settings import VERTEX_PROFILES

        profile = VERTEX_PROFILES.get(settings.llm_backend, {})
        return {
            "provider": settings.llm_backend,
            "model": profile.get("model", settings.model),
            "endpoint": profile.get("endpoint", settings.endpoint),
            "reasoning": settings.reasoning,
        }

    def review_extractions(self, cohort, store_path, progress, *, method="extraction_queries"):
        """Workup checks on precomputed extraction records, run in code.

        ``extraction_queries``: compiling a query (once per workup item, cached) sends
        guideline text to the concordance connection; no patient text reaches an LLM.
        ``extraction_sql``: the SQL agent explores the store's extracted values with
        read-only SQL (once per workup item and store, cached), and those exploration
        results reach the concordance connection; each accepted query then runs once over
        the cohort's records.
        """
        from .extraction_workup import (
            compile_space_queries, compile_space_sql, open_store, query_summary, sql_summary,
        )
        from matchminer_ai.population import (
            check_population_workup_with_extractions, check_population_workup_with_sql,
        )

        store = open_store(self.settings, store_path)
        ids = cohort.summaries.patient_id.tolist()
        progress(f"Reading extraction records for {len(ids)} patients")
        documents, patients = store.records(ids)
        check_cancelled()
        selected = self.settings.for_task("workup")
        config = selected.extraction_config()
        if method == "extraction_sql":
            compiled, metadata = compile_space_sql(cohort.space, self.settings, config, store,
                                                   progress)
            summary = sql_summary(compiled, metadata)
            check_cancelled()
            result = check_population_workup_with_sql(
                cohort, compiled, document_records=documents, patient_records=patients,
                config=config, progress_callback=progress,
            )
        else:
            compiled, metadata = compile_space_queries(cohort.space, self.settings, config,
                                                       progress)
            summary = query_summary(compiled, metadata)
            check_cancelled()
            result = check_population_workup_with_extractions(
                cohort, compiled, document_records=documents, patient_records=patients,
                progress_callback=progress,
            )
        source = store.manifest
        # The SQL method's only model calls are the SQL agent's, on its own connection.
        used = self.settings.for_task("workup_sql") if method == "extraction_sql" else selected
        return replace(result, report={
            **result.report,
            "connection": self.connection(used),
            "compile": {k: v for k, v in summary.items() if k != "queries"},
            "extraction_store": {
                "name": store.path.name, "created_utc": source["created_utc"],
                "levels": source["levels"], "counts": source["counts"],
                "model": source["extraction"].get("model"),
                "schema_version": source["extraction"].get("schema_version"),
            },
        })

    def review(self, cohort, note_source, progress, *, max_concurrent_patients=4):
        """Full-note review of every patient, as in the patient view's full-note method."""
        from matchminer_ai.population import review_population_workup

        selected = self.settings.for_task("workup")
        backup = self.settings.for_task("workup_full_backup")
        result = review_population_workup(
            cohort,
            config=selected.inference_config(),
            note_source=note_source,
            backup_config=backup.inference_config() if backup.backup_enabled else None,
            max_consecutive_failures=backup.failure_threshold,
            max_concurrent_patients=max_concurrent_patients,
            progress_callback=progress,
        )
        return replace(result, report={**result.report, "connection": self.connection(selected)})


class PopulationSessions:
    """One source population and one active cohort per session; no implicit disk storage."""

    def __init__(self, ttl):
        self.ttl, self.lock, self.entries = ttl, threading.RLock(), {}

    def clean(self, busy_owners=()):
        with self.lock:
            for owner, entry in list(self.entries.items()):
                if owner not in busy_owners and time.monotonic() - entry["updated"] > self.ttl:
                    del self.entries[owner]

    def get(self, owner):
        with self.lock:
            entry = self.entries.get(owner)
            if entry:
                entry["updated"] = time.monotonic()
            return entry

    def save(self, owner, source, active=None, *, expected=None):
        with self.lock:
            check_cancelled()
            if expected is not None and self.entries.get(owner) is not expected:
                raise InputError("The population changed; reload the view before retrying.")
            self.entries[owner] = {
                "source": source,
                "active": active,
                "updated": time.monotonic(),
                "version": secrets.token_urlsafe(12),
            }
            return self.view(owner)

    def clear(self, owner):
        with self.lock:
            self.entries.pop(owner, None)

    def view(self, owner):
        entry = self.get(owner)
        if not entry:
            return None
        source, active = entry["source"], entry["active"]
        on_demand = bool(source.provenance.get("notes_loaded_on_demand"))
        return {
            "version": entry["version"],
            "source_count": len(source.summaries),
            "source_note_count": source.provenance["note_count"] if on_demand else len(source.notes),
            "notes_on_demand": on_demand,
            "provenance": source.provenance,
            "active": None
            if active is None
            else {
                "patient_count": len(active.summaries),
                "note_count": len(active.notes),
                "space": active.space,
                "matches": active.matches,
                "compression": active.compression,
                "report": None
                if active.report is None
                else {k: v for k, v in active.report.items() if k != "patients"},
                "retrieval": active.provenance.get("retrieval", {}),
            },
        }
