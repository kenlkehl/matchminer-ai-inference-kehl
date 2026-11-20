"""PHI mode: private, preprocessed real-patient populations for fast population loading.

A cache is one directory (0700; files 0600) built by the preprocess_population command:

  manifest.json      format, source database, counts and the TrialSpace identity
  summaries.parquet  patient_id, cancer_history_summary,
                     general_exclusion_criteria_evidence, summary_sha256, note_count
  embeddings.npy     float32 TrialSpace vectors; row i belongs to summaries row i
  notes.parquet      patient_id, note_text, note_date, note_type, sorted by patient
                     in small row groups so matched patients' notes load quickly

Summaries, notes and dedup rules are the DuckDB sample loader's. The cache holds
PHI. Builds are atomic: a hidden partial directory replaces the output only after
every file and the manifest are complete.
"""

import json
import os
import re
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .population import (
    NOTE_TEXT, _check_schema, _connect, _file_info, _inside, clean_notes, clean_summaries,
    duckdb_roots, require_phi, resolve_database,
)
from .services import InputError

# The former guideline-fetcher-phi app's name: existing caches record it, so keep it.
FORMAT = "guideline-fetcher-phi population cache"
VERSION = 1
FILES = ("summaries.parquet", "embeddings.npy", "notes.parquet")
NOTE_ROW_GROUP = 2000
NOTE_COLUMNS = ["patient_id", "note_text", "note_date", "note_type"]


def _notes_schema():
    import pyarrow as pa

    return pa.schema([
        ("patient_id", pa.string()), ("note_text", pa.string()),
        ("note_date", pa.timestamp("us")), ("note_type", pa.string()),
    ])


def _private(path):
    os.chmod(path, 0o600)


def build_population_cache(settings, output, *, database=None, summaries=None, batch_size=1024,
                           memory_limit="128GB", limit=None, seed=23, overwrite=False,
                           log=print):
    """Read every current summary and its patient's notes, embed, and save atomically.

    summaries, a Parquet file or a folder of them (such as a summarize_patients run's
    summaries/), replaces the database's current patient_summary rows; notes still come
    from the database. limit samples that many patients (seeded) for a quick trial build.
    Only counts and file names are logged; patient text and identifiers never are.
    """
    from matchminer_ai._metadata import package_metadata
    from matchminer_ai.matching import embed_guideline_patients
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", memory_limit):
        raise InputError("Use a DuckDB memory limit such as 128GB.")
    path = resolve_database(settings, database)
    summary_files = _summary_files(settings, summaries) if summaries is not None else None
    output = Path(output).expanduser()
    if not output.is_absolute():
        raise InputError("Use an absolute cache directory.")
    output = output.resolve()
    if "no_phi" in output.parts:
        raise InputError("A population cache contains PHI; never write it to a no_phi folder.")
    if output.exists() and not overwrite:
        raise InputError(f"{output} exists. Pass --overwrite to replace it.")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = output.parent / f".{output.name}.partial-{secrets.token_hex(4)}"
    partial.mkdir(mode=0o700)
    started = time.monotonic()

    def say(message):
        seconds = int(time.monotonic() - started)
        log(f"[{seconds // 60}:{seconds % 60:02d}] {message}")

    try:
        con = _connect(path)
        try:
            _check_schema(con)
            # Spill the note sort inside the private partial directory, never /tmp.
            con.execute(f"SET memory_limit = '{memory_limit}'")
            spill = str(partial / "duckdb_spill").replace("'", "''")
            con.execute(f"SET temp_directory = '{spill}'")
            if summary_files is None:
                say(f"Reading current summaries from {path.name}")
                summaries = con.execute(
                    "SELECT CAST(mrn AS VARCHAR) AS patient_id, "
                    "patient_summary AS cancer_history_summary, "
                    "COALESCE(patient_boilerplate, '') AS general_exclusion_criteria_evidence "
                    "FROM patient_summary WHERE is_current AND patient_summary IS NOT NULL "
                    "AND mrn IS NOT NULL"
                ).df()
                summary_source = "patient_summary WHERE is_current"
            else:
                folder, files = summary_files
                say(f"Reading summaries from {len(files):,} Parquet file(s) in {folder.name}")
                summaries = _read_summary_files(files)
                summary_source = f"Parquet: {folder.name} ({len(files):,} files)"
            if summaries.patient_id.duplicated().any():
                raise InputError("The summaries have more than one row for a patient.")
            current = len(summaries)
            summaries = clean_summaries(summaries)
            blank = current - len(summaries)
            if limit:
                summaries = summaries.sample(n=min(limit, len(summaries)), random_state=seed)
            con.register("cache_patients", summaries.patient_id.astype("int64").to_frame("mrn"))
            total = con.execute(
                "SELECT count(*) FROM patient_notes WHERE mrn IN (SELECT mrn FROM cache_patients)"
            ).fetchone()[0]
            say(f"{len(summaries):,} patients with nonblank current summaries; "
                f"sorting and writing their {total:,} notes")
            reader = con.execute(
                f"SELECT CAST(mrn AS VARCHAR) AS patient_id, {NOTE_TEXT} AS note_text, "
                "event_date AS note_date, note_source AS note_type, id FROM patient_notes "
                "WHERE mrn IN (SELECT mrn FROM cache_patients) "
                "ORDER BY patient_id, note_date NULLS LAST, id"
            ).fetch_record_batch(100_000)
            counts, duplicates, read = {}, 0, 0
            with pq.ParquetWriter(partial / "notes.parquet", _notes_schema(),
                                  compression="zstd") as writer:

                def write(frame):
                    nonlocal duplicates
                    if frame.empty:
                        return
                    notes, removed = clean_notes(frame)
                    duplicates += removed
                    for pid, count in notes.patient_id.value_counts().items():
                        counts[pid] = counts.get(pid, 0) + int(count)
                    writer.write_table(
                        pa.Table.from_pandas(notes[NOTE_COLUMNS], schema=_notes_schema(),
                                             preserve_index=False),
                        row_group_size=NOTE_ROW_GROUP,
                    )

                carry = None
                for batch in reader:
                    frame = batch.to_pandas()
                    read += len(frame)
                    if carry is not None:
                        frame = pd.concat([carry, frame], ignore_index=True)
                    # Rows are sorted by patient: hold back the last patient, who may continue.
                    last = frame.patient_id.iloc[-1]
                    carry = frame.loc[frame.patient_id.eq(last)]
                    write(frame.loc[frame.patient_id.ne(last)])
                    say(f"Notes written: {read:,}/{total:,}")
                if carry is not None:
                    write(carry)
        finally:
            con.close()
        shutil.rmtree(partial / "duckdb_spill", ignore_errors=True)

        without_notes = len(summaries) - sum(pid in counts for pid in summaries.patient_id)
        summaries = summaries.loc[summaries.patient_id.isin(counts)].sort_values(
            "patient_id").reset_index(drop=True)
        summaries["note_count"] = summaries.patient_id.map(counts).astype("int64")
        say(f"Embedding {len(summaries):,} summaries with TrialSpace")
        last_report = [0.0]

        def embedding_progress(message):
            if message.startswith("Embedded") and time.monotonic() - last_report[0] < 30:
                return
            last_report[0] = time.monotonic()
            say(message)

        embeddings = embed_guideline_patients(
            summaries, config=settings.inference_config(), batch_size=batch_size,
            progress_callback=embedding_progress,
        )
        if embeddings.encoder_identity is None:
            raise InputError("The TrialSpace encoder cannot be verified, so the app would "
                             "refuse these vectors.")
        if embeddings.patient_ids != tuple(summaries.patient_id):
            raise InputError("Embeddings do not align with the cached summaries.")
        summaries["summary_sha256"] = embeddings.summary_sha256
        summaries[["patient_id", "cancer_history_summary", "general_exclusion_criteria_evidence",
                   "summary_sha256", "note_count"]].to_parquet(
            partial / "summaries.parquet", index=False, compression="zstd")
        np.save(partial / "embeddings.npy", embeddings.vectors)
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": {
                "database": _file_info(path),
                "tables": {"summaries": summary_source, "notes": "patient_notes"},
                "summary_files": None if summary_files is None else {
                    "folder": summary_files[0].name, "files": len(summary_files[1]),
                    "bytes": sum(f.stat().st_size for f in summary_files[1])},
                "current_summaries": current,
                "sample_limit": limit,
                "excluded_blank_summary_count": blank,
                "excluded_without_notes_count": without_notes,
                "duplicate_notes_removed": duplicates,
            },
            "counts": {"patients": len(summaries), "notes": int(sum(counts.values()))},
            "embedding": {
                "model_metadata": embeddings.model_metadata,
                "encoder_identity": embeddings.encoder_identity,
                "dimension": int(embeddings.vectors.shape[1]),
                "dtype": "float32",
            },
            "files": {name: (partial / name).stat().st_size for name in FILES},
            "builder": {"package": package_metadata()},
        }
        (partial / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        for name in (*FILES, "manifest.json"):
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
    say(f"Saved {manifest['counts']['patients']:,} patients and "
        f"{manifest['counts']['notes']:,} notes to {output}")
    return manifest


def _summary_files(settings, value):
    """The folder and Parquet files of a --summaries file or folder of files."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise InputError("Use an absolute summaries file or folder.")
    path = path.resolve()
    if not _inside(path, duckdb_roots(settings)):
        raise InputError("Choose summaries within the configured database folders.")
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files or not all(f.is_file() and f.suffix == ".parquet" for f in files):
        raise InputError("Choose a .parquet summaries file or a folder of them.")
    return (path if path.is_dir() else path.parent), files


def _read_summary_files(files):
    """patient_id, cancer_history_summary and general_exclusion_criteria_evidence of every file.

    patient_id must be the database's numeric MRN, since the patients' notes are read by it.
    """
    import pyarrow.parquet as pq

    frames = []
    for path in files:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        if not {"patient_id", "cancer_history_summary"} <= names:
            raise InputError(f"{path.name} has no patient_id and cancer_history_summary columns.")
        columns = ["patient_id", "cancer_history_summary"] + (
            ["general_exclusion_criteria_evidence"] if "general_exclusion_criteria_evidence" in names else [])
        frames.append(pd.read_parquet(path, columns=columns))
    summaries = pd.concat(frames, ignore_index=True)
    if "general_exclusion_criteria_evidence" not in summaries:
        summaries["general_exclusion_criteria_evidence"] = ""
    summaries["general_exclusion_criteria_evidence"] = (
        summaries.general_exclusion_criteria_evidence.fillna("").astype(str))
    summaries = summaries.loc[summaries.cancer_history_summary.notna()]
    summaries["patient_id"] = summaries.patient_id.astype(str).str.strip()
    if not summaries.patient_id.str.fullmatch(r"\d+").all():
        raise InputError("Summary patient_id values must be the database's numeric MRNs.")
    return summaries.reset_index(drop=True)


def resolve_cache(settings, value=None):
    require_phi(settings)
    path = Path(value).expanduser() if value else Path(settings.population_cache)
    if not path.is_absolute():
        raise InputError("Enter the absolute path of a population cache folder.")
    path = path.resolve()
    roots = duckdb_roots(settings)
    default = Path(settings.population_cache).expanduser().resolve()
    if not (_inside(path, roots) or path == default):
        raise InputError("Choose a cache within the configured database folders.")
    if not (path / "manifest.json").is_file():
        raise InputError("No population cache here. Build one with the preprocess_population command.")
    return path


def _manifest(path):
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError):
        raise InputError("This population cache manifest cannot be read.") from None
    if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise InputError("Unsupported population cache format. Rebuild the cache.")
    for name, size in manifest["files"].items():
        if not (path / name).is_file() or (path / name).stat().st_size != size:
            raise InputError("This population cache is incomplete or changed. Rebuild it.")
    return manifest


def inspect_cache(settings, value=None):
    """Manifest facts for the UI; no patient rows are read."""
    path = resolve_cache(settings, value)
    manifest = _manifest(path)
    stored = manifest["embedding"]["model_metadata"]
    # embed_guideline_patients records "embed_for_matching" (package 0.6+); older caches
    # recorded "embedding_model".
    model = stored.get("embed_for_matching") or stored.get("embedding_model") or {}
    return {
        "path": str(path),
        "name": path.name,
        "created_utc": manifest["created_utc"],
        "patients": manifest["counts"]["patients"],
        "notes": manifest["counts"]["notes"],
        "database": manifest["source"]["database"]["name"],
        "sample_limit": manifest["source"]["sample_limit"],
        "embedding_model": model.get("model_name"),
    }


@dataclass(frozen=True)
class PopulationCache:
    path: Path
    manifest: dict
    manifest_sha256: str
    summaries: pd.DataFrame
    embeddings: object

    def notes(self, patient_ids):
        """Source notes for these patients only, in cached (pipeline) order."""
        import pyarrow.dataset as ds

        dataset = ds.dataset(self.path / "notes.parquet", format="parquet")
        table = dataset.to_table(filter=ds.field("patient_id").isin(list(patient_ids)))
        return table.to_pandas()


_LOADED = {}
_LOCK = threading.Lock()


def _same(a, b):
    # As retrieval compares them: the manifest's identity round-tripped through JSON.
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def _require_current_encoder(settings, manifest):
    """Refuse a cache whose vectors another TrialSpace model or encoding made.

    Both modes embed with the inference package's default TrialSpace settings; the
    guideline spaces searched against these vectors are embedded with them now.
    """
    import matchminer_ai.matching as matching

    stored = manifest["embedding"].get("encoder_identity")
    current = matching.guideline_encoder_identity(settings.inference_config())
    if current is None or not _same(stored, current):
        model = (stored or {}).get("model_path") or "an unrecorded model"
        raise InputError(
            f"This population cache was embedded by another TrialSpace encoder ({model}), "
            "so its vectors cannot be compared with the current model's. Rebuild it with "
            "the preprocess_population command."
        )


def open_cache(settings, value=None, expected=None):
    """Load once per server process and share read-only across browser sessions.

    A cache made with another TrialSpace encoder than the current one is refused.
    """
    from matchminer_ai._storage import digest
    from matchminer_ai.matching import PatientEmbeddings

    path = resolve_cache(settings, value)
    raw = (path / "manifest.json").read_bytes()
    key = (str(path), digest(raw))
    if expected is not None and key[1] != expected:
        raise InputError("The population cache was rebuilt. Load the population again.")
    with _LOCK:
        cache = _LOADED.get(key)
        if cache is None:
            manifest = _manifest(path)
            _require_current_encoder(settings, manifest)
            summaries = pd.read_parquet(path / "summaries.parquet")
            vectors = np.load(path / "embeddings.npy")
            if vectors.shape != (len(summaries), manifest["embedding"]["dimension"]):
                raise InputError("Cached embeddings do not match the summaries. Rebuild the cache.")
            embeddings = PatientEmbeddings(
                patient_ids=tuple(summaries.patient_id),
                summary_sha256=tuple(summaries.summary_sha256),
                vectors=vectors,
                model_metadata=manifest["embedding"]["model_metadata"],
                encoder_identity=manifest["embedding"]["encoder_identity"],
            )
            cache = PopulationCache(path, manifest, key[1], summaries[[
                "patient_id", "cancer_history_summary", "general_exclusion_criteria_evidence"
            ]], embeddings)
            # One population is held at a time; each is several hundred megabytes.
            _LOADED.clear()
            _LOADED[key] = cache
        return cache


def cached_population(cache):
    """A summaries-only source cohort; notes load for matched patients at retrieval."""
    from matchminer_ai.population import PatientCohort

    source = cache.manifest["source"]
    return PatientCohort(
        cache.summaries,
        pd.DataFrame(columns=NOTE_COLUMNS),
        {
            "source": "population_cache",
            "cache": {"path": str(cache.path), "name": cache.path.name,
                      "created_utc": cache.manifest["created_utc"],
                      "manifest_sha256": cache.manifest_sha256},
            "database": source["database"],
            "sample_limit": source["sample_limit"],
            "note_count": cache.manifest["counts"]["notes"],
            "notes_loaded_on_demand": True,
        },
    )
