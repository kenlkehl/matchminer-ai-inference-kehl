"""Load and look up source-derived guideline catalogs without model calls."""

from __future__ import annotations

import copy
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Callable

from matchminer_ai._metadata import package_metadata
from matchminer_ai._storage import digest, read_json

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class _CachedCatalog:
    signature: tuple
    records: list[dict]
    artifact: dict
    source_bytes: int


# Process-local, guideline-only cache. Never store patient records or returned
# mutable objects here. Bounds also prevent old editions accumulating forever.
_CATALOG_CACHE: OrderedDict[Path, _CachedCatalog] = OrderedDict()
_CATALOG_CACHE_LOCK = RLock()
_CATALOG_CACHE_MAX_FILES = 128
_CATALOG_CACHE_MAX_BYTES = 64 * 1024 * 1024


def _file_signature(path, *, required=False):
    try:
        info = path.stat()
    except FileNotFoundError:
        if required:
            raise
        return None
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _catalog_signature(path):
    return (
        _file_signature(path, required=True),
        _file_signature(path.parent / "status.json"),
        _file_signature(path.parent / "validation.json"),
    )


def _load_catalog_file(path, *, refresh, progress):
    import pandas as pd

    # One reader populates a version at a time, including concurrent API calls.
    with _CATALOG_CACHE_LOCK:
        previous = _CATALOG_CACHE.pop(path, None)
        signature = _catalog_signature(path)
        if not refresh and previous is not None and previous.signature == signature:
            _CATALOG_CACHE[path] = previous
            return (
                copy.deepcopy(previous.records),
                copy.deepcopy(previous.artifact),
                True,
            )

        # Remove a superseded version before loading. Failed reads/validation
        # must never fall back to stale records from an earlier successful call.
        progress()
        content = path.read_bytes()
        file_hash = digest(content)
        if (
            signature[1] is not None
            and read_json(path.parent / "status.json").get("status") != "complete"
        ):
            raise ValueError(f"Catalog is not complete: {path.parent}")
        audit_status = "unavailable"
        if signature[2] is not None:
            audit = read_json(path.parent / "validation.json")
            if (
                audit.get("status") != "passed"
                or audit.get("paradigms_sha256") != file_hash
            ):
                raise ValueError(
                    "Recorded catalog audit did not pass or its export hash differs."
                )
            audit_status = "passed_export_hash_verified"
        records = []
        for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Catalog line {number} must be a record object.")
            records.append(row)
        if records:
            _validate_catalog(pd.DataFrame(records))
        if _catalog_signature(path) != signature:
            raise ValueError("Catalog files changed while loading; retry the request.")
        artifact = {
            "input_type": "jsonl",
            "path": str(path),
            "sha256": file_hash,
            "recorded_source_audit": audit_status,
        }
        if len(content) <= _CATALOG_CACHE_MAX_BYTES:
            _CATALOG_CACHE[path] = _CachedCatalog(
                signature, records, artifact, len(content)
            )
            while (
                len(_CATALOG_CACHE) > _CATALOG_CACHE_MAX_FILES
                or sum(entry.source_bytes for entry in _CATALOG_CACHE.values())
                > _CATALOG_CACHE_MAX_BYTES
            ):
                _CATALOG_CACHE.popitem(last=False)
        return copy.deepcopy(records), copy.deepcopy(artifact), False


def _validate_catalog(frame):
    from ._guideline_schema import format_space

    required = {
        "space_trial_id",
        "trial_id",
        "clinical_space_summary",
        "name",
        "space",
        "diagnostic_workup",
        "treatment_options",
        "evidence",
        "source",
        "uncertainties",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Guideline catalog is missing columns: {sorted(missing)}. Use paradigms.jsonl, not trial_spaces.csv."
        )
    reserved = {
        "patient_id",
        "cancer_history_summary",
        "rank",
        "retrieval_rank",
        "match_quality_score",
        "match_quality_pass",
        "similarity_score",
    }
    if reserved.intersection(frame.columns):
        raise ValueError(
            "Catalog must contain guideline records only, without patient or ranking columns."
        )
    if frame.empty:
        raise ValueError("Guideline catalog is empty.")
    for row in frame.to_dict("records"):
        for key in ("space_trial_id", "trial_id", "clinical_space_summary", "name"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Catalog {key} must be a nonempty string.")
        if row["clinical_space_summary"] != format_space(row["space"]):
            raise ValueError(
                "Catalog clinical_space_summary differs from its stored nine-field space."
            )
        if not isinstance(row["source"], dict) or not all(
            isinstance(row["source"].get(k), str) and row["source"][k]
            for k in (
                "disease",
                "title",
                "version",
                "source_sha256",
                "source_fingerprint",
            )
        ):
            raise ValueError(
                "Catalog records must retain source edition and fingerprints."
            )
        if not isinstance(row["uncertainties"], list) or not all(
            isinstance(s, str) for s in row["uncertainties"]
        ):
            raise ValueError("Catalog uncertainties must be a list of strings.")
        owners = [row]
        for menu in ("diagnostic_workup", "treatment_options"):
            if not isinstance(row[menu], list):
                raise ValueError(f"Catalog {menu} must be a list.")
            for item in row[menu]:
                if not isinstance(item, dict) or not all(
                    isinstance(item.get(key), str) and item[key].strip()
                    for key in ("name", "conditions", "category")
                ):
                    raise ValueError(
                        "Each catalog option needs its name, conditions, and category."
                    )
                owners.append(item)
        for owner in owners:
            if not isinstance(owner.get("evidence"), list) or not owner["evidence"]:
                raise ValueError("Each catalog state and option must retain evidence.")
            for evidence in owner["evidence"]:
                if (
                    not isinstance(evidence, dict)
                    or not isinstance(evidence.get("page_id"), str)
                    or not isinstance(evidence.get("line_ids"), list)
                    or not evidence["line_ids"]
                    or any(type(n) is not int or n < 1 for n in evidence["line_ids"])
                    or not isinstance(evidence.get("quote"), str)
                    or not evidence["quote"].strip()
                ):
                    raise ValueError(
                        "Catalog evidence must retain source page/line addresses and quotations."
                    )
    if frame["space_trial_id"].duplicated().any():
        raise ValueError(
            "Catalog space_trial_id values must be unique, including across supplied files."
        )


def load_guideline_catalog(
    catalog: str | Path | pd.DataFrame | list[str | Path],
    *,
    return_metadata: bool = False,
    refresh: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Load complete catalog records, retaining menus and source provenance.

    Parameters
    ----------
    catalog : path, DataFrame, or list of paths
        An external completed disease output directory, its ``paradigms.jsonl``,
        a list of those paths, or the DataFrame returned by ``summarize_guidelines``.
        The standalone prototype's JSONL exports are also supported. CSV space
        exports omit the menus and cannot be used here.
    return_metadata : bool, optional
        Also return the catalog fingerprint, file hashes, source editions, and
        any recorded audit status. No source PDFs or model calls are needed.
    refresh : bool, default False
        Force rereading and validating supplied files, replacing cached versions.
        Otherwise unchanged files reuse previously validated records in memory.
        DataFrame input is always copied and validated without caching.
    progress_callback : callable, optional
        Receives file-change checking, loading/validation, and cache reuse updates.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, dict]
        Complete, validated catalog records. Stored clinical text, conditions,
        categories, evidence, and uncertainties are preserved without rewriting.

    Notes
    -----
    A present ``status.json`` must say complete. A present ``validation.json``
    must report a passing audit whose export hash matches the JSONL. This loader
    checks stored structure and export integrity; it does not repeat the full
    source audit or establish clinical correctness. Bare JSONL/DataFrame inputs
    have no verified audit status. Catalogs remain source-derived local data.
    The process-local cache checks file identity, size, modification time, and
    change time for the JSONL and its status/audit files on every call. It follows
    the filesystem's metadata visibility (including network filesystem delays);
    use ``refresh=True`` to force a full read. Cache entries are bounded to 128
    files and 64 MiB of source JSONL, and disappear on process restart. Returned
    records are independent copies, including nested menus and evidence.
    """
    import pandas as pd

    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")

    def progress(message):
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                logging.getLogger(__name__).exception(
                    "Guideline catalog progress callback failed"
                )

    artifacts = []
    cached_files = 0
    loaded_files = 0
    if isinstance(catalog, pd.DataFrame):
        progress("Validating supplied guideline records")
        frame = pd.DataFrame(
            copy.deepcopy(catalog.to_dict("records")), columns=catalog.columns
        )
        artifacts.append(
            {"input_type": "dataframe", "recorded_source_audit": "unavailable"}
        )
        _validate_catalog(frame)
    else:
        paths = catalog if isinstance(catalog, (list, tuple)) else [catalog]
        if not paths:
            raise ValueError("Supply at least one guideline catalog.")
        progress("Checking guideline catalog files for changes")
        records = []
        for index, value in enumerate(paths, 1):
            if not isinstance(value, (str, Path)):
                raise TypeError("catalog must be a DataFrame, path, or list of paths.")
            # Resolve links before choosing sidecars, so a JSONL symlink cannot
            # bypass the completion/audit files beside its actual target.
            path = Path(value).resolve()
            if path.is_dir():
                path = path / "paradigms.jsonl"
            if path.suffix != ".jsonl":
                raise ValueError(
                    "Use a completed catalog directory or paradigms.jsonl; CSV lacks considerations."
                )
            rows, artifact, hit = _load_catalog_file(
                path,
                refresh=refresh,
                progress=lambda: progress(
                    f"Loading and validating guideline catalog {index}/{len(paths)}"
                ),
            )
            records.extend(rows)
            artifacts.append(artifact)
            cached_files += int(hit)
            loaded_files += int(not hit)
        frame = pd.DataFrame(records)
        if frame.empty:
            _validate_catalog(frame)
        # Individual files are validated once per version. Collection-level
        # uniqueness and legacy numbering still depend on this call's selection.
        if frame["space_trial_id"].duplicated().any():
            raise ValueError(
                "Catalog space_trial_id values must be unique, including across supplied files."
            )
    # Early standalone exports omitted these two standard trial-space columns.
    # Add bookkeeping only; no clinical field or menu is changed.
    if "clinical_space_number" not in frame:
        frame["clinical_space_number"] = None
    # Mixed old/new inputs can contain the column but lack it on legacy rows.
    # Allocate unused numbers within each guideline without changing stored ones.
    for _, group in frame.groupby("trial_id", sort=False):
        existing = group["clinical_space_number"].dropna()
        if (
            any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not float(number).is_integer()
                or number < 1
                for number in existing
            )
            or existing.duplicated().any()
        ):
            raise ValueError(
                "Catalog space numbers must be unique positive integers per trial_id."
            )
        used = set(existing)
        number = 1
        for index in group.index[group["clinical_space_number"].isna()]:
            while number in used:
                number += 1
            frame.loc[index, "clinical_space_number"] = number
            used.add(number)
    frame["clinical_space_number"] = frame["clinical_space_number"].astype(int)
    if "general_exclusion_criteria" not in frame:
        frame["general_exclusion_criteria"] = "NA"
    else:
        frame["general_exclusion_criteria"] = frame[
            "general_exclusion_criteria"
        ].fillna("NA")
    sources = {}
    for source in frame["source"]:
        sources[source["source_fingerprint"]] = {
            key: source[key]
            for key in (
                "disease",
                "title",
                "version",
                "source_sha256",
                "source_fingerprint",
            )
        }
    metadata = {
        "package": package_metadata(),
        "catalog_sha256": digest(frame.to_dict("records")),
        "catalog_spaces": len(frame),
        "artifacts": artifacts,
        "sources": list(sources.values()),
        "clinical_correctness": "not established",
        "validation_cache": {
            "cached_files": cached_files,
            "loaded_files": loaded_files,
            "storage": "process_memory",
        },
    }
    frame = frame.reset_index(drop=True)
    if artifacts and not isinstance(catalog, pd.DataFrame):
        progress(
            f"Guideline catalogs ready: reused {cached_files} validated catalogs; "
            f"loaded and validated {loaded_files} catalogs"
        )
    return (frame, metadata) if return_metadata else frame


def get_guideline_considerations(
    catalog: str | Path | pd.DataFrame | list[str | Path],
    *,
    space_trial_id: str | None = None,
    clinical_space_summary: str | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Return the stored diagnostic/treatment considerations for an exact space.

    Select by ``space_trial_id``, exact ``clinical_space_summary`` text, or both.
    No fuzzy lookup, model generation, or inferred menu applicability is used.
    An unknown ID/text or a mismatched ID/text pair raises ``KeyError``. If the
    exact definition occurs in multiple supplied source editions, text lookup
    returns all matching records with their provenance; use an ID to select one.
    Return conventions match ``load_guideline_catalog``.
    """
    if space_trial_id is None and clinical_space_summary is None:
        raise ValueError(
            "Supply space_trial_id or clinical_space_summary for an exact catalog lookup."
        )
    for value in (space_trial_id, clinical_space_summary):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("Space lookup values must be nonempty strings.")
    frame, metadata = load_guideline_catalog(catalog, return_metadata=True)
    keep = frame.index == frame.index
    if space_trial_id is not None:
        keep &= frame["space_trial_id"] == space_trial_id
    if clinical_space_summary is not None:
        keep &= frame["clinical_space_summary"] == clinical_space_summary
    result = frame.loc[keep].reset_index(drop=True)
    if result.empty:
        raise KeyError("The requested space is not present in this guideline catalog.")
    return (result, metadata) if return_metadata else result
