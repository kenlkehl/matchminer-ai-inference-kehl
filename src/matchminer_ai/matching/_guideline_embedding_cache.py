"""Persist only guideline vectors, keyed by encoder identity and exact input text."""

from __future__ import annotations

from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from matchminer_ai._storage import digest
from matchminer_ai.embedding.inference import _embedding_cache_identity


def _vector(value, dimension):
    vector = np.asarray(value, dtype="<f8")
    if (
        vector.shape != (dimension,)
        or not np.isfinite(vector).all()
        or not np.any(vector)
    ):
        raise ValueError("TrialSpace returned invalid guideline embeddings.")
    return vector


def _read(path, identity, keys, dimension):
    found = {}
    with closing(sqlite3.connect(path, timeout=5)) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)"
            )
            row = connection.execute("SELECT identity FROM metadata").fetchone()
            serialized = json.dumps(identity, sort_keys=True)
            if row is None:
                connection.execute("INSERT INTO metadata VALUES (?)", (serialized,))
            elif row[0] != serialized:
                raise sqlite3.DatabaseError("Cache identity mismatch")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(input_sha256 TEXT PRIMARY KEY, vector BLOB NOT NULL, checksum TEXT NOT NULL)"
            )
        # One read transaction and batched lookups avoid a filesystem lock round
        # trip per space, especially when the cache directory is on a slow mount.
        connection.execute("BEGIN")
        keys = list(keys)
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                "SELECT input_sha256, vector, checksum FROM embeddings "
                f"WHERE input_sha256 IN ({placeholders})",
                batch,
            )
            for key, data, checksum in rows:
                try:
                    if digest(key.encode() + data) != checksum:
                        continue
                    found[key] = _vector(np.frombuffer(data, dtype="<f8"), dimension)
                except (TypeError, ValueError):
                    continue  # A damaged entry is a miss, never an accepted vector.
        connection.commit()
    return found


def _write(path, vectors):
    rows = []
    for key, vector in vectors.items():
        data = vector.tobytes()
        rows.append((key, data, digest(key.encode() + data)))
    with closing(sqlite3.connect(path, timeout=5)) as connection:
        with connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)", rows
            )


def cached_guideline_embeddings(
    spaces, *, config, patient_model, dimension, directory, progress, embed
):
    """Load reusable catalog vectors; embed and atomically save only missing text."""
    root = Path(directory).expanduser().resolve()
    package = Path(__file__).resolve().parents[1]
    code_roots = [package]
    if (package.parent.parent / "pyproject.toml").exists():
        code_roots.append(package.parent.parent)
    if any(root.is_relative_to(path) for path in code_roots):
        raise ValueError(
            "Guideline embedding cache must stay outside the code repository."
        )

    identity = _embedding_cache_identity(config.embedding, patient_model)
    keys = [digest(text.encode()) for text in spaces.clinical_space_summary]
    unique = dict(zip(keys, spaces.index, strict=True))
    found = {}
    path = None
    status = "unverified_encoder"
    if identity is not None:
        identity = {**identity, "dimension": dimension}
        path = root / f"trialspace-{digest(identity)}.sqlite3"
        try:
            root.mkdir(parents=True, exist_ok=True)
            found = _read(path, identity, unique, dimension)
            status = "ready"
        except (OSError, sqlite3.Error):
            status = "unavailable"
            progress(
                "Guideline embedding cache unavailable; recomputing catalog vectors"
            )
            logging.getLogger(__name__).warning(
                "Guideline vector cache could not be read"
            )
    else:
        progress("Encoder revision is unverified; guideline vectors will not be cached")

    hit_spaces = sum(key in found for key in keys)
    missing = [key for key in unique if key not in found]
    progress(
        f"Reusing {hit_spaces} cached guideline embeddings; "
        f"embedding {len(missing)} new or changed guideline texts"
    )
    if missing:
        subset = spaces.loc[
            [unique[key] for key in missing],
            ["space_trial_id", "clinical_space_summary"],
        ]
        fresh, metadata = embed(
            subset, entity_type="trial", config=config, return_metadata=True
        )
        if metadata["model_metadata"]["embedding_model"] != patient_model:
            raise ValueError(
                "Patient and guideline embedding model metadata differ; refusing to mix embeddings."
            )
        if fresh.space_trial_id.duplicated().any() or set(fresh.space_trial_id) != set(
            subset.space_trial_id
        ):
            raise ValueError(
                "TrialSpace did not return exactly the requested guideline IDs."
            )
        indexed = fresh.set_index("space_trial_id")
        computed = {
            key: _vector(indexed.loc[identifier, "embedding"], dimension)
            for key, identifier in zip(missing, subset.space_trial_id, strict=True)
        }
        if status == "ready":
            try:
                _write(path, computed)
            except (OSError, sqlite3.Error):
                status = "write_failed"
                progress(
                    "Guideline embeddings computed, but the disk cache could not be saved"
                )
        found.update(computed)
    result = pd.DataFrame(
        {
            "space_trial_id": spaces.space_trial_id,
            "embedding": [found[k].tolist() for k in keys],
        }
    )
    return result, {
        "enabled": True,
        "status": status,
        "path": str(path) if path is not None else None,
        "identity_sha256": digest(identity) if identity is not None else None,
        "loaded_revision": identity.get("loaded_revision") if identity else None,
        "cached_spaces": hit_spaces,
        "embedded_texts": len(missing),
    }
