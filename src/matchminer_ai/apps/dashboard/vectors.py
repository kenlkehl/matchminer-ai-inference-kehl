"""TrialSpace vectors of the dashboard's real trial cohorts, made by exactly the encoder of the round in use.

The synthetic patients' vectors are the round's own (``patients.npy``); a real cohort is embedded once with the
round's encoder (identity-checked by ``core.encoder``) from its key criteria with the cohort prompt, as
training's mining embeds corpus cohorts. Vectors are kept per encoder identity and cohort text, so a later
round (a new encoder) re-embeds everything and an unchanged cohort is never embedded twice. They are written
owner-only next to the store: they derive from confidential protocol text.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading

import numpy as np

from .core.cache import text_key

BATCH = 8


def cohort_text_key(text: str) -> str:
    from matchminer_ai.embedding.trialspace_contract import COHORT_PROMPT

    return text_key("cohort", COHORT_PROMPT, text.strip())


class CohortVectors:
    def __init__(self, directory: Path, identity: dict | None):
        self.directory = Path(directory)
        self.identity = identity
        self.key = text_key(identity)[:16] if identity else None
        self.lock = threading.Lock()
        self.vectors: dict[str, np.ndarray] = {}
        self.state = "no round" if identity is None else "not embedded"
        self.error: str | None = None
        self._matrix: tuple[tuple, list[str], np.ndarray] | None = None
        if self.key:
            self._load()

    @property
    def path(self) -> Path:
        return self.directory / f"cohort_vectors_{self.key}.npz"

    def _load(self):
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                if json.loads(str(data["identity"])) != self.identity:
                    return
                self.vectors = dict(zip(data["keys"].tolist(), np.asarray(data["vectors"], dtype=np.float32)))
        except (OSError, ValueError, KeyError) as exc:
            self.error = f"could not read {self.path.name}: {exc}"

    def _save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        keys = list(self.vectors)
        matrix = np.stack([self.vectors[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)
        fd, temporary = tempfile.mkstemp(prefix=".cohort_vectors.", suffix=".npz", dir=self.directory)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, keys=np.array(keys), vectors=matrix, identity=np.array(json.dumps(self.identity)))
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def missing(self, store: dict) -> list[tuple[str, str]]:
        """(text key, text) of every store cohort without a vector, dashboard cohorts first."""
        todo, seen = [], set()
        for cohort in sorted(store["cohorts"].values(), key=lambda c: not c["primary"]):
            key = cohort_text_key(cohort["key"])
            if key not in self.vectors and key not in seen:
                seen.add(key)
                todo.append((key, cohort["key"]))
        return todo

    def embed(self, store: dict, encoder, progress=None) -> dict:
        """Embed every cohort that has no vector yet; saved every few batches so a restart keeps the work."""
        from matchminer_ai.embedding.trialspace_contract import COHORT_PROMPT

        todo = self.missing(store)
        if progress:
            progress(0, len(todo), "starting the TrialSpace encoder" if todo else "all cohorts embedded")
        self.state = "embedding" if todo else "ready"
        try:
            for start in range(0, len(todo), BATCH):
                part = todo[start:start + BATCH]
                vectors = encoder.encode([text for _, text in part], COHORT_PROMPT)
                with self.lock:
                    for (key, _), vector in zip(part, vectors):
                        self.vectors[key] = np.asarray(vector, dtype=np.float32)
                    self._matrix = None
                if (start // BATCH) % 10 == 9 or start + BATCH >= len(todo):
                    with self.lock:
                        self._save()
                if progress:
                    progress(min(start + BATCH, len(todo)), len(todo), "embedding trial cohorts")
        except Exception as exc:
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
            raise
        self.state, self.error = "ready", None
        return {"embedded": len(todo), "total": len(self.vectors)}

    def matrix(self, store: dict, cohort_ids: list[str]) -> tuple[list[str], np.ndarray]:
        """The embedded cohorts among ``cohort_ids`` (in order) and their vectors as one matrix."""
        signature = (store["built_at"], len(self.vectors), tuple(cohort_ids))
        with self.lock:
            if self._matrix and self._matrix[0] == signature:
                return self._matrix[1], self._matrix[2]
            ids, rows = [], []
            for cid in cohort_ids:
                vector = self.vectors.get(cohort_text_key(store["cohorts"][cid]["key"]))
                if vector is not None:
                    ids.append(cid)
                    rows.append(vector)
            matrix = np.stack(rows) if rows else np.zeros((0, 1), np.float32)
            self._matrix = (signature, ids, matrix)
            return ids, matrix

    def vector(self, store: dict, cohort_id: str) -> np.ndarray | None:
        return self.vectors.get(cohort_text_key(store["cohorts"][cohort_id]["key"]))

    def describe(self, store: dict | None = None) -> dict:
        out = {"state": self.state, "error": self.error, "vectors": len(self.vectors), "file": self.path.name if self.key else None}
        if store is not None:
            out["missing"] = len(self.missing(store))
        return out
