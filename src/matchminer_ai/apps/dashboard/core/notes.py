"""A synthetic patient's raw notes, read on demand from ``all_synthetic_notes.parquet``.

The 14 GB file is not sorted by patient, so a patient-to-row-group index is built once (reading only the
``patient_id`` column, about 30 s) and cached per file stamp in the app's cache directory; one patient's
notes then come from its ~8 row groups.
"""
from __future__ import annotations

from pathlib import Path
import threading

from .cache import JSONCache, file_stamp

COLUMNS = ["patient_id", "date", "event_type", "synthetic_note"]


class NotesIndex:
    def __init__(self, path: Path, cache: JSONCache):
        self.path, self.cache = Path(path), cache
        self.index: dict[str, list[int]] | None = None
        self.state = "absent" if not self.path.exists() else "not built"
        self.error: str | None = None
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        if self.path.exists():
            cached = cache.get("notes_index")
            if cached and cached.get("stamp") == file_stamp(self.path):
                self.index, self.state = cached["index"], "ready"

    def ensure(self) -> str:
        """Start building the index in the background if needed; returns the state."""
        with self.lock:
            if self.state in ("not built", "error") and self.path.exists():
                self.state, self.error = "building", None
                self.thread = threading.Thread(target=self._build, name="mmai-notes-index", daemon=True)
                self.thread.start()
        return self.state

    def _build(self):
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(self.path)
            index: dict[str, set[int]] = {}
            for group in range(parquet.metadata.num_row_groups):
                for pid in parquet.read_row_group(group, columns=["patient_id"]).column(0).to_pylist():
                    index.setdefault(pid, set()).add(group)
            self.index = {pid: sorted(groups) for pid, groups in index.items()}
            self.cache.put("notes_index", {"stamp": file_stamp(self.path), "index": self.index})
            self.state = "ready"
        except Exception as exc:
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"

    def notes(self, pid: str) -> list[dict]:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        if self.index is None:
            raise RuntimeError("the notes index is not ready")
        groups = self.index.get(pid)
        if not groups:
            return []
        table = pq.ParquetFile(self.path).read_row_groups(groups, columns=COLUMNS)
        rows = table.filter(pc.equal(table["patient_id"], pid)).to_pylist()
        rows.sort(key=lambda r: (str(r.get("date") or ""),))
        return [{"date": r.get("date"), "type": r.get("event_type"), "text": r.get("synthetic_note") or ""} for r in rows]

    def describe(self) -> dict:
        return {"path": str(self.path), "state": self.state, "error": self.error,
                "patients": len(self.index) if self.index else None}
