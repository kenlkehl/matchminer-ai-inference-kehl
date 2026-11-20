"""One completed note compression per browser session, held only in memory."""

import hashlib
import json
import threading
import time
from copy import deepcopy

from matchminer_ai.cancellation import check_cancelled


def _identity(text, notes, config):
    # Match exact source rows, including their order, dates and note types.
    source = {"text": text, "notes": notes}
    settings = {"patient": config.patient, "remote": config.remote, "version": 1}
    return tuple(
        hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
        for value in (source, settings)
    )


class NoteCompressions:
    def __init__(self, ttl=3600):
        self.ttl = ttl
        self.lock = threading.RLock()
        self.entries = {}

    def clean(self):
        with self.lock:
            for owner, entry in list(self.entries.items()):
                if time.monotonic() - entry["updated"] > self.ttl:
                    del self.entries[owner]

    def bind(self, owner):
        with self.lock:
            self.clean()
            entry = self.entries.setdefault(owner, {"snapshot": None})
            entry["updated"] = time.monotonic()
            return _BoundCompression(self, owner, entry)

    def clear(self, owner):
        with self.lock:
            # Removing the entry also invalidates handles held by running jobs.
            self.entries.pop(owner, None)

    def latest(self, owner):
        with self.lock:
            self.clean()
            entry = self.entries.get(owner)
            if entry is None or entry["snapshot"] is None:
                return None
            entry["updated"] = time.monotonic()
            return deepcopy(entry["snapshot"])


class _BoundCompression:
    def __init__(self, store, owner, entry):
        self.store, self.owner, self.entry = store, owner, entry

    def lookup(self, text, notes, config):
        identity = _identity(text, notes, config)
        with self.store.lock:
            self.store.clean()
            if self.store.entries.get(self.owner) is not self.entry:
                return None
            self.entry["updated"] = time.monotonic()
            if self.entry.get("identity") != identity:
                self.entry.update(identity=identity, snapshot=None)
            return deepcopy(self.entry["snapshot"])

    def save(self, text, notes, config, compressed_notes, metadata):
        identity = _identity(text, notes, config)
        with self.store.lock:
            check_cancelled()
            if self.store.entries.get(self.owner) is not self.entry:
                return
            self.entry.update(
                identity=identity,
                updated=time.monotonic(),
                snapshot=deepcopy(
                    {
                        "notes_input": {"text": text, "notes": notes},
                        "notes": compressed_notes,
                        "metadata": metadata,
                        "model": config.patient["remote"]["model_name"],
                    }
                ),
            )
