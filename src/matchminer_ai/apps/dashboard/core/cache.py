"""The app's own small cache directory: JSON values keyed by name, and a SQLite score cache.

Data files are identified by ``file_stamp`` (byte size and whole-second mtime), never hashed whole,
following the training repository's file-identity policy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time


def file_stamp(path: Path) -> str | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def text_key(*parts) -> str:
    """SHA-256 of in-memory values (texts and identities), never of data files."""
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class JSONCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def path(self, name: str) -> Path:
        return self.directory / f"{name}.json"

    def get(self, name: str):
        path = self.path(name)
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def put(self, name: str, value) -> None:
        with self.lock:
            fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=self.directory)
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, ensure_ascii=False)
            os.replace(temporary, self.path(name))


class ScoreCache:
    """Decision-model answers keyed by model fingerprint, question digest, family and the exact input texts."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
                                "created REAL NOT NULL)")
        self.connection.commit()

    def get_many(self, keys: list[str]) -> dict[str, dict]:
        found = {}
        with self.lock:
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                rows = self.connection.execute(
                    f"SELECT key, value FROM scores WHERE key IN ({','.join('?' * len(chunk))})", chunk).fetchall()
                found.update({key: json.loads(value) for key, value in rows})
        return found

    def put_many(self, values: dict[str, dict]) -> None:
        with self.lock:
            self.connection.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
                                        [(k, json.dumps(v), time.time()) for k, v in values.items()])
            self.connection.commit()

    def count(self) -> int:
        with self.lock:
            return self.connection.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
