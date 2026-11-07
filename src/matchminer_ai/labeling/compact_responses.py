"""Compressed SQLite response checkpoints on local disk, with complete trace fidelity."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import time
import zlib

FILENAME = 'responses.sqlite'


class CompactResponses:
    def __init__(self, root, *, read_only=False, reserve_bytes=16 * 1024**3):
        self.root = Path(root)
        self.path = self.root/FILENAME
        self.read_only = read_only
        self.lock = threading.RLock()
        self.reserve_bytes = reserve_bytes
        self.last_space_check = 0.0
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True)
        if read_only:
            self.connection = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro', uri=True,
                timeout=60, check_same_thread=False, isolation_level=None)
        else:
            self.connection = sqlite3.connect(self.path, timeout=60, check_same_thread=False, isolation_level=None)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.execute('CREATE TABLE IF NOT EXISTS responses '
                '(key TEXT PRIMARY KEY, body BLOB NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL) WITHOUT ROWID')
            self.connection.execute('PRAGMA user_version=1')
        if self.connection.execute('PRAGMA user_version').fetchone()[0] != 1:
            raise ValueError('Unsupported compact response format')

    def get_bytes(self, key):
        with self.lock:
            row = self.connection.execute('SELECT body,sha256,bytes FROM responses WHERE key=?', (key,)).fetchone()
        if row is None:
            return None
        raw = zlib.decompress(row[0])
        if len(raw) != row[2] or hashlib.sha256(raw).hexdigest() != row[1]:
            raise ValueError('Corrupt compact response: '+key)
        return raw

    def keys(self):
        with self.lock:
            return {row[0] for row in self.connection.execute('SELECT key FROM responses')}

    def put_many(self, items):
        if self.read_only:
            raise ValueError('Cannot write a read-only response store')
        encoded = []
        for key, raw in items:
            if json.loads(raw).get('key') != key:
                raise ValueError('Response key does not match its content')
            encoded.append((key, zlib.compress(raw, 1), hashlib.sha256(raw).hexdigest(), len(raw)))
        with self.lock:
            if time.monotonic()-self.last_space_check >= 1:
                if shutil.disk_usage(self.root).free < self.reserve_bytes:
                    raise OSError('Local response storage reached its free-disk reserve; checkpoints retained')
                self.last_space_check = time.monotonic()
            self.connection.execute('BEGIN IMMEDIATE')
            try:
                self.connection.executemany('INSERT INTO responses VALUES (?,?,?,?) '
                    'ON CONFLICT(key) DO UPDATE SET body=excluded.body,sha256=excluded.sha256,bytes=excluded.bytes', encoded)
                self.connection.execute('COMMIT')
            except BaseException:
                self.connection.execute('ROLLBACK')
                raise

    def close(self):
        with self.lock:
            if self.connection is not None:
                if not self.read_only:
                    self.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                    self.connection.execute('PRAGMA journal_mode=DELETE')
                self.connection.close()
                self.connection = None
