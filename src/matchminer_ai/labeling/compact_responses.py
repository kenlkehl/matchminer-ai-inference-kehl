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
    """One writer connection; concurrent writers share a commit (each still returns only once its rows are durable)
    and every reading thread has its own read-only connection. With one transaction and fsync per response, and
    reads waiting on the same lock, a labeler process stalled near 30 responses/s (2026-11-10)."""

    def __init__(self, root, *, read_only=False, reserve_bytes=16 * 1024**3):
        self.root = Path(root)
        self.path = self.root/FILENAME
        self.read_only = read_only
        self.lock = threading.RLock()
        self.reserve_bytes = reserve_bytes
        self.last_space_check = 0.0
        self.pending, self.pending_lock = [], threading.Lock()
        self.readers, self.reader_connections = threading.local(), []
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

    def _select(self, sql, parameters=()):
        if self.read_only:
            with self.lock:
                return self.connection.execute(sql, parameters).fetchall()
        reader = getattr(self.readers, 'connection', None)
        if reader is None:  # WAL: it reads committed rows without waiting for the writer
            reader = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro', uri=True, timeout=60,
                                     check_same_thread=False, isolation_level=None)
            with self.pending_lock:
                self.reader_connections.append(reader)
            self.readers.connection = reader
        return reader.execute(sql, parameters).fetchall()

    def get_bytes(self, key):
        rows = self._select('SELECT body,sha256,bytes FROM responses WHERE key=?', (key,))
        if not rows:
            return None
        row = rows[0]
        raw = zlib.decompress(row[0])
        if len(raw) != row[2] or hashlib.sha256(raw).hexdigest() != row[1]:
            raise ValueError('Corrupt compact response: '+key)
        return raw

    def keys(self):
        return {row[0] for row in self._select('SELECT key FROM responses')}

    def put_many(self, items):
        if self.read_only:
            raise ValueError('Cannot write a read-only response store')
        encoded = []
        for key, raw in items:
            if json.loads(raw).get('key') != key:
                raise ValueError('Response key does not match its content')
            encoded.append((key, zlib.compress(raw, 1), hashlib.sha256(raw).hexdigest(), len(raw)))
        batch = dict(rows=encoded, done=threading.Event(), error=None)
        with self.pending_lock:
            self.pending.append(batch)
        # Whoever holds the writer lock commits everything pending in one transaction, in arrival order.
        while not batch['done'].is_set():
            with self.lock:
                if batch['done'].is_set():
                    break
                with self.pending_lock:
                    batches, self.pending = self.pending, []
                try:
                    self._commit([row for waiting in batches for row in waiting['rows']])
                except BaseException as error:
                    for waiting in batches:
                        waiting['error'] = error
                finally:
                    for waiting in batches:
                        waiting['done'].set()
        if batch['error'] is not None:
            raise batch['error']

    def _commit(self, encoded):
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
            with self.pending_lock:
                readers, self.reader_connections = self.reader_connections, []
            for reader in readers:  # before the checkpoint: open readers would keep the WAL from truncating
                reader.close()
            if self.connection is not None:
                if not self.read_only:
                    self.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                    self.connection.execute('PRAGMA journal_mode=DELETE')
                self.connection.close()
                self.connection = None
