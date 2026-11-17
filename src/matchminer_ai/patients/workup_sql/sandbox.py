"""The only way model-written SQL runs: a persistent, isolated DuckDB worker process.

:class:`SQLSandbox` starts ``_sql_worker.py`` with ``-I -S``, loads a database
into it (a DuckDB file opened read-only, or :class:`ExtractionDatabase` tables
sent over a pipe and held in memory, never written to disk), and then sends one
statement at a time. The worker accepts exactly one read-only SELECT statement
(SELECT, WITH, FROM, DESCRIBE, SUMMARIZE, SHOW) and runs it with DuckDB's file
and network access, extensions and configuration changes disabled, under a
seccomp allowlist without open, socket or process creation, a memory limit and
a per-statement timeout. Results are capped in rows and characters.
"""

from __future__ import annotations

import io
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from matchminer_ai.cancellation import check_cancelled


class SQLSandboxError(RuntimeError):
    """The worker failed; the message never contains SQL, results or patient data."""


@dataclass(frozen=True)
class SandboxLimits:
    """Bounds of the SQL worker.

    ``memory_mb`` caps the worker's address space growth after start-up and
    ``duckdb_memory_mb`` DuckDB's own buffers (spilling to disk is disabled);
    ``query_timeout_seconds`` interrupts a statement, and the worker is killed
    ``kill_grace_seconds`` later if it does not answer.
    """

    memory_mb: int = 4096
    duckdb_memory_mb: int = 2048
    startup_timeout_seconds: float = 180.0
    query_timeout_seconds: float = 20.0
    kill_grace_seconds: float = 10.0
    max_sql_chars: int = 12000
    max_reply_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        if self.duckdb_memory_mb < 64 or self.memory_mb < self.duckdb_memory_mb:
            raise ValueError("memory_mb must be at least duckdb_memory_mb (>= 64).")
        if self.query_timeout_seconds <= 0 or self.startup_timeout_seconds <= 0:
            raise ValueError("Timeouts must be positive.")


def _package_paths():
    import duckdb
    import pyarrow

    return list(dict.fromkeys(
        str(Path(module.__file__).resolve().parent.parent) for module in (duckdb, pyarrow)
    ))


def _arrow_bytes(table) -> bytes:
    import pyarrow as pa

    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


class SQLSandbox:
    """A loaded, isolated DuckDB worker. Thread-safe: statements run one at a time.

    Pass ``database_path`` (a file from
    :func:`~matchminer_ai.patients.workup_sql.write_extraction_database`) or
    ``database`` (an :class:`ExtractionDatabase`, loaded in memory).
    ``info`` is the file's ``database_info`` table (empty for in-memory tables).
    """

    def __init__(self, *, database_path=None, database=None, limits: SandboxLimits | None = None):
        if (database_path is None) == (database is None):
            raise ValueError("Pass exactly one of database_path or database.")
        if sys.platform != "linux":
            raise SQLSandboxError("The isolated SQL worker requires Linux and libseccomp.")
        check_cancelled()
        self.limits = limits or SandboxLimits()
        self.lock = threading.Lock()
        payloads, tables, statements = [], [], []
        if database is not None:
            for name, table in database.tables.items():
                data = _arrow_bytes(table)
                payloads.append(data)
                tables.append({"name": name, "bytes": len(data)})
            statements = database.statements()
        setup = {
            "database": str(Path(database_path).resolve()) if database_path is not None else None,
            "tables": tables,
            "statements": statements,
            "memory_mb": self.limits.memory_mb,
            "duckdb_memory_mb": self.limits.duckdb_memory_mb,
        }
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-S", str(Path(__file__).with_name("_sql_worker.py")),
             json.dumps(_package_paths())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env={name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            cwd="/", close_fds=True, bufsize=0,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            payload = (json.dumps(setup) + "\n").encode() + b"".join(payloads)
            ready = self._exchange(payload, timeout=self.limits.startup_timeout_seconds)
            if ready.get("ready") is not True or not isinstance(ready.get("info"), dict):
                raise SQLSandboxError("SQL worker isolation unavailable; no SQL was run.")
            self.info = {str(k): str(v) for k, v in ready["info"].items()}
        except BaseException:
            self.close()
            raise

    def _exchange(self, payload: bytes, *, timeout):
        failed = threading.Event()

        def send():
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(self.process.stdin.fileno(), view)
                    view = view[written:]
            except (OSError, ValueError):
                failed.set()

        writer = threading.Thread(target=send, daemon=True)
        writer.start()
        deadline = time.monotonic() + timeout
        chunks, size = [], 0
        try:
            while True:
                check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SQLSandboxError("SQL worker exceeded its wall-time limit.")
                if not self.selector.select(min(remaining, 0.1)):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk or failed.is_set():
                    raise SQLSandboxError("SQL worker exited or its transport failed.")
                size += len(chunk)
                if size > self.limits.max_reply_bytes:
                    raise SQLSandboxError("SQL worker exceeded its output limit.")
                chunks.append(chunk)
                if chunk.endswith(b"\n"):
                    try:
                        reply = json.loads(b"".join(chunks))
                    except (ValueError, UnicodeError):
                        raise SQLSandboxError("SQL worker returned invalid data.") from None
                    if not isinstance(reply, dict):
                        raise SQLSandboxError("SQL worker returned invalid data.")
                    return reply
        except BaseException:
            self.close()
            raise
        finally:
            writer.join(timeout=1)

    def query(self, sql: str, *, max_rows: int, max_cell_chars: int | None = None,
              count_limit: int = 1_000_000, timeout: float | None = None) -> dict:
        """Run one statement: ``columns``, ``types``, up to ``max_rows`` ``rows``,
        ``row_count`` (counted up to ``count_limit``; ``more_rows`` beyond it),
        ``error`` (an exception type name) and ``error_detail`` (for the model only)."""
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("Empty SQL.")
        if len(sql) > self.limits.max_sql_chars:
            return {"columns": [], "types": [], "rows": [], "row_count": 0, "more_rows": False,
                    "error": "SQLTooLong", "seconds": 0.0,
                    "error_detail": f"SQL longer than {self.limits.max_sql_chars} characters."}
        timeout = timeout or self.limits.query_timeout_seconds
        request = {"sql": sql, "timeout": timeout, "max_rows": int(max_rows),
                   "count_limit": int(count_limit), "max_cell_chars": max_cell_chars}
        with self.lock:
            if self.process.poll() is not None:
                raise SQLSandboxError("SQL worker is not running.")
            reply = self._exchange((json.dumps(request, ensure_ascii=True) + "\n").encode(),
                                   timeout=timeout + self.limits.kill_grace_seconds)
        expected = {"columns", "types", "rows", "row_count", "more_rows", "error",
                    "error_detail", "seconds"}
        if (set(reply) != expected or not isinstance(reply["rows"], list)
                or len(reply["rows"]) > max_rows or type(reply["row_count"]) is not int
                or not isinstance(reply["columns"], list)
                or any(not isinstance(r, list) or len(r) != len(reply["columns"])
                       for r in reply["rows"])
                or (reply["error"] is not None and not isinstance(reply["error"], str))):
            self.close()
            raise SQLSandboxError("SQL worker returned an invalid result.")
        return reply

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            self.selector.close()
        except Exception:  # noqa: BLE001
            pass
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


__all__ = ["SQLSandbox", "SQLSandboxError", "SandboxLimits"]
