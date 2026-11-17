"""Standalone Linux worker that runs model-written SQL; launched with -I -S, never imported.

Everything that needs the file system happens first: imports, opening the
database (a DuckDB file read-only, or Arrow tables from stdin into memory),
DuckDB's own lockdown (no file or network access, no extensions, a frozen
configuration) and warming every result conversion. Then the process limits
itself (address space, no core or file writes) and installs a seccomp allowlist
without open, socket or process creation. The syscall allowlist and DuckDB's
lockdown, not this script's checks, are the isolation boundary.

Protocol: one JSON setup line (then the Arrow bytes it announces), one JSON
reply ``{"ready": ...}``; then one JSON request per line, one JSON reply each.
"""

import ctypes
import datetime
import decimal
import errno
import json
import math
import resource
import sys
import threading
import time
import uuid

# Paths come from the parent launcher, not model SQL.
sys.path[:0] = json.loads(sys.argv[1])
import duckdb  # noqa: E402 -- trusted bootstrap before isolation
import pyarrow as pa  # noqa: E402

ALLOWED_SYSCALLS = (
    "read", "pread64", "write", "close", "fstat", "lseek", "fcntl", "mmap", "mprotect",
    "munmap", "mremap", "brk", "madvise", "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
    "sigaltstack", "clock_gettime", "gettimeofday", "time", "futex", "sched_yield", "getpid",
    "gettid", "restart_syscall", "exit", "exit_group",
)
LOCKDOWN = {
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "allow_community_extensions": False,
    "allow_unsigned_extensions": False,
}


def _virtual_size():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmSize unavailable")


def isolate(memory_mb):
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    lib.seccomp_attr_set.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # DuckDB reserves address space at start; the limit bounds growth beyond it.
    limit = _virtual_size() + memory_mb * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    policy = lib.seccomp_init(0x00050000 | errno.EPERM)  # default: deny
    if not policy:
        raise RuntimeError("seccomp initialization failed")
    try:
        if lib.seccomp_attr_set(policy, 4, 1):  # SCMP_FLTATR_CTL_TSYNC: every thread
            raise RuntimeError("seccomp thread synchronization unavailable")
        for name in ALLOWED_SYSCALLS:
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(policy, 0x7FFF0000, number, 0):
                raise RuntimeError("seccomp rule failed")
        if lib.seccomp_load(policy):
            raise RuntimeError("seccomp load failed")
    finally:
        lib.seccomp_release(policy)


def plain(value, limit):
    """A JSON value of one result cell; text longer than ``limit`` is cut."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value if limit is None or len(value) <= limit else value[:limit] + "…"
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    if isinstance(value, (decimal.Decimal, datetime.timedelta, uuid.UUID)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(value)} bytes>"
    if isinstance(value, (list, tuple)):
        return [plain(v, limit) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v, limit) for k, v in value.items()}
    return str(value)


def main():
    incoming, outgoing = sys.stdin.buffer, sys.stdout
    setup = json.loads(incoming.readline())
    config = {**LOCKDOWN, "threads": 1}
    if setup.get("database"):
        connection = duckdb.connect(
            setup["database"], read_only=True,
            config={**config, "enable_external_access": False},
        )
    else:
        connection = duckdb.connect(":memory:", config=config)
        for table in setup["tables"]:
            data = incoming.read(table["bytes"])
            if len(data) != table["bytes"]:
                raise RuntimeError("short table payload")
            arrow = pa.ipc.open_stream(data).read_all()
            connection.register("_incoming", arrow)
            connection.execute(f'CREATE TABLE "{table["name"]}" AS SELECT * FROM _incoming')
            connection.unregister("_incoming")
            del arrow, data
        for statement in setup.get("statements", []):
            connection.execute(statement)
        connection.execute("SET enable_external_access = false")
    connection.execute(f"SET memory_limit = '{int(setup['duckdb_memory_mb'])}MB'")
    connection.execute("SET max_temp_directory_size = '0B'")
    connection.execute("SET lock_configuration = true")
    info = {}
    names = {r[0] for r in connection.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    if "database_info" in names:
        info = dict(connection.execute("SELECT key, value FROM database_info").fetchall())
    # Warm every result conversion and lazy module while imports still work.
    connection.execute(
        "SELECT 1::DECIMAL(10,2), DATE '2024-01-02', TIMESTAMP '2024-01-02 03:04:05', "
        "INTERVAL 3 DAY, uuid(), [1, 2], {'a': 1}, MAP {'k': 'v'}, 'x'::BLOB, 1::HUGEINT, "
        "TIME '01:02:03', 1.5::DOUBLE, regexp_matches('ab', 'a'), 'Ab' ILIKE '%b', "
        "strftime(DATE '2024-01-02', '%Y'), date_diff('day', DATE '2024-01-01', DATE '2024-02-01')"
    ).fetchall()
    connection.execute("SUMMARIZE SELECT 1 AS a, 'x' AS b").fetchall()
    connection.extract_statements("SELECT 1")
    deadline = [None]
    wake = threading.Event()

    def watchdog():
        while True:
            wake.wait(0.05)
            when = deadline[0]
            if when is not None and time.monotonic() > when:
                deadline[0] = None
                connection.interrupt()

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        isolate(int(setup["memory_mb"]))
    except Exception:
        outgoing.write('{"ready":false}\n')
        outgoing.flush()
        return
    outgoing.write(json.dumps({"ready": True, "info": info}) + "\n")
    outgoing.flush()
    for line in incoming:
        started = time.monotonic()
        reply = {"columns": [], "types": [], "rows": [], "row_count": 0, "more_rows": False,
                 "error": None, "error_detail": None}
        try:
            request = json.loads(line)
            sql = request["sql"]
            statements = connection.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                raise PermissionError("Only one read-only SELECT statement is allowed "
                                      "(SELECT, WITH, FROM, DESCRIBE, SUMMARIZE or SHOW).")
            deadline[0] = time.monotonic() + float(request["timeout"])
            try:
                result = connection.execute(sql)
                description = result.description or []
                keep, cap = int(request["max_rows"]), int(request["count_limit"])
                cell = request.get("max_cell_chars")
                rows, count = [], 0
                while count < cap:
                    batch = result.fetchmany(min(2048, cap - count))
                    if not batch:
                        break
                    for row in batch:
                        if len(rows) < keep:
                            rows.append([plain(x, cell) for x in row])
                    count += len(batch)
                more = count >= cap and bool(result.fetchmany(1))
            finally:
                deadline[0] = None
            reply.update(columns=[d[0] for d in description],
                         types=[str(d[1]) for d in description], rows=rows, row_count=count,
                         more_rows=more)
        except BaseException as exc:  # noqa: BLE001 -- every failure is a reply
            # Bounded diagnostics go only to the configured LLM, never to logs.
            reply["error"] = type(exc).__name__
            reply["error_detail"] = str(exc)[:600]
        reply["seconds"] = round(time.monotonic() - started, 4)
        outgoing.write(json.dumps(reply, ensure_ascii=True) + "\n")
        outgoing.flush()


if __name__ == "__main__":
    main()
