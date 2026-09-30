"""Bounded transport for a persistent, syscall-isolated Python process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import sysconfig
import threading
import time


class NoteREPLError(RuntimeError):
    """A worker failed without exposing note text or executed source code."""


class NoteREPL:
    def __init__(self, history, limits, *, notes=None, patient_summary=None):
        if sys.platform != "linux":
            raise NoteREPLError("The isolated note REPL requires Linux and libseccomp.")
        self.limits = limits
        self.history_length = len(history)
        # Trusted dependency paths only. Keep -I -S: do not execute site/.pth
        # startup hooks, inherit PYTHONPATH, or expose the parent's environment.
        import pandas

        package_paths = list(
            dict.fromkeys(
                [
                    sysconfig.get_path("purelib"),
                    sysconfig.get_path("platlib"),
                    str(Path(pandas.__file__).resolve().parent.parent),
                ]
            )
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(Path(__file__).with_name("_note_repl_worker.py")),
                json.dumps(package_paths),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={
                name: "1"
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "NUMEXPR_MAX_THREADS",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            cwd="/",
            close_fds=True,
            bufsize=0,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            ready = self._exchange(
                {
                    "history": history,
                    "notes": notes,
                    "patient_summary": patient_summary,
                    "memory_mb": limits.worker_memory_mb,
                    "max_output_chars": limits.max_output_chars,
                    "max_scan_patterns": getattr(limits, "max_scan_patterns", 128),
                },
                timeout=limits.worker_startup_timeout_seconds,
            )
            if ready != {"ready": True}:
                raise NoteREPLError(
                    "Worker isolation unavailable; no code was executed."
                )
        except BaseException:
            self.close()
            raise

    def _exchange(self, value, *, timeout):
        payload = (json.dumps(value, ensure_ascii=True) + "\n").encode()
        failed = threading.Event()

        def send():
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(self.process.stdin.fileno(), view)
                    view = view[written:]
            except (OSError, ValueError):
                failed.set()

        # A hostile or dead worker cannot block the caller by refusing to read.
        writer = threading.Thread(target=send, daemon=True)
        writer.start()
        deadline = time.monotonic() + timeout
        chunks, size = [], 0
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.selector.select(remaining):
                    raise NoteREPLError("Worker exceeded its wall-time limit.")
                chunk = os.read(self.process.stdout.fileno(), 8192)
                size += len(chunk)
                if not chunk or failed.is_set():
                    raise NoteREPLError("Worker exited or its transport failed.")
                if size > self.limits.max_output_chars * 12 + 4096:
                    raise NoteREPLError("Worker exceeded its output limit.")
                chunks.append(chunk)
                if b"\n" in chunk:
                    try:
                        response = json.loads(b"".join(chunks))
                    except (ValueError, UnicodeError):
                        raise NoteREPLError(
                            "Worker returned invalid transport data."
                        ) from None
                    if not isinstance(response, dict):
                        raise NoteREPLError("Worker returned invalid transport data.")
                    return response
        except BaseException:
            self.close()
            raise
        finally:
            # Killing a failed worker closes its read end and releases this writer.
            writer.join(timeout=1)

    def execute(self, code):
        if not isinstance(code, str) or len(code) > self.limits.max_code_chars:
            raise NoteREPLError("Cell exceeds the configured code limit.")
        result = self._exchange(
            {"code": code}, timeout=self.limits.cell_timeout_seconds
        )
        if (
            set(result)
            != {
                "output",
                "truncated",
                "error",
                "error_detail",
                "source_spans",
                "sources_truncated",
            }
            or not isinstance(result["output"], str)
            or len(result["output"]) > self.limits.max_output_chars
            or type(result["truncated"]) is not bool
            or type(result["sources_truncated"]) is not bool
            or not isinstance(result["source_spans"], list)
            or len(result["source_spans"]) > 64
            or any(
                not isinstance(span, list)
                or len(span) != 2
                or any(type(n) is not int for n in span)
                or not 0 <= span[0] < span[1] <= self.history_length
                for span in result["source_spans"]
            )
            or (result["error"] is not None and not isinstance(result["error"], str))
            or (
                result["error_detail"] is not None
                and (
                    not isinstance(result["error_detail"], str)
                    or len(result["error_detail"]) > 500
                )
            )
        ):
            raise NoteREPLError("Worker returned an invalid cell result.")
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
