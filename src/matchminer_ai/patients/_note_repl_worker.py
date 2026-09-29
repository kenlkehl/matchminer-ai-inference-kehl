"""Standalone Linux worker; launched with -I -S, never imported by the package.

All imports and syscall policy setup precede model code. The syscall allowlist,
not Python namespace restrictions, is the isolation boundary.
"""

import ast
import collections
import contextlib
import ctypes
import errno
import heapq
import io
import json
import math
import re
import resource
import sys


def isolate(memory_mb):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    policy = lib.seccomp_init(0x00050000 | errno.EPERM)  # default: deny
    if not policy:
        raise RuntimeError("seccomp initialization failed")
    try:
        for name in (
            "read",
            "write",
            "close",
            "fstat",
            "lseek",
            "fcntl",
            "mmap",
            "mprotect",
            "munmap",
            "mremap",
            "brk",
            "madvise",
            "rt_sigaction",
            "rt_sigprocmask",
            "rt_sigreturn",
            "sigaltstack",
            "clock_gettime",
            "gettimeofday",
            "time",
            "futex",
            "sched_yield",
            "getpid",
            "gettid",
            "restart_syscall",
            "exit",
            "exit_group",
        ):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and lib.seccomp_rule_add(policy, 0x7FFF0000, number, 0):
                raise RuntimeError("seccomp rule failed")
        if lib.seccomp_load(policy):
            raise RuntimeError("seccomp load failed")
    finally:
        lib.seccomp_release(policy)


class Output(io.TextIOBase):
    def __init__(self, limit):
        self.limit = limit
        self.parts = []
        self.size = 0
        self.truncated = False

    def write(self, text):
        remaining = max(0, self.limit - self.size)
        if remaining:
            self.parts.append(text[:remaining])
        self.size += min(len(text), remaining)
        self.truncated |= len(text) > remaining
        return len(text)

    def value(self):
        return "".join(self.parts)


def main():
    incoming, outgoing = sys.stdin, sys.stdout
    setup = json.loads(incoming.readline())
    history = setup.pop("history")
    output_limit = setup["max_output_chars"]
    max_scan_patterns = setup.get("max_scan_patterns", 128)
    try:
        if type(max_scan_patterns) is not int or max_scan_patterns < 1:
            raise ValueError("max_scan_patterns must be a positive integer")
        isolate(setup["memory_mb"])
    except Exception:
        outgoing.write('{"ready":false}\n')
        outgoing.flush()
        return

    source_spans, sources_truncated = [], False

    def retain(hits):
        nonlocal sources_truncated
        for hit in hits:
            span = [hit["start"], hit["end"]]
            if span not in source_spans:
                if len(source_spans) < 64:
                    source_spans.append(span)
                else:
                    sources_truncated = True

    def excerpt(start, end):
        """Read a bounded original character span (end exclusive)."""
        if type(start) is not int or type(end) is not int:
            raise ValueError("Offsets must be integers")
        if not 0 <= start < end <= len(history):
            raise ValueError("Offsets outside the history")
        stop = min(end, start + output_limit // 2)
        return {
            "start": start,
            "end": stop,
            "quote": history[start:stop],
            "truncated": stop < end,
        }

    def read(start, end):
        result = excerpt(start, end)
        retain([result])
        return result

    def search(pattern, start=0, context=250, limit=6, flags=re.I):
        """Regex search with original offsets, context, and pagination."""
        if not isinstance(pattern, str) or len(pattern) > 2000:
            raise ValueError("Supply a regex of at most 2000 characters")
        if (
            type(start) is not int
            or not 0 <= start <= len(history)
            or type(context) is not int
            or not 0 <= context <= 2000
            or type(limit) is not int
            or not 1 <= limit <= 20
        ):
            raise ValueError("Invalid search bounds")
        hits, more, next_start = [], False, None
        for match in re.compile(pattern, flags).finditer(history, start):
            if len(hits) == limit:
                more = True
                break
            left, right = (
                max(0, match.start() - context),
                min(len(history), match.end() + context),
            )
            if right > left:
                hits.append(
                    {
                        **excerpt(left, right),
                        "match_start": match.start(),
                        "match_end": match.end(),
                    }
                )
            next_start = min(len(history), max(match.end(), match.start() + 1))
        retain(hits)
        return {
            "hits": hits,
            "has_more": more,
            "next_start": next_start if more else None,
        }

    def scan(patterns, *, context=160, limit=12):
        """Scan every pattern over the full history; show bounded, spread-out hits.

        Counts include overlapping matches from different patterns. Sampling is
        by character position, not inferred event date. Keep only the first and
        last match per position bucket so memory does not grow with match count.
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        if (
            not isinstance(patterns, (list, tuple))
            or not 1 <= len(patterns) <= max_scan_patterns
            or any(not isinstance(p, str) or not 1 <= len(p) <= 2000 for p in patterns)
            or type(context) is not int
            or not 0 <= context <= 2000
            or type(limit) is not int
            or not 2 <= limit <= 20
        ):
            raise ValueError(
                f"scan expects 1-{max_scan_patterns} regex strings, "
                "context 0-2000, limit 2-20"
            )
        buckets, small = {}, []
        count, previous = 0, None
        matches = heapq.merge(
            *(re.compile(p, re.I).finditer(history) for p in patterns),
            key=lambda m: (m.start(), m.end()),
        )
        for match in matches:
            span = (match.start(), match.end())
            if span == previous:
                continue
            previous = span
            if span[0] == span[1]:
                raise ValueError("scan patterns must match nonempty text")
            count += 1
            if len(small) < limit:
                small.append(span)
            bucket = min(limit - 1, match.start() * limit // max(1, len(history)))
            first, last = buckets.get(bucket, (span, span))
            buckets[bucket] = (min(first, span), max(last, span))
        spans = (
            small
            if count <= limit
            else sorted({span for pair in buckets.values() for span in pair})
        )
        if len(spans) > limit:
            spans = [spans[i * (len(spans) - 1) // (limit - 1)] for i in range(limit)]
        # Budget quotes before rendering, preserving the earliest and latest
        # selected hits if a smaller output limit requires fewer excerpts.
        quote_limit = max(32, min(4000, (output_limit - 600) // limit - 180))
        hits = []
        for start, end in spans:
            left = max(0, start - min(context, quote_limit // 3))
            right = min(len(history), end + context, left + quote_limit)
            hits.append(
                {**excerpt(left, right), "match_start": start, "match_end": end}
            )
            hits[-1]["truncated"] |= right < end + min(context, len(history) - end)
        result = {
            "hits": hits,
            "match_count": count,
            "omitted": count > len(hits),
            "truncated": any(h["truncated"] for h in hits),
        }
        while hits and len(repr(result)) + 1 > output_limit:
            if len(hits) > 2:
                hits.pop(len(hits) // 2)
            elif max(len(h["quote"]) for h in hits) > 32:
                for hit in hits:
                    size = max(32, len(hit["quote"]) // 2)
                    hit["start"] = max(hit["start"], hit["match_start"] - size // 3)
                    hit["quote"] = history[
                        hit["start"] : min(hit["end"], hit["start"] + size)
                    ]
                    hit["end"] = hit["start"] + len(hit["quote"])
                    hit["truncated"] = True
            else:
                hits.clear()
            result.update(omitted=True, truncated=True)
        retain(hits)
        return result

    namespace = {
        "__builtins__": __builtins__,
        "history": history,
        "read": read,
        "search": search,
        "scan": scan,
        "re": re,
        "json": json,
        "math": math,
        "collections": collections,
    }
    outgoing.write('{"ready":true}\n')
    outgoing.flush()
    for line in incoming:
        output = Output(output_limit)
        source_spans, sources_truncated = [], False
        error = None
        error_detail = None
        try:
            code = json.loads(line)["code"]
            tree = ast.parse(code, mode="exec")
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    tail = tree.body.pop()
                    exec(compile(tree, "<note-cell>", "exec"), namespace)
                    result = eval(
                        compile(ast.Expression(tail.value), "<note-cell>", "eval"),
                        namespace,
                    )
                    if result is not None:
                        print(repr(result))
                else:
                    exec(compile(tree, "<note-cell>", "exec"), namespace)
        except BaseException as exc:
            # Bounded diagnostic text returns only to the configured LLM, never
            # to application logs or the public answer's failure message.
            error = type(exc).__name__
            error_detail = str(exc)[:500]
        outgoing.write(
            json.dumps(
                {
                    "output": output.value(),
                    "truncated": output.truncated,
                    "error": error,
                    "error_detail": error_detail,
                    "source_spans": source_spans,
                    "sources_truncated": sources_truncated,
                },
                ensure_ascii=True,
            )
            + "\n"
        )
        outgoing.flush()


if __name__ == "__main__":
    main()
