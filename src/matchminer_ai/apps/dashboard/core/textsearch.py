"""Regular-expression search over every corpus text (the patient and cohort lists' search box).

A query is one or more terms separated by spaces; every term must match somewhere in an entry's text (AND).
A term with spaces goes in double quotes. Terms are case-insensitive RE2 regular expressions (pyarrow's
``match_substring_regex``; no lookaround or backreferences), in multi-line mode: ``^`` and ``$`` match at line
starts and ends, and ``.`` stays within a line. Plain words therefore match as substrings, as before.

The texts are held as pyarrow chunks matched on several threads at once (about 0.05-0.1 s over the 87,890
synthetic patients' summaries, ~212 million characters, where Python's ``re`` takes 2-5 s).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import re
import threading
from typing import Iterable

import numpy as np

TERM = re.compile(r'"([^"]*)"?|(\S+)')  # "a quoted term" (closed or still being typed) or a bare term
THREADS = max(1, min(16, os.cpu_count() or 1))
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def parse_terms(query: str) -> list[str]:
    terms = []
    for found in TERM.finditer(query or ""):
        term = found.group(1) if found.group(1) is not None else found.group(2)
        if term.strip():
            terms.append(term)
    return terms


def _executor() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(THREADS, thread_name_prefix="textsearch")
        return _pool


class TextIndex:
    """Entry ``i``'s searchable text, matched by :meth:`match` as a boolean mask in entry order."""

    def __init__(self, texts: Iterable[str], parts: int = THREADS):
        import pyarrow as pa

        texts = list(texts)
        self.size = len(texts)
        step = max(1, -(-self.size // max(1, parts)))
        self.chunks = [pa.array(texts[i:i + step], type=pa.large_string()) for i in range(0, self.size, step)]

    def match(self, query: str) -> np.ndarray:
        mask = np.ones(self.size, dtype=bool)
        for term in parse_terms(query):
            mask &= self._term(term)
        return mask

    def _term(self, term: str) -> np.ndarray:
        import pyarrow as pa
        import pyarrow.compute as pc

        pattern = "(?m)" + term

        def run(chunk):
            return pc.match_substring_regex(chunk, pattern, ignore_case=True).to_numpy(zero_copy_only=False)

        try:
            parts = list(_executor().map(run, self.chunks)) if len(self.chunks) > 1 else [run(c) for c in self.chunks]
        except pa.ArrowInvalid as exc:
            reason = str(exc).split("Invalid regular expression:", 1)[-1].strip().replace(pattern, term)
            raise ValueError(f"invalid pattern {term!r}: {reason}") from None
        return np.concatenate(parts).astype(bool) if parts else np.zeros(0, dtype=bool)
