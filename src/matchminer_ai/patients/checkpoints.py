"""Filesystem checkpoints for resumable patient summarization.

A run saves its prepared chunks once, then each finished (patient, chunk) result
as one line of an append-only JSON-lines journal. Runs before the journal saved
one Parquet file per chunk round instead; those files are still read, so their
checkpoint directories resume.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


_PREPARED_CHUNKS_FILENAME = "prepared_chunks.parquet"
_ROUND_CHECKPOINT_PATTERN = re.compile(r"round_(\d+)\.parquet")
CHUNK_JOURNAL_FILENAME = "chunk_journal.jsonl"
#: A round checkpoint's columns; a journal line adds ``chunk_index``.
CHECKPOINT_COLUMNS = (
    "patient_id",
    "prior_summary",
    "patient_reasoning_text",
    "patient_answer_text",
    "finish_reason",
)
logger = logging.getLogger(__name__)


def prepare_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    """Create a patient checkpoint directory if needed and return its path."""
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_prepared_chunks(checkpoint_dir: str | Path) -> pd.DataFrame | None:
    """Load prepared patient note chunks, or return ``None`` if none were saved."""
    path = Path(checkpoint_dir) / _PREPARED_CHUNKS_FILENAME
    if not path.exists():
        return None
    logger.info("Loading prepared patient data from %s.", path)
    return pd.read_parquet(path)


def save_prepared_chunks(
    checkpoint_dir: str | Path,
    prepared_chunks: pd.DataFrame,
) -> None:
    """Save prepared patient note chunks for reuse by a later retry."""
    path = prepare_checkpoint_dir(checkpoint_dir) / _PREPARED_CHUNKS_FILENAME
    prepared_chunks.to_parquet(path, index=False)
    logger.info("Saved prepared patient data to %s.", path)


def load_round_checkpoints(
    checkpoint_dir: str | Path,
) -> dict[int, pd.DataFrame]:
    """Load completed round checkpoints keyed by their zero-based round index."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return {}

    # Sorting by filename preserves round order because indexes are zero-padded.
    checkpoints: dict[int, pd.DataFrame] = {}
    for path in sorted(checkpoint_path.glob("round_*.parquet")):
        match = _ROUND_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            checkpoints[int(match.group(1))] = pd.read_parquet(path)
    if checkpoints:
        logger.info(
            "Loaded %d completed patient summarization round(s) from %s.",
            len(checkpoints),
            checkpoint_path,
        )
    return checkpoints


def save_round_checkpoint(
    checkpoint_dir: str | Path,
    round_idx: int,
    round_results: pd.DataFrame,
) -> None:
    """Save all results from one completed patient summarization round."""
    checkpoint_path = prepare_checkpoint_dir(checkpoint_dir)
    path = checkpoint_path / f"round_{round_idx:04d}.parquet"
    round_results.to_parquet(path, index=False)
    logger.info(
        "Saved completed patient summarization round %d to %s.", round_idx, path
    )


def load_chunk_journal(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    """Load the journal's chunk results in the order they were written.

    A line without its newline is a write the process did not finish, and a
    line that does not parse is damaged; both are skipped, and counted in the
    log without their content.
    """
    path = Path(checkpoint_dir) / CHUNK_JOURNAL_FILENAME
    if not path.exists():
        return []
    lines = path.read_bytes().split(b"\n")
    # The final element is empty after a complete last line, else a torn write.
    torn = 1 if lines[-1] else 0
    rows: list[dict[str, Any]] = []
    damaged = 0
    for line in lines[:-1]:
        if not line:
            continue
        try:
            row = json.loads(line)
            row["chunk_index"] = int(row["chunk_index"])
            if any(column not in row for column in CHECKPOINT_COLUMNS):
                raise KeyError("missing column")
        except (ValueError, KeyError, TypeError):
            damaged += 1
            continue
        rows.append(row)
    if torn or damaged:
        logger.warning(
            "Skipped %d unfinished and %d damaged line(s) of the patient "
            "summarization journal %s.",
            torn,
            damaged,
            path,
        )
    logger.info(
        "Loaded %d patient summarization chunk result(s) from %s.", len(rows), path
    )
    return rows


def _drop_unfinished_line(path: Path) -> None:
    """Truncate a last line that an interrupted write left without its newline.

    The loader already skipped it; removing it keeps the next result from being
    appended to it.
    """
    with path.open("r+b") as handle:
        end = position = handle.seek(0, os.SEEK_END)
        keep = 0
        while position > 0:
            start = max(0, position - 65536)
            handle.seek(start)
            newline = handle.read(position - start).rfind(b"\n")
            if newline >= 0:
                keep = start + newline + 1
                break
            position = start
        if keep < end:
            handle.truncate(keep)


class ChunkJournal:
    """Append finished (patient, chunk) results to the checkpoint journal.

    Each result is one JSON line with the round checkpoint columns plus
    ``chunk_index``. Every :meth:`append` writes its lines in one call and syncs
    them to disk before returning, so an interrupted run loses only the requests
    it still had in flight. One file is kept open for the whole run, which suits
    checkpoint directories on NFS better than a file per request. Call from one
    thread at a time.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        path = prepare_checkpoint_dir(checkpoint_dir) / CHUNK_JOURNAL_FILENAME
        self.path = path
        path.touch()
        _drop_unfinished_line(path)
        self._handle = path.open("ab")

    def append(self, rows: Iterable[dict[str, Any]]) -> None:
        """Write and sync these results; each needs every journal column."""
        lines = [
            json.dumps(
                {
                    "patient_id": str(row["patient_id"]),
                    "chunk_index": int(row["chunk_index"]),
                    **{column: row[column] for column in CHECKPOINT_COLUMNS[1:]},
                },
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
            for row in rows
        ]
        if not lines:
            return
        self._handle.write(b"".join(lines))
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


__all__ = [
    "CHECKPOINT_COLUMNS",
    "CHUNK_JOURNAL_FILENAME",
    "ChunkJournal",
    "load_chunk_journal",
    "load_prepared_chunks",
    "load_round_checkpoints",
    "prepare_checkpoint_dir",
    "save_prepared_chunks",
    "save_round_checkpoint",
]
