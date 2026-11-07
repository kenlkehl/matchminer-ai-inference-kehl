"""Bounded parallel export of durable question checkpoints, including reasoning."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice
import json
import multiprocessing
import os
from pathlib import Path
import time


_READERS = None
_ROOT = None
_STORE = None


def _initialize(root, threads):
    global _READERS, _ROOT, _STORE
    _ROOT = Path(root)
    from .responses import ResponseStore
    _STORE = ResponseStore(root, read_only=True)
    _READERS = ThreadPoolExecutor(max_workers=threads)


def _read(key):
    _, row = _STORE.read(key)
    if row is None:
        raise FileNotFoundError('Missing response checkpoint: '+key)
    if row.get('key') != key:
        raise ValueError('Response checkpoint key differs from requested export key: ' + key)
    # Re-encode the entire object; never drop prompt, clinical context or reasoning.
    return (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')


def _batch(keys):
    return b''.join(_READERS.map(_read, keys)), len(keys)


def _batches(keys, size):
    iterator = iter(keys)
    while batch := list(islice(iterator, size)):
        yield batch


def export_labels(root, destination, keys, *, total, workers=8, reader_threads=8,
                  batch_size=128, progress=None):
    """Keep input order and publish atomically only after every trace is readable.

    At most two batches per process are outstanding. Neither all response text
    nor millions of futures are accumulated in the parent process.
    """
    if min(workers, reader_threads, batch_size) < 1 or total < 0:
        raise ValueError('Invalid response export limits')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    started = last_report = time.monotonic()
    count = byte_count = 0

    def report():
        elapsed = time.monotonic() - started
        state = dict(exported_questions=count, total_questions=total, exported_bytes=byte_count,
                     export_elapsed_seconds=elapsed, export_questions_per_second=count / max(elapsed, .001))
        if progress:
            progress(state)
        return state

    report()
    jobs = _batches(keys, batch_size)
    with temporary.open('wb') as stream:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize, initargs=(str(root), reader_threads)) as pool:
            while window := list(islice(jobs, workers * 2)):
                for raw, size in pool.map(_batch, window):
                    stream.write(raw)
                    count += size
                    byte_count += len(raw)
                    if time.monotonic() - last_report >= 10:
                        report()
                        last_report = time.monotonic()
        if count != total:
            raise ValueError(f'Expected {total} response checkpoints, exported {count}')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return report()
