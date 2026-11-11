"""Compact, durable completed-pair checkpoints, separate from reasoning traces.

The labeler's exclusive lock owns this append-only file. A pair is committed only
after all required question responses are saved. Resume reads this sequential
file, never the completed pairs' individual response files.
"""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import time

from .files import digest, write_json
from .scoring import validate_score
from matchminer_ai.llm.async_processing import BackgroundWork


def pending_candidates(pairs, scores):
    """Filter in a background worker before request dispatch; reject orphan scores."""
    unmatched, pending = set(scores), []
    for pair in pairs:
        key = digest([pair['patient_id'], pair['cohort_id']])
        if key in scores:
            unmatched.discard(key)
        else:
            pending.append(pair)
    if unmatched:
        raise ValueError('Completed-score journal contains pairs absent from the candidate set')
    return pending


class ScoreJournal:
    def __init__(self, output, signature, *, allow_import=False, shared=False):
        self.output = Path(output)
        self.root = self.output / 'score_journal'
        self.path = self.root / 'scores.jsonl'
        manifest = self.root / 'manifest.json'
        expected = dict(format='match-score-journal-v1', teacher_manifest=signature)
        if (self.root / 'IMPORTING').exists():
            raise ValueError('Score journal import is incomplete; finish the explicit import first')
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ValueError('Completed-score journal identity changed')
        else:
            responses = self.output / 'responses'
            if not allow_import and responses.exists() and next(responses.iterdir(), None) is not None:
                raise ValueError('Responses exist without a score journal; label into a new directory')
            if self.path.exists():
                raise ValueError('Score journal is missing its identity manifest')
            self.root.mkdir(parents=True, exist_ok=True)
            write_json(manifest, expected)
        self.signature = signature
        self.shared = shared  # several label client processes append (labeling.clients)
        self.queue = asyncio.Queue(maxsize=32768)
        self.background = BackgroundWork(1, 1, 'score-journal')
        self.error = None

    def load(self, patients, cohorts):
        """Validate compact records; recover only an uncommitted final fragment."""
        scores = {}
        if not self.path.exists():
            return scores
        with self.path.open('r+b') as stream:
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                if not raw.endswith(b'\n'):
                    # Append can be interrupted. Earlier complete records remain
                    # authoritative; never discard a malformed complete record.
                    (self.root / f'torn-tail-{time.time_ns()}.bin').write_bytes(raw)
                    stream.truncate(offset)
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                row = json.loads(raw)
                validate_score(row, patients, cohorts, self.signature['contract'],
                               self.signature['aggregation_contract'])
                key = digest([row['patient_id'], row['cohort_id']])
                if key in scores and scores[key] != row:
                    raise ValueError('Conflicting completed scores in journal')
                scores[key] = row
        return scores

    def append(self, rows):
        raw = ''.join(json.dumps(r, separators=(',', ':'), ensure_ascii=False) + '\n'
                      for r in rows).encode()
        fd = os.open(self.path, (os.O_RDWR if self.shared else os.O_WRONLY) | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if self.shared:  # one append at a time, never after another client's torn record
                fcntl.flock(fd, fcntl.LOCK_EX)
                self._cut_torn_tail(fd)
            remaining = memoryview(raw)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError('Incomplete score-journal write')
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

    def _cut_torn_tail(self, fd):
        size = end = os.fstat(fd).st_size
        if not size or os.pread(fd, 1, size - 1) == b'\n':
            return
        while end:
            start = max(0, end - 65536)
            index = os.pread(fd, end - start, start).rfind(b'\n')
            end = start + index + 1 if index >= 0 else start
            if index >= 0:
                break
        (self.root / f'torn-tail-{time.time_ns()}.bin').write_bytes(os.pread(fd, size - end, end))
        os.ftruncate(fd, end)
        os.fsync(fd)

    async def _writer(self):
        batch = []
        try:
            while True:
                batch = [await self.queue.get()]
                # Coalesce independent completed pairs into a single durable
                # write without reopening any saved reasoning files.
                deadline = asyncio.get_running_loop().time() + .25
                while len(batch) < 1024:
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(),
                            max(.001, deadline - asyncio.get_running_loop().time())))
                    except TimeoutError:
                        break
                    if asyncio.get_running_loop().time() >= deadline:
                        break
                await self.background.run(self.append, [row for row, _ in batch])
                for _, future in batch:
                    if not future.done():
                        future.set_result(None)
                batch = []
        except BaseException as error:
            self.error = error
            while not self.queue.empty():
                batch.append(self.queue.get_nowait())
            for _, future in batch:
                if not future.done():
                    if isinstance(error, asyncio.CancelledError):
                        future.cancel()
                    else:
                        future.set_exception(error)
            if not isinstance(error, asyncio.CancelledError):
                raise

    async def commit(self, row):
        if self.error is not None:
            raise RuntimeError('Completed-score writer stopped') from self.error
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((row, future))
        # The writer might have failed while a full queue held this producer.
        if self.error is not None:
            raise RuntimeError('Completed-score writer stopped') from self.error
        await future

    async def __aenter__(self):
        await self.background.__aenter__()
        self.writer = asyncio.create_task(self._writer())
        return self

    async def __aexit__(self, *exc):
        self.writer.cancel()
        result = await asyncio.gather(self.writer, return_exceptions=True)
        await self.background.__aexit__(*exc)
        if exc[0] is None and isinstance(result[0], Exception):
            raise result[0]


def import_archive(archive, output):
    """One-time migration from a validated export made against this exact corpus."""
    archive, output = Path(archive), Path(output)
    exported = json.loads((archive / 'manifest.json').read_text())
    signature = json.loads((output / 'labeling_manifest.json').read_text())
    if exported['format'] != 'completed-gated-scores-v1' or exported['teacher_manifest'] != signature:
        raise ValueError('Compact export identity differs from active labeling')
    if Path(exported['scores_file']).name != exported['scores_file']:
        raise ValueError('Invalid compact export filename')
    if (output / 'score_journal').exists():
        raise ValueError('Score journal already exists; refusing to replace it')
    journal = ScoreJournal(output, signature, allow_import=True)
    marker = journal.root / 'IMPORTING'
    marker.touch()
    temporary = journal.path.with_suffix('.jsonl.tmp')
    count = 0
    with (archive / exported['scores_file']).open('rb') as source, temporary.open('wb') as dest:
        for raw in source:
            if not raw.endswith(b'\n'):
                raise ValueError('Truncated compact export')
            dest.write(raw)
            count += 1
        dest.flush()
        os.fsync(dest.fileno())
    size = temporary.stat().st_size
    if size != exported['scores_bytes'] or count != exported['counts']['completed_pairs']:
        raise ValueError('Compact export size or pair count mismatch')
    temporary.replace(journal.path)
    write_json(journal.root / 'import.json', dict(archive=str(archive.resolve()),
               completed_pairs=count, bytes=size))
    marker.unlink()
    return count
