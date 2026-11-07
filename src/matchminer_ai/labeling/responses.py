"""Avoid repeated directory mutations on shared checkpoint filesystems."""
import json
from pathlib import Path

from .files import write_json


class ResponseStore:
    """Caller holds the label-directory lock; only this writer creates responses.

    Inspect legacy filenames once and create the 256 shards once. Resume reads
    only requested unfinished questions, without scanning completed shard files.
    A missing shard file costs one open instead of repeated stat/mkdir operations.
    """

    def __init__(self, root, *, compact=False, read_only=False):
        self.root = Path(root)
        from .compact_responses import CompactResponses, FILENAME
        self.database = (CompactResponses(self.root, read_only=read_only)
                         if compact or (self.root/FILENAME).exists() else None)
        if self.database is not None:
            self.legacy = {p.stem for p in self.root.glob('*.json')}
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self.legacy = {p.stem for p in self.root.iterdir() if p.suffix == '.json'}
        for value in range(256):
            (self.root / f'{value:02x}').mkdir(exist_ok=True)

    def path(self, key):
        return (self.root if key in self.legacy else self.root / key[:2]) / (key + '.json')

    def read(self, key):
        path = self.path(key)
        if self.database is not None:
            raw = self.database.get_bytes(key)
            if raw is not None:
                return path, json.loads(raw)
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return path, None
        return path, json.loads(raw)

    def write(self, path, row):
        if self.database is not None:
            self.database.put_many([(row['key'], json.dumps(row, ensure_ascii=False).encode())])
            return
        write_json(path, row, create_parents=False)

    def close(self):
        if self.database is not None:
            self.database.close()
