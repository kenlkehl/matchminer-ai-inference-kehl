"""JSON, JSONL and copy helpers of the teacher labeler, byte-compatible with the label directories
training wrote before the labeler moved here (2026-11-06).

Data files are identified by ``file_stamp`` (byte size and whole-second mtime) and copies are checked
by size; whole data files are never hashed or read back.
"""
import json
import os
from pathlib import Path
import shutil

from matchminer_ai._storage import digest


def write_json(path, value, *, compact=False, create_parents=True):
    path = Path(path)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    options = dict(separators=(',', ':')) if compact else dict(indent=2)
    temporary.write_text(json.dumps(value, ensure_ascii=False, **options) + '\n')
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def file_stamp(path):
    """Cheap change detector: byte size and whole-second mtime, without reading content.

    Whole seconds tolerate the differing timestamp precision of NFS clients on
    other hosts. A copy gets a new mtime, so compare a copy with its source by
    size alone.
    """
    stat = Path(path).stat()
    return f'{stat.st_size}:{int(stat.st_mtime)}'


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def require_free_space(path, required=0, reserve_gb=16):
    if reserve_gb < 0:
        raise ValueError('Retrieval disk reserve must be nonnegative')
    free = shutil.disk_usage(path).free
    if free < required + int(reserve_gb * 1024 ** 3):
        raise OSError(f'Insufficient local retrieval space at {path}: {free} bytes free; '
                      f'need {required} bytes plus {reserve_gb} GiB reserve')


def require_local_disk(path):
    """Do not silently put the hot working set back on a network mount."""
    path = Path(path).resolve()
    matches = []
    for line in Path('/proc/mounts').read_text().splitlines():
        fields = line.split()
        mount = Path(fields[1].replace('\\040', ' '))
        if path.is_relative_to(mount):
            matches.append((len(mount.parts), fields[2]))
    filesystem = max(matches)[1] if matches else ''
    if filesystem in {'nfs', 'nfs4', 'cifs', 'smb3', '9p', 'ceph', 'fuse.sshfs', 'fuse.glusterfs'}:
        raise ValueError(f'Retrieval scratch must be local disk, not {filesystem}: {path}')
    return filesystem


def copy_verified(source, target, expected=None):
    """Copy sequentially and fsync; check byte sizes before atomic rename.

    ``expected`` is the source's recorded file_stamp. The copy keeps the
    source's modification time, so it has the same stamp as its source and
    stamp-based completion records stay valid after publication.
    """
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    before = source.stat()
    if expected is not None and file_stamp(source) != expected:
        raise ValueError(f'Retrieval copy source changed: {source} -> {target}')
    with source.open('rb') as incoming, temporary.open('wb') as outgoing:
        while block := incoming.read(8 * 1024 ** 2):
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    after = source.stat()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or temporary.stat().st_size != before.st_size):
        raise ValueError(f'Retrieval copy verification failed: {source} -> {target}')
    os.utime(temporary, ns=(before.st_atime_ns, before.st_mtime_ns))
    temporary.replace(target)
    return dict(bytes=before.st_size, stamp=file_stamp(target))


__all__ = ['atomic_json', 'copy_verified', 'digest', 'file_stamp', 'read_jsonl', 'require_free_space',
           'require_local_disk', 'write_json', 'write_jsonl']
