"""Stage receipts: what a stage directory holds, so the stage can be run anywhere and its output adopted.

Every stage script (``trialspace.prepare``, ``trialspace.mine``, ``trialspace.label``, ``trialspace.project``,
``trialspace.finetune``, ``training.decision_model`` prepare|train) writes ``stage_receipt.json`` into its output
directory last. Its ``identity`` names the stage's settings and the receipts of the directories it read, and
contains no paths, hosts or devices; outputs are listed by relative name and byte size (data files are never
hashed or read back); ``provenance`` records where and with which code it ran. A directory produced on another
machine and copied back is therefore complete here when its identity matches what this machine computes.

``--check`` exit codes: 0 complete; 10 not complete (absent or resumable); 20 the identity or outputs differ,
which needs a fresh directory.
"""
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from matchminer_ai.labeling.files import digest, write_json

FORMAT = 'mmai-stage-receipt-v1'
NAME = 'stage_receipt.json'
COMPLETE, INCOMPLETE, MISMATCH = 0, 10, 20


def add_check_arg(parser):
    parser.add_argument('--check', action='store_true',
                        help='Only report whether the output is complete for these settings: exit 0 complete, '
                             '10 not yet, 20 settings or outputs differ (use a fresh directory)')


def read_receipt(directory):
    path = Path(directory) / NAME
    return json.loads(path.read_text()) if path.exists() else None


def receipt_id(directory):
    """The id of a complete directory's receipt (an upstream reference)."""
    receipt = read_receipt(directory)
    if receipt is None:
        raise ValueError(f'{directory} has no stage receipt; run its stage first')
    return receipt['receipt_id']


def check(directory, kind, identity):
    """0, 10 or 20 (see the module docstring), and the reason."""
    receipt = read_receipt(directory)
    if receipt is None:
        return INCOMPLETE, 'no stage receipt'
    if receipt.get('format') != FORMAT or receipt.get('kind') != kind:
        return MISMATCH, f'not a {kind} stage receipt'
    if receipt['identity_sha256'] != digest(identity):
        changed = sorted(k for k in set(identity) | set(receipt['identity'])
                         if identity.get(k) != receipt['identity'].get(k))
        return MISMATCH, 'settings or inputs differ: ' + ', '.join(changed)
    for name, size in receipt['outputs'].items():
        path = Path(directory) / name
        if not path.exists() or path.stat().st_size != size:
            return MISMATCH, 'output missing or changed: ' + name
    return COMPLETE, 'complete'


def exit_check(directory, kind, identity):
    code, reason = check(directory, kind, identity)
    print(f'[check] {kind} {directory}: {reason}', flush=True)
    sys.exit(code)


def require_new_or_same(directory, kind, identity):
    """Before running: refuse a directory completed (or begun) under other settings."""
    code, reason = check(directory, kind, identity)
    if code == MISMATCH:
        raise ValueError(f'{directory}: {reason}; use a fresh directory')
    return code == COMPLETE


def _git(path):
    try:
        root = Path(path).resolve()
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True,
                                    text=True, timeout=10).stdout.strip()) if commit else None
        return dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, dirty=None)


def write_receipt(directory, kind, identity, outputs, *, content=None, upstream=None):
    """Record a completed stage; ``outputs`` are names relative to ``directory``."""
    import matchminer_ai
    directory = Path(directory)
    sizes = {name: (directory / name).stat().st_size for name in sorted(outputs)}
    body = dict(format=FORMAT, kind=kind, identity=identity, identity_sha256=digest(identity),
                upstream=dict(sorted((upstream or {}).items())), outputs=sizes, content=content or {})
    body['receipt_id'] = digest({k: body[k] for k in ('kind', 'identity_sha256', 'upstream', 'outputs', 'content')})
    body['provenance'] = dict(
        host=socket.gethostname(), pid=os.getpid(), argv=sys.argv, directory=str(directory.resolve()),
        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        repository=_git(Path(matchminer_ai.__file__).parents[2]),
        package_version=getattr(matchminer_ai, '__version__', None))
    write_json(directory / NAME, body)
    return body


__all__ = ['COMPLETE', 'INCOMPLETE', 'MISMATCH', 'NAME', 'add_check_arg', 'check', 'exit_check', 'read_receipt',
           'receipt_id', 'require_new_or_same', 'write_receipt']
