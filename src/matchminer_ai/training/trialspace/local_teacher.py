"""Release one explicitly registered local teacher before student training."""
import json
import os
from pathlib import Path
import platform
import signal

from .data import write_json


def _pidfd_syscall(number, *args):
    # Portable Python builds may omit pidfd wrappers even on a capable kernel.
    # These syscall numbers are the Linux x86-64 ABI, verified on the production host.
    import ctypes
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This platform needs native Python pidfd support')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    import ctypes
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def send_pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    import ctypes
    return _pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def process_identity(pid):
    """Linux process identity, including start time to detect PID reuse."""
    try:
        folder = Path('/proc') / str(pid)
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        command = (folder / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(pid=pid, start_ticks=int(fields[19]), command=command)
    except FileNotFoundError:
        return None


def stop_registered_teacher(registration, receipt, *, training_gpus):
    """SIGTERM only the registered process; GPU availability is checked by the runner."""
    target = json.loads(Path(registration).read_text())
    if target.get('purpose') != 'stop_this_teacher_after_labeling' or str(target.get('gpu')) not in training_gpus:
        raise ValueError('Teacher shutdown must be explicitly registered for a selected training GPU')
    expected = target['process']
    pid = expected['pid']
    # A pidfd prevents signaling a different process if the PID disappears/recycles.
    try:
        fd = open_pidfd(pid)
    except ProcessLookupError:
        write_json(receipt, dict(state='already_stopped', registration=target))
        return
    try:
        current = process_identity(pid)
        if current is None:
            state = 'already_stopped'
        elif current != expected:
            raise ValueError('Registered teacher process identity changed; refusing to signal it')
        else:
            command = current['command']
            if 'serve' not in command or '--port' not in command or command[command.index('--port') + 1] != str(target['port']):
                raise ValueError('Registered process does not match the teacher server command')
            try:
                send_pidfd_signal(fd, signal.SIGTERM)
                state = 'termination_requested'
            except ProcessLookupError:
                state = 'already_stopped'
        write_json(receipt, dict(state=state, registration=target))
    finally:
        os.close(fd)
