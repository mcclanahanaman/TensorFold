"""The Thor NVMe n-gram row cache's warm-start snapshot: one file written atomically, reloaded only when every check
passes, a damaged file renamed ``<path>.bad`` (FR-OPS-17).

HD-2026-10-07-04 Phase 2: the row cache's bytes (``NVMeTable.snapshot``: the geometry and the 90-byte rows) are saved
next to the row file so the next start can skip the cold NVMe warm-up. The header carries the row file's manifest
sha256 as the ``key``: a snapshot taken against another checkpoint must never be restored. ``save`` writes to a
temporary file in the same directory, fsyncs it and ``os.replace``s it onto the path, so a crash leaves either the old
file or the new one, never a torn one. ``load`` never raises on a file problem: a wrong magic, an unparsable header, a
short payload or a checksum mismatch renames the file ``<path>.bad`` so it is never loaded again, and every other
refusal keeps the file. ``Saver`` saves periodically from a daemon thread and once more at exit.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import signal
import struct
import sys
import threading
import time

import numpy as np

__all__ = ['MAGIC', 'VERSION', 'Saver', 'load', 'save']

MAGIC = b'TFNGSNP1'
VERSION = 1

_GEOMETRY = ('sets', 'ways', 'row_bytes', 'rows', 'held')


def _fsync_dir(target: str) -> None:
    try:
        fd = os.open(os.path.dirname(target) or '.', os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _quarantine(target: str) -> None:
    try:
        os.replace(target, target + '.bad')
    except OSError:
        pass


def _report(exc: BaseException) -> None:
    print(f'[tensorfold] n-gram snapshot: {" ".join(str(exc).split())}', file=sys.stderr)


def save(table, path, key) -> dict:
    """Write the table's snapshot to ``path`` atomically and return what was written."""
    started = time.perf_counter()
    geometry, payload = table.snapshot()
    payload = np.asarray(payload, dtype=np.uint8)
    body = payload.tobytes()
    header = {'version': VERSION, 'key': key, **geometry, 'payload_bytes': payload.size,
              'sha256': hashlib.sha256(body).hexdigest()}
    text = json.dumps(header).encode('utf-8')
    blob = MAGIC + struct.pack('<I', len(text)) + text + body
    target = os.fspath(path)
    tmp = f'{target}.tmp.{os.getpid()}'
    with open(tmp, 'wb') as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)
    _fsync_dir(target)
    return {'saved': True, 'bytes': len(blob), 'held': geometry['held'],
            'seconds': time.perf_counter() - started}


def load(table, path, key, *, max_bytes: int = 4 << 30) -> dict:
    """Restore ``path`` into ``table``; every file problem is a reason, never an exception."""
    started = time.perf_counter()
    target = os.fspath(path)
    try:
        size = os.path.getsize(target)
    except OSError:
        return {'loaded': False, 'reason': 'missing'}
    if size > max_bytes:
        return {'loaded': False, 'reason': 'too large'}
    try:
        with open(target, 'rb') as f:
            raw = f.read()
    except OSError:
        return {'loaded': False, 'reason': 'missing'}
    if raw[:8] != MAGIC:
        _quarantine(target)
        return {'loaded': False, 'reason': 'magic'}
    if len(raw) < 12:
        _quarantine(target)
        return {'loaded': False, 'reason': 'header'}
    (n,) = struct.unpack('<I', raw[8:12])
    if 12 + n > len(raw):
        _quarantine(target)
        return {'loaded': False, 'reason': 'header'}
    try:
        header = json.loads(raw[12:12 + n].decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        _quarantine(target)
        return {'loaded': False, 'reason': 'header'}
    if not isinstance(header, dict):
        _quarantine(target)
        return {'loaded': False, 'reason': 'header'}
    if header.get('version') != VERSION:
        return {'loaded': False, 'reason': 'version'}
    if header.get('key') != key:
        return {'loaded': False, 'reason': 'key'}
    body = raw[12 + n:]
    expected = header.get('payload_bytes')
    if not isinstance(expected, int) or len(body) != expected:
        _quarantine(target)
        return {'loaded': False, 'reason': 'size'}
    if hashlib.sha256(body).hexdigest() != header.get('sha256'):
        _quarantine(target)
        return {'loaded': False, 'reason': 'checksum'}
    geometry = {name: header.get(name) for name in _GEOMETRY}
    payload = np.frombuffer(body, dtype=np.uint8)
    if not table.restore(geometry, payload):
        return {'loaded': False, 'reason': 'geometry'}
    return {'loaded': True, 'held': header.get('held'), 'seconds': time.perf_counter() - started}


class Saver:
    """Save the table every ``period_s`` from a daemon thread, and once more on stop or exit."""

    def __init__(self, table, path, key, period_s: float = 1800.0, min_new_rows: int = 0):
        self.table, self.path, self.key = table, path, key
        self.period_s, self.min_new_rows = float(period_s), int(min_new_rows)
        self.saves = 0
        self._last_inserted = 0
        self._lock = threading.Lock()
        self._stopped = False
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> 'Saver':
        self._thread = threading.Thread(target=self._run, name='tensorfold-ngram-snapshot', daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop_event.wait(self.period_s):
            self._tick()

    def _tick(self, force: bool = False) -> None:
        try:
            inserted = int(self.table.stats()['inserted'])
        except Exception as exc:
            _report(exc)
            return
        if not force and inserted - self._last_inserted < max(1, self.min_new_rows):
            return
        try:
            save(self.table, self.path, self.key)
        except Exception as exc:
            _report(exc)
            return
        self.saves += 1
        self._last_inserted = inserted

    def stop(self, final: bool = True) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        if final:
            self._tick(force=True)

    def install_exit_hooks(self) -> None:
        atexit.register(self.stop)
        if threading.current_thread() is not threading.main_thread():
            return
        previous = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum, frame):
            self.stop()
            if callable(previous):
                previous(signum, frame)
            elif previous is signal.SIG_IGN:
                return
            else:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, _on_sigterm)
