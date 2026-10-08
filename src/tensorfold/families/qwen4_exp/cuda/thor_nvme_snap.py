"""The Thor NVMe n-gram row cache's warm-start snapshot: one file written atomically, reloaded only when every check
passes, a damaged file renamed ``<path>.bad`` (FR-OPS-17).

HD-2026-10-07-04 Phase 2: the row cache's bytes (``NVMeTable.snapshot``: the geometry and the 90-byte rows) are saved
next to the row file so the next start can skip the cold NVMe warm-up. The header carries the row file's manifest
sha256 as the ``key``: a snapshot taken against another checkpoint must never be restored. ``save`` writes to a
temporary file in the same directory, fsyncs it and ``os.replace``s it onto the path, so a crash leaves either the old
file or the new one, never a torn one. ``load`` never raises on a file problem: a wrong magic, an unparsable header, a
short payload or a checksum mismatch renames the file ``<path>.bad`` so it is never loaded again, and every other
refusal keeps the file. ``Saver`` saves periodically from a daemon thread and once more at exit.

Memory: the serving process's memory is the KV pool's, so a save or a load holds ONE copy of the payload (0.53 GiB at
the default cache: no byte-string copies) and drops the file's pages from the page cache after it.

Switches (read by ``attach`` at load): ``TENSORFOLD_THOR_NGRAM_SNAPSHOT=<path>`` keeps the row cache of a repacked
table (``TENSORFOLD_THOR_NGRAM_FILE``) across restarts: restored at start (after a seeded sample of its rows is checked
against the row file), saved at exit and every ``TENSORFOLD_THOR_NGRAM_SNAPSHOT_S`` seconds (1800; 0: only at exit)
once the cache took ``MIN_NEW_ROWS`` new rows. Unset: no snapshot.
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
from pathlib import Path

import numpy as np

__all__ = ['ENV', 'MAGIC', 'MIN_NEW_ROWS', 'PERIOD_ENV', 'VERSION', 'Saver', 'attach', 'load', 'save']

MAGIC = b'TFNGSNP1'
VERSION = 1
ENV = 'TENSORFOLD_THOR_NGRAM_SNAPSHOT'
PERIOD_ENV = 'TENSORFOLD_THOR_NGRAM_SNAPSHOT_S'
MIN_NEW_ROWS = 100_000

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


def _drop_cache(fd: int) -> None:
    try:                                   # the file's pages out of the page cache: that memory is the KV pool's
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def _write_all(f, data) -> None:
    view = memoryview(data).cast('B')
    while view:
        view = view[f.write(view):]


def _read_exact(f, n: int) -> bytes:
    parts, got = [], 0
    while got < n:
        part = f.read(n - got)
        if not part:
            break
        parts.append(part)
        got += len(part)
    return b''.join(parts)


def _readinto_all(f, view) -> int:
    got = 0
    while got < len(view):
        n = f.readinto(view[got:])
        if not n:
            break
        got += n
    return got


def save(table, path, key) -> dict:
    """Write the table's snapshot to ``path`` atomically and return what was written."""
    started = time.perf_counter()
    geometry, payload = table.snapshot()
    payload = np.ascontiguousarray(payload, dtype=np.uint8).reshape(-1)
    header = {'version': VERSION, 'key': key, **geometry, 'payload_bytes': int(payload.size),
              'sha256': hashlib.sha256(payload).hexdigest()}
    text = json.dumps(header).encode('utf-8')
    target = os.fspath(path)
    tmp = f'{target}.tmp.{os.getpid()}'
    try:
        with open(tmp, 'wb', buffering=0) as f:
            _write_all(f, MAGIC + struct.pack('<I', len(text)) + text)
            _write_all(f, payload)
            os.fsync(f.fileno())
            size = f.tell()
            _drop_cache(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(target)
    return {'saved': True, 'bytes': size, 'held': geometry['held'],
            'seconds': time.perf_counter() - started}


def _read(target: str, size: int, key) -> tuple:
    """(reason or None, header, payload) of the file at ``target`` (``size`` bytes); OSError on a read failure."""
    with open(target, 'rb', buffering=0) as f:
        head = _read_exact(f, 12)
        if head[:8] != MAGIC:
            return 'magic', None, None
        if len(head) < 12:
            return 'header', None, None
        (n,) = struct.unpack('<I', head[8:12])
        if 12 + n > size:
            return 'header', None, None
        try:
            header = json.loads(_read_exact(f, n).decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            return 'header', None, None
        if not isinstance(header, dict):
            return 'header', None, None
        if header.get('version') != VERSION:
            return 'version', header, None
        if header.get('key') != key:
            return 'key', header, None
        expected = header.get('payload_bytes')
        if not isinstance(expected, int) or isinstance(expected, bool) or size - 12 - n != expected:
            return 'size', header, None
        payload = np.empty(expected, dtype=np.uint8)
        got = _readinto_all(f, memoryview(payload))
        _drop_cache(f.fileno())
    if got != expected:
        return 'size', header, None
    if hashlib.sha256(payload).hexdigest() != header.get('sha256'):
        return 'checksum', header, None
    return None, header, payload


_RENAMED = ('magic', 'header', 'size', 'checksum', 'rows')


def load(table, path, key, *, max_bytes: int = 4 << 30, check=None) -> dict:
    """Restore ``path`` into ``table``; every file problem is a reason, never an exception. ``check(geometry,
    payload)`` False (the rows differ from the row file) refuses it as ``'rows'`` (renamed ``<path>.bad``)."""
    started = time.perf_counter()
    target = os.fspath(path)
    try:
        size = os.path.getsize(target)
    except OSError:
        return {'loaded': False, 'reason': 'missing'}
    if size > max_bytes:
        return {'loaded': False, 'reason': 'too large'}
    try:
        reason, header, payload = _read(target, size, key)
    except OSError as exc:
        return {'loaded': False, 'reason': f'unreadable ({exc.strerror or exc})'}
    geometry = {name: header.get(name) for name in _GEOMETRY} if header is not None else {}
    if reason is None and check is not None and not check(geometry, payload):
        reason = 'rows'
    if reason in _RENAMED:
        _quarantine(target)
    if reason is not None:
        return {'loaded': False, 'reason': reason}
    if not table.restore(geometry, payload):
        return {'loaded': False, 'reason': 'geometry'}
    return {'loaded': True, 'held': header.get('held'), 'seconds': time.perf_counter() - started}


class Saver:
    """Save the table every ``period_s`` from a daemon thread, and once more on stop or exit."""

    def __init__(self, table, path, key, period_s: float = 1800.0, min_new_rows: int = 0):
        self.table, self.path, self.key = table, path, key
        self.period_s, self.min_new_rows = float(period_s), int(min_new_rows)
        self.saves = 0
        self.last: dict | None = None              # the last save's result, or {'error': ...}
        self._last_inserted = 0
        self._lock = threading.Lock()
        self._saving = threading.Lock()            # one save at a time (the thread, save_now, the final save)
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

    def _tick(self, force: bool = False) -> dict | None:
        try:
            inserted = int(self.table.stats()['inserted'])
        except Exception as exc:
            _report(exc)
            return None
        if not force and inserted - self._last_inserted < max(1, self.min_new_rows):
            return None
        if not self._saving.acquire(timeout=120.0):
            _report(TimeoutError('another save did not finish in 120 s'))
            return None
        try:
            got = save(self.table, self.path, self.key)
        except Exception as exc:
            _report(exc)
            self.last = {'saved': False, 'error': ' '.join(str(exc).split())}
            return self.last
        finally:
            self._saving.release()
        self.saves += 1
        self._last_inserted = inserted
        self.last = got
        return got

    def save_now(self) -> dict | None:
        """Save whatever the cache holds now (dev and benchmark routes); the result, or None."""
        return self._tick(force=True)

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

    def install_exit_hooks(self) -> 'Saver':
        atexit.register(self.stop)
        if threading.current_thread() is not threading.main_thread():
            return self
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
        return self


def attach(tables, path=None, period_s: float | None = None) -> tuple[list[Saver], str]:
    """``ENV`` set (or ``path``): each repacked NVMe table's row cache restored from its snapshot (a sample of its rows
    checked against the row file first), then kept by a ``Saver`` (every ``PERIOD_ENV`` seconds, default 1800, 0 only at
    exit; and at exit). The savers and one line for the log ('' when off)."""
    where = os.fspath(path) if path is not None else os.environ.get(ENV, '').strip()
    if not where:
        return [], ''
    tables = list(tables)
    if period_s is None:
        period_s = float(os.environ.get(PERIOD_ENV, '').strip() or 1800.0)
    savers, notes = [], []
    for i, table in enumerate(tables):
        target = Path(where if len(tables) == 1 else f'{where}.{i}')
        sha = str((getattr(table, 'manifest', None) or {}).get('sha256') or '')
        if not sha:
            notes.append(f'n-gram cache snapshot off: {ENV} needs a repacked row file (TENSORFOLD_THOR_NGRAM_FILE)')
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            notes.append(f'n-gram cache snapshot off: {target.parent} ({exc.strerror or exc})')
            continue
        got = load(table, target, sha, check=getattr(table, 'check_dump', None))
        saver = Saver(table, target, sha, period_s=max(period_s, 0.0), min_new_rows=MIN_NEW_ROWS)
        if period_s > 0:
            saver.start()
        savers.append(saver.install_exit_hooks())
        state = (f"restored {int(got.get('held') or 0):,} rows in {got['seconds']:.2f}s" if got['loaded']
                 else f"not restored ({got['reason']})")
        every = f'every {period_s:g}s with >= {MIN_NEW_ROWS:,} new rows and ' if period_s > 0 else ''
        notes.append(f'n-gram cache snapshot {target} (key {sha[:12]}): {state}; saved {every}at exit')
    return savers, '; '.join(notes)
