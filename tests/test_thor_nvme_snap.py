"""The contract of thor_nvme_snap: the warm-start snapshot of the Thor NVMe row
cache. A file: MAGIC, a little-endian uint32 header length, the UTF-8 JSON header, then the payload; written atomically;
loaded only when every check passes (versioned by the row file's manifest sha256, the "key"); a corrupt file is renamed
``<path>.bad`` and never raises. Pure Python: a fake table stands in for NVMeTable (snapshot/restore/stats)."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import struct
import sys
import time

import numpy as np
import pytest

from tensorfold.families.qwen4_exp.cuda import thor_nvme_snap as snap

KEY = "27fa5c08b71cff6741cc1027bf7fec9ece801a64427ca86f88b0593d55e563aa"


class Fake:
    def __init__(self, held: int = 7, size: int = 4096, seed: int = 0, ok: bool = True):
        self.payload = np.random.default_rng(seed).integers(0, 256, size, dtype=np.uint8)
        self.geometry = {"sets": 4, "ways": 16, "row_bytes": 90, "rows": 1000, "held": held}
        self.ok, self.restored, self.inserted = ok, [], 0

    def snapshot(self):
        return dict(self.geometry), self.payload.copy()

    def restore(self, geometry, payload):
        self.restored.append((dict(geometry), np.asarray(payload).copy()))
        return self.ok

    def stats(self):
        return {"inserted": self.inserted}


def _parts(path):
    raw = open(path, "rb").read()
    assert raw[:8] == snap.MAGIC
    (n,) = struct.unpack("<I", raw[8:12])
    return json.loads(raw[12:12 + n].decode()), raw[12 + n:]


def test_save_writes_the_format(tmp_path):
    t, p = Fake(), tmp_path / "s.snap"
    out = snap.save(t, p, KEY)
    assert out["saved"] is True and out["bytes"] == p.stat().st_size and out["held"] == 7
    header, payload = _parts(p)
    assert header["version"] == snap.VERSION == 1 and header["key"] == KEY
    for k, v in t.geometry.items():
        assert header[k] == v
    assert header["payload_bytes"] == len(payload) == t.payload.size
    assert header["sha256"] == hashlib.sha256(payload).hexdigest() and payload == t.payload.tobytes()
    assert not [f for f in os.listdir(tmp_path) if f != "s.snap"]        # no temporary file left


def test_round_trip(tmp_path):
    a, p = Fake(seed=1), tmp_path / "s.snap"
    snap.save(a, p, KEY)
    b = Fake(seed=2)
    got = snap.load(b, p, KEY)
    assert got["loaded"] is True and got["held"] == 7 and "seconds" in got
    (geometry, payload), = b.restored
    assert geometry == a.geometry and payload.tobytes() == a.payload.tobytes()


@pytest.mark.parametrize("damage,reason,renamed", [
    ("missing", "missing", False), ("magic", "magic", True), ("header", "header", True), ("truncate", "size", True),
    ("flip", "checksum", True), ("version", "version", False), ("key", "key", False)])
def test_load_refuses(tmp_path, damage, reason, renamed):
    a, p = Fake(), tmp_path / "s.snap"
    snap.save(a, p, KEY)
    raw = bytearray(p.read_bytes())
    header, payload = _parts(p)
    if damage == "missing":
        p.unlink()
    elif damage == "magic":
        raw[0] ^= 1
        p.write_bytes(bytes(raw))
    elif damage == "header":
        raw[14] = 0xFF                                  # inside the JSON text
        p.write_bytes(bytes(raw))
    elif damage == "truncate":
        p.write_bytes(bytes(raw[:-10]))
    elif damage == "flip":
        raw[-5] ^= 0x10
        p.write_bytes(bytes(raw))
    elif damage in ("version", "key"):
        header = dict(header, **({"version": 2} if damage == "version" else {"key": "0" * 64}))
        text = json.dumps(header).encode()
        p.write_bytes(snap.MAGIC + struct.pack("<I", len(text)) + text + payload)
    b = Fake()
    got = snap.load(b, p, KEY)
    assert got["loaded"] is False and got["reason"] == reason and not b.restored
    assert os.path.exists(str(p) + ".bad") is renamed
    if renamed:
        assert not p.exists()


def test_load_refuses_too_large_and_geometry(tmp_path):
    a, p = Fake(size=1 << 16), tmp_path / "s.snap"
    snap.save(a, p, KEY)
    got = snap.load(Fake(), p, KEY, max_bytes=1 << 12)
    assert got == {"loaded": False, "reason": "too large"} or (got["loaded"] is False and got["reason"] == "too large")
    assert p.exists()                                   # a valid file is never renamed for being large
    b = Fake(ok=False)                                  # the table refuses the geometry
    got = snap.load(b, p, KEY)
    assert got["loaded"] is False and got["reason"] == "geometry" and len(b.restored) == 1 and p.exists()


def test_save_replaces_atomically(tmp_path):
    p = tmp_path / "s.snap"
    snap.save(Fake(seed=1), p, KEY)
    snap.save(Fake(seed=2), p, KEY)
    b = Fake()
    assert snap.load(b, p, KEY)["loaded"] is True
    assert b.restored[0][1].tobytes() == Fake(seed=2).payload.tobytes()


def test_saver_periodic_and_final(tmp_path):
    t, p = Fake(), tmp_path / "s.snap"
    s = snap.Saver(t, p, KEY, period_s=0.05, min_new_rows=10).start()
    time.sleep(0.3)
    assert not p.exists()                               # nothing new inserted: no save
    t.inserted = 25
    deadline = time.time() + 3
    while not p.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert p.exists()
    first = p.stat().st_mtime_ns
    time.sleep(0.3)
    assert p.stat().st_mtime_ns == first                # inserted unchanged since the save: no new save
    t.payload[:] = 9
    s.stop()                                            # final save, whatever changed
    assert _parts(p)[1] == t.payload.tobytes() and s.saves >= 2
    s.stop()                                            # idempotent


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(pid, SIGTERM) ends the process on Windows")
def test_saver_sigterm_hook_saves_then_chains(tmp_path):
    t, p = Fake(), tmp_path / "s.snap"
    seen = []
    old = signal.signal(signal.SIGTERM, lambda *a: seen.append("prev"))
    try:
        s = snap.Saver(t, p, KEY, period_s=3600).start()
        s.install_exit_hooks()
        os.kill(os.getpid(), signal.SIGTERM)
        deadline = time.time() + 3
        while not seen and time.time() < deadline:
            time.sleep(0.02)
        assert seen == ["prev"] and p.exists()          # saved, then the previous handler ran
    finally:
        signal.signal(signal.SIGTERM, old)
        s.stop(final=False)


# Engineer additions (Phase 2 wiring): the row check, save_now, attach (the engine's switch).

def test_load_refuses_rows_that_differ(tmp_path):
    a, p = Fake(seed=1), tmp_path / "s.snap"
    snap.save(a, p, KEY)
    seen = []
    got = snap.load(Fake(), p, KEY, check=lambda g, pl: seen.append((g, pl.size)) or False)
    assert got == {"loaded": False, "reason": "rows"} and seen[0][1] == a.payload.size
    assert not p.exists() and (tmp_path / "s.snap.bad").exists()      # never loaded again
    snap.save(a, p, KEY)
    b = Fake()
    assert snap.load(b, p, KEY, check=lambda g, pl: True)["loaded"] and len(b.restored) == 1


def test_save_now_and_last(tmp_path):
    t, p = Fake(), tmp_path / "s.snap"
    s = snap.Saver(t, p, KEY, period_s=3600, min_new_rows=10**9)
    got = s.save_now()                                  # whatever the cache holds, now
    assert got["saved"] and s.last == got and s.saves == 1 and _parts(p)[1] == t.payload.tobytes()
    s.stop(final=False)


class Table(Fake):
    manifest = {"sha256": KEY}


def test_attach(tmp_path, monkeypatch):
    p = tmp_path / "d" / "s.snap"
    monkeypatch.setattr(snap.signal, "signal", lambda *a: None)       # no process-wide SIGTERM hook from a test
    monkeypatch.delenv(snap.ENV, raising=False)
    assert snap.attach([Table()]) == ([], "")                         # off unless asked
    monkeypatch.setenv(snap.ENV, str(p))
    monkeypatch.setenv(snap.PERIOD_ENV, "0")
    savers, note = snap.attach([Fake()])                              # no row file: off, said so
    assert savers == [] and "needs a repacked row file" in note
    a = Table(seed=3)
    savers, note = snap.attach([a])
    assert "not restored (missing)" in note and savers[0]._thread is None and p.parent.is_dir()
    savers[0].stop()                                                  # the exit save
    b = Table(seed=4)
    b.check_dump = lambda g, pl: pl.tobytes() == a.payload.tobytes()
    savers, note = snap.attach([b])
    assert "restored 7 rows" in note and b.restored[0][1].tobytes() == a.payload.tobytes()
    savers[0].stop(final=False)
