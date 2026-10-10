"""Serve TensorFold with ROWS-2 diagnostics: the same server, plus a JSON-lines record of its memory gate,
kept prompts, prompt passes, n-gram gathers and allocator calls. The wrappers only count and time: no computed value,
buffer or schedule changes.

    TF_DIAG_OUT=/out/diag.jsonl [TF_DIAG_EVERY=0.5] [TF_DIAG_BALLAST_GIB=3] python tools/thor/serve_diag.py serve ...

The arguments are tensorfold.cli's. One JSON object a line; "k" names the record, "t" is time.time():
- "start": pid, argv, ballast GiB, TENSORFOLD_PREFILL_ROWS; "md": the concurrent decoder's gate at start (room,
  reserve), prefill_rows, slots, one stream's bytes at the full window;
- "pass": one prompt pass (MultiDecoder._pass), i its index: t0 (wall), dt (s), its pieces' rows/starts/prompt
  lengths, during it: stage_s (host staging: n-gram gathers and the wait for the previous step's copies), the
  gathers (g_calls, g_rows table rows, g_s), comp_s (compute's host time), the process's majflt/minflt; after it torch
  allocated/reserved bytes; "pass_gpu": i and ms, CUDA-event time around compute (read later, no added sync);
- "grow": a stream's caches grown or refused (MultiDecoder._grow): rows asked, capacity before and after, ok, dt,
  the gate's held bytes after; "evict": a kept prompt end freed (ok, kept count after); "shrink": a slot back to its
  first rows (capacity before); "keep": a prompt end kept (its length, kept count after);
- "admit": a request admitted (prompt length); "finish": streams that ended;
- "st": every TF_DIAG_EVERY s: the cumulative counters (gathers, stages, passes, rounds and their seconds, grows,
  evictions, shrinks, empty_cache calls and seconds), the process's faults, the gate (room, reserve, held, waits,
  ends, the driver pool a wide-piece gate keeps), kept prompts (count, rows), live/filling streams, free slots, torch
  allocated/reserved/peak reserved bytes and its device allocs, frees and retries.
TF_DIAG_BALLAST_GIB allocates (and touches) that much GPU memory before the model loads: the memory a wider pass's
buffers take, without the wider pass.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time

OUT = os.environ.get("TF_DIAG_OUT", "")
EVERY = float(os.environ.get("TF_DIAG_EVERY", "") or 0.5)
BALLAST_GIB = float(os.environ.get("TF_DIAG_BALLAST_GIB", "") or 0)

C = {"g_calls": 0, "g_rows": 0, "g_s": 0.0, "stages": 0, "stage_s": 0.0, "comp_s": 0.0, "passes": 0, "pass_s": 0.0,
     "rounds": 0, "round_s": 0.0, "grows": 0, "grow_s": 0.0, "refused": 0, "evicts": 0, "shrinks": 0, "keeps": 0,
     "ec_calls": 0, "ec_s": 0.0}
_lock = threading.Lock()
_file = None
_md = None                     # the concurrent decoder, once made
_cur: dict = {}                # the pass in progress: its pieces and compute's events
_pending: list = []            # (pass index, start event, end event) not yet read


def emit(rec: dict) -> None:
    rec.setdefault("t", round(time.time(), 4))
    line = json.dumps(rec, separators=(",", ":"), default=str)
    with _lock:
        _file.write(line + "\n")


def faults() -> tuple[int, int]:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_majflt, r.ru_minflt


def resolve() -> None:
    """Emit the GPU time of every pass whose end event has completed (query only: never waits)."""

    with _lock:
        ready = [p for p in _pending if p[2].query()]
        for p in ready:
            _pending.remove(p)
    for i, e0, e1 in ready:
        emit({"k": "pass_gpu", "i": i, "ms": round(e0.elapsed_time(e1), 3)})


def install() -> None:
    import numpy as np
    import torch
    from tensorfold.families.qwen4_exp import host_table
    from tensorfold.families.qwen4_exp.cuda import multi

    MD = multi.MultiDecoder

    def timed_table(cls) -> None:
        orig = cls.__dict__.get("gather")
        if orig is None:
            return

        def gather(self, ids):
            t = time.perf_counter()
            try:
                return orig(self, ids)
            finally:
                C["g_s"] += time.perf_counter() - t
                C["g_calls"] += 1
                C["g_rows"] += int(np.size(ids))
        cls.gather = gather

    for name in ("BF16Table", "NVFP4Table"):
        if hasattr(host_table, name):
            timed_table(getattr(host_table, name))

    stage0, compute0 = multi.stage, multi.compute

    def stage(*a, **k):
        t = time.perf_counter()
        try:
            return stage0(*a, **k)
        finally:
            C["stages"] += 1
            C["stage_s"] += time.perf_counter() - t

    def compute(*a, **k):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t = time.perf_counter()
        e0.record()
        try:
            return compute0(*a, **k)
        finally:
            e1.record()
            C["comp_s"] += time.perf_counter() - t
            _cur["ev"] = (e0, e1)
    multi.stage, multi.compute = stage, compute

    init0 = MD.__init__

    def init(self, *a, **k):
        global _md
        init0(self, *a, **k)
        _md = self
        g = self.memory_gate
        emit({"k": "md", "room": g.room, "reserve": g.reserve, "prefill_rows": self.prefill_rows,
              "slots": len(self.free), "window_bytes": self.window_bytes, "slot_bytes": self.slot_bytes})

    note0 = MD._note_passed

    def note_passed(self, pieces):
        _cur["pieces"] = [(int(a), int(n), len(s.prompt)) for s, a, n in pieces]
        return note0(self, pieces)

    pass0 = MD._pass

    def _pass(self):
        i = C["passes"]
        _cur.pop("ev", None)
        g0, s0, c0, f0 = (C["g_calls"], C["g_rows"], C["g_s"]), C["stage_s"], C["comp_s"], faults()
        t0w, t0 = time.time(), time.perf_counter()
        try:
            return pass0(self)
        finally:
            dt = time.perf_counter() - t0
            f1 = faults()
            C["passes"] += 1
            C["pass_s"] += dt
            pieces = _cur.pop("pieces", [])
            emit({"k": "pass", "i": i, "t0": round(t0w, 4), "dt": round(dt, 4),
                  "rows": [n for _, n, _ in pieces], "starts": [a for a, _, _ in pieces],
                  "plens": [p for _, _, p in pieces], "stage_s": round(C["stage_s"] - s0, 4),
                  "g_calls": C["g_calls"] - g0[0], "g_rows": C["g_rows"] - g0[1], "g_s": round(C["g_s"] - g0[2], 4),
                  "comp_s": round(C["comp_s"] - c0, 4), "majflt": f1[0] - f0[0], "minflt": f1[1] - f0[1],
                  "alloc": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()})
            ev = _cur.pop("ev", None)
            if ev is not None:
                with _lock:
                    _pending.append((i, ev[0], ev[1]))
            resolve()

    round0 = MD.round

    def round_(self):
        t = time.perf_counter()
        try:
            return round0(self)
        finally:
            C["rounds"] += 1
            C["round_s"] += time.perf_counter() - t

    grow0 = MD._grow

    def _grow(self, st, rows, *, alone=False, protect=None):
        cap0, t, ok = st.capacity, time.perf_counter(), None
        try:
            ok = grow0(self, st, rows, alone=alone, protect=protect)
            return ok
        finally:
            dt = time.perf_counter() - t
            if st.capacity != cap0 or ok is not True:
                C["grows" if st.capacity != cap0 else "refused"] += 1
                C["grow_s"] += dt
                pool = getattr(self.memory_gate, "pool", None)
                emit({"k": "grow", "rows": int(rows), "cap0": int(cap0), "cap1": int(st.capacity), "ok": ok,
                      "alone": bool(alone), "dt": round(dt, 4), "held": int(self.memory_gate.held),
                      "pool": int(pool()) if pool is not None else None})

    evict0 = MD._evict_kept

    def _evict_kept(self, keep, *, protect=None):
        ok = evict0(self, keep, protect=protect)
        C["evicts"] += bool(ok)
        emit({"k": "evict", "ok": bool(ok), "kept": len(self.kept), "held": int(self.memory_gate.held)})
        return ok

    shrink0 = MD._shrink

    def _shrink(self, st, *, force=False):
        cap0 = st.capacity
        shrink0(self, st, force=force)
        C["shrinks"] += 1
        emit({"k": "shrink", "cap0": int(cap0), "cap1": int(st.capacity), "held": int(self.memory_gate.held)})

    remember0 = MD._remember

    def _remember(self, ids, st, snap, tail):
        out = remember0(self, ids, st, snap, tail)
        C["keeps"] += 1
        emit({"k": "keep", "len": len(ids), "kept": len(self.kept), "held": int(self.memory_gate.held)})
        return out

    admit0 = MD.admit

    def admit(self, s):
        emit({"k": "admit", "plen": len(s.prompt)})
        return admit0(self, s)

    finish0 = MD.finish

    def finish(self, done):
        emit({"k": "finish", "n": len(done)})
        return finish0(self, done)

    MD.__init__, MD._note_passed, MD._pass, MD.round = init, note_passed, _pass, round_
    MD._grow, MD._evict_kept, MD._shrink, MD._remember, MD.admit, MD.finish = (
        _grow, _evict_kept, _shrink, _remember, admit, finish)

    empty0 = torch.cuda.empty_cache

    def empty_cache():
        t = time.perf_counter()
        try:
            return empty0()
        finally:
            C["ec_calls"] += 1
            C["ec_s"] += time.perf_counter() - t
    torch.cuda.empty_cache = empty_cache


def status() -> None:
    import torch

    while True:
        time.sleep(EVERY)
        try:
            rec = {"k": "st", **{k: round(v, 4) if isinstance(v, float) else v for k, v in C.items()}}
            rec["majflt"], rec["minflt"] = faults()
            md = _md
            if md is not None:
                g, kept = md.memory_gate, list(md.kept)
                rec.update(room=int(g.room), reserve=int(g.reserve), held=int(g.held), waits=int(g.waits),
                           ends=int(g.ends), kept=len(kept), kept_rows=sum(len(k[0]) for k in kept),
                           live=len(md.streams), filling=len(md.filling), free_slots=len(md.free))
                pool = getattr(g, "pool", None)
                rec["pool"] = int(pool()) if pool is not None else None
            if torch.cuda.is_initialized():
                s = torch.cuda.memory_stats()
                rec.update(alloc=s.get("allocated_bytes.all.current", 0),
                           reserved=s.get("reserved_bytes.all.current", 0),
                           reserved_peak=s.get("reserved_bytes.all.peak", 0),
                           dev_allocs=s.get("num_device_alloc", 0), dev_frees=s.get("num_device_free", 0),
                           retries=s.get("num_alloc_retries", 0))
                resolve()
            emit(rec)
        except Exception as exc:                 # noqa: BLE001  (a diagnostic never stops the server)
            emit({"k": "st_error", "error": repr(exc)[:300]})


def main() -> int:
    global _file
    if not OUT:
        raise SystemExit("set TF_DIAG_OUT to the JSON-lines file to write")
    _file = open(OUT, "a", buffering=1)
    emit({"k": "start", "pid": os.getpid(), "argv": sys.argv[1:], "ballast_gib": BALLAST_GIB,
          "rows_env": os.environ.get("TENSORFOLD_PREFILL_ROWS")})
    ballast = None
    if BALLAST_GIB > 0:
        import torch

        ballast = torch.empty(int(BALLAST_GIB * 2**30), dtype=torch.uint8, device="cuda")
        ballast.fill_(1)
        torch.cuda.synchronize()
        emit({"k": "ballast", "bytes": ballast.numel()})
    install()
    threading.Thread(target=status, name="diag-status", daemon=True).start()
    from tensorfold import cli

    return cli.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
