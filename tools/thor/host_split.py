"""A decode round's host time by call, split by whether a forward runs on the GPU (DEC-2).

``decode_phases.py --host-split`` installs this in place of its phase timer. ``install`` wraps the round's host calls
(``multi.py`` and what it calls: staging, the round tables, graphs, sampling, accept/commit, MTP drafting); the
forwards (eager ``compute``/``mtp_compute`` and graph replays) also record a CUDA event before their first launch and
after their last. Per round, ``begin`` (after a device sync: the GPU is idle) records an event and reads the host clock
together, and ``end`` (after the round's sync) maps the forwards' event windows onto the host clock and charges each
stretch of host time to the innermost wrapped call running then (its exclusive time), split into the part with no
forward on the GPU and the part under one. The no-forward total is the round's wall time less the forwards' GPU spans.
Kernels, launches and bits are untouched; the wrappers cost about a microsecond a call.
"""

from __future__ import annotations

import resource
import time

import torch


class Split:
    def __init__(self) -> None:
        self.marks: list = []          # (host seconds, +1 enter / -1 exit, name)
        self.windows: list = []        # (event before the first launch, event after the last)
        self.t0 = 0.0
        self.e0 = None
        self.faults: dict = {}         # name: [thread minor, thread major, process minor, process major]

    def begin(self) -> None:
        self.marks.clear()
        self.windows.clear()
        self.faults.clear()
        self.e0 = torch.cuda.Event(enable_timing=True)
        self.e0.record()
        self.t0 = time.perf_counter()

    def wrap(self, name: str, fn, forward: bool = False, faults: bool = False):
        marks, windows, clock, counts = self.marks, self.windows, time.perf_counter, self.faults

        def wrapped(*args, **kwargs):
            if faults:
                f0 = _faults()
            marks.append((clock(), 1, name))
            if forward:
                e0 = torch.cuda.Event(enable_timing=True)
                e0.record()
            try:
                return fn(*args, **kwargs)
            finally:
                if forward:
                    e1 = torch.cuda.Event(enable_timing=True)
                    e1.record()
                    windows.append((e0, e1))
                marks.append((clock(), -1, name))
                if faults:
                    got = counts.setdefault(name, [0, 0, 0, 0])
                    for i, (a, b) in enumerate(zip(f0, _faults())):
                        got[i] += b - a
        wrapped.__wrapped__ = fn
        return wrapped

    def end(self) -> dict:
        """After the round's device sync: {"calls": {name: [no-forward ms, under-forward ms, calls]}, "wall_ms",
        "forward_ms" (the forwards' merged GPU spans), "idle_ms" (no forward: wall less forward_ms)}."""

        t1 = time.perf_counter()
        spans = sorted((self.t0 + self.e0.elapsed_time(a) / 1e3, self.t0 + self.e0.elapsed_time(b) / 1e3)
                       for a, b in self.windows)
        merged: list = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])

        def under(a: float, b: float) -> float:
            return sum(max(0.0, min(b, y) - max(a, x)) for x, y in merged)

        calls: dict = {}
        stack, last = ["(outside)"], self.t0
        for t, step, name in self.marks + [(t1, 0, None)]:
            got = calls.setdefault(stack[-1], [0.0, 0.0, 0])
            o = under(last, t)
            got[0] += (t - last - o) * 1e3
            got[1] += o * 1e3
            last = t
            if step > 0:
                stack.append(name)
                calls.setdefault(name, [0.0, 0.0, 0])[2] += 1
            elif step < 0 and len(stack) > 1:
                stack.pop()
        wall = (t1 - self.t0) * 1e3
        fwd = sum(min(b, t1) - max(a, self.t0) for a, b in merged if b > self.t0 and a < t1) * 1e3
        return {"calls": calls, "wall_ms": wall, "forward_ms": fwd, "idle_ms": wall - fwd,
                "faults": {k: list(v) for k, v in self.faults.items()}}


def _faults() -> tuple:
    """(this thread's minor, major, the process's minor, major) page faults so far (getrusage; Linux)."""

    t, p = resource.getrusage(resource.RUSAGE_THREAD), resource.getrusage(resource.RUSAGE_SELF)
    return t.ru_minflt, t.ru_majflt, p.ru_minflt, p.ru_majflt


def _patch(split: Split, owner, attr: str, name: str, forward: bool = False, faults: bool = False) -> None:
    fn = getattr(owner, attr, None)
    if fn is None or getattr(fn, "__wrapped__", None) is not None:
        return
    setattr(owner, attr, split.wrap(name, fn, forward, faults))


def install(split: Split) -> None:
    from tensorfold.cuda import streams
    from tensorfold.cuda.kernels import gdn as shared
    from tensorfold.families.qwen4_exp import host_table
    from tensorfold.families.qwen4_exp.cuda import attn_multi, forward, gdn_multi, multi, ngram, state, thor_graphs

    md = multi.MultiDecoder
    for attr in ("round", "finish", "_make_room", "_draft_all", "_mtp", "_picks", "_k", "_pick_dev", "_timed",
                 "_ends", "_pieces", "_pass_rows"):
        _patch(split, md, attr, attr.lstrip("_"))
    for attr in ("stage", "mtp_stage", "sample_streams", "accept", "capture", "commit", "choose_rows"):
        _patch(split, multi, attr, attr)
    for attr in ("compute", "mtp_compute", "compute_mixed"):
        _patch(split, multi, attr, attr, forward=True)
    _patch(split, torch.cuda.CUDAGraph, "replay", "graph.replay", forward=True)
    _patch(split, attn_multi.Step, "__init__", "Step")
    _patch(split, gdn_multi.Tables, "__init__", "Tables")
    for attr in ("keep", "normalize"):
        _patch(split, gdn_multi, attr, "gdn." + attr)
    for attr in ("to_device", "plan_host"):
        _patch(split, shared, attr, attr)
    _patch(split, thor_graphs, "settle", "settle")
    for attr in ("run", "key", "_load"):
        _patch(split, thor_graphs.DecodeGraphs, attr, "graphs." + attr.lstrip("_"))
    for attr in ("stage_ple_rows", "stage_ahead"):
        _patch(split, forward, attr, attr)
    _patch(split, ngram.NGram, "ids", "ngram.ids")
    for cls in vars(host_table).values():
        if isinstance(cls, type) and "gather" in vars(cls):
            _patch(split, cls, "gather", "table.gather", faults=True)
    _patch(split, torch.cuda.Event, "synchronize", "event.sync")
    _patch(split, torch.cuda.Stream, "synchronize", "stream.sync")
    for attr in ("take", "counted"):
        _patch(split, streams.Stream, attr, "stream." + attr)
    _patch(split, state.State, "set_mtp_len", "set_mtp_len")


def summarize(rounds: list, top: int = 14) -> dict:
    """Means over rounds: wall, forward and no-forward ms, and the calls with the most no-forward time."""

    n = len(rounds) or 1
    tot: dict = {}
    for r in rounds:
        for name, (idle, busy, calls) in r["calls"].items():
            got = tot.setdefault(name, [0.0, 0.0, 0])
            got[0] += idle
            got[1] += busy
            got[2] += calls
    ranked = sorted(tot.items(), key=lambda kv: -kv[1][0])
    faults: dict = {}
    for r in rounds:
        for name, got in r.get("faults", {}).items():
            acc = faults.setdefault(name, [0] * len(got))
            for i, v in enumerate(got):
                acc[i] += v
    return {"rounds": len(rounds), "wall_ms": sum(r["wall_ms"] for r in rounds) / n,
            "forward_ms": sum(r["forward_ms"] for r in rounds) / n, "idle_ms": sum(r["idle_ms"] for r in rounds) / n,
            "top": [[name, round(idle / n, 3), round(busy / n, 3), round(calls / n, 2)]
                    for name, (idle, busy, calls) in ranked[:top]],
            "faults": {name: [round(v / n, 2) for v in acc] for name, acc in faults.items()}}
