"""Flash Next's concurrent decode rounds as CUDA graphs on Jetson Thor (sm_110) (THOR-K3).

The concurrent server (``--parallel``) runs its rounds eagerly: a verification forward over every stream's window,
then an MTP step a draft depth, each one launch a kernel, about 2,300 launches a round for one stream. Here each such
forward replays a CUDA graph recorded for its exact launch structure (a verification forward that shares its experts
with a prompt pass stays eager):

- the key holds everything a launch takes from the host: each stream's window rows and the pointers the forward
  bakes in (its PLE tail and position; a sparse stream's pooled keys too), whether its rows are sparse, the DeltaNet
  scratch parity, whether last round's kept rows fold in, the image flag and the context bucket; the round's device
  tables (rows' streams and positions, cache and state pointers, DeltaNet plans) are copied into the graph's own
  before each replay;
- a graphed forward bounds its attention launches by a power-of-two context bucket, as the single-stream graphs do,
  and keeps the DeltaNet fold table at its widest. Every kernel reads its rows' real lengths on the device (chunks past
  a row's keys write nothing and the merge never reads them; scores past a row's complete blocks are never written;
  the fold reads only its counted rows), so the bits are the eager path's;
- a forward folds last round's kept DeltaNet rows into the states in place, so it must never run twice: a key's
  first forward runs eagerly (and compiles every launch shape its graph holds), a later one records the graph, which
  runs nothing, and replays it once. At most ``cap`` graphs are kept; past that, new keys stay eager.

``TENSORFOLD_THOR_DECODE_GRAPH=0`` keeps every round eager, as before; ``TENSORFOLD_THOR_DECODE_GRAPH_STREAMS=N``
graphs only forwards of at most N streams (``_CAP``: the most graphs kept). Events go to stderr as
``[thor-graph] {json}`` lines (``tools/thor/graph_stats.py`` digests them).
"""

from __future__ import annotations

import copy
import ctypes
import dataclasses
import functools
import gc
import json
import os
import sys
import time
from collections import OrderedDict
from typing import Callable, Sequence

import torch

ENV = "TENSORFOLD_THOR_DECODE_GRAPH"
CAP_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_CAP"
STREAMS_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_STREAMS"
SPIN_ENV = "TENSORFOLD_THOR_DECODE_SPIN"
CAPABILITY = (11, 0)
SCHED_MASK, SCHED_SPIN = 0x07, 0x01  # CU_CTX_SCHED_MASK, CU_CTX_SCHED_SPIN
CAP = 256                   # graphs kept, verification and MTP together
MIN_SEEN = 2                # a key's forwards before its graph is recorded (the first runs eagerly)
STREAMS = 1                 # the most streams a graphed forward holds (more stay eager: their keys rarely repeat)
SEEN_MAX = 1 << 14          # keys counted before the counts start over
STATS_EVERY = 512           # forwards between stats events
MIB = float(1 << 20)


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110 and not switched off: decided once."""

    return not _off() and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY


@functools.lru_cache(maxsize=None)
def spinning(device: int = 0) -> str:
    """How the host waits for the GPU in decode on sm_110 with ``TENSORFOLD_THOR_DECODE_SPIN=1`` (THOR-DEC-1).

    A synchronize that sleeps lets the governor clock the waiting core down to its floor (972 MHz on Thor), and every
    host phase of a round then runs about 2x slower. ``"ctx"``: the CUDA context's waits spin (its scheduling flag,
    set here once); ``"poll"``: the driver refused the flag, so ``settle`` polls an event before the decode
    read-backs; ``""``: off. Host only: the GPU's work and its bits are unchanged."""

    on = os.environ.get(SPIN_ENV, "0").strip().lower() in ("1", "on", "true", "yes")
    if not (on and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY):
        return ""
    torch.cuda.init()
    mode = "poll"
    try:
        cu = ctypes.CDLL("libcuda.so.1")
        flags = ctypes.c_uint()
        if cu.cuCtxGetFlags(ctypes.byref(flags)) == 0 and \
                cu.cuCtxSetFlags(ctypes.c_uint((flags.value & ~SCHED_MASK) | SCHED_SPIN)) == 0 and \
                cu.cuCtxGetFlags(ctypes.byref(flags)) == 0 and flags.value & SCHED_MASK == SCHED_SPIN:
            mode = "ctx"
    except (OSError, AttributeError):
        pass
    print("[thor-graph] " + json.dumps({"event": "spin", "mode": mode}, separators=(",", ":")), file=sys.stderr,
          flush=True)
    return mode


def settle() -> None:
    """Before a decode read-back: busy-wait until the current stream's queued work is done, when the context's own
    waits could not be made to spin (``spinning() == "poll"``)."""

    if spinning() != "poll":
        return
    ev = torch.cuda.Event()
    ev.record()
    while not ev.query():
        pass


def bucket(end: int) -> int:
    """A graphed forward's attention bound: the next power of two at or past ``end`` keys, at least 8,192."""

    return max(8192, 1 << (max(int(end), 1) - 1).bit_length())


def _base(t: torch.Tensor) -> torch.Tensor:
    return t if t._base is None else t._base


class _Twin:
    """A round object (``attn_multi.Step``, ``gdn_multi.Tables``) whose device tables live in a graph's own storage:
    a shallow copy whose tensors view clones of the round's buffers. ``load`` copies a later round's buffers in."""

    def __init__(self, obj) -> None:
        self.obj = copy.copy(obj)
        self.paths: list[tuple[str, str | None, torch.Tensor]] = []     # (attribute, dataclass field, static buffer)
        clones: dict[int, torch.Tensor] = {}
        for name, value in vars(obj).items():
            if isinstance(value, torch.Tensor):
                setattr(self.obj, name, self._view(name, None, value, clones))
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                moved = {f.name: self._view(name, f.name, getattr(value, f.name), clones)
                         for f in dataclasses.fields(value) if isinstance(getattr(value, f.name), torch.Tensor)}
                setattr(self.obj, name, dataclasses.replace(value, **moved))

    def _view(self, name: str, field: str | None, t: torch.Tensor, clones: dict) -> torch.Tensor:
        base = _base(t)
        if not base.is_contiguous() or base.storage_offset():
            raise RuntimeError(f"a decode graph cannot hold the round's {name} table (not a whole contiguous buffer)")
        if id(base) not in clones:
            clones[id(base)] = base.clone()
            self.paths.append((name, field, clones[id(base)]))
        static = clones[id(base)]
        return static.as_strided(t.size(), t.stride(), t.storage_offset())

    def load(self, obj) -> None:
        for name, field, static in self.paths:
            value = getattr(obj, name)
            src = _base(value if field is None else getattr(value, field))
            if src.numel() != static.numel() or src.dtype != static.dtype:
                raise RuntimeError(f"a decode graph's {name} table does not match this round's")
            static.copy_(src, non_blocking=True)


@dataclasses.dataclass
class _Entry:
    graph: torch.cuda.CUDAGraph
    out: torch.Tensor
    step: _Twin
    tables: _Twin | None


class DecodeGraphs:
    """The graphs of one ``MultiDecoder``: verification forwards (``"verify"``) and MTP steps (``"mtp"``)."""

    def __init__(self, width: int, *, cap: int | None = None, min_seen: int = MIN_SEEN) -> None:
        self.width = width                          # the DeltaNet fold table's rows a stream (the widest window)
        self.cap = int(os.environ.get(CAP_ENV, "0") or 0) or (CAP if cap is None else cap)
        self.min_seen = max(1, min_seen)
        self.streams = int(os.environ.get(STREAMS_ENV, "0") or 0) or STREAMS
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: OrderedDict[tuple, _Entry] = OrderedDict()
        self.seen: dict[tuple, int] = {}
        self.forwards: dict[str, int] = {}
        self.replays: dict[str, int] = {}
        self.captures: dict[str, int] = {}
        self.capture_s = 0.0
        self.graph_mib = 0.0                        # device memory the graphs took (free before less after capture)
        self.failed: set[str] = set()
        self.calls = 0
        self._emit({"event": "on", "cap": self.cap, "min_seen": self.min_seen, "streams": self.streams,
                    "free_mib": _free_mib()})

    @staticmethod
    def _emit(event: dict) -> None:
        event["t"] = round(time.time(), 3)          # wall clock: a client's request times place each event
        print("[thor-graph] " + json.dumps(event, separators=(",", ":")), file=sys.stderr, flush=True)

    def stats(self) -> dict:
        return {"event": "stats", "forwards": dict(self.forwards), "replays": dict(self.replays),
                "captures": dict(self.captures), "evictions": 0, "capture_s": round(self.capture_s, 4),
                "graph_mib": round(self.graph_mib, 1),
                "graphs": len(self.graphs), "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1),
                "free_mib": _free_mib()}

    def key(self, kind: str, segs: Sequence, b, step, tables) -> tuple:
        """Everything this forward's launches take from the host (see the module docstring)."""

        sc = b.attn
        top = sc.budget // sc.ratio
        mtp = kind == "mtp"
        per = []
        for (st, a0, a1), end in zip(segs, step.ends):
            sparse = bool(sc.qsa) and end // sc.ratio > top
            pos = st.mtp_pos if mtp else st.pos_dev
            tail = getattr(st, "ple_tail", None)
            sig = (a1 - a0, sparse, pos.data_ptr(), 0 if tail is None else tail.data_ptr())
            if sparse:
                sig += (st.mtp_pooled.data_ptr(),) if mtp else tuple(p.data_ptr() for p in st.pooled)
            per.append(sig)
        k = (kind, tuple(per), bool(step.vision), step.context)
        if tables is not None:
            k += (tables.cur, bool(tables.folds), tables.plan.slots, tables.plan.max_rows, tables.held.shape[1])
        return k

    def run(self, kind: str, segs: Sequence, b, fn: Callable[[], torch.Tensor], *, step,
            tables=None) -> torch.Tensor:
        """``fn()``'s forward (it reads ``b``'s step and tables, which hold ``step`` and ``tables``): this round's
        replay of the graph recorded for its launch structure, or an eager run."""

        self.forwards[kind] = self.forwards.get(kind, 0) + 1
        self.calls += 1
        if self.calls % STATS_EVERY == 0:
            self._emit(self.stats())
        if len(segs) > self.streams:                # eager as before (the widest fold table changes no bits)
            return fn()
        step.context = bucket(max(step.ends))
        key = self.key(kind, segs, b, step, tables)
        entry = self.graphs.get(key)
        if entry is not None:
            entry.step.load(step)
            if entry.tables is not None:
                entry.tables.load(tables)
            entry.graph.replay()
            self.replays[kind] = self.replays.get(kind, 0) + 1
            return entry.out
        seen = self.seen.get(key, 0) + 1
        if kind in self.failed or seen < self.min_seen or len(self.graphs) >= self.cap:
            if len(self.seen) >= SEEN_MAX:
                self.seen.clear()
            self.seen[key] = seen
            return fn()
        self.seen.pop(key, None)
        entry = self._record(kind, b, step, tables, fn, segs)
        if entry is None:
            return fn()
        self.graphs[key] = entry
        entry.graph.replay()                        # this round's tables are the ones it was recorded with
        self.replays[kind] = self.replays.get(kind, 0) + 1
        return entry.out

    def _record(self, kind: str, b, step, tables, fn, segs) -> _Entry | None:
        t0 = time.perf_counter()
        free0 = _free_mib()
        twin_step = _Twin(step)
        twin_tables = _Twin(tables) if tables is not None else None
        saved = (b.attn_step, getattr(b, "gdn_tables", None))
        b.attn_step = twin_step.obj
        if twin_tables is not None:
            b.gdn_tables = twin_tables.obj
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        enabled = gc.isenabled()
        gc.disable()                                # a collected graph's destructor would invalidate the capture
        failed = None
        try:
            # recorded on a side stream, nothing runs (torch.cuda.graph would also sync, collect and empty the cache)
            with torch.cuda.stream(side):
                graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
                try:
                    out = fn()
                except Exception as exc:            # noqa: BLE001
                    failed = exc
                finally:
                    try:
                        graph.capture_end()
                    except Exception as exc:        # noqa: BLE001
                        failed = failed or exc
        finally:
            if enabled:
                gc.enable()
            b.attn_step = saved[0]
            if twin_tables is not None:
                b.gdn_tables = saved[1]
        torch.cuda.current_stream().wait_stream(side)
        if failed is not None:                      # this kind stays eager from here on
            self.failed.add(kind)
            self._emit({"event": "fail", "kind": kind, "error": f"{type(failed).__name__}: {failed}"[:400]})
            return None
        spent = time.perf_counter() - t0
        free1 = _free_mib()
        self.capture_s += spent
        self.graph_mib += free0 - free1
        self.captures[kind] = self.captures.get(kind, 0) + 1
        self._emit({"event": "capture", "kind": kind, "n": len(segs), "rows": segs[-1][2],
                    "graphs": len(self.graphs) + 1, "capture_s": round(spent, 4), "graph_mib": round(free0 - free1, 1),
                    "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1), "free_mib": free1})
        return _Entry(graph, out, twin_step, twin_tables)


def _free_mib() -> float:
    free, _ = torch.cuda.mem_get_info()
    return round(free / MIB, 1)
