"""Flash Next's concurrent decode rounds as CUDA graphs on Jetson Thor (sm_110) (THOR-K3).

The concurrent server (``--parallel``) runs its rounds eagerly: a verification forward over every stream's window,
then an MTP step a draft depth, each one launch a kernel, about 2,300 launches a round for one stream. Here each such
forward replays a CUDA graph recorded for its exact launch structure (a verification forward that shares its experts
with a prompt pass stays eager):

- the key holds everything a launch takes from the host: each stream's window rows, whether its rows are sparse (a
  sparse stream's own select reads its slot's position and pooled keys, so those pointers join its key), the DeltaNet
  scratch parity, whether last round's kept rows fold in, the image flag and the context bucket; the round's device
  tables (rows' streams and positions, cache and state pointers, DeltaNet plans) and each stream's n-gram conv tail
  are copied into the graph's own before each replay, so a dense forward's graph serves any slot (THOR-DEC-1);
- a graphed forward bounds its attention launches by a power-of-two context bucket, as the single-stream graphs do,
  and keeps the DeltaNet fold table at its widest. Every kernel reads its rows' real lengths on the device (chunks past
  a row's keys write nothing and the merge never reads them; scores past a row's complete blocks are never written;
  the fold reads only its counted rows), so the bits are the eager path's;
- a forward folds last round's kept DeltaNet rows into the states in place, so it must never run twice: a key's
  first forward runs eagerly (and compiles every launch shape its graph holds), a later one records the graph, which
  runs nothing, and replays it once. At most ``cap`` graphs are kept; past that, new keys stay eager;
- every capture runs on one side stream into one pool: the caching allocator reuses a block only on the stream that
  freed it, so the captures share their temporaries instead of each keeping its own (THOR-DEC-1);
- ``prerecord`` records the one-stream keys of short contexts at startup, on a free slot reset after, so requests
  never pay a recording. It runs after the memory gate measured the room, and the room is charged with the graphs'
  own memory (their captures and uploads); the eager forward before each capture takes the decode path's first-use
  memory, which eager decoding also takes at its first requests, after its gate. Other keys then stay eager
  (``TENSORFOLD_THOR_DECODE_GRAPH_LAZY=1`` records them on their second forward, as before;
  ``TENSORFOLD_THOR_DECODE_GRAPH_STARTUP=0`` records every key that way). ``TENSORFOLD_THOR_DECODE_GRAPH_ROWS``
  (default ``2-W``: every window from two rows to the widest ``W``; a drafter that stops its chain early leaves the
  windows between, which ran eagerly under ``2,W``, the earlier default and now the kill switch; or ``3,4,5``) picks
  the verification windows recorded: on Thor a graph executable holds driver memory of its own, about 20 MiB a
  verification forward;
- past the startup set, a one-stream key (a longer context's bucket; a sparse stream's, which holds its slot's pointers;
  a first round's) records on its second forward on demand (PAR-1 s4; ``TENSORFOLD_THOR_DECODE_GRAPH_DEMAND=0`` keeps it
  eager, 8f73033's path and the kill switch) when the memory gate's room, past the streams' caches and its reserve, can
  pay it: the room is charged the free memory its capture and upload took, and the host memory the driver keeps for
  its executable (glibc's bytes in use grew by them, at most ``HOST_MAX_MIB`` a graph: on Thor ~5 MiB a graph, 451 MiB
  over 93 graphs that the free memory never showed [vmem-a1]; ``TENSORFOLD_THOR_DECODE_GRAPH_HOST=0`` charges the free
  memory only, the earlier path). The on-demand graphs take at most ``TENSORFOLD_THOR_DECODE_GRAPH_DEMAND_MIB``
  (default 1,024 MiB); past it the least recently replayed one, idle for ``RECENT`` forwards, goes, and the room gets
  back the memory its release returned;
- a one-stream verification forward always writes DeltaNet scratch half 0 (``gdn_multi.normalize`` copies last
  round's pending rows to half 1 first, exact copies), so one graph a window size serves both parities;
- an MTP step's graph may hold its draft picks' reductions and their copy to pinned host memory (``extra`` keys
  them by the picks' width), so a chain step is one replay and one read of host memory.
- a concurrent forward (two or more streams; DPUSH MS-GRAPH) is keyed by its stream count and rows only
  (``canonical``), never by a stream's window or slot: its n-gram convs run as one launch that reads each row's window
  start and each stream's conv tail address from the step's tables, the pooled-key grid spans the widest window and
  the DeltaNet plan's shared memory the widest chain, so one graph serves every split of its rows over its streams.
  A concurrent forward with a sparse stream stays eager. Such graphs record on demand like any other key, charged to
  the same budget. By default only MTP chain steps (one row a stream) replay graphs, every other concurrent forward
  runs eagerly with its one-launch convs (``multi_graphed``; ``TENSORFOLD_THOR_DECODE_GRAPH_MULTI=all``: every
  kind); ``TENSORFOLD_THOR_DECODE_GRAPH_MULTI=0`` keeps concurrent forwards eager with their per-stream convs
  (e4fb57f's path, the kill switch);

Graphs are on by default on Thor (THOR-DEC-1 G-D ruling, 2026-10-03; ``TENSORFOLD_THOR_DECODE_GRAPH=0`` keeps every
round eager, the r37 path and the kill switch); ``TENSORFOLD_THOR_DECODE_GRAPH_STREAMS=N`` graphs only forwards of at
most N streams (default: every forward, or one stream with ``MULTI`` off; ``_CAP``: the most graphs kept). Events go
to stderr as ``[thor-graph] {json}`` lines (``tools/thor/graph_stats.py`` digests them).
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
import threading
import time
from collections import OrderedDict
from typing import Callable, Sequence

import torch

from . import gdn_multi

ENV = "TENSORFOLD_THOR_DECODE_GRAPH"
CAP_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_CAP"
STREAMS_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_STREAMS"
SPIN_ENV = "TENSORFOLD_THOR_DECODE_SPIN"
STARTUP_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_STARTUP"
LAZY_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_LAZY"
ROWS_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_ROWS"
CAPTURE_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_CAPTURE"   # diagnostics: 0 = prerecord's eager warm-ups only, no graphs
DEMAND_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_DEMAND"     # after prerecord, other one-stream keys on demand (default on)
DEMAND_MIB_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_DEMAND_MIB"
HOST_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_HOST"         # on-demand graphs' host memory charged too (default on)
MEM_ENV = "TENSORFOLD_THOR_MEM_PROBE"   # diagnostics: a trigger file path; [thor-mem] breakdowns at startup and on it
MULTI_ENV = "TENSORFOLD_THOR_DECODE_GRAPH_MULTI"       # concurrent forwards under canonical keys (default on)
CAPABILITY = (11, 0)
SCHED_MASK, SCHED_SPIN = 0x07, 0x01  # CU_CTX_SCHED_MASK, CU_CTX_SCHED_SPIN
CAP = 256                   # graphs kept, verification and MTP together
MIN_SEEN = 2                # a key's forwards before its graph is recorded (the first runs eagerly)
STREAMS = 1                 # the most streams a graphed forward holds (more stay eager: their keys rarely repeat)
MULTI_STREAMS = 64          # ... with MULTI on: every concurrent forward, keyed by its stream count and rows
SEEN_MAX = 1 << 14          # keys counted before the counts start over
STATS_EVERY = 512           # forwards between stats events
DEMAND_MIB = 1024           # the on-demand graphs' memory at most (MiB)
EST_MIB = 24                # an on-demand graph's memory before one is measured (a verification forward's ~20 MiB)
RECENT = 2048               # forwards an on-demand graph sits unreplayed before it may go
MISSED_MAX = 64             # eager one-stream keys counted by name in stats (the rest as "other")
HOST_MAX_MIB = 32           # an on-demand graph's host charge at most (another thread's allocation in the window)
MIB = float(1 << 20)


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")   # on by default (DEC-1 G-D)


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "off", "false", "no")


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110 and not switched off: decided once."""

    return not _off() and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY


@functools.lru_cache(maxsize=None)
def spinning(device: int = 0) -> str:
    """How the host waits for the GPU in decode on sm_110 (THOR-DEC-1; ``TENSORFOLD_THOR_DECODE_SPIN=0``: sleeps).

    A synchronize that sleeps lets the governor clock the waiting core down to its floor (972 MHz on Thor), and every
    host phase of a round then runs about 2x slower. ``"ctx"``: the CUDA context's waits spin (its scheduling flag,
    set here once); ``"poll"``: the driver refused the flag, so ``settle`` polls an event before the decode
    read-backs; ``""``: off. Host only: the GPU's work and its bits are unchanged."""

    on = os.environ.get(SPIN_ENV, "1").strip().lower() not in ("0", "off", "false", "no")
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


@functools.lru_cache(maxsize=None)
def multi_on(device: int = 0) -> bool:
    """Concurrent forwards (two or more streams) graphed under canonical keys, their n-gram convs one launch (DPUSH
    MS-GRAPH): sm_110 with graphs on, unless ``TENSORFOLD_THOR_DECODE_GRAPH_MULTI=0`` (e4fb57f's eager concurrent
    rounds, the kill switch). Decided once."""

    return available(device) and _flag(MULTI_ENV, "1")


def multi_graphed(kind: str, streams: int, rows: int, every: bool) -> bool:
    """Whether a concurrent forward replays graphs (MS-GRAPH): by default only MTP chain steps (one row a stream: at
    most one key a stream count, so nothing churns the budget); ``TENSORFOLD_THOR_DECODE_GRAPH_MULTI=all`` also
    verification forwards and first MTP steps under ``canonical`` keys (dpush-3-dq1: their keys filled the 1,024 MiB
    budget at 4-8 streams, 104-124 evictions a session and their recordings cost 2-6%)."""

    return every or (kind == "mtp" and rows == streams)


def canonical(kind: str, segs: Sequence, step, extra: tuple = ()) -> tuple:
    """A concurrent forward's key (DPUSH MS-GRAPH): its stream count and rows, never a stream's window or slot.

    Every launch of such a forward takes its rows' streams, positions, window starts, conv tails and cache pointers
    from the step's and tables' device tables (a replay copies this round's in), and sizes its grid by the rows, the
    streams, the widest window and the context bucket; so any split of the same rows over the same number of streams,
    in any slots, replays one graph. (Per-stream windows in the key gave keys that never repeated: 0.94-0.98x.)"""

    return (kind, ("multi", len(segs), segs[-1][2]), bool(step.vision), step.context) + tuple(extra)


def settle() -> None:
    """Before a decode read-back: busy-wait until the current stream's queued work is done, when the context's own
    waits could not be made to spin (``spinning() == "poll"``)."""

    if spinning() != "poll":
        return
    ev = torch.cuda.Event()
    ev.record()
    while not ev.query():
        pass


def _rows(spec: str, width: int) -> list[int]:
    """``ROWS_ENV``'s verification windows: ``a-b`` and/or comma-separated sizes, kept within 2..``width``."""

    out: set[int] = set()
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(r for r in out if 2 <= r <= width)


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
    shx: int = 0                                # fused shared-expert tails this graph captured (each replay runs them)
    swap: bool = True                           # its forward reads the graph's own conv tails (copied in a replay)


class DecodeGraphs:
    """The graphs of one ``MultiDecoder``: verification forwards (``"verify"``) and MTP steps (``"mtp"``)."""

    def __init__(self, width: int, *, cap: int | None = None, min_seen: int = MIN_SEEN,
                 ple: dict | None = None) -> None:
        self.width = width                          # the DeltaNet fold table's rows a stream (the widest window)
        self.cap = int(os.environ.get(CAP_ENV, "0") or 0) or (CAP if cap is None else cap)
        self.min_seen = max(1, min_seen)
        self.multi = multi_on()                     # concurrent forwards too, under canonical keys (MS-GRAPH)
        self.multi_all = os.environ.get(MULTI_ENV, "").strip().lower() == "all"   # ... every kind, not chain steps
        self.streams = int(os.environ.get(STREAMS_ENV, "0") or 0) or (MULTI_STREAMS if self.multi else STREAMS)
        self.multi_eager = 0                        # concurrent forwards left eager (a sparse stream)
        self.pool = torch.cuda.graph_pool_handle()
        self.side = torch.cuda.Stream()             # every capture's stream (a pool block is reused on its own stream)
        self.ple = {"verify": True, "mtp": True} if ple is None else dict(ple)   # kind -> its forward reads conv tails
        self.tails: list[torch.Tensor] = []         # the graphs' n-gram conv tails, a stream's copied in before a replay
        self.startup = _flag(STARTUP_ENV, "1")      # prerecord the one-stream short-context keys at startup
        self.lazy = _flag(LAZY_ENV, "0")            # after prerecord, record other keys on their second forward too
        self.capture = _flag(CAPTURE_ENV, "1")      # prerecord records graphs (0: its eager warm-ups only; diagnostics)
        self.rows = _rows(os.environ.get(ROWS_ENV, "") or f"2-{width}", width)   # verification windows prerecorded
        self.hist: dict[str, list[int]] = {}        # one-stream forwards by kind and rows: [forwards, replays]
        self.missed: dict[str, int] = {}            # diagnostics: one-stream eager forwards by key (pointers left out)
        self.fixed = False                          # prerecord ran: unrecorded keys stay eager unless ``lazy``
        self.demand = _flag(DEMAND_ENV, "1")        # ... or the room can pay them (on demand, after prerecord)
        self.budget = float(os.environ.get(DEMAND_MIB_ENV, "") or DEMAND_MIB) * MIB
        self.gate = None                            # the decoder's memory gate (prerecord attaches it)
        self.room_at = None                         # its room then (diagnostics)
        self.paid: OrderedDict[tuple, list] = OrderedDict()     # on demand: key -> [bytes charged, last call]
        self.charged = self.refunded = 0            # bytes on-demand graphs took off the room, and gave back
        self.host = _flag(HOST_ENV, "1")            # their host memory charged too (glibc's in-use growth)
        self.host_mib = 0.0                         # host memory on-demand captures took (measured even when off)
        self.est = EST_MIB * MIB                    # an on-demand graph's bytes before it is measured (the most seen)
        self.evicted = 0
        self.ctx: dict[str, list[int]] = {}         # one-stream forwards by kind:bucket:sparse: [forwards, replays]
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
                    "multi": self.multi, "multi_all": self.multi_all,
                    "startup": self.startup, "lazy": self.lazy, "rows": self.rows, "demand": self.demand,
                    "demand_mib": round(self.budget / MIB, 1), "ple": self.ple,
                    "free_mib": _free_mib()})

    @staticmethod
    def _emit(event: dict) -> None:
        event["t"] = round(time.time(), 3)          # wall clock: a client's request times place each event
        print("[thor-graph] " + json.dumps(event, separators=(",", ":")), file=sys.stderr, flush=True)

    def stats(self) -> dict:
        return {"event": "stats", "forwards": dict(self.forwards), "replays": dict(self.replays),
                "captures": dict(self.captures), "evictions": 0, "capture_s": round(self.capture_s, 4),
                "graph_mib": round(self.graph_mib, 1),
                "graphs": len(self.graphs), "hist": dict(self.hist), "missed": dict(self.missed),
                "ctx": {k: list(v) for k, v in self.ctx.items()}, "multi_eager": self.multi_eager,
                "demand": {"graphs": len(self.paid), "mib": round((self.charged - self.refunded) / MIB, 1),
                           "evicted": self.evicted, "host_mib": round(self.host_mib, 1), "host": self.host},
                "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1),
                "free_mib": _free_mib()}

    def key(self, kind: str, segs: Sequence, b, step, tables, extra: tuple = ()) -> tuple:
        """Everything this forward's launches take from the host (see the module docstring)."""

        sc = b.attn
        top = sc.budget // sc.ratio
        mtp = kind == "mtp"
        if self.multi and len(segs) > 1:            # MS-GRAPH: rows and streams only; a sparse stream stays eager
            if sc.qsa and any(end // sc.ratio > top for end in step.ends):
                return None
            k = canonical(kind, segs, step, extra)
            if tables is not None:
                k += (tables.cur, bool(tables.folds), tables.plan.slots, tables.plan.max_rows, tables.held.shape[1])
            return k
        per = []
        for (st, a0, a1), end in zip(segs, step.ends):
            sparse = bool(sc.qsa) and end // sc.ratio > top
            sig = (a1 - a0, sparse)
            if sparse:                              # its own select reads the slot's position and pooled keys
                pos = st.mtp_pos if mtp else st.pos_dev
                sig += (pos.data_ptr(),) + ((st.mtp_pooled.data_ptr(),) if mtp else
                                            tuple(p.data_ptr() for p in st.pooled))
            per.append(sig)
        k = (kind, tuple(per), bool(step.vision), step.context) + tuple(extra)
        if tables is not None:
            k += (tables.cur, bool(tables.folds), tables.plan.slots, tables.plan.max_rows, tables.held.shape[1])
        return k

    def _miss(self, key: tuple) -> None:
        """Diagnostics: a one-stream forward ran eagerly; ``stats`` counts it by its key, pointers left out."""

        per = key[1] if key[1][:1] == ("multi",) else tuple(p[:2] for p in key[1])
        k = str((key[0], per) + key[2:])
        if k not in self.missed and len(self.missed) >= MISSED_MAX:
            k = "other"
        self.missed[k] = self.missed.get(k, 0) + 1

    def run(self, kind: str, segs: Sequence, b, fn: Callable[[], torch.Tensor], *, step,
            tables=None, extra: tuple = ()) -> torch.Tensor:
        """``fn()``'s forward (it reads ``b``'s step and tables, which hold ``step`` and ``tables``): this round's
        replay of the graph recorded for its launch structure, or an eager run."""

        self.forwards[kind] = self.forwards.get(kind, 0) + 1
        self.calls += 1
        if self.calls % STATS_EVERY == 0:
            self._emit(self.stats())
        if len(segs) > self.streams:                # eager as before (the widest fold table changes no bits)
            return fn()
        if self.multi and len(segs) > 1 and not multi_graphed(kind, len(segs), segs[-1][2], self.multi_all):
            return fn()                             # eager (its n-gram convs still one launch)
        if self.multi and len(segs) > 1:            # MS-GRAPH: grids by the widest window, whatever the split
            step.most = max(step.most, self.width)  # (a pooled-key block past a stream's rows writes nothing)
            if tables is not None and tables.plan.max_rows < self.width:
                tables.plan = dataclasses.replace(tables.plan, max_rows=self.width)   # a chain plan: shared memory only
        if tables is not None and tables.cur:       # one parity a graph: this round writes half 0
            gdn_multi.normalize(tables)
        step.context = bucket(max(step.ends))
        key = self.key(kind, segs, b, step, tables, extra)
        if key is None:                             # a concurrent forward with a sparse stream
            self.multi_eager += 1
            return fn()
        h = self.hist.setdefault(f"{kind}{segs[-1][2]}" if len(segs) == 1 else f"{kind}x{len(segs)}", [0, 0])
        h[0] += 1
        sparse = key[1][:1] != ("multi",) and any(p[1] for p in key[1])      # (a concurrent key is dense)
        c = self.ctx.setdefault(f"{kind}:{step.context}:{'s' if sparse else 'd'}", [0, 0])
        c[0] += 1
        entry = self.graphs.get(key)
        if entry is not None:
            self._load(entry, kind, segs, step, tables)
            self._replay(entry)
            self.replays[kind] = self.replays.get(kind, 0) + 1
            h[1] += 1
            c[1] += 1
            if key in self.paid:                    # on demand: the least recently replayed goes first
                self.paid[key][1] = self.calls
                self.paid.move_to_end(key)
            return entry.out
        paid = self.fixed and not self.lazy         # past the startup set: on demand, the room pays each graph
        if paid and (not self.demand or self.gate is None):     # 8f73033: the rest stay eager
            self._miss(key)
            return fn()
        seen = self.seen.get(key, 0) + 1
        if kind in self.failed or seen < self.min_seen or len(self.graphs) >= self.cap or paid and not self._afford():
            if len(self.seen) >= SEEN_MAX:
                self.seen.clear()
            self.seen[key] = seen
            self._miss(key)
            return fn()
        self.seen.pop(key, None)
        free0, host0 = _free_mib(), _host_mib()
        entry = self._record(kind, b, step, tables, fn, segs)
        if entry is None:
            self._miss(key)
            return fn()
        self.graphs[key] = entry
        self._replay(entry)                         # this round's tables are the ones it was recorded with
        self.replays[kind] = self.replays.get(kind, 0) + 1
        c[1] += 1
        if paid:
            self._charge(key, kind, free0, host0)
        return entry.out

    def _afford(self) -> bool:
        """On demand: whether one more graph fits the budget (the least recently replayed on-demand graph, idle for
        ``RECENT`` forwards, goes first) and the memory gate (its room past the streams' caches and the reserve, and the
        host's live memory)."""

        while self.paid and self.charged - self.refunded + self.est > self.budget and self._evict():
            pass
        if self.charged - self.refunded + self.est > self.budget:
            return False
        if getattr(self.gate, "pooled", False):     # Thor KV pool: the graphs live outside it, in the budget it left
            return self.gate.outside(int(self.est))
        return self.gate.fits(int(self.est))

    def _charge(self, key: tuple, kind: str, free0: float, host0: float | None = None) -> None:
        """An on-demand graph's own memory (the free memory its capture and upload took, and the host memory the
        driver keeps for its executable) off the gate's room."""

        torch.cuda.synchronize()                    # the upload landed
        cost = int(max(0.0, free0 - _free_mib()) * MIB)
        host = 0.0 if host0 is None else min(float(HOST_MAX_MIB), max(0.0, _host_mib() - host0))
        self.host_mib += host
        if self.host:
            cost += int(host * MIB)
        if not getattr(self.gate, "pooled", False):  # Thor KV pool: the pool left this budget out already
            self.gate.room -= cost
        self.charged += cost
        self.paid[key] = [cost, self.calls]
        self.est = max(self.est, float(cost))
        self._emit({"event": "demand", "kind": kind, "context": key[3], "graphs": len(self.paid),
                    "charged_mib": round(cost / MIB, 1), "host_mib": round(host, 1),
                    "demand_mib": round((self.charged - self.refunded) / MIB, 1),
                    "room_mib": round(self.gate.room / MIB, 1), "held_mib": round(self.gate.held / MIB, 1),
                    "free_mib": _free_mib()})

    def _evict(self) -> bool:
        """The least recently replayed on-demand graph goes, unless it replayed within ``RECENT`` forwards; the room
        gets back the memory its release returned (pool blocks stay for later captures, still charged)."""

        key, (cost, last) = next(iter(self.paid.items()))
        if self.calls - last < RECENT:
            return False
        del self.paid[key]
        entry = self.graphs.pop(key)
        torch.cuda.synchronize()
        free0, host0 = _free_mib(), _host_mib()
        entry.graph.reset()
        del entry
        torch.cuda.synchronize()
        back = int(max(0.0, _free_mib() - free0) * MIB)
        if self.host:                               # the executable's host memory came back too
            back += int(min(float(HOST_MAX_MIB), max(0.0, host0 - _host_mib())) * MIB)
        back = min(cost, back)
        if not getattr(self.gate, "pooled", False):
            self.gate.room += back
        self.refunded += back
        self.evicted += 1
        return True

    def _load(self, entry: _Entry, kind: str, segs: Sequence, step, tables) -> None:
        """This round's tables and n-gram conv tails into the graph's own storage, before its replay."""

        entry.step.load(step)
        if entry.tables is not None:
            entry.tables.load(tables)
        if self.ple.get(kind, True) and entry.swap:
            for tail, (st, _, _) in zip(self.tails, segs):
                tail.copy_(st.ple_tail, non_blocking=True)

    def _replay(self, entry: _Entry) -> None:
        """Replay ``entry``. Each fused shared-expert tail it captured runs again; ``thor_shx`` counts those."""

        entry.graph.replay()
        if entry.shx:
            from . import thor_shx
            thor_shx.add_replays(entry.shx)

    def _record(self, kind: str, b, step, tables, fn, segs) -> _Entry | None:
        t0 = time.perf_counter()
        free0, alloc0, res0 = _free_mib(), torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        twin_step = _Twin(step)
        twin_tables = _Twin(tables) if tables is not None else None
        # the graph reads the tails it owns, this round's copied in; a step with a tail table (MS-GRAPH) reads each
        # stream's own tail through it, so nothing is swapped
        ple = self.ple.get(kind, True) and getattr(step, "tails", None) is None
        if ple:
            while len(self.tails) < len(segs):
                self.tails.append(torch.empty_like(segs[0][0].ple_tail))
            for tail, (st, _, _) in zip(self.tails, segs):
                tail.copy_(st.ple_tail)
        own = [st.ple_tail for st, _, _ in segs]
        saved = (b.attn_step, getattr(b, "gdn_tables", None))
        b.attn_step = twin_step.obj
        if twin_tables is not None:
            b.gdn_tables = twin_tables.obj
        graph = torch.cuda.CUDAGraph()
        side = self.side
        side.wait_stream(torch.cuda.current_stream())
        enabled = gc.isenabled()
        gc.disable()                                # a collected graph's destructor would invalidate the capture
        failed = None
        shx = 0
        from . import thor_shx
        try:
            if ple:
                for tail, (st, _, _) in zip(self.tails, segs):
                    st.ple_tail = tail
            # recorded on a side stream, nothing runs (torch.cuda.graph would also sync, collect and empty the cache)
            with torch.cuda.stream(side):
                thor_shx.begin_capture()
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
                    shx = thor_shx.take_capture()
        finally:
            if enabled:
                gc.enable()
            for (st, _, _), tail in zip(segs, own):
                st.ple_tail = tail
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
                    "kept_mib": round((torch.cuda.memory_allocated() - alloc0) / MIB, 2),
                    "pool_mib": round((torch.cuda.memory_reserved() - res0) / MIB, 1),
                    "reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1), "free_mib": free1})
        return _Entry(graph, out, twin_step, twin_tables, shx, swap=ple)

    def prerecord(self, dec, mtp_fn: Callable | None = None) -> int:
        """Record the one-stream keys of short contexts now (``dec``: the ``MultiDecoder``, every slot free): a
        verification window of each ``rows`` size (scratch half 0, last round's rows folding in) and an MTP step of
        1..``width`` rows (``mtp_fn(segs)``: the step's forward and picks, keyed by ``dec.picks_key``), at the first
        context bucket. Each forward runs eagerly once on a free
        slot (it compiles its launches), is recorded, then replays once (the graph's upload); the slot and the scratch
        parity are reset after. A first round (nothing folds in), a one-row verification, a sparse or image stream
        and a longer context are left to ``run`` (on demand, or ``lazy``). Returns the graphs' own memory in bytes:
        the free memory their captures and uploads took (the eager forwards' is the decode path's first-use memory),
        0 if nothing ran."""

        from .attn_multi import Step
        from .forward import compute, stage
        from .gdn_multi import Tables
        from .mtp import mtp_compute, mtp_stage

        self.gate = getattr(dec, "memory_gate", None)     # on demand: later graphs are charged to its room
        self.room_at = None if self.gate is None else self.gate.room
        if not self.startup or self.fixed or not dec.free:
            return 0
        w, st = dec.w, dec.free[-1]
        torch.cuda.synchronize()
        t0, free0, res0 = time.perf_counter(), _free_mib(), torch.cuda.memory_reserved()
        made, mem = 0, {"verify": [0.0, 0.0, 0.0], "mtp": [0.0, 0.0, 0.0]}     # free MiB lost: eager, capture, upload
        try:
            for rows in self.rows:
                segs = stage(w, dec.buf, [(st, [0] * rows)])
                dec.gdn.parity = 0
                tables = dec.buf.gdn_tables = Tables(w, dec.gdn, segs, [[0]], width=self.width)
                step = dec.buf.attn_step = Step(w, segs, mtp=False)
                made += self._prerecord("verify", segs, dec.buf, step, tables,
                                        lambda segs=segs: compute(w, segs, dec.buf), mem)
            if dec.mbuf is not None:
                for rows in range(1, self.width + 1):
                    segs = mtp_stage(w, dec.mbuf, [(st, [0] * rows, dec.buf.streams[:rows])])
                    step = dec.mbuf.attn_step = Step(w, segs, mtp=True)
                    fn = (lambda segs=segs: mtp_compute(w, segs, dec.mbuf)) if mtp_fn is None else \
                        (lambda segs=segs: mtp_fn(segs))
                    made += self._prerecord("mtp", segs, dec.mbuf, step, None, fn, mem,
                                            extra=() if mtp_fn is None else dec.picks_key)
        finally:
            dec.buf.gdn_tables = dec.buf.attn_step = None
            if dec.mbuf is not None:
                dec.mbuf.attn_step = None
            dec.gdn.parity = 0
            st.reset(w)
            torch.cuda.synchronize()
        self.fixed = True
        held = _free_mib()
        torch.cuda.empty_cache()                    # the warm-ups' cached blocks back to the driver (graphs keep theirs)
        own = max(0.0, sum(v[1] + v[2] for v in mem.values()))    # captures and uploads: the room is charged this
        if made:
            self.est = max(self.est, own * MIB / made)
        self._emit({"event": "startup", "graphs": made, "s": round(time.perf_counter() - t0, 3),
                    "graph_mib": round(free0 - _free_mib(), 1), "released_mib": round(_free_mib() - held, 1),
                    "phase_mib": {k: [round(x, 1) for x in v] for k, v in mem.items()}, "charged_mib": round(own, 1),
                    "pool_mib": round((torch.cuda.memory_reserved() - res0) / MIB, 1), "free_mib": _free_mib()})
        return int(own * MIB)

    def _prerecord(self, kind: str, segs: Sequence, b, step, tables, fn: Callable[[], torch.Tensor], mem: dict,
                   extra: tuple = ()) -> int:
        step.context = bucket(max(step.ends))
        key = self.key(kind, segs, b, step, tables, extra)
        if key in self.graphs or kind in self.failed:
            return 0
        f0 = _free_mib()
        fn()                                        # eagerly first: every launch shape compiled before the capture
        torch.cuda.synchronize()
        f1 = _free_mib()
        if not self.capture:                        # diagnostics: the warm-up alone, every forward stays eager
            mem[kind][0] += f0 - f1
            return 0
        entry = self._record(kind, b, step, tables, fn, segs)
        if entry is None:
            return 0
        self.graphs[key] = entry
        f2 = _free_mib()
        self._replay(entry)                         # the upload, now rather than in a request
        torch.cuda.synchronize()
        f3 = _free_mib()
        for i, lost in enumerate((f0 - f1, f1 - f2, f2 - f3)):
            mem[kind][i] += lost
        return 1


def _free_mib() -> float:
    free, _ = torch.cuda.mem_get_info()
    return round(free / MIB, 1)


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
                                                       "fsmblks", "uordblks", "fordblks", "keepcost")]


@functools.lru_cache(maxsize=None)
def _mallinfo2():
    try:
        fn = ctypes.CDLL("libc.so.6").mallinfo2
    except (OSError, AttributeError):
        return None
    fn.restype = _Mallinfo2
    return fn


def _host_mib() -> float:
    """glibc's bytes in use (its heaps' and its own mmaps'), MiB: what the driver's host allocations, a graph
    executable's among them, hold; 0 without mallinfo2 (another libc), so nothing is charged."""

    fn = _mallinfo2()
    if fn is None:
        return 0.0
    m = fn()
    return (m.uordblks + m.hblkhd) / MIB


_MEM: list = []                                     # diagnostics: [the decoder, the trigger path]


def mem_probe(dec, tag: str) -> None:
    """Diagnostics (``TENSORFOLD_THOR_MEM_PROBE=<path>``): a ``[thor-mem]`` breakdown of the process's memory now (the
    driver's free memory, MemAvailable, torch's segments by pool, pinned host memory, the gate's room and held bytes);
    the first call also starts a thread that emits one, tagged with the file's text, whenever ``<path>`` appears."""

    path = os.environ.get(MEM_ENV, "")
    if not path:
        return
    if not _MEM:
        _MEM.extend([dec, path])
        threading.Thread(target=_mem_watch, args=(path,), daemon=True, name="thor-mem").start()
    _MEM[0] = dec
    _mem_emit(tag)


def _mem_watch(path: str) -> None:
    while True:
        time.sleep(0.5)
        try:
            with open(path) as f:
                tag = f.read().strip() or "probe"
            os.remove(path)
        except OSError:
            continue
        _mem_emit(tag)
        g = getattr(_MEM[0], "graphs", None)
        if g is not None:                           # the graphs' counters at the same edge (a phase's counts)
            try:
                g._emit(dict(g.stats(), tag=tag))
            except Exception:                       # noqa: BLE001  diagnostics never break serving
                pass


def _mem_emit(tag: str) -> None:
    event: dict = {"event": "mem", "tag": tag, "t": round(time.time(), 3)}
    try:
        pools: dict[str, list[int]] = {}
        for seg in torch.cuda.memory_snapshot():
            pid = tuple(seg.get("segment_pool_id", (0, 0)) or (0, 0))
            p = pools.setdefault("default" if pid == (0, 0) else "graph", [0, 0, 0])
            p[0] += int(seg["total_size"])
            p[1] += int(seg["allocated_size"])
            p[2] += 1
        m = {r.split()[0].rstrip(":"): int(r.split()[1]) for r in open("/proc/meminfo")}
        own = {r.split(":")[0]: int(r.split()[1]) for r in open("/proc/self/status")
               if r.startswith(("VmRSS", "RssAnon", "RssFile", "RssShmem"))}
        host = getattr(torch.cuda, "host_memory_stats", None)
        h = {k: v for k, v in (host() if host is not None else {}).items() if "bytes" in k and k.endswith("current")}
        dec = _MEM[0]
        gate = getattr(dec, "memory_gate", None)
        event.update({
            "free_mib": _free_mib(), "memavail_mib": round(m["MemAvailable"] / 1024, 1),
            "memfree_mib": round(m["MemFree"] / 1024, 1), "cached_mib": round(m.get("Cached", 0) / 1024, 1),
            "torch_mib": {k: [round(v[0] / MIB, 1), round(v[1] / MIB, 1), v[2]] for k, v in pools.items()},
            "pinned_mib": {k: round(v / MIB, 2) for k, v in h.items()},
            "pins_kib": round(sum(t.numel() * t.element_size() for t in getattr(dec, "pins", {}).values()) / 1024, 1),
            "rss_mib": {k: round(v / 1024, 1) for k, v in own.items()},
            "room_mib": round(gate.room / MIB, 1) if gate is not None else None,
            "held_mib": round(gate.held / MIB, 1) if gate is not None else None,
            # par-1-fix diagnostics: the pool-guard's own live signal (MemoryGate.pool, e.g. driver_pool), beside
            # memavail_mib, to see whether short() is reading real pressure or a noisy proxy.
            "pool_mib": round(gate.pool() / MIB, 1) if gate is not None and gate.pool is not None else None,
            "kept": len(getattr(dec, "kept", ())), "streams": len(getattr(dec, "streams", ())),
            "graphs": len(dec.graphs.graphs) if getattr(dec, "graphs", None) is not None else 0})
    except Exception as exc:                        # diagnostics never break serving
        event["error"] = repr(exc)[:200]
    print("[thor-mem] " + json.dumps(event, separators=(",", ":")), file=sys.stderr, flush=True)
