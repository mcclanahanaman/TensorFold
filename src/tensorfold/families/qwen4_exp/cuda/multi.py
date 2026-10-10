"""Flash Next's concurrent rounds on one GPU: every stream keeps exactly its own accepted prefix."""

from __future__ import annotations

import time

import numpy as np
import torch

from tensorfold.cuda.logprobs import capture

from tensorfold.cuda.capacity import LIMIT_ENV, available_bytes, cuda_limit_bytes
from tensorfold.cuda.kernels import thor_gdn
from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.memory_gate import MEMORY_RETRY_S, SLOTS_RETRY_S, MemoryGate, NoRoom, driver_pool, torch_live
from tensorfold.server.errors import CONTEXT_LIMIT, ContextLengthError
from tensorfold.cuda.sampling import sample_streams
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from .decode import PREFILL_ROWS, WARM_TAIL, Engine, draft, entry_end, prefill_begin
from . import attn_multi, gdn_multi, image_rows, prefixes
from .forward import Cut, commit, compute, compute_mixed, converges, cut_snapshot, stage
from .mtp import MtpRoomError, mtp_compute, mtp_stage
from .prompt_plan import keeps_pool, pass_limit
from .state import ENDS, Buffers, State
from . import thor_dec2, thor_graphs, thor_mtp_fed
from ..cuda import CONFIDENCE, DEPTH

FIRST, STEP = 256, 8192          # rows an idle slot keeps; rows a stream's caches grow by at a time
GIB = 1024**3
SHARE = 0.0                      # --decode-share: a round alone takes this share of its pass's time (0: whole passes)
PASS_MIN = 128                   # the fewest prompt rows a round's pass takes
FILL_GUARD = 8                   # a prompt passed over this many passes takes the next one (no starvation), as on Macs


def _mapped_tables(w) -> int:
    """Bytes of every layer's mapped n-gram table (0 without PLE): host memory, not charged to the CUDA budget."""

    return sum(layer.ple.table.nbytes for layer in w.layers if layer.ple is not None)


def _table_aware_pool(torch, base_pool, mapped: int):
    """``base_pool`` (``driver_pool``'s own freed-page signal), or true MemAvailable headroom past the mapped
    n-gram tables when that is larger (par-1-fix).

    ROWS-2's own gate (T-033/T-034, r22) validated ``driver_pool`` at ``--parallel 4`` with no vision and no
    mapped table: there, ``KReclaimable - SReclaimable`` plus torch's own cushion tracked real pressure, because
    there was nothing else to subtract. A vision-capable, 8-stream recipe with a 26.8 GiB mapped table keeps that
    cushion structurally small (``_grow`` calls ``empty_cache()`` right after this check, by design) regardless of
    true room: measured on Thor (par1fix-verify-b, ``[thor-mem]`` probes during an 80k prompt_reuse growth),
    ``pool_mib`` sat at 0.9-1.7 GiB while ``memavail_mib`` stayed at 29-31 GiB the whole time, so ``short()`` fired
    on nearly every qualifying grow and evicted a wanted kept prefix instead of an idle one (80k turns 2% cached
    instead of 99%+). Past that true margin, a low ``base_pool`` reading is reported as a protective "short" only
    when it is NOT a false alarm, i.e. only when MemAvailable minus the table is ALSO tight: both signals track
    the same underlying pressure, so neither usually reads healthy while the other reads truly short. ``base_pool``
    alone decides where there is no mapped table (``mapped == 0``) or on a non-unified GPU (``page_room`` is None),
    which keeps ROWS-2's own gate exactly reproduced there."""

    from tensorfold.cuda.capacity import page_room

    def pool() -> int:
        base = base_pool()
        if not mapped:
            return base
        room = page_room(torch)
        return base if room is None else max(base, room - mapped)

    return pool


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int, prefill_rows: int) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, prefill_rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    e.stops = ()
    return e


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", prefill_rows: int = PREFILL_ROWS,
                 share: float = SHARE, points=None, vision=None, workspace_bytes: int = 0, pool: bool = False,
                 vision_bytes: int = 0, vision_host: int = 0) -> None:
        if w.comm is not None:
            raise ValueError("concurrent Flash Next runs on one GPU for now")
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.vision = vision
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1)
        # a round's window and a prompt pass share each layer's expert launch: the pass's buffers hold both
        self.converged, self.prefill_rows = converges(w), prefill_rows
        # rounds beside a filling prompt size its pass so decoding keeps ``share`` of the pass's time (0: whole passes)
        self.share, self.round_s, self.row_s = share, None, None
        self.buf = Buffers(w, rows, capacity, moe_prefill=True)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows + (rows if self.converged else 0), capacity, prefill=True)
        self.gdn = gdn_multi.Scratch(w, rows)            # every stream's DeltaNet rows, one launch a step
        # Thor (sm_110): a round's forwards may replay CUDA graphs (thor_graphs; on by default, TENSORFOLD_THOR_DECODE_GRAPH=0: off)
        self.slots = slots
        head = w.head if w.draft_head is None else w.draft_head
        self.draft_n = head.n                            # the draft head's columns (the picks' widest top-k)
        self.pins: dict[int, torch.Tensor] = {}          # picks width -> pinned host rows a graphed MTP step writes
        self.picks_key = (1,)                            # the startup MTP graphs' picks width (greedy streams)
        self.graphs = thor_graphs.DecodeGraphs(depth + 1, ple={     # which forwards read the streams' n-gram conv tails
            "verify": any(layer.ple is not None for layer in w.layers),
            "mtp": w.mtp is not None and w.mtp.layer.ple is not None}) if thor_graphs.available() else None
        thor_graphs.spinning()                           # Thor: host waits spin (TENSORFOLD_THOR_DECODE_SPIN=0: off)
        # Thor DPUSH-6 (thor_mtp_fed): the device-fed draft chain (TENSORFOLD_THOR_MTP_FED=0: the host-fed chain)
        self.fed = thor_mtp_fed.mode() if self.mbuf is not None and depth > 1 else "0"
        self.fed_check = None
        if self.fed != "0":
            dev = self.mbuf.ids.device
            self.picks_key = ("fed",)                    # its MTP graphs' picks: thor_mtp_fed.picks, no host copy
            self.fed_got = torch.zeros((slots, 3), dtype=torch.float32, device=dev)     # a step's picks rows
            self.fed_tok = torch.zeros((slots,), dtype=torch.int32, device=dev)         # their draft ids
            self.fed_map = None if w.draft_ids is None else w.draft_ids.to(device=dev, dtype=torch.int32).contiguous()
            self.fed_ar = torch.arange(slots, dtype=torch.int32, device=dev)
            self.fed_host = torch.zeros((2 * slots,), dtype=torch.int32, pin_memory=True)   # gather rows
            self.fed_dev = torch.zeros((2 * slots,), dtype=torch.int32, device=dev)
            self.fed_staged = torch.cuda.Event()
            self.fed_pins = [torch.zeros((slots, 3), dtype=torch.float32, pin_memory=True) for _ in range(2)]
            self.fed_ev = [torch.cuda.Event(), torch.cuda.Event()]
            self.fed_check = thor_mtp_fed.Check() if self.fed == "check" else None
        self.held: dict[int, list[int]] = {}             # stream id -> last round's kept rows, folded in next round
        # slots start small and grow with their stream's context, up to the window, while the gate has room
        self.free = [State(w, min(capacity, FIRST), depth + 1, kv_dtype, limit=capacity) for _ in range(slots)]
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.window_bytes = self.free[0].cache_bytes(capacity)          # one stream's caches at the full window
        # par-1-fix: a stable reference for cache_bytes(rows)/layer_bytes(rows) math regardless of which slot is
        # free or busy later (the formula only reads the model's own config, fixed and the same on every slot).
        self._ref_state = self.free[0]
        free = torch_live(torch, available_bytes) if torch.cuda.is_available() else None
        # the mapped n-gram tables are not held back (they barely fit on a Spark); lookups page from disk instead
        live = free
        self.memory_gate = MemoryGate(live() if live is not None else 1 << 62,
                                      reserve=max(2 * GIB, workspace_bytes), live=live)
        # Thor KV pool (thor_kv: the tables locked, the caches' memory reserved at load by ``reserve_pool``): a grow
        # never takes the driver's pages, so the wide-piece guard has nothing to keep
        self.kv_pool, self.pool_bytes, self.vision_bytes = bool(pool), 0, int(vision_bytes)
        self.vision_host = int(vision_host)          # THOR-VISION-MEM: the workspace's host part, left out of the pool
        if pool:
            self.memory_gate.pooled = True
            self.memory_gate.system = lambda: available_bytes(torch)
        elif torch.cuda.is_available() and keeps_pool(prefill_rows, torch.cuda.get_device_capability()):
            self.memory_gate.pool = _table_aware_pool(torch, driver_pool(torch), _mapped_tables(w))
        self.streams: dict[int, Stream] = {}
        self.filling: list[Stream] = []                  # admitted, prompts still prefilling (oldest first)
        self.fills: dict[int, list] = {}                 # stream id -> [its engine, drafts?, next row, kept state]
        self.passed: dict[int, int] = {}                 # stream id -> prompt passes since its last rows
        self.arrived = lambda: False                     # a request waits to be admitted (the scheduler sets this)
        self.next_id = 0
        self.draft_host = w.draft_ids.cpu().numpy() if w.draft_ids is not None else None
        self.kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []   # (ids, slot, snapshot, tail)
        self.keep = keep
        self.used: dict[int, float] = {}                 # id(slot) -> when it last started or finished a request
        self.evicted_by = {"memory": 0, "slots": 0, "entries": 0}     # kept chains freed, by cause (/v1/capacity)
        self.points_dropped = 0                          # kept points dropped while their chain stays (prefixes.remember)
        thor_graphs.mem_probe(self, "gate")              # diagnostics (TENSORFOLD_THOR_MEM_PROBE=<trigger path>)
        if self.graphs is not None:     # Thor: one-stream short-context graphs, recorded after the gate measured the room
            self.memory_gate.room -= self._prerecord()      # the room pays the graphs' own memory (THOR-DEC-1)
            thor_graphs.mem_probe(self, "prerecord")

    def _prerecord(self) -> int:
        """Thor: record the one-stream short-context graphs (MTP steps with greedy picks) on a free slot now; the
        bytes their captures and uploads took, which the memory gate's room is charged (the eager warm-ups before each
        capture take the decode path's first-use memory, which eager decoding also takes at its first requests)."""

        self._pin(1)
        if getattr(self, "fed", "0") != "0":
            return self.graphs.prerecord(self, mtp_fn=lambda segs: self._pick_fed(mtp_compute(self.w, segs, self.mbuf)))
        return self.graphs.prerecord(self, mtp_fn=lambda segs: self._pick_dev(mtp_compute(self.w, segs, self.mbuf), 1))

    def _busy(self) -> set[int]:
        return {id(s.st) for s in [*self.streams.values(), *self.filling]}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st: State, rows: int, *, alone: bool = False, protect: State | None = None) -> bool:
        """Grow caches while the gate has room; one stream may use its startup allowance within an explicit cap."""

        if rows <= st.capacity or st.capacity >= st.limit:       # admission's count keeps a stream within its window
            return True
        size = min(st.limit, -(-rows // STEP) * STEP)
        before = st.cache_bytes()
        grow = st.cache_bytes(size) - before
        while not self.memory_gate.fits(grow + st.layer_bytes(size)):     # a layer's old buffers stay until its copy
            if not self._evict_kept(st, protect=protect):
                if alone:
                    limit = cuda_limit_bytes() if torch.cuda.is_available() else None
                    if limit is not None:
                        # A lone stream may use its startup reserve without exceeding the copy peak cap.
                        peak = int(torch.cuda.memory_allocated()) + grow + st.layer_bytes(st.capacity)
                        if peak > limit:
                            if self.filling:
                                return False             # a filling request can finish and release its slot
                            raise NoRoom(
                                f"Growing this request's attention caches to {size} tokens would exceed "
                                f"{LIMIT_ENV}={limit / GIB:g}: the estimated copy peak is {peak / GIB:.2f} GiB. "
                                "Shorten the prompt or max_tokens, reduce --context or --parallel, or raise "
                                f"{LIMIT_ENV} if more GPU memory is available.", retry_after_s=MEMORY_RETRY_S)
                    break
                return False
        if self.memory_gate.pool is not None:    # Thor's wide pieces: kept ends go before the n-gram tables' pages
            while self.memory_gate.short(grow + st.layer_bytes(size)) and self._evict_kept(st, protect=protect):
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()         # freed blocks back to the driver, where the grow takes its pages
        try:
            added = st.resize(size)
        except Exception:
            self.memory_gate.take(st.cache_bytes() - before)
            raise
        self.memory_gate.take(added)
        if torch.cuda.is_available() and not getattr(self, "kv_pool", False):
            torch.cuda.empty_cache()             # the old buffers back to the system: MemAvailable stays true
        return True

    def _shrink(self, st: State, *, force: bool = False) -> None:
        """An idle slot back to its first rows: its caches' memory returns to the gate."""

        st.reset(self.w)
        if force or st.capacity > FIRST:
            self.memory_gate.give(-st.resize(min(FIRST, st.limit)))

    def _evict_kept(self, keep: State | None, *, protect: State | None = None, cause: str = "memory") -> bool:
        """Free the least recently used idle kept slot (never ``keep`` or ``protect``): its prompt ends go, its caches
        shrink and it joins the free list. Counted by cause ("memory": a grow's room, "slots": a request's slot,
        "entries": past ``keep`` points) for /v1/capacity, so a client can tell its next turn will run cold. Slots no
        request has used yet count as oldest, in the kept list's order. False when no idle kept slot is left."""

        busy = self._busy()
        used = getattr(self, "used", {})
        idle = [st for _, st, _, _ in self.kept if st is not keep and st is not protect and id(st) not in busy]
        if not idle:
            return False
        st = min(idle, key=lambda x: used.get(id(x), 0.0))
        self._drop_kept(st)
        self._shrink(st)
        if all(f is not st for f in self.free):
            self.free.append(st)
        gate = getattr(self, "memory_gate", None)        # some tests build a decoder with no gate at all
        if gate is not None:
            gate.evicted_kept = getattr(gate, "evicted_kept", 0) + 1
        by = getattr(self, "evicted_by", None)
        if by is not None:
            by[cause] = by.get(cause, 0) + 1
        return True

    def _make_room(self) -> list[Stream]:
        """Before a round: grow each live window oldest-first; a stream that can't grow makes the newest end."""

        live = sorted((s for s in self.streams.values() if not s.done), key=lambda s: s.sid)
        blocked, ended = False, []
        for s in live:
            rows = max(s.st.pos, s.st.mtp_len) + len(s.drafts) + self.depth + 2
            try:
                s.waiting = rows > s.st.capacity if blocked else not self._grow(s.st, rows, alone=len(live) == 1)
            except NoRoom as exc:
                s.error, s.done, s.waiting = exc, True, False
                self.held.pop(s.sid, None)
                self._drop_kept(s.st)
                self.memory_gate.ends += 1
                ended.append(s)                          # finish() releases only this request's slot
                continue
            blocked = blocked or s.waiting
        if live and live[0].waiting and len(live) > 1:        # even the oldest can't grow: the newest ends
            newest = live[-1]
            newest.error = RuntimeError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, after "
                f"{len(newest.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt or "
                "max_tokens, or start the server with a smaller --parallel.")
            newest.done, newest.waiting = True, False
            self.memory_gate.ends += 1
            self.streams.pop(newest.sid, None)
            self.held.pop(newest.sid, None)
            self._drop_kept(newest.st)
            self._shrink(newest.st)
            self.free.append(newest.st)
            return [*ended, newest, *self._make_room()]
        self.memory_gate.waits += any(s.waiting for s in live)
        return ended

    def _slot_for(self, prompt: list[int], reuse: bool):
        """Reuse the longest kept point, copying a fork into a free slot when the memory gate permits it."""

        return prefixes.slot_for(self, prompt, reuse)

    def _remember(self, ids: list[int], st: State, snap: dict, tail) -> None:
        prefixes.remember(self, ids, st, snap, tail)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def _snapshot_bytes(self) -> int:
        """One kept point's snapshot (``State.snapshot``: a DeltaNet state, the conv and n-gram tails)."""

        st = getattr(self, "_ref_state", None)
        if st is None:
            return 0
        return st.rec[0].numel() * st.rec.element_size() + st.conv.numel() * st.conv.element_size() + \
            st.ple_tail.numel() * st.ple_tail.element_size()

    def _worst(self, ref: State, prompt: int, count: int) -> int:
        """A request's worst case: its caches at the step its prompt and reply reach (and the drafts' rows), with one
        layer's old buffers during the last grow's copy."""

        rows = min(self.capacity, -(-(prompt + count + self.depth + 2) // STEP) * STEP)
        return ref.cache_bytes(rows) + ref.layer_bytes(rows)

    def room_tokens(self) -> int:
        """The most prompt plus reply tokens one request's caches hold within the room admission promises against."""

        ref, room = getattr(self, "_ref_state", None), self.memory_gate.admission_room()
        if ref is None:
            return self.capacity
        low, high = 0, self.capacity
        while low < high:
            mid = (low + high + 1) // 2
            if self._worst(ref, mid, 0) <= room:
                low = mid
            else:
                high = mid - 1
        return low

    def _too_long(self, prompt: int, count: int) -> str:
        n, room = self.room_tokens(), self.memory_gate.admission_room()
        return (f"{CONTEXT_LIMIT} {n} tokens for one request at this server's memory setting (its window is "
                f"{self.capacity} tokens; the caches' room is {room / GIB:.2f} GiB): this request's "
                f"{prompt} prompt tokens with up to {count} reply tokens do not fit it. Shorten the prompt or "
                "max_tokens.")

    def reserve_pool(self) -> str:
        """Thor KV pool (``thor_kv``): after the warm-ups, the gate's live room past the on-demand graphs' budget is
        allocated once into torch's cache, so later grows split it instead of taking the driver's pages; the room
        becomes that block (and what the caches already hold) less the kept snapshots' share and the image tower's
        workspace (its calls run beside the rounds, a video's as its frames decode). THOR-VISION-MEM: the workspace's
        host part (``vision_host``) is left out of the block, free for vision requests' host buffers, and only its GPU
        part comes off the room: the same room as the whole workspace in the block. The host's idle free memory is then
        landed on its design point, the host reserve plus those two budgets (``thor_kv.fit``). Returns the note."""

        from tensorfold.cuda.capacity import reserve_bytes, unified, _meminfo
        from . import thor_kv

        gate = self.memory_gate
        thor_kv.trim_host()                              # the load's freed host memory back first, then measure
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        graphs = getattr(self, "graphs", None)
        demand = int(graphs.budget) if graphs is not None and getattr(graphs, "demand", False) else 0
        live = int(gate.live()) if gate.live is not None else 0
        vhost = int(getattr(self, "vision_host", 0))     # THOR-VISION-MEM's host part stays out of the block
        memory = _meminfo()
        host = reserve_bytes(memory["MemTotal"]) if memory else 0
        read = _meminfo if memory is not None and unified(torch) else None
        size = thor_kv.reserve(live - demand - vhost)
        target = host + demand + vhost                   # the idle's design point: what the pool must leave free
        size, steps = thor_kv.fit(size, target, memory, read=read)
        self._pool_fit = (target, memory, read)          # ``refit_pool``'s design point, swap baseline and reader
        self._pool_set(size, steps, read is not None)
        n = self.room_tokens()
        return (f"every stream's caches from one {size / GIB:.2f} GiB pool reserved at load: {gate.room / GIB:.2f} GiB "
                f"for caches past {self.keep} kept snapshots and {self.vision_bytes / GIB:.2f} GiB of image tower "
                f"workspace ({n} tokens, {gate.room / max(1, self.window_bytes):.2f} full {self.capacity}-token "
                f"windows); {demand / GIB:.2f} GiB left for on-demand graphs, "
                + (f"{vhost / GIB:.2f} GiB for vision requests' host buffers (THOR-VISION-MEM) " if vhost else "")
                + f"and {host / GIB:.2f} GiB for the host (TENSORFOLD_MEMORY_RESERVE_GIB){self._idle_note(steps, target)}")

    def refit_pool(self) -> str | None:
        """THOR-KV-FIT, after the server's own init (the CLI builds its App past the engine: tokenizer, chat template):
        the idle read again and the pool re-reserved by the miss (``thor_kv.fit``, the same design point and swap
        baseline), so that init's host memory comes out of the pool, not out of the design point. Only before any
        request: an allocation since ``reserve_pool`` may hold a piece of the pool's block, and then nothing moves.
        None when there is nothing to refit (no fitted pool, ``TENSORFOLD_THOR_KV_POOL_FIT=0``). Returns the note."""

        from . import thor_kv

        fitted = getattr(self, "_pool_fit", None)
        if fitted is None or fitted[2] is None or not thor_kv.fit_on() or not getattr(self, "pool_fit", None):
            return None
        if torch.cuda.memory_allocated() > self._pool_allocated:
            return (f"the pool was not refit after the server's init: {torch.cuda.memory_allocated() - self._pool_allocated} "
                    "bytes were allocated after it was reserved")
        target, before, read = fitted
        torch.cuda.synchronize()
        size, steps = thor_kv.fit(self.pool_bytes, target, before, read=read)
        self._pool_set(size, steps, True)
        gate = self.memory_gate
        return (f"the pool refit after the server's init: every stream's caches from one {size / GIB:.2f} GiB pool: "
                f"{gate.room / GIB:.2f} GiB for caches ({self.room_tokens()} tokens){self._idle_note(steps, target)}")

    def _pool_set(self, size: int, steps: list[dict], fitted: bool) -> None:
        """The gate's room from a pool of ``size`` bytes, and the fit's record; ``fitted`` (a unified GPU's fit): what
        live tensors hold now, so ``refit_pool`` moves nothing once more is allocated."""

        from . import thor_kv

        gate = self.memory_gate
        snaps = self.keep * self._snapshot_bytes()
        gate.room = max(0, size + gate.held - snaps - self.vision_bytes)
        gate.reserve = thor_kv.SLACK
        self.pool_bytes = size
        self.pool_fit = steps
        self._pool_allocated = torch.cuda.memory_allocated() if fitted else 0

    @staticmethod
    def _idle_note(steps: list[dict], target: int) -> str:
        from . import thor_kv

        if not steps:
            return ""
        last, pools = steps[-1], ", ".join(f"{x['pool'] / GIB:.2f}" for x in steps)
        idle = (f"; idle free memory {last['idle'] / GIB:.2f} GiB, {(last['idle'] - target) / 2**20:+.0f} MiB from "
                f"its design point {target / GIB:.2f} GiB after {len(steps)} reading(s) (pools {pools} GiB"
                + (f", {last['swapped'] / 2**20:.0f} MiB in swap on disk counted as used" if last["swapped"] else "")
                + f"; {thor_kv.FIT_ENV}=0 sizes once)")
        if last["idle"] < target:
            idle += " -- BELOW the design point: this start cannot leave it free"
        return idle

    def kept_view(self) -> list[dict]:
        """The kept conversations, one per slot: tokens (its longest point), points, cache bytes, idle seconds and
        whether a live request holds it. Read without the engine's lock (as /health): a round old at worst."""

        now, used, busy = time.monotonic(), dict(getattr(self, "used", {})), self._busy()
        out: dict[int, dict] = {}
        for ids, st, _, _ in list(self.kept):
            e = out.get(id(st))
            if e is None:
                e = out[id(st)] = {"tokens": 0, "points": 0, "bytes": int(st.cache_bytes()),
                                   "idle_s": round(now - used[id(st)], 1) if id(st) in used else None,
                                   "busy": id(st) in busy}
            e["tokens"] = max(e["tokens"], len(ids))
            e["points"] += 1
        return sorted(out.values(), key=lambda e: -e["tokens"])

    def preview(self, prompt: list[int], count: int, *, max_streams: int | None = None) -> dict:
        """What admitting ``prompt`` now would do, without admitting it (/v1/capacity's POST): the kept prefix it would
        resume, its worst-case cache bytes, and whether it is admissible now or why not (``reason``: "length", it
        can never fit; "slots", every seat is live; "memory", the room is promised to live requests). A preview
        that finds its conversation's kept prefix counts as that conversation's use (the LRU clock): a client
        about to send its next turn moves to the back of the eviction queue."""

        out = {"prompt_tokens": len(prompt), "cached_tokens": 0, "max_tokens": None, "need_bytes": None,
               "admissible_now": False, "reason": None, "retry_after_s": None}
        room = self.capacity - len(prompt) - self.depth - 1
        if room < 1:
            out["reason"] = "length"
            return out
        count = max(1, min(int(count), room))
        out["max_tokens"] = count
        best = prefixes._best(list(self.kept), list(prompt))
        out["cached_tokens"] = len(best[0]) if best is not None else 0
        used = getattr(self, "used", None)
        if best is not None and used is not None:
            used[id(best[1])] = time.monotonic()
        ref, gate = getattr(self, "_ref_state", None), self.memory_gate
        need = self._worst(ref, len(prompt), count) if ref is not None else 0
        out["need_bytes"] = need
        if need > gate.admission_room():
            out["reason"] = "length"
        elif self.live() >= (max_streams or self.slots):
            out.update(reason="slots", retry_after_s=SLOTS_RETRY_S)
        elif not gate.admits(need) or any(x.waiting for x in list(self.streams.values())):
            out.update(reason="memory", retry_after_s=MEMORY_RETRY_S)
        else:
            out["admissible_now"] = True
        return out

    @torch.no_grad()
    def warm(self) -> None:
        """A synthetic greedy request through prefill (a full chunk, then a partial one cut at the kept point), its drafts and one round, then forgotten, so no request compiles or loads a kernel."""

        s = Stream([0] * min(self.prefill_rows + WARM_TAIL + 1, self.capacity - self.depth - 2), 2)
        self.admit(s)
        if not s.done:
            self.round()                                 # the whole prompt (nothing else decodes), then a round
        self.memory_gate.unpromise(id(s))                # what admit promised it (no finish() here: torn down by hand)
        self.held.pop(s.sid, None)
        self.streams.pop(s.sid, None)
        self._drop_kept(s.st)
        self._shrink(s.st)
        if all(f is not s.st for f in self.free):
            self.free.append(s.st)
        self._prerecord_experts()

    def _prerecord_experts(self) -> None:
        """Thor (sm_110): every decode window width's routed-expert graphs now, before the pool takes the room
        (``nvfp4_moe.prerecord_thor``): the verification buffers' for the model's layers, the drafts' for the MTP
        head's. Their memory is gone before ``reserve_pool`` measures what is left, so the room never counts it."""

        from .nvfp4_moe import prerecord_thor, thor_layers

        layers = thor_layers(self.w)
        if not layers:
            return
        head = getattr(getattr(self.w, "mtp", None), "layer", None)
        t0 = time.perf_counter()
        n, took = prerecord_thor(self.w, self.buf, [x for x in layers if x is not head])
        if self.mbuf is not None and head is not None and any(x is head for x in layers):
            m, more = prerecord_thor(self.w, self.mbuf, [head])
            n, took = n + m, took + more
        if n:
            print(f"[tensorfold] Thor (sm_110): {n} decode expert graphs (every window width, 1 to "
                  f"{self.buf.moe.thor.rows} rows) recorded at load in {time.perf_counter() - t0:.1f}s, "
                  f"{took / 2**20:.0f} MiB (TENSORFOLD_THOR_FI_MOE_PRERECORD=0: on first use)", flush=True)

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request in a free slot (a kept prompt end it extends, if any); rounds prefill its prompt."""

        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            # too long for ANY configuration, not a transient capacity refusal: OpenAI's context_length_exceeded (HTTP
            # 400), as the HTTP layer's own window check refuses it before submit, not NoRoom/503.
            raise ContextLengthError(f"{CONTEXT_LIMIT} {self.capacity - self.depth - 1} tokens: a prompt of "
                                     f"{len(s.prompt)} tokens leaves no room for a reply; shorten the prompt")
        s.count = max(1, min(s.count, room))
        if any(x.waiting for x in self.streams.values()):
            raise NoRoom("streams already wait for memory; a new request waits until one finishes", reason="memory",
                         retry_after_s=MEMORY_RETRY_S)
        # par-1-fix admission control: reserve this stream's full worst-case extent (its prompt plus its declared
        # max tokens, clamped to the window) before touching a slot, so a later request cannot squeeze a stream
        # already promised that room (see MemoryGate.reserved's docstring). Freed in finish()/drop(). ``_ref_state``
        # is set by __init__; a hand-built decoder in a test (MultiDecoder.__new__, __init__ skipped) does not have
        # it, and skips this reservation layer too, unchanged from before this feature existed.
        ref = getattr(self, "_ref_state", None)
        if ref is not None:
            worst_case = self._worst(ref, len(s.prompt), s.count)
            if worst_case > self.memory_gate.admission_room():   # past what admission ever promises: no wait helps
                raise ContextLengthError(self._too_long(len(s.prompt), s.count))
            if not self.memory_gate.admits(worst_case):
                raise NoRoom(
                    f"accepting a {len(s.prompt)}-token prompt (up to {s.count} reply tokens) would need more cache "
                    f"room than this server's budget allows with its other live and reserved sessions; retry "
                    "shortly, or after an in-flight session finishes", reason="memory", retry_after_s=MEMORY_RETRY_S)
            self.memory_gate.promise(id(s), worst_case)
        try:
            t0 = time.perf_counter()
            st, resume, s.cached = self._slot_for(list(s.prompt), s.draft and s.vision is None)
            used = getattr(self, "used", None)
            if used is not None:
                used[id(st)] = time.monotonic()
            try:
                if not self._grow(st, len(s.prompt) + self.depth + 2, alone=not self.streams and not self.filling):
                    raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes",
                                reason="memory", retry_after_s=MEMORY_RETRY_S)
            except NoRoom:
                if resume is None:
                    self.free.append(st)
                else:                                        # the kept prompt end stays kept
                    self._remember(list(s.prompt[:s.cached]), st, resume["state"], resume["tail"])
                raise
            # Thor NVMe tables: the first uncached rows read from here on, once no refusal can follow (a request held
            # for memory is admitted again every scheduler pass and would start its reads again each time)
            s.ahead = self._ahead(s)
            e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows)
            mtp = s.draft and self.depth > 0 and self.mbuf is not None
            try:
                begin = prefill_begin(e, s.prompt, mtp=mtp, resume=resume)
                image_rows.begin(e, s, self.vision)
            except Exception:
                self._drop_kept(st)
                self.free.append(st)
                raise
        except Exception:
            if ref is not None:                # the reservation above was only made when ref was not None either
                self.memory_gate.unpromise(id(s))
            raise
        e.stops = (thor_gdn.stops(self.points(s.prompt), begin + MIN_GAP, entry_end(s.prompt)) if thor_gdn.use() else
                   sorted({p for p in self.points(s.prompt) if begin + MIN_GAP <= p < entry_end(s.prompt)})) \
            if s.draft and st.image_positions is None and self.points is not None else []
        s.sid, s.st = self.next_id, st
        self.next_id += 1
        s.prefill_s = time.perf_counter() - t0
        same = resume is not None and self._keep_at(s) == begin         # the same prompt again: its own point
        self.fills[s.sid] = [e, mtp, begin, (resume["state"], resume["tail"]) if same else None]
        self.filling.append(s)

    def _ahead(self, s: Stream) -> list:
        """Thor NVMe n-gram tables: start reading the rows of the prompt's first
        ``arrival_rows`` uncached tokens at admission, so the reads overlap it; the pass's stage waits (``_arrivals``).
        A hint only: the stage's gather reads whatever is still missing, so the rows are the same either way."""

        out = []
        for layer in self.w.layers:
            p = layer.ple
            rows = getattr(getattr(p, "table", None), "arrival_rows", 0) if p is not None else 0
            start = min(s.cached or 0, len(s.prompt))
            if not rows or start >= len(s.prompt):
                continue
            lo, hi = max(0, start - (p.ngram.n - 1)), min(len(s.prompt), start + rows)
            try:
                ids = p.ngram.ids(p.ngram.initial_history(), np.asarray(s.prompt[lo:hi], dtype=np.int64))[start - lo:]
                out.append((p.table, p.table.ahead(ids)))
            except Exception:                        # noqa: BLE001  (a hint: never fails the request)
                continue
        return out

    @staticmethod
    def _arrivals(pieces) -> None:
        """Before a pass's stage: its prompts' arrival reads finished (what is left read now, at full width)."""

        for s, _, _ in pieces:
            for table, ticket in getattr(s, "ahead", None) or ():
                table.wait(ticket)
            s.ahead = None

    def _fill(self) -> list[Stream]:
        """Fill ordered prompt pieces until a stream decodes or a waiting request needs admission."""

        ended: list[Stream] = []
        while self.filling:
            ended += self._pass()
            if any(not x.done and not x.waiting for x in self.streams.values()) or self.arrived():
                break
        return ended

    def _pass_rows(self) -> int:
        """A round's prompt rows: its decode (a round alone) takes ``share`` of the pass's time, by the last rounds."""

        live = any(not s.done for s in self.streams.values())
        return pass_limit(self.prefill_rows, live, self.share, self.round_s, self.row_s, PASS_MIN)

    def _timed(self, seconds: float, rows: int) -> None:
        """A round's wall time: a round alone updates its estimate, a round with a pass the seconds a row adds."""

        if rows:
            extra = max(0.0, seconds - (self.round_s or 0.0)) / rows
            self.row_s = extra if self.row_s is None else 0.7 * self.row_s + 0.3 * extra
        else:
            self.round_s = seconds if self.round_s is None else 0.7 * self.round_s + 0.3 * seconds

    def _order(self) -> list[Stream]:
        """Order overdue prompts first, then foreground, fewest rows left and stable arrival order."""

        def key(s: Stream):
            passed = self.passed.get(s.sid, 0)
            due = passed >= FILL_GUARD
            return (not due, -passed if due else 0, s.background, len(s.prompt) - self.fills[s.sid][2])

        return sorted(self.filling, key=key)                         # stable: ties keep arrival order

    def _pieces(self, rows: int | None = None) -> list[tuple[Stream, int, int]]:
        """The next pass: rows from the filling prompts in ``_order``, up to ``rows`` and ENDS ending prompts."""

        pieces, room = [], self._pass_rows() if rows is None else rows
        for s in self._order():
            e, mtp, start, _ = self.fills[s.sid]
            n = min(next((p for p in e.stops if p > start), len(s.prompt)) - start, room)
            if thor_gdn.use():                  # Thor's chunked DeltaNet chain: a piece ends on its 64-row grid
                n = thor_gdn.piece_rows(start, n, len(s.prompt))
            ends = sum(1 for x, a, k in pieces if a + k == len(x.prompt))
            if n == 0 or (start + n == len(s.prompt) and ends == ENDS):
                break
            pieces.append((s, start, n))
            room -= n
        return pieces

    def _pass(self) -> list[Stream]:
        """One prompt pass alone; prompts that end sample their first token, draft and join the rounds."""

        pieces = self._pieces()
        self._note_passed(pieces)
        self._arrivals(pieces)
        t0 = time.perf_counter()
        try:
            segs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
            ends, cuts = self._end_rows(pieces, segs), self._cuts(pieces, segs)
            logits = compute(self.w, segs, self.pbuf, logits=bool(ends), ends=ends, cuts=cuts)
            heads = logits[:len(ends)].clone() if ends else None
            lasts = self._absorb(pieces, segs, cuts)
        except Exception as exc:                         # noqa: BLE001  (these requests fail, the others go on)
            return self._failed(pieces, exc)
        return self._joined(pieces, heads, lasts, (time.perf_counter() - t0) / len(pieces))

    def _note_passed(self, pieces) -> None:
        """Count a pass against every filling prompt it left out; one it took starts over."""

        took = {s.sid for s, _, _ in pieces}
        for s in self.filling:
            self.passed[s.sid] = 0 if s.sid in took else self.passed.get(s.sid, 0) + 1
        for sid in [k for k in self.passed if k not in {s.sid for s in self.filling}]:
            del self.passed[sid]

    @staticmethod
    def _end_rows(pieces, segs) -> list[int]:
        """The pass rows that end a prompt (each gets the head)."""

        return [a1 - 1 for (s, a, n), (_, _, a1) in zip(pieces, segs) if a + n == len(s.prompt)]

    @staticmethod
    def _keep_at(s: Stream) -> int | None:
        """Where a drafting stream's prompt state is kept: one token before its end, which a next turn extends."""

        return entry_end(s.prompt) if s.draft and s.st.image_positions is None else None

    def _point(self, s: Stream, start: int) -> int | None:
        """The next message-start or prompt-end snapshot this prompt piece can reach."""

        if not s.draft or s.st.image_positions is not None:
            return None
        return next((p for p in self.fills[s.sid][0].stops if p > start), self._keep_at(s))

    def _cuts(self, pieces, segs) -> list[Cut]:
        """The kept points strictly inside the pass's pieces, where their DeltaNet chains split."""

        return [Cut(k - a, at=a0) for (s, a, n), (_, a0, _) in zip(pieces, segs)
                if (k := self._point(s, a)) is not None and a < k < a + n]

    def _absorb(self, pieces, segs, cuts=()) -> list[torch.Tensor]:
        """After a pass's forward: each prompt's last row and kept point, the MTP head's absorb, the commits."""

        lasts = [self.pbuf.streams[a1 - 1:a1].clone() for _, _, a1 in segs]
        at, points = {cut.at: cut for cut in cuts}, []
        for (s, a, n), (st, a0, _) in zip(pieces, segs):      # before the MTP head writes the pass's streams
            k = self._point(s, a)
            if k is None or not a < k <= a + n:
                continue
            row, mtp = k - a, self.fills[s.sid][1]
            mtp_len = st.mtp_len + row - 1 if mtp else st.mtp_len       # every row but the point's last
            tail = self.pbuf.streams[a0 + row - 1:a0 + row].clone() if mtp else None
            cut = at.get(a0)
            snap = None if cut is None else cut_snapshot(self.w, st, self.pbuf, cut, mtp_len)
            points.append((s, mtp_len, tail, snap))
        absorb = [(s.st, s.prompt[a + 1:a + n + 1], self.pbuf.streams[a0:a0 + n])
                  for (s, a, n), (_, a0, _) in zip(pieces, segs) if self.fills[s.sid][1] and a + 1 < len(s.prompt)]
        if absorb:                   # the MTP head absorbs each prompt's rows (its cache in position order)
            absorb = [(st, nxt, streams[:len(nxt)]) for st, nxt, streams in absorb]
            mtp_compute(self.w, mtp_stage(self.w, self.pbuf, absorb), self.pbuf)
            for st, nxt, _ in absorb:
                st.set_mtp_len(st.mtp_len + len(nxt))
        for (s, a, n), (st, a0, _) in zip(pieces, segs):
            commit(self.w, st, self.pbuf, n, n, at=a0)
        for s, mtp_len, tail, snap in points:            # a point that ends its piece: the state as committed
            self.fills[s.sid][3] = (snap if snap is not None else {**s.st.snapshot(), "mtp_len": mtp_len}, tail)
        return lasts

    def _failed(self, pieces, exc: Exception) -> list[Stream]:
        failed = [s for s, _, _ in pieces]
        for s in failed:
            s.error, s.done = exc, True
            self.filling.remove(s)
            self.fills.pop(s.sid)
            self._drop_kept(s.st)
        return failed                                    # finish() frees their slots

    def _joined(self, pieces, heads, lasts, spent: float) -> list[Stream]:
        """Prompts that ended sample their first token, draft and join the rounds; returns those already done."""

        joined, head = [], 0
        for (s, a, n), last in zip(pieces, lasts):
            s.prefill_s += spent
            e, mtp, _, kept = self.fills[s.sid]
            self.fills[s.sid][2] = a + n
            if kept is not None and a < kept[0]["pos"] <= a + n:
                self._remember(list(s.prompt[:kept[0]["pos"]]), s.st, *kept)
            self.fills[s.sid][3] = None                    # only the bounded cache owns a stored snapshot
            if a + n < len(s.prompt):
                continue
            self.filling.remove(s)
            self.fills.pop(s.sid)
            st, e.last_streams = s.st, last
            logits = heads[head:head + 1]
            if s.constraint is not None:                 # a reply's grammar: the first token too
                logits = s.constraint.mask(logits, None, self.w.meta.get("vocab_offset", 0))
            first = e.sample(logits, [len(s.prompt)], s.sampling)[0]
            if s.probabilities is not None:
                capture(logits, [first], [len(s.prompt)], s.probabilities)
            if s.constraint is not None:
                s.constraint.advance([first])
            head += 1
            image_rows.finish(st)
            s.context = list(s.prompt)
            s.drafts = draft(e, last, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                             self.confidence) if mtp and s.count > 1 else []
            s.started = time.perf_counter()
            self.streams[s.sid] = s
            s.take([first], self._ends(s))
            if s.done:
                joined.append(s)
        return joined

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams, with the next prompt pass in the same forward while prompts fill."""

        ended = self._make_room()                      # every stream's caches hold this round, or the newest wait
        live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if self.filling and (not live or not self.converged):
            ended += self._fill()                      # passes alone; a prompt that ends here joins this round
            live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if not live:
            return ended
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None:
                tokens = [s.out[-1]] + list(s.drafts)
                try:
                    grammars[s.sid] = s.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                except GrammarError as exc:              # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    self.held.pop(s.sid, None)
                    failed.append(s)
                    continue
                s.drafts = grammars[s.sid].tokens[1:]
        live = [s for s in live if not s.done]
        if not live:
            return failed + ended
        t0 = time.perf_counter()
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        segs = stage(self.w, self.buf, windows)
        # a pass shares the round's forward only where their experts share a launch; else _fill ran it between rounds
        pieces, psegs = (self._pieces(self._pass_rows()) if self.filling and self.converged else []), None
        cuts = []
        if pieces:
            self._note_passed(pieces)
            self._arrivals(pieces)
            try:

                psegs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                cuts = self._cuts(pieces, psegs)
            except Exception as exc:                     # noqa: BLE001  (the pass's requests fail, the round goes on)
                ended += self._failed(pieces, exc)
                pieces = []
        held = [self.held.pop(s.sid, []) for s in live]
        graphs = None if pieces else self.graphs         # a round that shares its experts with a pass stays eager
        tables = self.buf.gdn_tables = gdn_multi.Tables(self.w, self.gdn, segs, held,
                                                        width=None if graphs is None else graphs.width)
        step = self.buf.attn_step = attn_multi.Step(self.w, segs, mtp=False)
        try:
            if pieces:                                 # the window and the pass: each layer's experts once for both
                pends = self._end_rows(pieces, psegs)
                logits, heads = compute_mixed(self.w, segs, self.buf, psegs, self.pbuf, ends=pends, cuts=cuts)
                heads = heads[:len(pends)].clone() if pends else None
            elif graphs is not None:
                logits = graphs.run("verify", segs, self.buf, lambda: compute(self.w, segs, self.buf), step=step,
                                    tables=tables)
            else:
                logits = compute(self.w, segs, self.buf)
        finally:
            self.buf.gdn_tables = self.buf.attn_step = None
        lasts = self._absorb(pieces, psegs, cuts) if pieces else None
        starts = [a0 for _, a0, _ in segs] + [segs[-1][2]]
        for s, (_, a0, a1) in zip(live, segs):
            if s.sid in grammars:
                s.constraint.mask(logits[a0:a1], grammars[s.sid])
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        thor_graphs.settle()                             # Thor switch: the host polls through the forward (DEC-1)
        sampled = sample_streams(logits, starts, positions, [s.sampling for s in live])
        paths = [accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self._ends(s))
                 for s, (_, tokens), rows in zip(live, windows, sampled)]
        for s, (_, tokens), (_, a0, _), (path, end), pos in zip(live, windows, segs, paths, positions):
            if s.probabilities is not None:
                capture(logits, [tokens[r] for r in path[1:]] + [end], [pos[r] for r in path],
                        s.probabilities, rows=[a0 + r for r in path])
        for s, rows in zip(live, gdn_multi.keep(tables, [len(path) for path, _ in paths])):
            self.held[s.sid] = rows                      # the next round's trees fold these rows in first
        kept = []
        for s, (_, tokens), (st, a0, a1), rows, (path, end) in zip(live, windows, segs, sampled, paths):
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0, states=False)
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = [tokens[r] for r in path[1:]] + [end]
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            last = s.error is not None or len(s.out) + len(new) >= s.count or end in self._ends(s)
            kept.append((s, a0, rows[:len(path)], new, last))
        drafting = [(s, a0, keep, new) for s, a0, keep, new, last in kept if s.draft and not last]
        self._draft_all([(s, a0, keep) for s, a0, keep, _ in drafting], {s.sid: new[-1] for s, _, _, new in drafting})
        for s, _, _, new, _ in kept:
            if s.error is not None:
                s.done = True
                continue
            s.take(new, self._ends(s))
        spent = time.perf_counter() - t0
        self._timed(spent, sum(n for _, _, n in pieces))
        if pieces:                                     # prompts that ended in this round's pass join the next
            ended += self._joined(pieces, heads, lasts, spent / len(pieces))
        done = [s for s in live if s.done]
        for s in done:                                   # a finished stream's state is never read again
            self.held.pop(s.sid, None)
        return failed + done + ended

    def _draft_all(self, streams: list, nexts: dict | None = None) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one step a depth; ``nexts``:
        each stream's next window's first token (Thor DEC-2 gathers the window's n-gram rows while the steps run).

        A stream's chain ends where its MTP rows would pass its caches (rcfix-1: a prompt that joins a round from its
        pass has the caches admission sized, prompt + depth + 2 rows, not ``_make_room``'s, while its chain can take up
        to prompt + 2 x depth): the drafts a chain that stays inside them takes are unchanged. A stream whose MTP step
        still finds no room fails alone (``MtpRoomError``), and every other stream's chain runs again from its start."""

        for s, _, _ in streams:
            s.drafts = []
        room = {s.sid: min(self.depth, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
        todo = [(s, a0, keep) for s, a0, keep in streams if room[s.sid] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, keep in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
            # the absorb ends at mtp_len + keep rows and each later step adds one: the last at mtp_len + keep + room - 1
            room[s.sid] = min(room[s.sid], st.capacity - st.mtp_len - len(keep) + 1)
        while todo:
            base = [s.st.mtp_len for s, _, _ in todo]
            try:
                return self._draft_chain(todo, room, nexts)
            except MtpRoomError as exc:                  # (a step raises before it stages: nothing ran for it)
                bad = [s for s, _, _ in todo if s.st is exc.state]
                if not bad:
                    raise
                for (s, _, _), n in zip(todo, base):
                    s.drafts = []
                    s.st.set_mtp_len(n)
                    s.st.mtp_drafted = 0
                for s in bad:                            # this request ends with the error, the others draft again
                    s.error = exc
                    self._drop_kept(s.st)
                todo = [t for t in todo if t[0].st is not exc.state]

    def _draft_chain(self, todo: list, room: dict, nexts: dict | None) -> None:
        """The draft chain for ``todo``, every stream's ``room`` inside its caches: the device-fed one (Thor, greedy
        streams) or the host-fed one."""

        fed = getattr(self, "fed", "0")                  # (a decoder built without __init__: the host-fed chain)
        if fed != "0" and all(s.sampling is None or s.sampling.temperature <= 0 for s, _, _ in todo) \
                and all(s.st.mtp_len + len(keep) + room[s.sid] <= s.st.capacity for s, _, keep in todo):
            if self.fed_check is not None:
                return self._draft_check(todo, room, nexts)
            return self._draft_fed(todo, room, nexts, fed != "sync")
        self._draft_host(todo, room, nexts)

    def _draft_host(self, todo: list, room: dict, nexts: dict | None, record: dict | None = None) -> None:
        """The host-fed chain (ae716c1's): each step launched after the host read the step before it; Thor
        ``TENSORFOLD_THOR_MTP_FED=0``, a round with a sampled stream, or a cache near its capacity. ``record``:
        ``thor_mtp_fed.chain``'s."""

        windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
        segs = mtp_stage(self.w, self.mbuf, windows)
        # Thor graphs: each MTP step's graph also holds its picks' reductions and their copy to pinned host memory
        k = self._k([s.sampling for s, _, _ in todo]) if self.graphs is not None else None
        logits = self._mtp(segs, k)
        ahead = bool(nexts) and thor_dec2.ahead_on(self.w)
        if ahead:                                        # Thor DEC-2: while the step runs, the next windows' first rows
            thor_dec2.ple_ahead(self.w, [(s.st, nexts[s.sid]) for s, _, _ in todo if s.sid in nexts], start=True)
        for (s, _, keep), (st, a0, a1) in zip(todo, segs):
            st.set_mtp_len(st.mtp_len + len(keep))
        active = [(s, a1 - 1) for s, (_, _, a1) in zip([t[0] for t in todo], segs)]
        for j in range(self.depth):
            picks = self._picks(logits, [s.st.pos + 1 + j for s, _ in active], [s.sampling for s, _ in active], k)
            nxt, took = [], []
            for (s, row), (d, p) in zip(active, picks):
                if record is not None:
                    record.setdefault(s.sid, []).append((d, p))
                low = self.confidence > 0 and p < self.confidence
                if low and j > 0:
                    continue
                s.drafts.append(d)
                took.append((s.st, d))
                if not low and j + 1 < room[s.sid]:
                    nxt.append((s, row, d))
            if not nxt:
                return
            windows = [(s.st, [d], self.mbuf.streams[row:row + 1]) for s, row, d in nxt]
            segs = mtp_stage(self.w, self.mbuf, windows)
            k = self._k([s.sampling for s, _, _ in nxt]) if self.graphs is not None else None
            logits = self._mtp(segs, k)
            if ahead:                                    # the drafts just picked: their rows while this step runs
                thor_dec2.ple_ahead(self.w, took, start=False)
            for s, _, _ in nxt:
                s.st.set_mtp_len(s.st.mtp_len + 1)
                s.st.mtp_drafted += 1
            active = [(s, a0) for (s, _, _), (_, a0, _) in zip(nxt, segs)]

    def _draft_fed(self, todo: list, room: dict, nexts: dict | None, spec: bool, record: dict | None = None) -> None:
        """Thor DPUSH-6 (``thor_mtp_fed``): the device-fed chain, greedy streams; ``spec``: each step launched for the
        streams with room before the host reads the step before it."""

        w = self.w
        segs = mtp_stage(w, self.mbuf, [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo])
        self._fed_step(segs, 0)
        ahead = bool(nexts) and thor_dec2.ahead_on(w)
        if ahead:                                        # Thor DEC-2: while the step runs, the next windows' first rows
            thor_dec2.ple_ahead(w, [(s.st, nexts[s.sid]) for s, _, _ in todo if s.sid in nexts], start=True)
        for (s, _, keep), (st, a0, a1) in zip(todo, segs):
            st.set_mtp_len(st.mtp_len + len(keep))
        live = [(s, i, a1 - 1) for i, ((s, _, _), (_, _, a1)) in enumerate(zip(todo, segs))]
        after = (lambda took: thor_dec2.ple_ahead(w, [(s.st, d) for s, d in took], start=False)) if ahead else None
        thor_mtp_fed.chain(live, room, self.depth, self.confidence, self._fed_launch, self._fed_read, self._fed_revert,
                           spec, after=after, record=record)

    def _draft_check(self, todo: list, room: dict, nexts: dict | None) -> None:
        """``TENSORFOLD_THOR_MTP_FED=check``: the speculative, ``sync`` and host-fed chains on the same inputs (each
        from the same lengths); the host-fed one is kept."""

        base = [(s.st.mtp_len, s.st.mtp_drafted) for s, _, _ in todo]
        runs, lens = {}, {}
        for how in ("spec", "sync", "host"):
            for (s, _, _), (n, d) in zip(todo, base):
                s.drafts = []
                s.st.set_mtp_len(n)
                s.st.mtp_drafted = d
            rec: dict = {}
            if how == "host":
                self._draft_host(todo, room, nexts, record=rec)
            else:
                self._draft_fed(todo, room, None, how == "spec", record=rec)
            runs[how] = {s.sid: (list(s.drafts), rec.get(s.sid, [])) for s, _, _ in todo}
            lens[how] = {s.sid: (s.st.mtp_len, s.st.mtp_drafted) for s, _, _ in todo}
        self.fed_check.add(runs, lens)

    def _fed_step(self, segs: list, par: int) -> None:
        """An MTP step whose picks kernel writes each row's draft id (``fed_tok``, the next step's ids) and picks
        (``fed_got``), then queues the picks to ``fed_pins[par]`` (``fed_ev[par]``: copied)."""

        mb = self.mbuf
        step = mb.attn_step = attn_multi.Step(self.w, segs, mtp=True)
        try:
            if self.graphs is not None:
                self.graphs.run("mtp", segs, mb, lambda: self._pick_fed(mtp_compute(self.w, segs, mb)), step=step,
                                extra=self.picks_key)
            else:
                self._pick_fed(mtp_compute(self.w, segs, mb))
        finally:
            mb.attn_step = None
        n = len(segs)
        self.fed_pins[par][:n].copy_(self.fed_got[:n], non_blocking=True)
        self.fed_ev[par].record()

    def _pick_fed(self, logits: torch.Tensor) -> torch.Tensor:
        return thor_mtp_fed.picks(logits, self.fed_got, self.fed_tok, self.fed_map)

    def _fed_launch(self, cands: list, par: int) -> list:
        """An MTP step for ``cands`` [(stream, picks row, streams row)] of the step before it, staged on the device
        (that step's draft ids and output streams), launched into ``par``; the streams' MTP lengths bumped."""

        mb, n = self.mbuf, len(cands)
        segs = []
        for i, (s, _, _) in enumerate(cands):
            if s.st.mtp_len + 1 > s.st.capacity:
                raise MtpRoomError(s.st)
            segs.append((s.st, i, i + 1))
        src, rows = [i for _, i, _ in cands], [r for _, _, r in cands]
        if src == rows == list(range(n)):
            si = ri = self.fed_ar[:n]
        else:
            self.fed_staged.synchronize()                # the last staging's copy has read the pinned rows
            h = self.fed_host.numpy()
            h[:n], h[n:2 * n] = src, rows
            self.fed_dev[:2 * n].copy_(self.fed_host[:2 * n], non_blocking=True)
            self.fed_staged.record()
            si, ri = self.fed_dev[:n], self.fed_dev[n:2 * n]
        torch.index_select(self.fed_tok, 0, si, out=mb.ids[:n])
        torch.index_select(mb.streams, 0, ri, out=mb.mtp_in[:n])
        mb.last[:n].copy_(self.fed_ar[:n])
        self._fed_step(segs, par)
        for s, _, _ in cands:
            s.st.set_mtp_len(s.st.mtp_len + 1)
            s.st.mtp_drafted += 1
        return [s for s, _, _ in cands]

    def _fed_read(self, par: int, rows: int) -> list[tuple[int, float]]:
        """A launched step's picks: its draft and probability at temperature 1 a row (``_picks``' greedy math)."""

        ev = self.fed_ev[par]
        if thor_graphs.spinning() == "poll":
            while not ev.query():
                pass
        else:
            ev.synchronize()
        out = []
        for top, col, lse in self.fed_pins[par][:rows].numpy().tolist():
            c = int(col)
            out.append((int(self.draft_host[c]) if self.draft_host is not None else c, float(np.exp(top - lse))))
        return out

    @staticmethod
    def _fed_revert(s) -> None:
        s.st.set_mtp_len(s.st.mtp_len - 1)
        s.st.mtp_drafted -= 1

    def _mtp(self, segs: list, k: int | None = None) -> torch.Tensor:
        """An MTP step over every drafting stream, its attention one launch a kernel for all of them; with ``k`` (Thor
        graphs) the step also reduces its picks ``k`` wide and queues them to pinned host memory (``_picks`` reads)."""

        step = self.mbuf.attn_step = attn_multi.Step(self.w, segs, mtp=True)
        try:
            if self.graphs is not None:
                if k is not None:
                    self._pin(k)
                    return self.graphs.run("mtp", segs, self.mbuf,
                                           lambda: self._pick_dev(mtp_compute(self.w, segs, self.mbuf), k), step=step,
                                           extra=(k,))
                return self.graphs.run("mtp", segs, self.mbuf, lambda: mtp_compute(self.w, segs, self.mbuf), step=step)
            return mtp_compute(self.w, segs, self.mbuf)
        finally:
            self.mbuf.attn_step = None

    def _k(self, samplings: list) -> int:
        """The picks' top-k width for these streams (one for greedy streams)."""

        k = max([int(s.top_k) + MARGIN for s in samplings if s is not None and s.temperature > 0 and s.top_k] or [1])
        return min(k, self.draft_n)

    def _pin(self, k: int) -> None:
        if k not in self.pins:                           # before any capture that writes it; never reallocated
            self.pins[k] = torch.empty((self.slots, 2 * k + 3), dtype=torch.float32, pin_memory=True)

    @staticmethod
    def _reduce(logits: torch.Tensor, k: int) -> torch.Tensor:
        row = logits.float()
        vals, idx = torch.topk(row, k, dim=-1, sorted=False)
        top, col = row.max(dim=-1, keepdim=True)
        lse = torch.logsumexp(row, dim=-1, keepdim=True)
        return torch.cat([vals, idx.float(), top, col.float(), lse], dim=1)

    def _pick_dev(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """The picks' reductions and their copy to pinned host memory, queued (a graph may hold both)."""

        got = self._reduce(logits, k)
        self.pins[k][:got.shape[0]].copy_(got, non_blocking=True)
        return got

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list,
               k: int | None = None) -> list[tuple[int, float]]:
        """Each row's keyed draft and its probability at temperature 1, one read-back (drafts change speed only);
        with ``k`` the MTP step already queued the reductions to ``pins[k]``."""

        if k is None:
            k = self._k(samplings)
            got = self._reduce(logits, k)
            thor_graphs.settle()
            got = got.cpu().numpy()
        else:
            thor_graphs.settle()
            torch.cuda.current_stream().synchronize()
            got = self.pins[k][:len(positions)].numpy()
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            g = got[i]
            lse_i = float(g[2 * k + 2])
            if smp is None or smp.temperature <= 0:
                c = int(g[2 * k + 1])
                out.append((int(self.draft_host[c]) if self.draft_host is not None else c,
                            float(np.exp(float(g[2 * k]) - lse_i))))
                continue
            cols = g[k:2 * k].astype(np.int64)
            ids = self.draft_host[cols] if self.draft_host is not None else cols
            tok = choose_rows(g[None, :k].astype(np.float32), ids[None, :], [pos], smp)[0]
            hit = np.nonzero(ids == tok)[0]
            out.append((int(tok), float(np.exp(float(g[hit[0]]) - lse_i)) if len(hit) else 0.0))
        return out

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt state is kept stays with it, the rest are free again."""

        gate = getattr(self, "memory_gate", None)
        used = getattr(self, "used", None)
        held = getattr(self, "held", None)
        for s in done:
            if gate is not None:
                gate.unpromise(id(s))      # par-1-fix: its admission reservation, freed once and for all
            if held is not None:
                held.pop(s.sid, None)      # its kept DeltaNet rows (a lane yield finishes one that is not done)
            self.streams.pop(s.sid, None)
            if any(k[1] is s.st for k in self.kept):
                if used is not None:
                    used[id(s.st)] = time.monotonic()        # its conversation's last use: the LRU clock
            elif all(f is not s.st for f in self.free):
                self._shrink(s.st)
                self.free.append(s.st)

    def drop(self) -> list[Stream]:
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.filling, self.fills = [], {}
        gate = getattr(self, "memory_gate", None)
        for s in live:
            if gate is not None:
                gate.unpromise(id(s))      # par-1-fix: as finish(), for the streams a dead server drops
            self.streams.pop(s.sid, None)
            self.held.pop(s.sid, None)
            self._drop_kept(s.st)
            self._shrink(s.st)
            self.free.append(s.st)
        return live


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))
