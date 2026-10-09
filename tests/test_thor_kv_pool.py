"""Thor's serving memory for concurrent Flash Next (FR-OPS-17 par-1-fix): kept conversations go least recently used
first and are counted by cause, a request past the whole room is refused as too long instead of waiting forever, the
capacity preview says what a request would get before it is sent, and the Thor KV pool's room arithmetic. CPU only."""

import importlib
import time
from types import SimpleNamespace

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)
from tests.test_cuda_growing_caches import decoder, weights

pytestmark = pytest.mark.torch
GIB = 1 << 30


def mods():
    import torch

    return (importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi"),
            importlib.import_module("tensorfold.families.qwen4_exp.cuda.state"), torch)


def kept_slots(dec, state, n, *, used=None, rows=300):
    """``n`` idle kept slots holding ``rows`` committed rows each, kept in list order 0..n-1; ``used`` their LRU clock."""

    slots = []
    for i in range(n):
        st = state.State(weights(), 256, 4, "bf16", limit=65536)
        dec.memory_gate.take(st.ensure(rows))
        dec.kept.append(([i + 1] * (10 + i), st, {"pos": 10 + i, "mtp_len": 9 + i}, None))
        slots.append(st)
    dec.used = {id(st): t for st, t in zip(slots, used)} if used is not None else {}
    dec.evicted_by = {"memory": 0, "slots": 0, "entries": 0}
    return slots


def test_memory_eviction_takes_the_least_recently_used_slot_not_the_oldest_entry(allocations):  # noqa: F811
    multi, state, torch = mods()
    dec = decoder(multi, state, torch, room=1 << 40)
    a, b, c = kept_slots(dec, state, 3, used=[30.0, 10.0, 20.0])      # b used longest ago, though kept second
    assert dec._evict_kept(None) and [k[1] for k in dec.kept] == [a, c]
    assert b.capacity == 256 and b in dec.free
    assert dec.evicted_by == {"memory": 1, "slots": 0, "entries": 0} and dec.memory_gate.evicted_kept == 1
    assert dec._evict_kept(None, cause="slots") and [k[1] for k in dec.kept] == [a]      # then c (20 < 30)
    assert dec.evicted_by["slots"] == 1


def test_unused_slots_go_in_kept_order_and_protected_or_busy_ones_never(allocations):  # noqa: F811
    multi, state, torch = mods()
    dec = decoder(multi, state, torch, room=1 << 40)
    a, b, c = kept_slots(dec, state, 3)                                # no clock yet: the kept list's order
    dec.streams = {7: SimpleNamespace(st=a, waiting=False)}            # a is busy
    assert dec._evict_kept(None, protect=b) and [k[1] for k in dec.kept] == [a, b]
    assert not dec._evict_kept(None, protect=b)                        # a busy, b protected: nothing left


def test_a_new_request_without_a_free_slot_takes_the_least_recently_used_idle_one(allocations):  # noqa: F811
    multi, state, torch = mods()
    dec = decoder(multi, state, torch, room=1 << 40)
    dec.capacity = 65536
    a, b = kept_slots(dec, state, 2, used=[5.0, 1.0])
    st, resume, cached = dec._slot_for([99] * 50, True)
    assert st is b and resume is None and cached == 0
    assert [k[1] for k in dec.kept] == [a] and dec.evicted_by["slots"] == 1


def test_a_shared_point_forks_into_a_taken_slot_and_never_consumes_the_longer_chain(allocations):  # noqa: F811
    multi, state, torch = mods()
    dec = decoder(multi, state, torch, room=1 << 40)
    dec.capacity = 65536
    shared, other = kept_slots(dec, state, 2, used=[9.0, 1.0])
    system = [7] * 8                                                   # shared's chain: a system point, then more
    dec.kept[0] = (system, shared, {"pos": 8, "mtp_len": 7}, None)
    dec.kept.append((system + [1, 2, 3], shared, {"pos": 11, "mtp_len": 10}, None))
    copies = []
    shared_copy = state.State.copy_prefix
    state.State.copy_prefix = lambda self, src, pos, mtp: copies.append((self, src, pos, mtp))
    try:
        st, resume, cached = dec._slot_for(system + [5, 5], True)
    finally:
        state.State.copy_prefix = shared_copy
    assert st is other and cached == 8 and copies == [(other, shared, 8, 7)]
    assert [len(k[0]) for k in dec.kept if k[1] is shared] == [8, 11]  # the longer chain stays kept
    assert dec.evicted_by["slots"] == 1


def test_past_keep_points_inner_points_go_before_any_conversation(allocations):  # noqa: F811
    multi, state, torch = mods()
    prefixes = importlib.import_module("tensorfold.families.qwen4_exp.cuda.prefixes")
    dec = decoder(multi, state, torch, room=1 << 40)
    dec.keep = 3
    a, b = kept_slots(dec, state, 2, used=[1.0, 2.0])                  # a: [1]*10, b: [2]*11
    dec.kept.append(([1] * 20, a, {}, None))                           # a's chain goes on: [1]*10 is inner now
    prefixes.remember(dec, [2] * 30, b, {}, None)                      # 4 points: b's [2]*11 inner (a's older first)
    assert sorted((id(k[1]) == id(a), len(k[0])) for k in dec.kept) == [(False, 11), (False, 30), (True, 20)]
    assert dec.evicted_by == {"memory": 0, "slots": 0, "entries": 0}


def test_a_request_past_the_whole_room_is_refused_as_too_long_not_held(allocations):  # noqa: F811
    multi, state, torch = mods()
    errors = importlib.import_module("tensorfold.server.errors")
    dec = decoder(multi, state, torch, room=0)
    dec.capacity, dec.slots = 65536, 8
    ref = dec._ref_state = state.State(weights(), 256, 4, "bf16", limit=65536)
    dec.memory_gate.room = dec._worst(ref, 20000, 0)                   # 24,576 rows' caches and one layer's copy
    assert dec.room_tokens() == 24576 - dec.depth - 2
    with pytest.raises(errors.ContextLengthError, match=r"maximum context length is 24571 tokens") as exc:
        dec.admit(SimpleNamespace(prompt=[1] * 20000, count=8000, draft=False, vision=None))
    assert "Shorten the prompt or max_tokens" in str(exc.value)
    assert dec.memory_gate.reserved_total == 0


def test_preview_names_why_a_request_would_not_start_now(allocations):  # noqa: F811
    multi, state, torch = mods()
    dec = decoder(multi, state, torch, room=0)
    dec.capacity, dec.slots = 65536, 8
    ref = dec._ref_state = state.State(weights(), 256, 4, "bf16", limit=65536)
    dec.memory_gate.room = dec._worst(ref, 30000, 0)
    slots = kept_slots(dec, state, 1)
    prompt = [1] * 10 + [4] * 90
    ok = dec.preview(prompt, 50)
    assert ok["admissible_now"] and ok["cached_tokens"] == 10 and ok["prompt_tokens"] == 100 and ok["reason"] is None
    assert ok["need_bytes"] == dec._worst(ref, 100, 50) == ref.cache_bytes(8192) + ref.layer_bytes(8192)
    assert ok["max_tokens"] == 50
    assert dec.preview([1] * 65534, 50)["reason"] == "length"                     # the window
    assert dec.preview([1] * 25000, 9000)["reason"] == "length"                   # the whole room
    dec.memory_gate.promise(1, dec.memory_gate.room - ok["need_bytes"] + 1)
    assert dec.preview(prompt, 50)["reason"] == "memory"
    dec.memory_gate.unpromise(1)
    dec.streams = {i: SimpleNamespace(st=slots[0], waiting=False) for i in range(8)}
    assert dec.preview(prompt, 50, max_streams=8)["reason"] == "slots"


def test_kept_view_is_one_entry_per_conversation_with_its_idle_time(allocations):  # noqa: F811
    multi, state, torch = mods()
    dec = decoder(multi, state, torch, room=1 << 40)
    now = time.monotonic()
    a, b = kept_slots(dec, state, 2, used=[now - 30, now - 5])
    dec.kept.append(([1] * 40, a, {}, None))
    view = dec.kept_view()
    assert [(e["tokens"], e["points"]) for e in view] == [(40, 2), (11, 1)]
    assert 29 <= view[0]["idle_s"] <= 31 and view[0]["bytes"] == a.cache_bytes() and view[0]["busy"] is False


def test_finishing_a_kept_conversation_moves_it_to_the_back_of_the_queue(allocations):  # noqa: F811
    multi, state, torch = mods()
    dec = decoder(multi, state, torch, room=1 << 40)
    a, b = kept_slots(dec, state, 2, used=[1.0, 2.0])
    dec.finish([SimpleNamespace(sid=3, st=a)])
    assert dec.used[id(a)] > 2.0
    assert dec._evict_kept(None) and [k[1] for k in dec.kept] == [a]


def test_the_pool_is_the_live_room_past_graphs_and_snapshots(allocations, monkeypatch):  # noqa: F811
    multi, state, torch = mods()
    thor_kv = importlib.import_module("tensorfold.families.qwen4_exp.cuda.thor_kv")
    dec = decoder(multi, state, torch, room=0)
    dec.capacity, dec.keep, dec.window_bytes, dec.vision_bytes = 65536, 8, 10 * GIB, GIB // 2
    dec._ref_state = state.State(weights(), 256, 4, "bf16", limit=65536)
    dec.graphs = SimpleNamespace(budget=GIB, demand=True)
    dec.memory_gate.live = lambda: 12 * GIB
    asked = []
    monkeypatch.setattr(thor_kv, "reserve", lambda size: asked.append(size) or size - size % (2 << 20))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    # not a unified GPU: the room arithmetic alone, no fit against this machine's memory (on a CUDA machine the fake
    # triton fixture breaks torch's lazy init, and a fit would measure the real host)
    monkeypatch.setattr(importlib.import_module("tensorfold.cuda.capacity"), "unified", lambda torch: False)
    note = dec.reserve_pool()
    snap = dec._snapshot_bytes()
    assert asked == [11 * GIB] and dec.pool_bytes == 11 * GIB
    assert dec.memory_gate.room == 11 * GIB - 8 * snap - GIB // 2 and dec.memory_gate.reserve == thor_kv.SLACK
    assert "11.00 GiB pool" in note and "1.00 GiB left for on-demand graphs" in note and "0.50 GiB of image" in note


def _fit(monkeypatch, host, *, before_swap=0, before_cached=0):
    """thor_kv.fit against a fake host: ``host(pool)`` -> (MemAvailable, SwapFree[, SwapCached]) once a pool of that size
    is held."""

    import torch

    thor_kv = importlib.import_module("tensorfold.families.qwen4_exp.cuda.thor_kv")
    held, asked = {"pool": 0}, []

    def reserve(size):
        size = max(0, int(size)) // thor_kv.ALIGN * thor_kv.ALIGN
        asked.append(size)
        held["pool"] = size
        return size

    monkeypatch.setattr(thor_kv, "reserve", reserve)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def read():
        avail, swap, *cached = host(held["pool"])
        return {"MemAvailable": avail, "SwapFree": swap, "SwapCached": cached[0] if cached else 0}

    def run(size, target):
        before = {"MemAvailable": 0, "SwapFree": before_swap, "SwapCached": before_cached}
        return thor_kv.fit(reserve(size), target, before, read=read)

    return thor_kv, asked, run


def test_the_pool_is_refit_until_the_idle_lands_on_its_design_point(monkeypatch):
    # the reservation took 1 GiB more than MemAvailable foretold (vmem-a2's on start): the pool comes down by it, the
    # re-reservation aiming FIT_AIM above the design point
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (40 * GIB - pool - GIB, 0))
    target, aim = 4 * GIB, thor_kv.FIT_AIM
    size, steps = run(36 * GIB, target)
    assert asked == [36 * GIB, 35 * GIB - aim] and size == 35 * GIB - aim
    assert [s["idle"] for s in steps] == [3 * GIB, target + aim] and steps[-1]["swapped"] == 0
    # a surplus (the reservation took less): the pool grows by it
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (40 * GIB - pool + GIB // 2, 0))
    size, steps = run(36 * GIB, target)
    assert asked == [36 * GIB, 36 * GIB + GIB // 2 - aim] and steps[-1]["idle"] == target + aim


def test_swap_the_reservation_filled_is_not_free_memory(monkeypatch):
    # vmem-a2's off start: the reservation swapped 1.5 GiB of the startup heap out; MemAvailable looks 1.5 GiB richer
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (40 * GIB - pool + 3 * GIB // 2, GIB // 2), before_swap=2 * GIB)
    size, steps = run(36 * GIB, 4 * GIB)
    assert asked == [36 * GIB] and steps == [{"pool": 36 * GIB, "available": 4 * GIB + 3 * GIB // 2,
                                              "swapped": 3 * GIB // 2, "idle": 4 * GIB}]


def test_swapped_pages_still_in_ram_are_not_counted_twice(monkeypatch):
    # vmem-a3-c: the reservation filled 1600 MiB of swap and 750 MiB of those pages stayed in RAM as swap cache, which
    # MemAvailable already leaves out: only the 850 MiB on disk count against the idle
    MIB = 1 << 20
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (40 * GIB - pool + 850 * MIB, 2 * GIB - 1600 * MIB, 760 * MIB),
                               before_swap=2 * GIB, before_cached=10 * MIB)
    size, steps = run(36 * GIB, 4 * GIB)
    assert asked == [36 * GIB] and steps[0]["swapped"] == 850 * MIB and steps[0]["idle"] == 4 * GIB
    # a swap cache that shrank is no credit: the whole SwapFree drop counts
    before, now = {"SwapFree": 2 * GIB, "SwapCached": 100 * MIB}, {"SwapFree": GIB, "SwapCached": 50 * MIB}
    assert thor_kv.swapped(before, now) == GIB
    assert thor_kv.swapped({"SwapFree": GIB}, {"SwapFree": 2 * GIB, "SwapCached": 5 * MIB}) == 0


def test_the_idle_lands_on_or_just_above_its_design_point_and_the_switch_measures_once(monkeypatch):
    MIB = 1 << 20
    for over in (0, 32 * MIB, 64 * MIB):                    # anywhere in [target, target + FIT_TOL]: kept
        thor_kv, asked, run = _fit(monkeypatch, lambda pool, over=over: (40 * GIB - pool + over, 0))
        assert run(36 * GIB, 40 * GIB - 36 * GIB)[0] == 36 * GIB and asked == [36 * GIB]
    assert thor_kv.FIT_TOL == 64 * MIB
    # 2 MiB under the design point is re-reserved (a symmetric window let the refit's App init stay under it)
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (40 * GIB - pool - 2 * MIB, 0))
    size, steps = run(36 * GIB, 4 * GIB)
    assert asked == [36 * GIB, 36 * GIB - 2 * MIB - thor_kv.FIT_AIM] and steps[-1]["idle"] == 4 * GIB + thor_kv.FIT_AIM
    # past FIT_TOL above: the pool grows
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (40 * GIB - pool + 66 * MIB, 0))
    assert run(36 * GIB, 4 * GIB)[0] == 36 * GIB + 66 * MIB - thor_kv.FIT_AIM
    monkeypatch.setenv(thor_kv.FIT_ENV, "0")
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (40 * GIB - pool - GIB, 0))
    size, steps = run(36 * GIB, 4 * GIB)
    assert size == 36 * GIB and asked == [36 * GIB] and steps[0]["idle"] == 3 * GIB and len(steps) == 1
    # never more than FIT_TRIES re-reservations, whatever the host does
    monkeypatch.delenv(thor_kv.FIT_ENV)
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (GIB, 0))
    run(36 * GIB, 4 * GIB)
    assert len(asked) == 1 + thor_kv.FIT_TRIES
    # not a unified GPU (read None): nothing measured, the pool kept
    assert thor_kv.fit(8 * GIB, 4 * GIB, {"SwapFree": 0}, read=None) == (8 * GIB, [])


def test_a_failed_meminfo_read_keeps_the_last_good_one_or_the_pool(monkeypatch):
    """GLM review LOW 1: ``capacity._meminfo`` returns None on a failed read; the settle loop must not index it."""
    thor_kv, asked, run = _fit(monkeypatch, lambda pool: (40 * GIB - pool, 0))
    reads = iter([{"MemAvailable": 3 * GIB, "SwapFree": 0}, None])
    assert thor_kv._settled(lambda: next(reads)) == {"MemAvailable": 3 * GIB, "SwapFree": 0}
    assert thor_kv._settled(lambda: None) is None
    reads = iter([{"MemAvailable": 3 * GIB, "SwapFree": 0}, None, None, None, None, None, None, None, None])
    size, steps = thor_kv.fit(36 * GIB, 4 * GIB, {"SwapFree": 0}, read=lambda: next(reads))
    assert [s["idle"] for s in steps] == [3 * GIB] and asked == [36 * GIB - GIB - thor_kv.FIT_AIM]
    asked.clear()
    assert thor_kv.fit(36 * GIB, 4 * GIB, {"SwapFree": 0}, read=lambda: None) == (36 * GIB, []) and asked == []


def _pooled(monkeypatch, host, *, allocated=lambda: 0):
    """A pooled decoder on a fake unified host: ``host(pool)`` -> MemAvailable once a pool of that size is held."""
    import torch

    multi, state, _ = mods()
    thor_kv = importlib.import_module("tensorfold.families.qwen4_exp.cuda.thor_kv")
    capacity = importlib.import_module("tensorfold.cuda.capacity")
    dec = decoder(multi, state, torch, room=0)
    dec.capacity, dec.keep, dec.window_bytes, dec.vision_bytes = 65536, 8, 10 * GIB, GIB // 2
    dec._ref_state = state.State(weights(), 256, 4, "bf16", limit=65536)
    dec.graphs = SimpleNamespace(budget=GIB, demand=True)
    dec.vision_host = GIB // 2
    dec.memory_gate.live = lambda: 12 * GIB
    held, asked = {"pool": 0}, []

    def reserve(size):
        size = max(0, int(size)) // thor_kv.ALIGN * thor_kv.ALIGN
        asked.append(size)
        held["pool"] = size
        return size

    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "2.5")
    monkeypatch.setattr(thor_kv, "reserve", reserve)
    monkeypatch.setattr(thor_kv, "trim_host", lambda: None)
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 40 * GIB, "MemAvailable": host(held["pool"]),
                                                       "SwapFree": 0, "SwapCached": 0})
    monkeypatch.setattr(capacity, "unified", lambda torch: True)
    for name in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: allocated())
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return dec, asked, thor_kv


def test_the_pool_is_refit_after_the_servers_own_init(allocations, monkeypatch):  # noqa: F811
    """THOR-KV-FIT: the CLI builds its App (tokenizer, chat template) after the engine; vmem-a3-c measured 94 MiB of
    server memory there, after the fit's last reading. ``refit_pool`` reads the idle again and lands it back."""
    MIB = 1 << 20
    app = {"bytes": 0}
    free = int(14.5 * GIB)                                      # MemAvailable without the pool
    dec, asked, thor_kv = _pooled(monkeypatch, lambda pool: free - pool - app["bytes"])
    target = int(2.5 * GIB) + GIB + GIB // 2                    # reserve + graphs + vision host
    note = dec.reserve_pool()
    assert asked == [11 * GIB - GIB // 2] and free - dec.pool_bytes == target
    room, first = dec.memory_gate.room, dec.pool_bytes
    assert "caches from one" in note and "from its design point" in note
    app["bytes"] = 94 * MIB                                     # the App's init
    asked.clear()
    note = dec.refit_pool()
    idle = free - dec.pool_bytes - app["bytes"]
    assert asked == [first - 94 * MIB - thor_kv.FIT_AIM] and idle == target + thor_kv.FIT_AIM
    assert dec.memory_gate.room == room - (first - dec.pool_bytes) and dec.pool_fit[-1]["idle"] == idle
    assert note.startswith("the pool refit after the server's init: every stream's caches from one") and \
        f"{dec.pool_bytes / GIB:.2f} GiB pool" in note
    # within the window already: measured, nothing moves
    asked.clear()
    assert "from its design point" in dec.refit_pool() and asked == []


def test_the_refit_leaves_the_pool_alone_when_switched_off_or_something_allocated(allocations, monkeypatch):  # noqa: F811
    MIB = 1 << 20
    app, alloc = {"bytes": 0}, {"bytes": 5 * GIB}
    dec, asked, thor_kv = _pooled(monkeypatch, lambda pool: int(14.5 * GIB) - pool - app["bytes"],
                                  allocated=lambda: alloc["bytes"])
    dec.reserve_pool()
    pool, room = dec.pool_bytes, dec.memory_gate.room
    app["bytes"] = 200 * MIB
    asked.clear()
    alloc["bytes"] += 2 * MIB                                   # a tensor allocated since: it may sit in the pool's block
    note = dec.refit_pool()
    assert "not refit" in note and asked == [] and dec.pool_bytes == pool and dec.memory_gate.room == room
    alloc["bytes"] -= 2 * MIB
    monkeypatch.setenv(thor_kv.FIT_ENV, "0")                    # the kill switch: sized once, as before
    assert dec.refit_pool() is None and asked == [] and dec.pool_bytes == pool
    monkeypatch.delenv(thor_kv.FIT_ENV)
    multi, state, torch = mods()
    assert decoder(multi, state, torch, room=0).refit_pool() is None     # never pooled: nothing to refit


def test_outside_the_pool_asks_the_hosts_memory_past_its_reserve():
    gate = importlib.import_module("tensorfold.cuda.memory_gate").MemoryGate(10, reserve=0)
    assert gate.outside(10**15)                                        # no system reading: never blocks
    gate.system = lambda: 5
    assert gate.outside(5) and not gate.outside(6)


def test_pooled_on_demand_graphs_leave_the_room_alone():
    tg = importlib.import_module("tensorfold.families.qwen4_exp.cuda.thor_graphs")
    g = object.__new__(tg.DecodeGraphs)
    g.paid, g.charged, g.refunded, g.est, g.budget = {}, 0, 0, 24 * (1 << 20), GIB
    asked = []
    g.gate = SimpleNamespace(pooled=True, room=7, outside=lambda n: asked.append(n) or True,
                             fits=lambda n: pytest.fail("a pooled gate's room is the caches'"))
    assert g._afford() and asked == [24 * (1 << 20)]
    g.gate.outside = lambda n: False
    assert not g._afford()


def test_on_demand_graphs_charge_the_host_memory_their_executables_keep(monkeypatch):
    """vmem-a1: an on-demand graph's executable keeps ~5 MiB of driver host memory (glibc) that the free memory never
    showed; the charge is the free memory's drop plus glibc's in-use growth (at most HOST_MAX_MIB: another thread's
    allocation in the window), an eviction gives both back (at most the charge), and HOST=0 charges the drop only."""
    from collections import OrderedDict

    tg = importlib.import_module("tensorfold.families.qwen4_exp.cuda.thor_graphs")
    mib = 1 << 20
    free, host = [1000.0], [50.0]
    monkeypatch.setattr(tg, "_free_mib", lambda: free[0])
    monkeypatch.setattr(tg, "_host_mib", lambda: host[0])
    monkeypatch.setattr(tg.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(tg.DecodeGraphs, "_emit", staticmethod(lambda event: None))

    def graphs(on: bool):
        g = object.__new__(tg.DecodeGraphs)
        g.paid, g.graphs, g.charged, g.refunded, g.est, g.budget = OrderedDict(), OrderedDict(), 0, 0, 24 * mib, GIB
        g.calls, g.evicted, g.host, g.host_mib = 10, 0, on, 0.0
        g.gate = SimpleNamespace(pooled=False, room=100 * mib, held=0)
        return g

    A, B = ("verify", (), 0, 4096), ("verify", (), 1, 4096)    # keys: kind, segments, flags, context bucket
    g = graphs(True)
    free[0], host[0] = 990.0, 55.0                       # the capture: 10 MiB of free memory, 5 MiB of glibc in use
    g._charge(A, "verify", 1000.0, 50.0)
    assert g.paid[A][0] == 15 * mib and g.charged == 15 * mib and g.gate.room == 85 * mib and g.host_mib == 5
    host[0] = 55.0 + 90                                   # a vision request's 90 MiB array lands in the window
    g._charge(B, "verify", 990.0, 55.0)
    assert g.paid[B][0] == tg.HOST_MAX_MIB * mib and g.est == tg.HOST_MAX_MIB * mib

    def reset(dfree, dhost):
        def run():
            free[0] += dfree
            host[0] -= dhost
        return SimpleNamespace(graph=SimpleNamespace(reset=run))

    g.graphs[A], g.graphs[B] = reset(10, 5), reset(0, 120)
    g.calls = 10 + tg.RECENT
    assert g._evict() and g.refunded == 15 * mib and g.gate.room == (85 - tg.HOST_MAX_MIB + 15) * mib
    assert g._evict() and g.refunded == (15 + tg.HOST_MAX_MIB) * mib     # never more back than its charge

    g = graphs(False)                                     # TENSORFOLD_THOR_DECODE_GRAPH_HOST=0: the earlier charge
    free[0], host[0] = 990.0, 55.0
    g._charge(A, "verify", 1000.0, 50.0)
    assert g.paid[A][0] == 10 * mib and g.host_mib == 5          # measured, not charged
    g.graphs[A], g.calls = reset(10, 5), 10 + tg.RECENT
    assert g._evict() and g.refunded == 10 * mib


def test_capacity_preview_route_renders_the_body_and_skips_images(allocations):  # noqa: F811
    cap = importlib.import_module("tensorfold.cuda.capacity_status")
    errors = importlib.import_module("tensorfold.server.errors")
    seen = []

    class Decoder:
        memory_gate = SimpleNamespace(evicted_kept=3)

        def preview(self, prompt, count, *, max_streams=None):
            seen.append((prompt, count, max_streams))
            return {"prompt_tokens": len(prompt), "cached_tokens": 2, "admissible_now": True, "reason": None}

    def prepare(body, chat):
        assert "stream" not in body and "stream_options" not in body
        if body.get("too_long"):
            raise errors.ContextLengthError("This server's maximum context length is 10 tokens")
        return SimpleNamespace(prompt=[1, 2, 3], max_tokens=64)

    app = SimpleNamespace(engine=SimpleNamespace(scheduler=SimpleNamespace(decoder=Decoder(), max_streams=8)),
                          prepare=prepare)
    code, body = cap.preview(app, {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    assert code == 200 and body["admissible_now"] and body["cached_tokens"] == 2 and body["previewed"]
    assert seen == [([1, 2, 3], 64, 8)] and body["evicted_kept_sessions_total"] == 3
    code, body = cap.preview(app, {"messages": [], "too_long": True})
    assert code == 200 and body["reason"] == "length" and body["admissible_now"] is False
    img = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}]}
    code, body = cap.preview(app, img)
    assert code == 200 and body["previewed"] is False


def test_a_pooled_gate_admits_only_what_still_grows_past_its_reserve():
    gate = importlib.import_module("tensorfold.cuda.memory_gate").MemoryGate(10 * GIB, reserve=GIB)
    assert gate.admits(10 * GIB)                                       # the earlier path: the whole room
    gate.pooled = True
    assert gate.admits(9 * GIB) and not gate.admits(9 * GIB + 1)


def test_a_preview_that_finds_its_conversation_moves_it_to_the_back_of_the_queue(allocations):  # noqa: F811
    multi, state, torch = mods()
    dec = decoder(multi, state, torch, room=1 << 40)
    dec.capacity, dec.slots = 65536, 8
    dec._ref_state = state.State(weights(), 256, 4, "bf16", limit=65536)
    a, b = kept_slots(dec, state, 2, used=[1.0, 2.0])
    assert dec.preview([1] * 10 + [5] * 40, 16)["cached_tokens"] == 10          # a's conversation, about to return
    assert dec.used[id(a)] > 2.0
    assert dec._evict_kept(None) and [k[1] for k in dec.kept] == [a]           # b goes, not a


def test_lock_tables_locks_every_format_all_or_nothing(monkeypatch, tmp_path):
    import ctypes

    import numpy as np

    thor_kv = importlib.import_module("tensorfold.families.qwen4_exp.cuda.thor_kv")
    arrs = [np.zeros(4096, dtype=np.uint8) for _ in range(3)]
    table = SimpleNamespace(values=arrs[:2], scales=arrs[2:], prefetch=lambda: None)
    calls = []

    def Libc(fail_at=None):
        def mlock(at, n):
            calls.append(("mlock", at))
            return -1 if len([c for c in calls if c[0] == "mlock"]) == fail_at else 0

        def munlock(at, n):
            calls.append(("munlock", at))
            return 0

        return SimpleNamespace(mlock=mlock, munlock=munlock)

    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: Libc())
    ok, note = thor_kv.lock_tables([table], read=True)
    assert ok and [c[1] for c in calls] == [a.ctypes.data for a in arrs] and "locked in memory" in note
    calls.clear()
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: Libc(fail_at=3))
    ok, note = thor_kv.lock_tables([table], read=True)
    assert not ok and calls[-2:] == [("munlock", arrs[0].ctypes.data), ("munlock", arrs[1].ctypes.data)]
    assert "stay evictable" in note
