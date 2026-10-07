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
    note = dec.reserve_pool()
    snap = dec._snapshot_bytes()
    assert asked == [11 * GIB] and dec.pool_bytes == 11 * GIB
    assert dec.memory_gate.room == 11 * GIB - 8 * snap - GIB // 2 and dec.memory_gate.reserve == thor_kv.SLACK
    assert "11.00 GiB pool" in note and "1.00 GiB left for on-demand graphs" in note and "0.50 GiB of image" in note


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
