"""AGX Thor's wide prompt pieces keep the GPU driver's free pages for cache grows (ROWS-2): a grow first frees kept
prompt ends, oldest first, while the driver's pool can't hold it and the reserve; released pieces, other devices and
the growing or protected slot are untouched."""

import importlib
from types import SimpleNamespace

import pytest

from tests.test_cuda_geometry import allocations  # noqa: F401  (fixture: fake triton, so the module imports)
from tests.test_cuda_growing_caches import decoder, weights

GIB = 1 << 30


def test_short_needs_a_pool():
    gate = importlib.import_module("tensorfold.cuda.memory_gate").MemoryGate(100, reserve=10)
    assert not gate.short(5) and not gate.short(10**12)                 # no pool: today's gate
    gate.pool = lambda: 14
    assert gate.short(5) and not gate.short(4)                         # pool < extra + reserve
    gate.pool = lambda: 15
    assert not gate.short(5)


def test_driver_pool_is_the_drivers_freed_pages_plus_torch_cache():
    gate = importlib.import_module("tensorfold.cuda.memory_gate")
    torch = SimpleNamespace(cuda=SimpleNamespace(memory_reserved=lambda: 7 * GIB, memory_allocated=lambda: 6 * GIB))
    pool = gate.driver_pool(torch, meminfo=lambda: {"KReclaimable": 5 * GIB, "SReclaimable": GIB, "MemFree": 9 * GIB})
    assert pool() == 4 * GIB + GIB
    assert gate.driver_pool(torch, meminfo=lambda: {"MemFree": GIB})() == 1 << 62    # unknown: never short
    assert gate.driver_pool(torch, meminfo=lambda: None)() == 1 << 62
    pool = gate.driver_pool(torch, meminfo=lambda: {"KReclaimable": GIB, "SReclaimable": 2 * GIB})
    assert pool() == GIB                                                # slab above KReclaimable counts as none


@pytest.mark.parametrize("rows,capability,on", [(4096, (11, 0), True), (2112, (11, 0), True), (8192, (11, 0), True),
                                                (2048, (11, 0), False), (4096, (12, 1), False), (4096, (10, 0), False),
                                                (4096, [11, 0], True)])
def test_keeps_pool_only_for_wide_pieces_on_thor(monkeypatch, rows, capability, on):
    plan = importlib.import_module("tensorfold.families.qwen4_exp.cuda.prompt_plan")
    monkeypatch.delenv(plan.POOL_ENV, raising=False)
    assert plan.keeps_pool(rows, capability) is on
    for value in ("0", "off", " false "):                               # the kill switch keeps today's gate
        monkeypatch.setenv(plan.POOL_ENV, value)
        assert plan.keeps_pool(rows, capability) is False
    monkeypatch.setenv(plan.POOL_ENV, "1")
    assert plan.keeps_pool(rows, capability) is on


def kept_decoder(n: int):
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, state, torch, room=1 << 40)
    dec.capacity = 65536
    slots = []
    for i in range(n):                                                  # kept ends, oldest first, 8192 rows each
        st = state.State(weights(), 256, 4, "bf16", limit=65536)
        dec.memory_gate.take(st.ensure(300))
        dec.kept.append(([i] * 3, st, {}, None))
        slots.append(st)
    return dec, slots, state


def grow_need(state):
    st = state.State(weights(), 256, 4, "bf16", limit=65536)
    size = 16384                                                       # 9000 rows: two 8192-row steps
    return st, st.cache_bytes(size) - st.cache_bytes() + st.layer_bytes(size), st.cache_bytes(8192) - st.cache_bytes()


def test_a_grow_frees_the_oldest_kept_ends_until_the_pool_holds_it(allocations):  # noqa: F811
    dec, slots, state = kept_decoder(3)
    st, need, each = grow_need(state)
    gate = dec.memory_gate
    gate.reserve = 3 * each
    held0 = gate.held
    base = need + gate.reserve - each - each // 2                     # one end short, two enough
    gate.pool = lambda: base + (held0 - gate.held)                     # freed caches return to the pool
    assert dec._grow(st, 9000) and st.capacity == 16384
    assert [k[1] for k in dec.kept] == [slots[2]]                       # the newest end stays
    assert slots[0].capacity == 256 and slots[1].capacity == 256 and slots[2].capacity == 8192
    assert all(any(f is s for f in dec.free) for s in slots[:2])


def test_a_full_pool_frees_nothing(allocations):  # noqa: F811
    dec, slots, state = kept_decoder(3)
    st, need, _ = grow_need(state)
    dec.memory_gate.pool = lambda: need + dec.memory_gate.reserve
    assert dec._grow(st, 9000) and st.capacity == 16384
    assert [k[1] for k in dec.kept] == slots


def test_released_pieces_keep_todays_gate(allocations):  # noqa: F811
    dec, slots, state = kept_decoder(3)
    st, _, _ = grow_need(state)
    assert dec.memory_gate.pool is None
    assert dec._grow(st, 9000) and st.capacity == 16384
    assert [k[1] for k in dec.kept] == slots                           # nothing freed without the pool


def test_the_growing_and_protected_slots_are_never_freed(allocations):  # noqa: F811
    dec, slots, state = kept_decoder(4)
    dec.memory_gate.pool = lambda: 0                                   # always short
    grower, guarded = slots[1], slots[2]
    assert dec._grow(grower, 9000, protect=guarded) and grower.capacity == 16384
    assert [k[1] for k in dec.kept] == [grower, guarded]               # the others went, oldest first
    assert slots[0].capacity == 256 and slots[3].capacity == 256


# par-1-fix: driver_pool's own reading can be structurally small (torch's own emptied cushion, by design of the
# empty_cache() right after this check) on a vision-capable, 8-stream, large-n-gram-table recipe, regardless of
# true headroom (measured on Thor: pool_mib 0.9-1.7 GiB while memavail_mib stayed 29-31 GiB for the whole 80k
# prompt_reuse growth): short() then fired on nearly every qualifying grow and evicted a wanted kept prefix instead
# of an idle one (80k turns fell from 99%+ cached to 2%). _table_aware_pool adds the one signal driver_pool cannot
# see without it: true MemAvailable past the mapped table's own bytes.

def test_mapped_tables_sums_every_ple_layers_table_nbytes():
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    layers = [SimpleNamespace(ple=None),
              SimpleNamespace(ple=SimpleNamespace(table=SimpleNamespace(nbytes=5 * GIB))),
              SimpleNamespace(ple=SimpleNamespace(table=SimpleNamespace(nbytes=3 * GIB)))]
    assert multi._mapped_tables(SimpleNamespace(layers=layers)) == 8 * GIB


def test_mapped_tables_is_zero_without_ple():
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    w = SimpleNamespace(layers=[SimpleNamespace(ple=None), SimpleNamespace(ple=None)])
    assert multi._mapped_tables(w) == 0


def test_table_aware_pool_keeps_driver_pool_without_a_mapped_table():
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    pool = multi._table_aware_pool(SimpleNamespace(), lambda: 7, 0)    # torch is never touched: mapped == 0
    assert pool() == 7


def test_table_aware_pool_keeps_driver_pool_off_a_unified_gpu(monkeypatch):
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    capacity = importlib.import_module("tensorfold.cuda.capacity")
    monkeypatch.setattr(capacity, "page_room", lambda torch: None)     # a discrete GPU: no page cache to share
    pool = multi._table_aware_pool(SimpleNamespace(), lambda: 7, 5 * GIB)
    assert pool() == 7


def test_table_aware_pool_uses_memavailable_past_the_table_when_that_is_larger(monkeypatch):
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    capacity = importlib.import_module("tensorfold.cuda.capacity")
    monkeypatch.setattr(capacity, "page_room", lambda torch: 30 * GIB)
    pool = multi._table_aware_pool(SimpleNamespace(), lambda: GIB, 27 * GIB)
    assert pool() == 3 * GIB                                           # 30 - 27, not driver_pool's 1 GiB reading


def test_table_aware_pool_still_reports_short_when_both_signals_are_tight(monkeypatch):
    """ROWS-2's own scenario, reproduced: real pressure shows up in both signals, so the guard still protects."""

    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    capacity = importlib.import_module("tensorfold.cuda.capacity")
    monkeypatch.setattr(capacity, "page_room", lambda torch: 27.2 * GIB)
    pool = multi._table_aware_pool(SimpleNamespace(), lambda: 0.15 * GIB, 27 * GIB)
    assert abs(pool() - 0.2 * GIB) < 1024                               # max(0.15, 0.2) GiB: still a tight reading
