"""Thor DEC-2's ``attn_ptrs`` cache key (CPU, no CUDA): a resize must never serve a stale pointer table.

par-1-fix: with 8 concurrent streams sharing one caching allocator, a resize's freed block for one attention layer
can be handed straight back by the allocator (the same address) while a sibling layer's block goes elsewhere (and
vice versa). A cache key that samples only one layer's address to detect "the caches moved" can then match by
coincidence while another layer's pointer has actually changed, serving a stale table to the kernel. Measured on
Thor (92160-context, 8-stream soak, TENSORFOLD_THOR_DEC2_CHECK=1): 31/802 stale tables, all from ``attn_ptrs``
(``gdn_ptrs`` recomputes fresh every call and never mismatched). The fix keys on ``State.version`` (already
incremented by every ``State.resize``, per its own docstring) instead of a sampled address: a monotonic counter
cannot repeat by allocator coincidence the way a freed-and-reused address can.
"""

from __future__ import annotations

import numpy as np

from tensorfold.families.qwen4_exp.cuda.thor_dec2 import attn_ptrs


class _Ptr:
    """Stands in for a CUDA tensor: only ``data_ptr()`` is read."""

    def __init__(self, addr: int) -> None:
        self._addr = addr

    def data_ptr(self) -> int:
        return self._addr


class _KC:
    def __init__(self, k: int, v: int, ks: int, vs: int) -> None:
        self.k, self.v, self.ks, self.vs = _Ptr(k), _Ptr(v), _Ptr(ks), _Ptr(vs)


class _Layer:
    def __init__(self, index: int) -> None:
        self.index = index


class _State:
    """The ``attn_ptrs``-relevant slice of ``State``: capacity, the three per-layer lists, ``att_index``, ``version``."""

    def __init__(self, capacity: int, kc: list, ikc: list, pooled: list, att_index: dict, version: int = 0) -> None:
        self.capacity = capacity
        self.kc, self.ikc, self.pooled = kc, ikc, pooled
        self.att_index = att_index
        self.version = version


def _state() -> tuple[_State, list]:
    layers = [_Layer(0), _Layer(1)]
    kc = [_KC(100, 101, 102, 103), _KC(200, 201, 202, 203)]
    ikc, pooled = [_Ptr(110), _Ptr(210)], [_Ptr(120), _Ptr(220)]
    st = _State(4096, kc, ikc, pooled, att_index={0: 0, 1: 1}, version=1)
    return st, layers


def test_attn_ptrs_matches_the_live_buffers() -> None:
    st, layers = _state()
    out = attn_ptrs(st, layers)
    np.testing.assert_array_equal(out, [[100, 101, 102, 103, 110, 120], [200, 201, 202, 203, 210, 220]])


def test_attn_ptrs_cache_hits_when_nothing_moved() -> None:
    st, layers = _state()
    first = attn_ptrs(st, layers)
    second = attn_ptrs(st, layers)
    assert second is first                     # the cached array itself, not a recompute


def test_attn_ptrs_recomputes_when_only_a_sibling_layer_moved() -> None:
    """The adversarial case: layer 0's address and the slot's capacity are unchanged (the allocator coincidence),
    but layer 1's cache moved and ``version`` (a real resize's signature) bumped. The stale table must not be served."""

    st, layers = _state()
    first = attn_ptrs(st, layers)
    assert first[1].tolist() == [200, 201, 202, 203, 210, 220]

    st.kc[1] = _KC(900, 901, 902, 903)          # layer 1 resized to a new buffer; layer 0 and capacity untouched
    st.version += 1                             # State.resize's own bookkeeping

    second = attn_ptrs(st, layers)
    assert second[0].tolist() == [100, 101, 102, 103, 110, 120]   # layer 0: unchanged, as expected
    assert second[1].tolist() == [900, 901, 902, 903, 210, 220]   # layer 1: the NEW buffer, not the stale cached one


def test_attn_ptrs_recomputes_when_layer_zero_moved_back_to_its_old_address() -> None:
    """The other half of the coincidence: layer 0's address returns to a previously-cached value by allocator luck.
    ``version`` must still force a recompute rather than trust the coincidental match."""

    st, layers = _state()
    attn_ptrs(st, layers)
    st.kc[0] = _KC(999, 998, 997, 996)
    st.version += 1
    attn_ptrs(st, layers)                       # cache now keyed on the moved layer 0 and the bumped version

    st.kc[0] = _KC(100, 101, 102, 103)          # "resized" back to its original address (allocator coincidence)
    st.version += 1

    third = attn_ptrs(st, layers)
    assert third[0].tolist() == [100, 101, 102, 103, 110, 120]
    assert third[1].tolist() == [200, 201, 202, 203, 210, 220]
