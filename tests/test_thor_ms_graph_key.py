"""Thor DPUSH MS-GRAPH (CPU, no CUDA): a concurrent forward's graph key holds its stream count and rows only.

Per-stream windows (or slots) in the key gave keys that never repeated at two or more streams (0.94-0.98x, dec1-p1);
``canonical`` keys by what every launch of such a forward takes from the host, so any split of the same rows over the
same number of streams, in any slots, replays one graph.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

from tensorfold.families.qwen4_exp.cuda import thor_graphs


def _segs(windows, slots):
    segs, a0 = [], 0
    for n, slot in zip(windows, slots):
        segs.append((NS(slot=slot), a0, a0 + n))
        a0 += n
    return segs


def test_canonical_key_ignores_the_split_and_the_slots():
    step = NS(vision=False, context=8192)
    a = thor_graphs.canonical("verify", _segs([3, 1, 7, 2], [0, 1, 2, 3]), step)
    assert a == thor_graphs.canonical("verify", _segs([4, 4, 4, 1], [5, 2, 7, 1]), step)
    assert a == thor_graphs.canonical("verify", _segs([2, 7, 1, 3], [3, 2, 1, 0]), step)
    assert a != thor_graphs.canonical("verify", _segs([3, 1, 7, 3], [0, 1, 2, 3]), step)       # rows
    assert a != thor_graphs.canonical("verify", _segs([6, 7], [0, 1]), step)                   # streams
    assert a != thor_graphs.canonical("mtp", _segs([3, 1, 7, 2], [0, 1, 2, 3]), step, (1,))   # kind and picks
    assert a != thor_graphs.canonical("verify", _segs([3, 1, 7, 2], [0, 1, 2, 3]), NS(vision=False, context=16384))
    assert a != thor_graphs.canonical("verify", _segs([3, 1, 7, 2], [0, 1, 2, 3]), NS(vision=True, context=8192))


def test_multi_kill_switch(monkeypatch):
    """``TENSORFOLD_THOR_DECODE_GRAPH_MULTI=0`` keeps concurrent forwards eager (e4fb57f's path)."""

    try:
        for value in ("0", "off", "false", "no"):
            monkeypatch.setenv(thor_graphs.MULTI_ENV, value)
            thor_graphs.multi_on.cache_clear()
            assert not thor_graphs.multi_on()
    finally:
        thor_graphs.multi_on.cache_clear()
    assert thor_graphs.STREAMS == 1 and thor_graphs.MULTI_STREAMS >= 8


def test_multi_default_graphs_only_mtp_chain_steps():
    """Default MS-GRAPH: MTP chain steps (one row a stream) only; ``all``: every concurrent forward."""

    assert thor_graphs.multi_graphed("mtp", 4, 4, False)
    assert not thor_graphs.multi_graphed("mtp", 4, 9, False)          # a first MTP step (kept rows)
    assert not thor_graphs.multi_graphed("verify", 4, 4, False)
    assert thor_graphs.multi_graphed("verify", 4, 13, True) and thor_graphs.multi_graphed("mtp", 4, 9, True)
