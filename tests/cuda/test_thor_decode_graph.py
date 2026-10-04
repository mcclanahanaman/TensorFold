"""Thor decode graphs (THOR-K3): concurrent rounds replayed from CUDA graphs keep the eager rounds' bits."""

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_flashnext_forward import V, _model  # noqa: E402
from test_flashnext_multi import PROMPTS  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import thor_graphs  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

DEPTH = 3


def _decode(w, prompts, counts, samplings, *, graphs: bool, capacity: int, cap: int | None = None,
            kv_dtype: str = "bf16"):
    """Every stream to its count; after each round, a copy of the verification and MTP logits buffers."""

    dec = MultiDecoder(w, slots=len(prompts), capacity=capacity, depth=DEPTH, confidence=0.3, kv_dtype=kv_dtype)
    dec.graphs = thor_graphs.DecodeGraphs(DEPTH + 1, cap=cap) if graphs else None
    dec.buf.logits.zero_()                  # rows no round writes compare equal (each decoder's buffers start empty)
    dec.mbuf.logits.zero_()
    streams = [Stream(p, n, smp) for p, n, smp in zip(prompts, counts, samplings)]
    for s in streams:
        dec.admit(s)
    rounds = []
    while dec.live():
        dec.finish(dec.round())
        rounds.append((dec.buf.logits.clone(), dec.mbuf.logits.clone()))
    return [list(s.out) for s in streams], rounds, dec.graphs


def _same_rounds(a, b) -> None:
    assert len(a) == len(b)
    for i, ((va, ma), (vb, mb)) in enumerate(zip(a, b)):
        assert torch.equal(va, vb), f"round {i}: verification logits differ"
        assert torch.equal(ma, mb), f"round {i}: MTP logits differ"


@pytest.mark.parametrize("cap", [None, 2])
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_graphed_rounds_keep_the_eager_rounds_bits(cap, kv_dtype):
    """Dense streams ending at different rounds, greedy and sampled: graph replays (all kept, or two then eager) give
    the eager rounds' logits, bit for bit, every round."""

    w = _model()
    samplings = [None, Sampling(seed=1234, top_k=20, top_p=0.95), None, Sampling(seed=7, top_k=20, top_p=0.95)]
    counts = [40, 24, 33, 12]
    off_out, off_rounds, _ = _decode(w, PROMPTS, counts, samplings, graphs=False, capacity=1024, kv_dtype=kv_dtype)
    on_out, on_rounds, g = _decode(w, PROMPTS, counts, samplings, graphs=True, capacity=1024, cap=cap,
                                   kv_dtype=kv_dtype)
    assert on_out == off_out
    _same_rounds(on_rounds, off_rounds)
    assert not g.failed
    assert g.replays.get("verify", 0) > 0 and g.replays.get("mtp", 0) > 0, (g.replays, g.forwards)
    assert len(g.graphs) <= (cap or thor_graphs.CAP)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_graphed_rounds_with_a_sparse_stream_keep_the_eager_rounds_bits(kv_dtype):
    """A stream past the attention budget (its own block selection, its pooled keys in the key) beside dense ones."""

    w = _model(7)
    g = torch.Generator().manual_seed(9)
    prompts = [torch.randint(1, V, (2400,), generator=g).tolist(), PROMPTS[0], PROMPTS[1]]
    samplings = [Sampling(seed=5, top_k=20, top_p=0.95), None, None]
    counts = [36, 20, 28]
    off_out, off_rounds, _ = _decode(w, prompts, counts, samplings, graphs=False, capacity=4096, kv_dtype=kv_dtype)
    on_out, on_rounds, gr = _decode(w, prompts, counts, samplings, graphs=True, capacity=4096, kv_dtype=kv_dtype)
    assert on_out == off_out
    _same_rounds(on_rounds, off_rounds)
    assert not gr.failed and gr.replays.get("verify", 0) > 0


def test_one_stream_replays_most_rounds():
    """One greedy stream: past each key's first (eager) round, its rounds replay graphs."""

    w = _model()
    off_out, off_rounds, _ = _decode(w, [PROMPTS[1]], [64], [None], graphs=False, capacity=1024)
    on_out, on_rounds, g = _decode(w, [PROMPTS[1]], [64], [None], graphs=True, capacity=1024)
    assert on_out == off_out
    _same_rounds(on_rounds, off_rounds)
    assert g.replays["verify"] >= g.forwards["verify"] // 2, (g.replays, g.forwards)


def _sequential(w, prompts, counts, *, graphs: bool, capacity: int):
    """One greedy request at a time (each alone in its rounds), graphs recorded at startup; the slot each one used."""

    dec = MultiDecoder(w, slots=len(prompts), capacity=capacity, depth=DEPTH, confidence=0.3)
    dec.graphs = thor_graphs.DecodeGraphs(DEPTH + 1) if graphs else None
    if graphs:
        dec._prerecord()
    dec.buf.logits.zero_()
    dec.mbuf.logits.zero_()
    outs, rounds, slots = [], [], []
    for prompt, count in zip(prompts, counts):
        s = Stream(prompt, count, None)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
            rounds.append((dec.buf.logits.clone(), dec.mbuf.logits.clone()))
        outs.append(list(s.out))
        slots.append(id(s.st))
    return outs, rounds, dec.graphs, slots


def test_startup_graphs_serve_any_slot(monkeypatch):
    """Graphs recorded at startup on one free slot (slot-free keys, each stream's n-gram conv tail copied in) replay
    for requests on other slots with the eager rounds' bits; keys outside the startup set stay eager."""

    monkeypatch.setenv(thor_graphs.ROWS_ENV, f"2-{DEPTH + 1}")     # every verification window (the default)
    monkeypatch.setenv(thor_graphs.DEMAND_ENV, "0")                 # 8f73033: the rest stay eager (the kill switch)
    w = _model()
    prompts, counts = PROMPTS[:3], [40, 33, 48]
    off_out, off_rounds, _, _ = _sequential(w, prompts, counts, graphs=False, capacity=1024)
    on_out, on_rounds, g, slots = _sequential(w, prompts, counts, graphs=True, capacity=1024)
    assert on_out == off_out
    _same_rounds(on_rounds, off_rounds)
    assert len(set(slots)) > 1, "the requests shared one slot"
    assert g.fixed and not g.failed
    # verification windows of 2..DEPTH+1 rows (one DeltaNet parity), MTP steps of 1..DEPTH+1 rows with their greedy
    # picks: nothing recorded later
    assert g.captures == {"verify": DEPTH, "mtp": DEPTH + 1}, g.captures
    # the diagnostics count every eager one-stream forward by its key
    assert sum(g.missed.values()) == sum(g.forwards.values()) - sum(g.replays.values()), (g.missed, g.forwards)
    assert g.replays.get("verify", 0) > 0 and g.replays.get("mtp", 0) > 0, (g.replays, g.forwards)


def test_rows_spec():
    assert thor_graphs._rows("2-7", 7) == [2, 3, 4, 5, 6, 7]
    assert thor_graphs._rows("3,5-6,9", 7) == [3, 5, 6]
    assert thor_graphs._rows("1-2", 7) == [2]
    assert thor_graphs._rows("2,7", 7) == [2, 7]                     # the kill switch: e8f9144's two windows


def test_bucket():
    assert thor_graphs.bucket(1) == 8192 and thor_graphs.bucket(8192) == 8192
    assert thor_graphs.bucket(8193) == 16384 and thor_graphs.bucket(40000) == 65536


def test_prerecord_after_the_gate_charges_the_room(monkeypatch):
    """Startup graphs are recorded after the memory gate measured the room, and the room is charged with exactly what
    prerecord reports as the graphs' own memory (captures and uploads); by default the verification windows recorded
    are every window from two rows to the widest."""

    from tensorfold.families.qwen4_exp.cuda import multi

    monkeypatch.setenv(thor_graphs.ENV, "1")
    monkeypatch.delenv(thor_graphs.ROWS_ENV, raising=False)
    monkeypatch.setattr(multi, "torch_live", lambda torch, available: (lambda: 64 << 30))
    charged, pre = [], thor_graphs.DecodeGraphs.prerecord

    def prerecord(self, *a, **k):
        charged.append(pre(self, *a, **k))
        return charged[-1]

    monkeypatch.setattr(thor_graphs.DecodeGraphs, "prerecord", prerecord)
    thor_graphs.available.cache_clear()
    try:
        dec = MultiDecoder(_model(), slots=2, capacity=1024, depth=DEPTH, confidence=0.3)
    finally:
        thor_graphs.available.cache_clear()
    g = dec.graphs
    assert g is not None and g.fixed and not g.failed and g.rows == list(range(2, DEPTH + 2))
    assert g.captures == {"verify": DEPTH, "mtp": DEPTH + 1}, g.captures
    assert len(charged) == 1 and 0 <= charged[0] < 1 << 30
    assert dec.memory_gate.room == (64 << 30) - charged[0]


def test_graphs_default_on_for_one_stream(monkeypatch):
    """THOR-DEC-1 G-D ruling: graphs are on unless TENSORFOLD_THOR_DECODE_GRAPH is 0 (the r37 path, the kill switch),
    for forwards of one stream by default."""

    monkeypatch.delenv(thor_graphs.ENV, raising=False)
    assert not thor_graphs._off()
    assert thor_graphs.STREAMS == 1
    for value in ("0", "off", "false", "no"):
        monkeypatch.setenv(thor_graphs.ENV, value)
        assert thor_graphs._off()
    monkeypatch.setenv(thor_graphs.ENV, "1")
    assert not thor_graphs._off()


def test_demand_graphs_keep_the_bits_and_charge_the_room(monkeypatch):
    """Past the startup set, on demand (the default): a sparse stream's keys (its slot's pooled keys in the key) and
    first rounds record on their second forward, each charged to the memory gate's room, with the eager rounds' bits;
    with the least recently replayed graph evicted before each new one (``RECENT`` 0) the bits hold and the room gets
    back what each release returned. ``TENSORFOLD_THOR_DECODE_GRAPH_DEMAND=0`` is 8f73033's path
    (``test_startup_graphs_serve_any_slot``)."""

    w = _model(7)
    g = torch.Generator().manual_seed(9)
    long = torch.randint(1, V, (2400,), generator=g).tolist()
    prompts, counts = [long, PROMPTS[0], long[:2200] + PROMPTS[1]], [36, 20, 28]
    off_out, off_rounds, _, _ = _sequential(w, prompts, counts, graphs=False, capacity=4096)
    startup = 2 * DEPTH + 1                     # verification windows 2..DEPTH+1, MTP steps 1..DEPTH+1
    afford = thor_graphs.DecodeGraphs._afford
    for evict in (False, True):
        if evict:
            def forced(self):
                if self.paid:
                    self._evict()
                return afford(self)
            monkeypatch.setattr(thor_graphs, "RECENT", 0)
            monkeypatch.setattr(thor_graphs.DecodeGraphs, "_afford", forced)
        on_out, on_rounds, gr, _ = _sequential(w, prompts, counts, graphs=True, capacity=4096)
        assert on_out == off_out
        _same_rounds(on_rounds, off_rounds)
        assert not gr.failed and gr.demand and gr.gate is not None
        assert sum(gr.captures.values()) > startup + 1, gr.captures
        assert any(k.endswith(":s") and v[1] > 0 for k, v in gr.ctx.items()), gr.ctx    # sparse rounds replayed
        assert gr.gate.room == gr.room_at - gr.own - gr.charged + gr.refunded    # startup, then on demand
        assert sum(gr.missed.values()) == sum(gr.forwards.values()) - sum(gr.replays.values())
        if evict:
            assert gr.evicted > 0 and len(gr.paid) <= 1, (gr.evicted, len(gr.paid))
