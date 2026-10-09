"""Thor decode graphs (THOR-K3): concurrent rounds replayed from CUDA graphs keep the eager rounds' bits."""

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_flashnext_forward import V, _Rand, _bf16_table, _cfg, _model, _ple  # noqa: E402
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
        assert gr.gate.room == gr.room_at - gr.charged + gr.refunded     # _sequential's own prerecord charges nothing
        assert sum(gr.missed.values()) == sum(gr.forwards.values()) - sum(gr.replays.values())
        if evict:
            assert gr.evicted > 0 and len(gr.paid) <= 1, (gr.evicted, len(gr.paid))


def _bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def test_shx_tail_matches_torch_and_replays(monkeypatch):
    """The fused SwiGLU writes the act slot and the cuBLAS pad; the down GEMM's y matches torch's thirteen kernels,
    on a fresh call and on a CUDA-graph replay."""

    from tensorfold.cuda import prompt_precision
    from tensorfold.cuda.nvfp4 import thor_dense
    from tensorfold.families.qwen4_exp.cuda import thor_shx

    if not thor_dense.available():
        pytest.skip("sm_110 with TENSORFOLD_THOR_DENSE on only")
    monkeypatch.delenv(thor_shx.ENV, raising=False)
    assert thor_shx.active()
    assert not prompt_precision.fp8()                   # the default: bf16 prompts. The FP8 case is below.
    old = (thor_dense.S.pad, thor_dense.S.staging, thor_dense.S.active)
    thor_dense.S.pad = torch.zeros((thor_dense.PAD * 2560,), dtype=torch.bfloat16, device="cuda")
    thor_dense.S.staging = torch.empty((1280 * 2560,), dtype=torch.bfloat16, device="cuda")
    thor_dense.S.active = True
    try:
        for width, hidden, rows in ((64, 256, 1), (64, 256, 6), (640, 2560, 2)):
            gu, down = _mx_pair(width, hidden, seed=width + rows)
            x = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
            y_off, act_off = _tail_once(x, gu, down, width, fused=False)
            y_on, act_on = _tail_once(x, gu, down, width, fused=True)
            assert _bits(y_on, y_off) and _bits(act_on, act_off), (width, rows)
            _replay_matches(x, gu, down, width, y_off, act_off)
        _fp8_prompt_matches_the_kill_switch(monkeypatch)
    finally:
        thor_dense.S.pad, thor_dense.S.staging, thor_dense.S.active = old


def _mx_pair(width: int, hidden: int, seed: int):
    from tensorfold.cuda.nvfp4 import linear as lin_mod
    from tensorfold.cuda.nvfp4 import thor_dense

    def mx(n, k, salt):
        g = torch.Generator().manual_seed(seed + salt)
        codes = (torch.randn(n, k, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
        scales = torch.randint(110, 132, (n, k // 32), generator=g, dtype=torch.uint8)
        lin = lin_mod.Mx8Linear.from_checkpoint(codes.cuda(), scales.cuda())
        lin.thor = True
        lin.thor_w = thor_dense.dequantize(lin)
        return lin

    return mx(2 * width, hidden, 1), mx(hidden, width, 2)


def _tail_once(x, gu, down, width: int, *, fused: bool):
    """One shared-expert tail from a zeroed PAD buffer, torch's kernels or the fused ones."""

    from tensorfold.cuda.nvfp4 import thor_dense
    from tensorfold.families.qwen4_exp.cuda import thor_shx
    from tensorfold.families.qwen4_exp.cuda.nvfp4_moe import _shared

    rows = x.shape[0]
    thor_dense.S.pad.zero_()
    g = thor_dense.linear(x, gu)
    act = torch.empty((rows, 3, width), dtype=torch.bfloat16, device=x.device)
    held = torch.empty((rows, 3, down.n), dtype=torch.bfloat16, device=x.device)
    slot = act[:, 2]
    prompt = held.dtype != torch.float32                     # moe(): bf16 slots are a prompt
    if fused:
        thor_shx.store(g, width, slot, rows)
        held[:, 2] = thor_shx.project(down, rows)
    else:
        gate, up = g[:, :width].to(torch.float32), g[:, width:].to(torch.float32)
        slot.copy_(((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32) * up).to(torch.bfloat16))
        held[:, 2] = _shared(down, slot, prompt, not prompt)  # moe()'s down: Linear.prefill, not thor_dense.linear
    return held[:, 2].contiguous(), slot.contiguous()


def _replay_matches(x, gu, down, width: int, reference: torch.Tensor, eager_slot: torch.Tensor) -> None:
    from tensorfold.cuda.nvfp4 import thor_dense
    from tensorfold.families.qwen4_exp.cuda import thor_shx

    rows = x.shape[0]
    static_x = x.clone()
    static_act = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):                     # compile, then a second eager pass, before the capture
            got = thor_dense.linear(static_x, gu)
            thor_shx.store(got, width, static_act, rows)
            thor_shx.project(down, rows)
        thor_dense.S.pad.zero_()
        thor_shx.begin_capture()
        graph.capture_begin(pool=torch.cuda.graph_pool_handle(), capture_error_mode="thread_local")
        try:
            got = thor_dense.linear(static_x, gu)
            thor_shx.store(got, width, static_act, rows)
            y = thor_shx.project(down, rows)
        finally:
            graph.capture_end()
            recorded = thor_shx.take_capture()
    torch.cuda.current_stream().wait_stream(side)
    graph.replay()
    torch.cuda.synchronize()
    assert recorded == 1                                   # the capture recorded the one store and did not run it
    assert _bits(y, reference)
    assert _bits(static_act, eager_slot)                   # the replay rewrote the act slot, not only y


def _fp8_prompt_matches_the_kill_switch(monkeypatch) -> None:
    """``--prefill-fp8``: ``want`` is false, so the fuse is not taken and y matches ``TENSORFOLD_THOR_SHX_FUSE=0``."""

    from types import SimpleNamespace

    from tensorfold.cuda import prompt_precision
    from tensorfold.cuda.nvfp4 import thor_dense
    from tensorfold.families.qwen4_exp.cuda import thor_shx
    from tensorfold.families.qwen4_exp.cuda.nvfp4_moe import MoE4, _shared

    def tail(x, gu, down, width: int):
        rows = x.shape[0]
        top_k = 2
        ex = SimpleNamespace(shared=SimpleNamespace(gu=gu, down=down), width=width, dims=down.n)
        buf = SimpleNamespace(act=torch.empty((rows, top_k + 1, width), dtype=torch.bfloat16, device=x.device),
                              y=torch.empty((rows, top_k + 1, down.n), dtype=torch.bfloat16, device=x.device))
        prompt = buf.y.dtype != torch.float32
        thor_dense.S.pad.zero_()
        if thor_shx.want(ex, x, buf, rows, top_k, prompt):
            thor_shx.store(_shared(ex.shared.gu, x, prompt, False), width, buf.act[:rows, top_k], rows)
            buf.y[:rows, top_k] = thor_shx.project(down, rows)
            taken = True
        else:
            buf.act[:rows, top_k] = MoE4.shared_act(ex, x, prompt)
            buf.y[:rows, top_k] = _shared(down, buf.act[:rows, top_k], prompt, not prompt)
            taken = False
        return buf.y[:rows, top_k].contiguous(), buf.act[:rows, top_k].contiguous(), taken, ex, buf

    with prompt_precision.using(True):
        # K of 64 makes prefill8's scale tensor non-contiguous; 256 is a shape that kernel already runs.
        for width, hidden, rows in ((256, 256, 1), (256, 256, 6)):
            gu, down = _mx_pair(width, hidden, seed=width + rows)
            x = torch.randn(rows, hidden, device="cuda").to(torch.bfloat16)
            with prompt_precision.using(False):
                _, _, bf16_taken, ex, buf = tail(x, gu, down, width)
            assert bf16_taken, (width, rows)               # the same fixture fuses on a bf16 prompt
            assert not thor_shx.want(ex, x, buf, x.shape[0], 2, True)
            monkeypatch.setenv(thor_shx.ENV, "0")
            y_off, act_off, off_taken, _, _ = tail(x, gu, down, width)
            monkeypatch.delenv(thor_shx.ENV, raising=False)
            assert thor_shx.active()
            before = thor_shx.S.calls
            y_on, act_on, on_taken, _, _ = tail(x, gu, down, width)
            assert not off_taken and not on_taken
            assert thor_shx.S.calls == before              # the fused kernel did not run
            assert _bits(y_on, y_off) and _bits(act_on, act_off), (width, rows)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_shx_fuse_keeps_every_rounds_bits(kv_dtype, tmp_path, monkeypatch):
    """Switch on, switch off, and eager: the same tokens and the same logits every round, both KV dtypes."""

    from tensorfold.cuda.nvfp4 import linear as lin_mod
    from tensorfold.cuda.nvfp4 import thor_dense
    from tensorfold.families.qwen4_exp.cuda import thor_shx
    from tensorfold.families.qwen4_exp.cuda.weights import load

    if not thor_dense.available():
        pytest.skip("sm_110 with TENSORFOLD_THOR_DENSE on only")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nvfp4_tiny import write

    w = load(write(tmp_path / "shx", mxfp8=True, ple_nvfp4=True, hidden=512), mtp=True)
    thor_dense.prepare(w)
    marked = 0
    for layer in w.layers:
        shared = layer.moe.experts.shared
        for lin in (shared.gu, shared.down):
            if isinstance(lin, lin_mod.Mx8Linear):
                lin.thor = True
                if getattr(lin, "thor_w", None) is None:
                    lin.thor_w = thor_dense.dequantize(lin)
                marked += 1
    assert marked >= 2
    prompts, counts = [[5, 17, 9, 3, 8, 2]], [20]
    runs = {}
    for name, graphs, fuse in (("eager", False, False), ("off", True, False), ("on", True, True)):
        monkeypatch.setenv(thor_shx.ENV, "1" if fuse else "0")
        assert thor_shx.active() is fuse
        before, before_replays = thor_shx.S.calls, thor_shx.S.replays
        thor_dense.S.pad.zero_()
        out, rounds, gr = _decode(w, prompts, counts, [None], graphs=graphs, capacity=256, kv_dtype=kv_dtype)
        runs[name] = (out, rounds, gr)
        if fuse:
            assert thor_shx.S.calls > before
            assert thor_shx.S.replays > before_replays      # graph replays ran the fused tail; a capture alone does not
        else:
            assert thor_shx.S.calls == before
            assert thor_shx.S.replays == before_replays
    monkeypatch.delenv(thor_shx.ENV, raising=False)
    eager, off, on = runs["eager"], runs["off"], runs["on"]
    assert on[0] == off[0] == eager[0]
    _same_rounds(on[1], off[1])
    _same_rounds(on[1], eager[1])
    assert on[2] is not None and on[2].replays.get("verify", 0) > 0, on[2].replays if on[2] else None
    assert not on[2].failed and not off[2].failed


def test_ple_conv_multi_keeps_each_streams_bits():
    """DPUSH MS-GRAPH: one launch over a concurrent step's rows (each row's stream and window start from tables, each
    stream's conv tail by address) writes the per-stream launches' rows and normed rows, bit for bit."""

    import numpy as np

    from tensorfold.families.qwen4_exp.cuda import glue

    g = torch.Generator(device="cuda").manual_seed(11)
    S, D, taps, dil = 4, 512, 4, 3
    wide, nt = S * D, (taps - 1) * dil
    windows = [3, 1, 7, 2]
    rows = sum(windows)

    def bf16(*shape, scale=1.0):
        return (torch.randn(shape, generator=g, device="cuda") * scale).to(torch.bfloat16)

    gated, h, nc, conv = bf16(rows, wide), bf16(rows, wide), bf16(wide), bf16(wide, taps, scale=0.3)
    pss = torch.rand((rows, S), generator=g, device="cuda") * D + 1.0
    tails = [bf16(nt, wide) for _ in windows]
    ref_h, ref_n, starts, a0 = h.clone(), torch.zeros_like(h), [], 0
    for s, n in enumerate(windows):
        glue.ple_conv(gated[a0:a0 + n], pss[a0:a0 + n], nc, tails[s], conv, ref_h[a0:a0 + n], ref_h[a0:a0 + n],
                      ref_n[a0:a0 + n], 1e-6, S, dil)
        starts.append(a0)
        a0 += n
    sid = torch.tensor(np.repeat(np.arange(len(windows)), windows), dtype=torch.int32, device="cuda")
    row0 = torch.tensor(np.repeat(starts, windows), dtype=torch.int32, device="cuda")
    ptrs = torch.tensor([t.data_ptr() for t in tails], dtype=torch.int64, device="cuda")
    out_h, out_n = h.clone(), torch.zeros_like(h)
    glue.ple_conv_multi(gated, pss, nc, ptrs, sid, row0, conv, out_h, out_h, out_n, 1e-6, S, dil)
    assert torch.equal(out_h, ref_h)
    assert torch.equal(out_n, ref_n)


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_concurrent_graphs_keep_the_per_stream_rounds_bits(kv_dtype, tmp_path, monkeypatch):
    """DPUSH MS-GRAPH: with the n-gram layer live, concurrent forwards replayed from graphs keyed by stream count and
    rows (one conv launch a step) give the eager rounds of e4fb57f's path (MULTI=0: per-stream convs) bit for bit,
    every round, and both verification forwards and MTP steps of two or more streams replay."""

    c = _cfg(ple=True)
    table = _bf16_table(tmp_path / "shard_0.safetensors", c.ngram(0).rows, c.ngram(0).dims)
    w = _model(ple=_ple(c, table, _Rand(3)))
    samplings = [None, Sampling(seed=1234, top_k=20, top_p=0.95), None, Sampling(seed=7, top_k=20, top_p=0.95)]
    counts = [48, 30, 40, 16]
    try:
        monkeypatch.setenv(thor_graphs.MULTI_ENV, "0")
        thor_graphs.multi_on.cache_clear()
        off_out, off_rounds, _ = _decode(w, PROMPTS, counts, samplings, graphs=False, capacity=1024,
                                         kv_dtype=kv_dtype)
        monkeypatch.setenv(thor_graphs.MULTI_ENV, "1")
        thor_graphs.multi_on.cache_clear()
        on_out, on_rounds, g = _decode(w, PROMPTS, counts, samplings, graphs=True, capacity=1024, kv_dtype=kv_dtype)
    finally:
        thor_graphs.multi_on.cache_clear()
    assert g.multi and g.streams > len(PROMPTS) and not g.failed
    assert on_out == off_out
    _same_rounds(on_rounds, off_rounds)
    multi = {k: v for k, v in g.hist.items() if "x" in k}
    assert sum(v[1] for k, v in multi.items() if k.startswith("verifyx")) > 0, g.hist
    assert sum(v[1] for k, v in multi.items() if k.startswith("mtpx")) > 0, g.hist
