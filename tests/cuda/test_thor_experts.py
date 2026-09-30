"""Thor's routed NVFP4 experts (FlashInfer's W4A16 grouped GEMM on sm_110): the checkpoint's weights exactly, each
pair's product to fp32 sums, a pair's bits alone or among any other pairs and under CUDA graph replays, the switch."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4 import format as fmt
from tensorfold.cuda.nvfp4 import thor_experts as thor

if not thor.available():
    pytest.skip("Jetson Thor (sm_110) with FlashInfer's W4A16 MoE only", allow_module_level=True)

E, D, NI, TOP = 16, 2560, 640, 4          # Flash Next's expert shapes, fewer experts


def _proj(n: int, k: int, g: torch.Generator, glob: torch.Tensor):
    words = torch.randint(0, 256, (E, n, k // 2), generator=g, dtype=torch.uint8)
    scales = torch.randint(0x28, 0x40, (E, n, k // 16), generator=g, dtype=torch.uint8)
    return words, scales, glob


def _dense(p, e: int) -> torch.Tensor:
    w, s, g = p
    return torch.from_numpy(fmt.dequant("nvfp4", w[e].numpy(), s[e].numpy(), float(g[e]))).double()


@pytest.fixture(scope="module")
def layer():
    g = torch.Generator().manual_seed(11)
    shared = torch.rand(E, generator=g) * 0.02 + 0.005                  # gate and up share weight_scale_2
    gate, up = _proj(NI, D, g, shared), _proj(NI, D, g, shared)
    down = _proj(D, NI, g, torch.rand(E, generator=g) * 0.02 + 0.005)
    cuda = [tuple(t.cuda() for t in p) for p in (gate, up, down)]
    return thor.make(*cuda), nvx.make(*cuda), gate, up, down


def _picks(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    routed = torch.stack([torch.randperm(E, generator=g)[:TOP] for _ in range(rows)])
    return torch.cat([routed, torch.full((rows, 1), E)], dim=1).to(torch.int32).cuda().contiguous()


def _run(ex, x, picks, scratch=None):
    rows = x.shape[0]
    plan = grouped.Plan(rows, TOP + 1, E + 1, "cuda", prefill=True)
    grouped.route(picks, plan, thor.tactic(rows)[0])
    y = torch.zeros((rows, TOP + 1, D), dtype=torch.bfloat16, device="cuda")
    scratch = scratch or thor.Scratch(rows, TOP, E, NI, D, "cuda")
    thor.run(x, ex, plan, y, rows, TOP, scratch)
    torch.cuda.synchronize()
    return y


def _x(rows: int, seed: int) -> torch.Tensor:
    return (torch.randn(rows, D, generator=torch.Generator().manual_seed(seed)) * 0.5).to(torch.bfloat16).cuda()


def test_layout_holds_the_checkpoint_weights_exactly(layer):
    ex, _, gate, up, down = layer
    for e in (0, 7, 15):
        for which, p in (("gate", gate), ("up", up), ("down", down)):
            assert torch.equal(thor.dense(ex, e, which).cpu().double(), _dense(p, e)), (e, which)


def test_each_pair_is_its_experts_swiglu_product(layer):
    ex, _, gate, up, down = layer
    x, picks = _x(6, 1), _picks(6, 2)
    y = _run(ex, x, picks)
    assert torch.equal(y[:, TOP], torch.zeros_like(y[:, TOP]))                     # the shared slot untouched
    for r in range(6):
        for s in range(TOP):
            e = int(picks[r, s])
            xv = x[r].double().cpu()
            gv, uv = xv @ _dense(gate, e).t(), xv @ _dense(up, e).t()
            h = (gv / (1 + torch.exp(-gv)) * uv).float().to(torch.bfloat16).double()
            want = h @ _dense(down, e).t()
            err = float((y[r, s].double().cpu() - want).abs().max() / want.abs().max())
            assert err < 1e-2, (r, s, err)


def test_pairs_keep_their_bits_alone_among_others_and_across_tactics(layer):
    ex = layer[0]
    x, picks = _x(700, 3), _picks(700, 4)
    big = _run(ex, x, picks)                                                    # a prompt-sized tactic
    for lo, n in ((0, 1), (3, 4), (100, 64), (200, 512)):
        small = _run(ex, x[lo:lo + n].contiguous(), picks[lo:lo + n].contiguous())
        assert torch.equal(small[:, :TOP], big[lo:lo + n, :TOP]), (lo, n)


def test_a_captured_call_replays_any_routing(layer):
    ex = layer[0]
    rows = 8
    scratch = thor.Scratch(rows, TOP, E, NI, D, "cuda")
    plan = grouped.Plan(rows, TOP + 1, E + 1, "cuda", prefill=True)
    x = _x(rows, 5)
    picks = _picks(rows, 6)
    y = torch.zeros((rows, TOP + 1, D), dtype=torch.bfloat16, device="cuda")
    tile = thor.tactic(rows)[0]

    def call():
        grouped.route(picks, plan, tile)
        thor.run(x, ex, plan, y, rows, TOP, scratch)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for seed in (7, 8, 9):
        x.copy_(_x(rows, seed))
        picks.copy_(_picks(rows, seed + 100))
        y.zero_()
        graph.replay()
        torch.cuda.synchronize()
        want = _run(ex, x.clone(), picks.clone())
        assert torch.equal(y[:, :TOP], want[:, :TOP]), seed


def test_the_portable_kernel_agrees_within_bf16(layer):
    ex, stock = layer[0], layer[1]
    x, picks = _x(32, 10), _picks(32, 11)
    y = _run(ex, x, picks)
    plan = grouped.Plan(32, TOP + 1, E + 1, "cuda", prefill=True)
    grouped.route(picks, plan, nvx.PREFILL_TILE)
    act = torch.zeros((32 * (TOP + 1), NI), dtype=torch.bfloat16, device="cuda")
    ref = torch.zeros((32, TOP + 1, D), dtype=torch.bfloat16, device="cuda")
    nvx.gate_up(x, stock, plan, act, 32, skip=E)
    nvx.down(act, stock, plan, ref.view(-1, D), 32, skip=E)
    a, b = y[:, :TOP].double(), ref[:, :TOP].double()
    assert float((a - b).norm() / b.norm()) < 1e-2


def test_the_switch_keeps_the_portable_kernel(monkeypatch):
    monkeypatch.setenv(thor.ENV, "0")
    thor.available.cache_clear()
    try:
        assert not thor.available()
    finally:
        monkeypatch.delenv(thor.ENV)
        thor.available.cache_clear()
    assert thor.available()


def test_scratch_covers_every_row_count():
    for rows in (1, 4, 5, 64, 65, 512, 513, 2048, 2049, 4096):
        need = thor.slots(rows, 10, 512)
        assert thor.scratch_rows(4096, 10, 512) >= need
        assert thor.scratch_rows(rows, 10, 512) >= need
    assert np.isclose(thor.scratch_bytes(2048, 10, 512, 640, 2560) / 2**20,
                      thor.scratch_rows(2048, 10, 512) * 3200 * 2 / 2**20, rtol=1e-3)
