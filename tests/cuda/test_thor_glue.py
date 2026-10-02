"""Jetson Thor's prompt expert glue (``thor_glue``): with the scatter skipped, the routed outputs sit in the experts'
permuted rows where the plan's tiles point; the write-back that reads them there stores the bytes of
``glue.hc_writeback``'s mode 2 on the scattered slots; the one-kernel SwiGLU stores the shared expert's torch bits on
every bf16 gate; and a Flash Next MoE layer written back with the path on equals the scatter path byte for byte, while
decode buffers and more than one GPU keep the scatter (sm_110 only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import experts as grouped  # noqa: E402
from tensorfold.cuda import moe as moe_mod  # noqa: E402
from tensorfold.cuda.nvfp4 import thor_experts as thor  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import glue, nvfp4_moe, thor_glue  # noqa: E402

if not (thor_glue.available() and thor.available()):
    pytest.skip("Jetson Thor (sm_110), FlashInfer's W4A16 MoE and TENSORFOLD_THOR_MOE_GLUE on only",
                allow_module_level=True)

DEV = "cuda"
E, D, NI, TOP, S = 16, 2560, 640, 4, 4      # Flash Next's expert and stream shapes, fewer experts and slots


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def _proj(n: int, k: int, g: torch.Generator, glob: torch.Tensor):
    words = torch.randint(0, 256, (E, n, k // 2), generator=g, dtype=torch.uint8)
    scales = torch.randint(0x28, 0x40, (E, n, k // 16), generator=g, dtype=torch.uint8)
    return words.to(DEV), scales.to(DEV).view(torch.float8_e4m3fn), glob.to(DEV)


@pytest.fixture(scope="module")
def layer():
    g = torch.Generator().manual_seed(23)
    shared_s2 = torch.rand(E, generator=g) * 0.02 + 0.005             # gate and up share weight_scale_2
    gate, up = _proj(NI, D, g, shared_s2), _proj(NI, D, g, shared_s2)
    down = _proj(D, NI, g, torch.rand(E, generator=g) * 0.02 + 0.005)
    shared = tuple((torch.randn(o, i, generator=g) * 0.02).to(torch.bfloat16).to(DEV)
                   for o, i in ((NI, D), (NI, D), (D, NI)))
    ex = nvfp4_moe.moe4_from_checkpoint(gate, up, down, shared)
    assert isinstance(ex.routed_experts, thor.ThorExperts)
    router = (torch.randn(E + 1, D, generator=g) * 0.1).to(torch.bfloat16).to(DEV)
    cfg = SimpleNamespace(num_experts_per_tok=TOP, num_experts=E, moe_intermediate_size=NI, hidden_size=D)
    return ex, router, cfg


@pytest.fixture()
def ready(monkeypatch):
    """The path as ``prepare`` leaves it on this device (the check passed), restored after the test."""

    monkeypatch.setattr(thor_glue.S, "ready", True)
    monkeypatch.setattr(thor_glue.S, "active", True)


def _x(rows: int, seed: int) -> torch.Tensor:
    return (torch.randn(rows, D, generator=torch.Generator().manual_seed(seed)) * 0.5).to(torch.bfloat16).to(DEV)


def _buffers(rows: int, cfg, prefill: bool = True):
    buf = moe_mod.MoEBuffers(rows, cfg, DEV, prefill=prefill)
    buf.thor = thor.Scratch(rows, TOP, E, NI, D, DEV)
    return buf


def _streams(rows: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    h = (torch.randn((rows, S * D), generator=g, device=DEV) * 1.5).to(torch.bfloat16)
    inj = (torch.rand((rows, S), generator=g, device=DEV) * 2.0).to(torch.bfloat16)
    return h, inj


@pytest.mark.parametrize("rows", [6, 700, 4096])
def test_without_the_scatter_each_pairs_output_sits_in_the_row_its_tile_names(layer, rows):
    ex, router, cfg = layer
    routed = ex.routed_experts
    x = _x(rows, rows)
    g = torch.Generator().manual_seed(rows + 1)
    picks = torch.stack([torch.randperm(E, generator=g)[:TOP] for _ in range(rows)])
    picks = torch.cat([picks, torch.full((rows, 1), E)], dim=1).to(torch.int32).to(DEV).contiguous()
    plan = grouped.Plan(rows, TOP + 1, E + 1, DEV, prefill=True)
    grouped.route(picks, plan, thor.tactic(rows)[0])
    y0 = torch.zeros((rows, TOP + 1, D), dtype=torch.bfloat16, device=DEV)
    y1 = torch.zeros_like(y0)
    scr = thor.Scratch(rows, TOP, E, NI, D, DEV)
    thor.run(x, routed, plan, y0, rows, TOP, scr)
    thor.run(x, routed, plan, y1, rows, TOP, scr, scatter=False)
    torch.cuda.synchronize()
    assert not y1.any()                                                             # y is not written
    pos = scr.pos[:rows * (TOP + 1)].view(rows, TOP + 1)[:, :TOP].long()
    assert _same(scr.hidden[pos], y0[:, :TOP])


@pytest.mark.parametrize("rows", [1, 77, 1000, 4096])
def test_the_write_back_from_the_experts_rows_stores_mode_2s_bytes(rows):
    got = thor_glue.check(rows, D, S, TOP + 1, NI, torch.device(DEV))
    assert got["streams"] and got["pss"] and got["split"], got


def test_the_one_kernel_swiglu_stores_the_torch_bits_on_every_bf16_gate():
    got = thor_glue.check(64, D, S, 11, NI, torch.device(DEV))
    assert got["act"] and got["act_slot"], got


@pytest.mark.parametrize("rows", [5, 4096])
def test_the_swiglu_on_projected_rows_and_into_a_strided_slot(rows):
    g = (torch.randn((rows, 2 * NI), generator=torch.Generator(device=DEV).manual_seed(rows), device=DEV) * 3.0
         ).to(torch.bfloat16)
    want = thor_glue.swiglu_ref(g, NI)
    slot = torch.full((rows, TOP + 1, NI), 7.0, dtype=torch.bfloat16, device=DEV)
    thor_glue.swiglu(g, NI, slot[:, TOP])
    assert _same(slot[:, TOP], want)
    assert bool((slot[:, :TOP] == 7.0).all())                                     # the other slots untouched


@pytest.mark.parametrize("rows", [9, 700, 4096])
def test_a_moe_layer_written_back_with_the_path_on_equals_the_scatter_path(layer, ready, rows):
    ex, router, cfg = layer
    x = _x(rows, 100 + rows)
    h, inj = _streams(rows, 200 + rows)
    out = {}
    for on in (False, True):
        thor_glue.set_active(on)
        buf = _buffers(rows, cfg)
        nvfp4_moe.moe(x, None, router, ex, buf, cfg, unscattered=True)
        assert (buf.kept is not None) == on
        hh = h.clone()
        pss = torch.full((rows, D // 256, S), float("nan"), dtype=torch.float32, device=DEV)
        glue.hc_writeback(hh, hh, pss, S, 2, inject=inj, y=buf.kept if on else buf.y[:rows], wts=buf.wts[:rows])
        shared = buf.kept.shared if on else buf.y[:rows, TOP]
        out[on] = (hh, pss, buf.act[:rows, TOP].clone(), shared.clone())
    for a, b in zip(out[False], out[True]):
        assert _same(a, b)


def test_decode_buffers_and_more_than_one_gpu_keep_the_scatter(layer, ready):
    ex, router, cfg = layer
    x = _x(8, 5)
    buf = _buffers(8, cfg)
    nvfp4_moe.moe(x, None, router, ex, buf, cfg, unscattered=False)               # ranks gather buf.y
    assert buf.kept is None
    dec = _buffers(8, cfg, prefill=False)                                          # a decode window's plan
    nvfp4_moe.moe(x, None, router, ex, dec, cfg, unscattered=True)
    assert dec.kept is None


def test_the_engines_decode_windows_keep_the_scatter(layer, ready):
    """The engine's decode windows run their experts on prompt-arithmetic buffers (``Buffers(moe_prefill=True)``: bf16
    slots and a prompt plan, as ``_buffers`` builds them); ``forward.moe_block`` leaves the slots in the experts' rows
    for prompt buffers (``Buffers.prefill``) only, so a decode window's MoE is the scatter path."""

    from tensorfold.families.qwen4_exp.cuda import forward

    ex, router, cfg = layer
    w = SimpleNamespace(cfg=SimpleNamespace(top_k=TOP, experts=E, moe_width=NI, hidden=D), comm=None, x3=None)
    lw = SimpleNamespace(moe=SimpleNamespace(router=router, experts=ex))
    x = _x(8, 7)
    for prefill in (False, True):
        b = SimpleNamespace(prefill=prefill, moe=_buffers(8, cfg), mixed=x, xs_mixed=x)
        mode, y, wts = forward.moe_block(lw, w, b, 8)
        assert mode == 2 and wts.shape == (8, TOP + 1)
        if prefill:
            assert isinstance(y, thor_glue.Rows) and b.moe.kept is y
        else:
            assert isinstance(y, torch.Tensor) and b.moe.kept is None and y.data_ptr() == b.moe.y.data_ptr()


def test_the_switch_keeps_the_scatter(layer, monkeypatch):
    ex, router, cfg = layer
    monkeypatch.setattr(thor_glue.S, "ready", True)
    monkeypatch.setattr(thor_glue.S, "active", thor_glue.S.active)                 # restored after the test
    thor_glue.set_active(False)
    buf = _buffers(8, cfg)
    nvfp4_moe.moe(_x(8, 6), None, router, ex, buf, cfg, unscattered=True)
    assert buf.kept is None
    monkeypatch.setattr(thor_glue.S, "ready", False)
    thor_glue.set_active(True)                                                     # not prepared: stays off
    assert not thor_glue.S.active
