"""Jetson Thor's warp-specialized prompt hyper-connection kernels (``thor_hc2``): the down projection with its activation
stores TensorFold's read-out's bits (glue.hc_normed, bf16.matmul's split-K sums in slice order, glue.hc_act) and the up
projection with the mix stores thor_hc's fused kernel's bits, at any row count, with zero rows and negative scales,
and a Flash Next prompt keeps its bits with them on (sm_110 only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import bf16, glue, thor_hc  # noqa: E402

if not thor_hc.available() or thor_hc._hc2() is None:
    pytest.skip("Jetson Thor (sm_110) with Gluon and TENSORFOLD_THOR_HC_FUSE2 on only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import thor_hc2  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write  # noqa: E402

DEV = "cuda"
S, D, LOW, EPS = 4, 2560, 320, 1e-6       # Flash Next's streams, hidden size and low rank


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def _weights(seed: int, inject: bool = True, negative: bool = False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    wide = S * D
    down = bf16.make_b16(torch.randn((LOW + (S if inject else 0), wide), generator=g, device=DEV) * 0.02)
    up = bf16.b16_from_rows(torch.randn((wide, LOW), generator=g, device=DEV) * 0.05)
    scale = 1.0 + 0.1 * torch.randn((wide,), generator=g, device=DEV)
    if negative:
        scale[::5] *= -1.0                                # negative products of zero streams: signed zeros
    return down, up, scale.contiguous()


def _h(rows: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    h = (torch.randn((rows, S * D), generator=g, device=DEV) * 1.5).to(torch.bfloat16)
    h[:, :7] = 0
    h[rows // 2] = 0                                      # a zero row: every product of its down projection is zero
    pss = torch.empty((rows, D // 256, S), dtype=torch.float32, device=DEV)
    glue.hc_writeback(h, h, pss, S, 0)
    return h, pss


def _reference(h, pss, scale, down, inject: bool):
    rows = h.shape[0]
    normed = torch.empty_like(h)
    glue.hc_normed(h, pss, scale, normed, torch.empty((rows, S * D // 32), dtype=torch.float32, device=DEV), S, EPS)
    dn = bf16.matmul(normed, down, f32=True)
    act = torch.empty((rows, LOW), dtype=torch.bfloat16, device=DEV)
    inj = torch.empty((rows, S), dtype=torch.bfloat16, device=DEV) if inject else None
    glue.hc_act(dn, act, torch.empty((rows, LOW // 32), dtype=torch.float32, device=DEV), inj, S, LOW)
    return act, inj


def _down(h, pss, scale, down, act, inj):
    thor_hc2.down_act(h, pss, scale, down, act, inj, S, LOW, EPS, bf16.split_k(down.n, down.k))


@pytest.mark.parametrize("rows", [1, 7, 128, 129, 1000, 4096])
@pytest.mark.parametrize("inject", [True, False])
def test_the_down_projection_stores_the_readouts_act_and_gates(rows: int, inject: bool) -> None:
    down, _, scale = _weights(11, inject)
    h, pss = _h(rows, rows)
    act0, inj0 = _reference(h, pss, scale, down, inject)
    act = torch.full_like(act0, 7.0)
    inj = torch.full_like(inj0, 7.0) if inject else None
    _down(h, pss, scale, down, act, inj)
    assert _same(act, act0)
    if inject:
        assert _same(inj, inj0)


def test_signed_zeros_match_with_negative_scales() -> None:
    down, _, scale = _weights(12, True, negative=True)
    h, pss = _h(600, 5)
    act0, inj0 = _reference(h, pss, scale, down, True)
    act, inj = torch.empty_like(act0), torch.empty_like(inj0)
    _down(h, pss, scale, down, act, inj)
    assert _same(act, act0) and _same(inj, inj0)


def test_a_row_keeps_its_bits_whatever_rows_share_its_tile() -> None:
    down, _, scale = _weights(13)
    h, pss = _h(700, 9)
    act, inj = torch.empty((700, LOW), dtype=torch.bfloat16, device=DEV), torch.empty((700, S), dtype=torch.bfloat16,
                                                                                         device=DEV)
    _down(h, pss, scale, down, act, inj)
    part, pinj = torch.empty_like(act[:300]), torch.empty_like(inj[:300])
    _down(h[101:401], pss[101:401], scale, down, part, pinj)
    assert _same(part, act[101:401]) and _same(pinj, inj[101:401])


@pytest.mark.parametrize("rows", [1, 7, 130, 1000, 4096])
@pytest.mark.parametrize("negative", [False, True])
def test_the_up_projection_mixes_as_thor_hcs_fused_kernel(rows: int, negative: bool) -> None:
    down, up, scale = _weights(18, True, negative)
    h, pss = _h(rows, 21 + rows)
    act, _ = _reference(h, pss, scale, down, True)
    want, got = torch.empty((rows, D), dtype=torch.bfloat16, device=DEV), torch.full((rows, D), 7.0, device=DEV,
                                                                                    dtype=torch.bfloat16)
    upv = torch.empty((rows, S * D), dtype=torch.bfloat16, device=DEV)
    thor_hc.upmix(act, up.weight, h, pss, scale, want, S, EPS, up=upv)        # its projection stored too
    thor_hc2.upmix(act, up.weight, h, pss, scale, got, S, EPS)
    assert _same(got, want)
    normed = torch.empty_like(h)
    glue.hc_normed(h, pss, scale, normed, torch.empty((rows, S * D // 32), dtype=torch.float32, device=DEV), S, EPS)
    mixed = torch.empty_like(want)
    glue.hc_mix(upv, normed, mixed, torch.empty((rows, D // 32), dtype=torch.float32, device=DEV), S)
    assert _same(got, mixed)                              # glue.hc_mix of the projection, from h


def test_the_check_compares_them_with_the_readout() -> None:
    down, up, scale = _weights(16)
    hc = type("HC", (), {"down": bf16._Routed(down), "up": up, "scale": scale, "inject": True})()
    got = thor_hc.check(hc, 1000, EPS, S, LOW, torch.device(DEV))
    assert got["act2"]
    assert not got["up_fused"] or got["mix_fused2"]


def test_small_prompt_passes_keep_the_triton_kernels() -> None:
    assert not thor_hc2.fits(thor_hc2.MIN_ROWS - 1, S * D, S, LOW, D // 256, LOW + S, 32)
    assert thor_hc2.fits(thor_hc2.MIN_ROWS, S * D, S, LOW, D // 256, LOW + S, 32)
    assert not thor_hc2.fits(4096, 3 * D, 3, LOW, D // 256, LOW + 3, 32)           # three streams


def test_a_prompt_keeps_its_bits_with_the_warp_specialized_kernels(tmp_path: Path) -> None:
    """A tiny NVFP4 checkpoint and a prompt with a 512-row pass: the first token and every stream after the prompt are
    the same bytes with thor_hc2 as with thor_hc's Triton kernels and TensorFold's read-out."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.weights import load

    tiny = write(tmp_path / "hc2", ple_nvfp4=True, hidden=512)      # 512-wide streams, low rank 64
    w = load(tiny, mtp=True, draft_vocab=None)
    assert thor_hc.prepare(w), thor_hc.S.note
    assert thor_hc.S.fuse2, thor_hc.S.note
    prompt = [(7 * i + 5) % 250 for i in range(600)]
    got = {}
    for name, active, two in (("hc2", True, True), ("hc", True, False), ("tf", False, False)):
        thor_hc.set_active(active)
        thor_hc.S.fuse2 = two
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=512, graphs=False)
        first = prefill(e, prompt, None)
        got[name] = (first, e.last_streams.clone())
    thor_hc.set_active(True)
    thor_hc.S.fuse2 = True
    assert got["hc2"][0] == got["hc"][0] == got["tf"][0]
    assert _same(got["hc2"][1], got["hc"][1]) and _same(got["hc2"][1], got["tf"][1])
