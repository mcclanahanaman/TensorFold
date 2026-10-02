"""Jetson Thor's fused prompt hyper-connections (``thor_hc``): the fused down projection with its activation, the mix
from the streams and the fused up projection store TensorFold's read-out's bits (glue.hc_normed, bf16.matmul's
split-K sums, glue.hc_act, glue.hc_mix) at any row count, and a Flash Next prompt keeps its bits with the path on
(sm_110 only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import bf16, glue, thor_hc  # noqa: E402

if not thor_hc.available():
    pytest.skip("Jetson Thor (sm_110) with TENSORFOLD_THOR_HC_FUSE on only", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write  # noqa: E402

DEV = "cuda"
S, D, LOW, EPS = 4, 2560, 320, 1e-6       # Flash Next's streams, hidden size and low rank


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))


def _weights(seed: int, inject: bool = True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    wide = S * D
    down = bf16.make_b16(torch.randn((LOW + (S if inject else 0), wide), generator=g, device=DEV) * 0.02)
    up = bf16.b16_from_rows(torch.randn((wide, LOW), generator=g, device=DEV) * 0.05)
    scale = (1.0 + 0.1 * torch.randn((wide,), generator=g, device=DEV)).contiguous()
    return down, up, scale


def _reference(h, pss, scale, down, up, inject: bool):
    """TensorFold's prompt read-out on bf16 kernels (forward._readout_b16 without the dense path): act, inject, up,
    mixed, normed."""

    rows = h.shape[0]
    normed = torch.empty_like(h)
    glue.hc_normed(h, pss, scale, normed, torch.empty((rows, S * D // 32), dtype=torch.float32, device=DEV), S, EPS)
    dn = bf16.matmul(normed, down, f32=True)
    act = torch.empty((rows, LOW), dtype=torch.bfloat16, device=DEV)
    inj = torch.empty((rows, S), dtype=torch.bfloat16, device=DEV) if inject else None
    glue.hc_act(dn, act, torch.empty((rows, LOW // 32), dtype=torch.float32, device=DEV), inj, S, LOW)
    upv = bf16.matmul(act, up.b)
    mixed = torch.empty((rows, D), dtype=torch.bfloat16, device=DEV)
    glue.hc_mix(upv, normed, mixed, torch.empty((rows, D // 32), dtype=torch.float32, device=DEV), S)
    return act, inj, upv, mixed, normed


def _h(rows: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    h = (torch.randn((rows, S * D), generator=g, device=DEV) * 1.5).to(torch.bfloat16)
    h[:, :7] = 0                                          # zeros and a quiet stretch inside the first slice
    pss = torch.empty((rows, D // 256, S), dtype=torch.float32, device=DEV)
    glue.hc_writeback(h, h, pss, S, 0)
    return h, pss


@pytest.mark.parametrize("rows", [1, 7, 128, 1000, 4096])
@pytest.mark.parametrize("inject", [True, False])
def test_the_fused_down_projection_stores_the_readouts_act_and_gates(rows: int, inject: bool) -> None:
    down, up, scale = _weights(11, inject)
    h, pss = _h(rows, rows)
    act0, inj0, _, _, _ = _reference(h, pss, scale, down, up, inject)
    act = torch.full_like(act0, 7.0)
    inj = torch.full_like(inj0, 7.0) if inject else None
    thor_hc.down_act(h, pss, scale, down, act, inj, S, LOW, EPS)
    assert _same(act, act0)
    if inject:
        assert _same(inj, inj0)


@pytest.mark.parametrize("bm,block_n,warps,stages", [(64, 128, 8, 3), (128, 64, 4, 3), (64, 64, 4, 2),
                                                     (128, 128, 8, 4)])
def test_the_fused_down_projections_bits_do_not_depend_on_its_tiles(bm: int, block_n: int, warps: int,
                                                                   stages: int) -> None:
    down, up, scale = _weights(12)
    h, pss = _h(300, 5)
    act0, inj0, _, _, _ = _reference(h, pss, scale, down, up, True)
    act, inj = torch.empty_like(act0), torch.empty_like(inj0)
    thor_hc.down_act(h, pss, scale, down, act, inj, S, LOW, EPS, bm=bm, block_n=block_n, num_warps=warps,
                     num_stages=stages)
    assert _same(act, act0) and _same(inj, inj0)


@pytest.mark.parametrize("bm,db,warps,stages", [(64, 64, 8, 2), (128, 64, 4, 2), (64, 32, 4, 2), (128, 32, 8, 3)])
def test_the_four_stream_up_projection_mixes_as_the_general_kernel(bm: int, db: int, warps: int, stages: int) -> None:
    down, up, scale = _weights(18)
    h, pss = _h(333, 21)
    act, _, upv, _, _ = _reference(h, pss, scale, down, up, True)
    want, got = torch.empty((333, D), dtype=torch.bfloat16, device=DEV), torch.full((333, D), 7.0, device=DEV,
                                                                                   dtype=torch.bfloat16)
    thor_hc.upmix(act, up.weight, h, pss, scale, want, S, EPS, up=torch.empty_like(upv))     # _upmix_h
    thor_hc.upmix(act, up.weight, h, pss, scale, got, S, EPS, bm=bm, db=db, num_warps=warps, num_stages=stages)
    assert _same(got, want)


def test_a_row_keeps_its_bits_whatever_rows_share_its_tile() -> None:
    down, up, scale = _weights(13)
    h, pss = _h(260, 9)
    act, inj = torch.empty((260, LOW), dtype=torch.bfloat16, device=DEV), torch.empty((260, S), dtype=torch.bfloat16,
                                                                                         device=DEV)
    thor_hc.down_act(h, pss, scale, down, act, inj, S, LOW, EPS)
    part, pinj = torch.empty_like(act[:37]), torch.empty_like(inj[:37])
    thor_hc.down_act(h[101:138], pss[101:138], scale, down, part, pinj, S, LOW, EPS)
    assert _same(part, act[101:138]) and _same(pinj, inj[101:138])


@pytest.mark.parametrize("rows", [1, 7, 300])
def test_the_mix_from_the_streams_is_hc_mix_on_the_normed_streams(rows: int) -> None:
    down, up, scale = _weights(14)
    h, pss = _h(rows, 3 * rows)
    _, _, upv, mixed0, _ = _reference(h, pss, scale, down, up, True)
    mixed = torch.full_like(mixed0, 7.0)
    thor_hc.mix(upv, h, pss, scale, mixed, S, EPS)
    assert _same(mixed, mixed0)


@pytest.mark.parametrize("rows", [1, 130, 1000])
def test_the_fused_up_projection_mixes_exactly_what_it_projects(rows: int) -> None:
    """Whatever the up projection's own bits, the fused kernel's mix is hc_mix of its rounded projection; its
    projection is TensorFold's bf16 kernel's up to the order of the products' sums."""

    down, up, scale = _weights(15)
    h, pss = _h(rows, 17)
    act, _, upv, _, normed = _reference(h, pss, scale, down, up, True)
    got_up, mixed = torch.empty_like(upv), torch.empty((rows, D), dtype=torch.bfloat16, device=DEV)
    thor_hc.upmix(act, up.weight, h, pss, scale, mixed, S, EPS, up=got_up)
    want = torch.empty_like(mixed)
    glue.hc_mix(got_up, normed, want, torch.empty((rows, D // 32), dtype=torch.float32, device=DEV), S)
    assert _same(mixed, want)
    ref = act.double() @ up.weight.double().t()
    assert float((got_up.double() - ref).norm() / ref.norm()) < 4e-3       # bf16 rounding of fp32 sums
    alone = torch.empty_like(mixed)
    thor_hc.upmix(act, up.weight, h, pss, scale, alone, S, EPS)             # without storing the projection
    assert _same(alone, mixed)


def test_the_check_compares_with_the_dense_path_and_prepare_follows_it() -> None:
    down, up, scale = _weights(16)
    hc = type("HC", (), {"down": bf16._Routed(down), "up": up, "scale": scale, "inject": True})()
    got = thor_hc.check(hc, 1000, EPS, S, LOW, torch.device(DEV))
    assert got["act"] and got["inject"] and got["mix"]
    assert not got["up_fused"] or got["mix_fused"]


def test_a_prompt_keeps_its_bits_with_the_path_on(tmp_path: Path) -> None:
    """A tiny NVFP4 checkpoint: the first token and every stream after the prompt are the same bytes with the fused
    read-out as with TensorFold's."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.weights import load

    tiny = write(tmp_path / "hc", ple_nvfp4=True, hidden=512)      # PLE kernels: 512-wide streams
    w = load(tiny, mtp=True, draft_vocab=None)
    assert thor_hc.prepare(w), thor_hc.S.note
    prompt = [(7 * i + 5) % 250 for i in range(53)]
    got = {}
    for on in (True, False):
        thor_hc.set_active(on)
        e = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=False)
        first = prefill(e, prompt, None)
        got[on] = (first, e.last_streams.clone())
    assert got[True][0] == got[False][0]
    assert _same(got[True][1], got[False][1])
