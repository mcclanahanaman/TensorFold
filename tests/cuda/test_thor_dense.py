"""Jetson Thor's prompt GEMMs (``thor_dense``): MXFP8 weights exact in bf16, rows that keep their bits at any row count,
bf16 copies and per-call dequantizing alike, errors within TensorFold's prompt GEMM's (sm_110 only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.nvfp4 import linear as lin_mod  # noqa: E402
from tensorfold.cuda.nvfp4 import thor_dense  # noqa: E402

if torch.cuda.get_device_capability() != thor_dense.CAPABILITY or not thor_dense.available():
    pytest.skip("Jetson Thor (sm_110) with TENSORFOLD_THOR_DENSE on only", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))
from nvfp4_tiny import write  # noqa: E402


def _mx8(n: int, k: int, seed: int) -> lin_mod.Mx8Linear:
    g = torch.Generator().manual_seed(seed)
    codes = (torch.randn(n, k, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    scales = torch.randint(110, 132, (n, k // 32), generator=g, dtype=torch.uint8)
    return lin_mod.Mx8Linear.from_checkpoint(codes.cuda(), scales.cuda())


def _exact(m: lin_mod.Mx8Linear) -> torch.Tensor:
    return m.w8_rows().float() * torch.exp2(m.scale_rows().float() - 127).repeat_interleave(32, dim=1)


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y.double() - ref).norm() / ref.norm())


@pytest.fixture
def ready():
    old = (thor_dense.S.pad, thor_dense.S.staging, thor_dense.S.active)
    thor_dense.S.pad = torch.zeros((thor_dense.PAD * 10240,), dtype=torch.bfloat16, device="cuda")
    thor_dense.S.staging = torch.empty((2560 * 6144,), dtype=torch.bfloat16, device="cuda")
    thor_dense.S.active = True
    yield
    thor_dense.S.pad, thor_dense.S.staging, thor_dense.S.active = old


@pytest.mark.parametrize("n,k", [(2560, 640), (1280, 2560), (300, 6144)])
def test_an_mxfp8_weight_dequantizes_exactly(n: int, k: int) -> None:
    m = _mx8(n, k, n + k)
    assert torch.equal(thor_dense.dequantize(m).float(), _exact(m))


@pytest.mark.parametrize("n,k", [(2560, 6144), (1280, 2560)])
def test_rows_keep_their_bits_at_any_row_count(ready, n: int, k: int) -> None:
    m = _mx8(n, k, 7)
    x = torch.randn(2048 + 64, k, device="cuda").to(torch.bfloat16)
    full = thor_dense.linear(x[:2048], m)
    for rows in (1, 3, 64, 100, 128, 129, 777, 2047):
        assert torch.equal(thor_dense.linear(x[:rows], m), full[:rows]), rows
    assert torch.equal(thor_dense.linear(x[64:64 + 2048], m)[:2048 - 64], full[64:])


def test_copies_and_dequantizing_give_the_same_bits(ready) -> None:
    m = _mx8(2560, 6144, 3)
    x = torch.randn(600, 6144, device="cuda").to(torch.bfloat16)
    each_call = thor_dense.linear(x, m)
    m.thor_w = thor_dense.dequantize(m)
    try:
        assert torch.equal(thor_dense.linear(x, m), each_call)
    finally:
        del m.thor_w


@pytest.mark.parametrize("rows", [128, 2048])
def test_the_error_stays_within_the_portable_prompt_gemm(ready, rows: int) -> None:
    m = _mx8(1280, 2560, 5)
    x = torch.randn(rows, 2560, device="cuda").to(torch.bfloat16)
    ref = x.double() @ _exact(m).double().t()
    portable = lin_mod._prompt(lin_mod.MXFP8, m.w8, m.bs, 1.0, m.n, m.npad, x)
    assert _rel(thor_dense.linear(x, m), ref) <= 1.5 * _rel(portable, ref)


def test_a_prompt_on_the_thor_path_matches_the_portable_one(tmp_path: Path) -> None:
    """A tiny MXFP8 checkpoint: every prompt GEMM is marked, copies and per-call dequantizing prefill to the same
    bits, and the first token and streams stay close to TensorFold's portable prompt GEMMs."""

    from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill
    from tensorfold.families.qwen4_exp.cuda.weights import load

    tiny = write(tmp_path / "k2", mxfp8=True, ple_nvfp4=True, hidden=512)      # PLE kernels: 512-wide streams
    w = load(tiny, mtp=True, draft_vocab=None)
    old = (thor_dense.S.pad, thor_dense.S.staging, thor_dense.S.active)
    try:
        held = thor_dense.prepare(w)
        assert held["mxfp8"] > 0 and held["bf16"] > 0 and not held["copy"]
        downs = [lin for name, lin, f32 in thor_dense.faces(w) if f32]
        assert downs and not any(getattr(lin, "thor", False) for lin in downs)      # fp32 sums: the split-K kernel
        prompt = [(7 * i + 5) % 250 for i in range(53)]
        got = {}
        for label in ("dequant", "copy", "portable"):
            if label == "copy":
                for _, lin, _ in thor_dense.faces(w):
                    if getattr(lin, "thor", False) and getattr(lin, "w8", None) is not None:
                        lin.thor_w = thor_dense.dequantize(lin)
            thor_dense.set_active(label != "portable")
            e = Engine(w, capacity=256, max_rows=8, prefill_rows=16, graphs=False)
            first = prefill(e, prompt, None)
            got[label] = (first, e.last_streams.clone().float())
        assert got["copy"][0] == got["dequant"][0] and torch.equal(got["copy"][1], got["dequant"][1])
        assert torch.allclose(got["dequant"][1], got["portable"][1], rtol=0.05, atol=0.05)
    finally:
        thor_dense.release(w)
        thor_dense.S.pad, thor_dense.S.staging, thor_dense.S.active = old
