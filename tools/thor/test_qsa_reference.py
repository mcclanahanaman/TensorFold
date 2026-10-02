"""qsa_reference.py on a small QSA layer, on the CPU (or a GPU when present): no checkpoint, no TensorFold kernels.

    python -m pytest -q tools/thor/test_qsa_reference.py

``select`` and ``attend`` against a direct float64 computation (each row's blocks by float64 score, ties to lower ids,
plus its unfinished tail; a dense row's keys up to its own), for a bf16 and an 8-bit cache, and the case the K4-QSA
gate's referee run hit on 2026-10-01 (``thor_divergence.py reference ... --qsa64``: a CUDA "index out of bounds" in
a scatter, exit 134, after the wiring check had passed): in the slice of rows where a fresh prompt turns sparse, a
dense row's list slots still hold an earlier prompt's selections, keys past this prompt's end, and ``attend`` must not
read them. The scratch starts zeroed, so only a later prompt of the process met them.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsa_reference as qr  # noqa: E402
from dense_reference import bf16_rounded  # noqa: E402

BUDGET, RATIO = 16, 4                 # the rule's sizes, scaled down: 4 blocks of 4 keys a sparse row
H, HK, D, HI, DI = 4, 2, 32, 2, 8     # query heads, KV heads, head dim (one H32 group), index heads, index dim
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _scratch(rows: int, dev: str, stale: int | None = None) -> SimpleNamespace:
    ids = torch.zeros((rows, BUDGET + RATIO), dtype=torch.int32, device=dev)
    if stale is not None:
        ids.fill_(stale)                          # an earlier prompt's lists, keys past this prompt's end
    return SimpleNamespace(ratio=RATIO, budget=BUDGET, qsa=True, ids=ids,
                           nk=torch.zeros((rows,), dtype=torch.int32, device=dev),
                           sparse=torch.zeros((rows,), dtype=torch.int32, device=dev))


def _layer(n: int, dev: str, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    t = {"q": torch.randn((n, H, D), generator=g), "k": torch.randn((n, HK, D), generator=g),
         "v": torch.randn((n, HK, D), generator=g), "iq": torch.randn((n, HI, DI), generator=g),
         "pooled": torch.randn((n // RATIO, DI), generator=g)}
    return {k: v.to(torch.bfloat16).to(dev) for k, v in t.items()}


def _quantized(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """8-bit codes and fp16 scales a 32-value group (the referee dequantizes (code + 0.5) * s / 128)."""

    rows, hk, d = x.shape
    grp = x.float().view(rows, hk, d // 32, 32)
    s = (grp.abs().amax(-1) / 127.0).clamp_min(1e-3).half()
    code = (grp / (s.float()[..., None] / 128.0) - 0.5).round().clamp(-128, 127).to(torch.int8)
    return code.view(rows, hk, d), s


def _expected(layer: dict, p0: int, rows: int, keys: torch.Tensor, vals: torch.Tensor, scale: float,
              rotated: bool) -> torch.Tensor:
    """Each row's attention in float64, its keys chosen here from scratch (not from the scratch lists)."""

    top = BUDGET // RATIO
    out = torch.empty((rows, H, D), dtype=torch.bfloat16)
    for i in range(rows):
        end = p0 + i + 1
        complete = end // RATIO
        if complete > top:                        # sparse: the top blocks by float64 score, ties to lower ids
            q = layer["iq"][p0 + i].double().cpu()
            s = torch.relu(q @ layer["pooled"][:complete].double().cpu().T).sum(0) / math.sqrt(DI)
            order = sorted(range(complete), key=lambda b: (-float(s[b]), b))[:top]
            cols = sorted(k for b in order for k in range(b * RATIO, b * RATIO + RATIO))
            cols += list(range(RATIO * complete, end))
        else:
            cols = list(range(end))
        for j in range(HK):
            qq = layer["q"][p0 + i, j * (H // HK):(j + 1) * (H // HK)].double().cpu()
            w = torch.softmax(qq @ keys[cols, j].T * scale, dim=-1)
            o = w @ vals[cols, j]
            out[i, j * (H // HK):(j + 1) * (H // HK)] = bf16_rounded(qr._h32(o) if rotated else o)
    return out


def _run(layer: dict, p0: int, rows: int, dev: str, bits: int, stale: int | None = None):
    sc = _scratch(rows, dev, stale)
    qr.select(layer["iq"][p0:p0 + rows], layer["pooled"], p0, sc, rows)
    out = torch.empty((rows, H, D), dtype=torch.bfloat16, device=dev)
    if bits:
        kc, ks = _quantized(layer["k"])
        vc, vs = _quantized(layer["v"])
    else:
        kc, vc, ks, vs = layer["k"], layer["v"], None, None
    qr.attend(layer["q"][p0:p0 + rows], kc, vc, p0, sc, rows, D ** -0.5, out, ks, vs, bits)
    keys = qr._values(kc, ks, p0 + rows, bits).cpu()
    vals = qr._values(vc, vs, p0 + rows, bits).cpu()
    return sc, out.cpu(), _expected(layer, p0, rows, keys, vals, D ** -0.5, bool(bits))


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("bits", [0, 8])
def test_the_slice_where_rows_turn_sparse_ignores_an_earlier_prompts_lists(dev, bits):
    """Rows 16..31: 17..19 keys dense, the rest sparse (more than 4 complete blocks); lists left at 10000."""

    layer = _layer(64, dev)
    sc, got, want = _run(layer, 16, 16, dev, bits, stale=10000)
    assert sc.sparse.cpu().tolist() == [0] * 3 + [1] * 13
    assert torch.equal(got, want)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("bits", [0, 8])
@pytest.mark.parametrize("p0, rows", [(0, 16), (16, 16), (28, 36)])
def test_float64_qsa_equals_a_direct_computation(dev, bits, p0, rows):
    """Dense rows, the turn, and sparse rows past it (two ``ROWS`` slices) from a fresh, zeroed scratch."""

    layer = _layer(64, dev, seed=1)
    _, got, want = _run(layer, p0, rows, dev, bits)
    assert torch.equal(got, want)


@pytest.mark.parametrize("dev", DEVICES)
def test_a_list_past_the_rows_end_is_refused_by_name(dev):
    """A sparse row's own list holding a key past its end (a selection bug) stops the run with a message, not a
    device assert."""

    layer = _layer(64, dev)
    sc = _scratch(16, dev)
    qr.select(layer["iq"][32:48], layer["pooled"], 32, sc, 16)
    sc.ids[5, 0] = 47                              # row 5 ends at key 37
    out = torch.empty((16, H, D), dtype=torch.bfloat16, device=dev)
    with pytest.raises(RuntimeError, match="past"):
        qr.attend(layer["q"][32:48], layer["k"], layer["v"], 32, sc, 16, D ** -0.5, out, None, None, 0)
