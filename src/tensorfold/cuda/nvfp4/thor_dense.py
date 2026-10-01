"""Flash Next's dense prompt GEMMs on Jetson Thor (sm_110): cuBLAS bf16 on the exact weights, fp32 sums (THOR-K2).

A prompt pass's DeltaNet and attention projections and shared expert (MXFP8 in local-inference-lab's checkpoint), and
its hyper-connection up, n-gram key/value and MTP projections (bf16), run as bf16 tensor-core GEMMs on cuBLAS with
fp32 accumulation (``allow_bf16_reduced_precision_reduction`` off) instead of TensorFold's portable prompt GEMMs;
decode keeps its lane kernels. The hyper-connection down projection, whose fp32 sums feed its activation, stays on
TensorFold's split-K kernel: cuBLAS's fp32 sums over its K of 10240 had 4-39x that kernel's error and changed bits with
the row count. Each value of an MXFP8 weight, e4m3 x 2^(e-127), is exact in bf16 (``fold``):
``copy`` (the default) keeps a bf16 copy of each such weight, made at load and counted in the startup estimate (about
5.5 GiB on local-inference-lab's checkpoint, that much less room for the KV caches, for about 3% faster prompts);
``dequant`` rebuilds the one a call needs in a staging buffer first (82 MiB). Both give the same bits. A call of fewer
than PAD rows runs on PAD: from about 100 rows up cuBLAS keeps a row's bits whatever the call's row count or first row
at these shapes (tools/thor/cublas_row_invariance.py); below 64 it did not.

``TENSORFOLD_THOR_DENSE=0`` keeps the portable kernels; ``TENSORFOLD_THOR_DENSE_WEIGHTS=dequant`` keeps no copies.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import torch

ENV = "TENSORFOLD_THOR_DENSE"
WEIGHTS_ENV = "TENSORFOLD_THOR_DENSE_WEIGHTS"
CAPABILITY = (11, 0)
PAD = 128                   # the fewest rows a call runs on


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


def copies() -> bool:
    """Whether MXFP8 weights keep bf16 copies (``copy``, the default) or are dequantized at each call (``dequant``)."""

    value = os.environ.get(WEIGHTS_ENV, "copy").strip().lower()
    if value not in ("copy", "dequant"):
        raise ValueError(f"{WEIGHTS_ENV}={value}: copy or dequant")
    return value == "copy"


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110 and not switched off, decided once; then cuBLAS's bf16 GEMMs sum in fp32 in this process."""

    if _off() or not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != CAPABILITY:
        return False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return True


class _State:
    active = False                          # prompt GEMMs marked ``thor`` take this path (``prepare`` turns it on)
    staging: torch.Tensor | None = None     # one MXFP8 weight dequantized for its call (``dequant``)
    pad: torch.Tensor | None = None         # PAD rows of the widest input, for short calls


S = _State()


def use(lin) -> bool:
    """Whether a prompt GEMM on ``lin`` runs here."""

    return S.active and getattr(lin, "thor", False)


def set_active(on: bool) -> None:
    """Prepared weights on this path (True) or on TensorFold's portable kernels (False): for A/B runs and tests."""

    S.active = bool(on) and S.pad is not None


@functools.lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def dequant(W8, BS, OUT, N, K: tl.constexpr):
        """Program (64 rows, 64 inputs): the lane matmul's fragment order holds a row's 64 e4m3 bytes together
        (column c * 16 + half * 8 + byte for input half * 32 + (byte // 4) * 16 + (byte % 4 // 2) * 8 + c * 2 +
        byte % 2); each byte times its 32 inputs' 2^(e - 127), exact in bf16, into OUT [N, K] row-major."""

        nb = tl.program_id(0)
        kb = tl.program_id(1)
        tile = (nb * (K // 64) + kb).to(tl.int64)
        r = tl.arange(0, 64)
        s = tl.arange(0, 64)
        code = tl.load(W8 + tile * 4096 + r[:, None] * 64 + s[None, :])
        v = code.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        half = (s // 8) % 2
        e = tl.load(BS + tile * 128 + r[:, None] * 2 + half[None, :]).to(tl.int32)
        scale = (e << 23).to(tl.float32, bitcast=True)        # 2^(e - 127), exact (``fold``: e in 3..134)
        byte = s % 8
        k = half * 32 + (byte // 4) * 16 + (byte % 4 // 2) * 8 + (s // 16) * 2 + byte % 2
        n = nb * 64 + r
        tl.store(OUT + n[:, None].to(tl.int64) * K + kb * 64 + k[None, :], (v * scale).to(tl.bfloat16),
                 mask=(n < N)[:, None])

    return dequant


def dequantize(lin, out: torch.Tensor | None = None) -> torch.Tensor:
    """An MXFP8 weight (``Mx8Linear``) as its exact bf16 rows [n, K]; ``out``: a flat buffer of n * K or more."""

    n, k = lin.n, lin.k
    dst = (torch.empty((n * k,), dtype=torch.bfloat16, device=lin.w8.device) if out is None else out)[:n * k]
    _kernel()[(lin.npad // 64, k // 64)](lin.w8, lin.bs, dst, n, K=k, num_warps=4)
    return dst.view(n, k)


def _weight(lin) -> torch.Tensor:
    """``lin``'s bf16 rows [N, K]: its copy (``copy``), the staging buffer (an MXFP8 weight) or the stored bf16."""

    held = getattr(lin, "thor_w", None)
    if held is not None:
        return held
    if getattr(lin, "w8", None) is not None:
        return dequantize(lin, S.staging)
    return lin.weight


def linear(x: torch.Tensor, lin, out: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] @ lin.T on cuBLAS: [M, N] bf16, summed in fp32; fewer than PAD rows run on PAD rows (the rest the pad
    buffer's), so a row's bits do not depend on M."""

    w = _weight(lin)
    if x.dtype != torch.bfloat16 or x.stride(-1) != 1:
        x = x.to(torch.bfloat16).contiguous()
    m, k = x.shape
    if m < PAD:
        rows = S.pad[:PAD * k].view(PAD, k)
        rows[:m].copy_(x)
        y = torch.mm(rows, w.t())[:m]
    elif out is not None and out.is_contiguous() and out.dtype == torch.bfloat16 and out.shape == (m, w.shape[0]):
        return torch.mm(x, w.t(), out=out)
    else:
        y = torch.mm(x, w.t())
    if out is None:
        return y
    out.copy_(y)
    return out


def faces(w) -> list:
    """(name, linear, fp32 sums) of every dense prompt GEMM, in layer order (the MTP head's last)."""

    out = []

    def hc(h, where: str) -> None:
        if h is not None:
            out.append((f"{where}.hc_down", getattr(h.down, "b", h.down), True))
            out.append((f"{where}.hc_up", h.up, False))

    layers = [*w.layers, *([w.mtp.layer] if w.mtp is not None else [])]
    for i, layer in enumerate(layers):
        where = f"layer{layer.index}" if i < len(w.layers) else "mtp"
        hc(layer.attn_hc, where + ".attn")
        hc(layer.mlp_hc, where + ".mlp")
        if layer.gdn is not None:
            out += [(f"{where}.gdn_in", layer.gdn.proj, False), (f"{where}.gdn_out", layer.gdn.out, False)]
        if layer.attn is not None:
            out += [(f"{where}.attn_qkv", layer.attn.proj, False), (f"{where}.attn_o", layer.attn.o, False)]
        if layer.ple is not None:
            out += [(f"{where}.ple_key", layer.ple.key, False), (f"{where}.ple_value", layer.ple.value, False)]
        shared = getattr(getattr(layer.moe, "experts", None), "shared", None)
        if shared is not None:
            out += [(f"{where}.shared_gu", shared.gu, False), (f"{where}.shared_down", shared.down, False)]
    hc(w.mixer, "mixer")
    if w.mtp is not None:
        hc(w.mtp.mixer, "mtp.mixer")
        out += [("mtp.fc_e", w.mtp.fc_e, False), ("mtp.fc_h", w.mtp.fc_h, False)]
    return out


def prepare(w) -> dict:
    """Mark the prompt GEMMs this path takes (``thor`` on each linear: bf16 rows, or MXFP8 rows exact in bf16) and,
    with ``copy``, make the bf16 copies; the staging and pad buffers are made here too, at load, before the caches
    are sized. Returns what it holds."""

    from .linear import Mx8Linear

    held = staging = widest = 0
    marked = {"mxfp8": 0, "bf16": 0, "kept": 0}
    copy = copies()
    for _, lin, f32 in faces(w):
        if f32:                                   # fp32 sums: TensorFold's split-K kernel (see the module docstring)
            marked["kept"] += 1
            continue
        if isinstance(lin, Mx8Linear):
            if not bool(((lin.bs >= 3) & (lin.bs <= 134)).all()):     # an exponent past bf16's exact range
                marked["kept"] += 1
                continue
            if copy:
                lin.thor_w = dequantize(lin)
                held += lin.thor_w.numel() * 2
            staging = max(staging, lin.n * lin.k)
            marked["mxfp8"] += 1
        elif getattr(lin, "kernel", "") == "b16":
            marked["bf16"] += 1
        else:
            marked["kept"] += 1
            continue
        lin.thor = True
        widest = max(widest, int(lin.k))
    device = w.inv_freq.device
    S.staging = None if copy or not staging else torch.empty((staging,), dtype=torch.bfloat16, device=device)
    S.pad = torch.zeros((PAD * max(widest, 1),), dtype=torch.bfloat16, device=device)
    S.active = True
    scratch = (0 if S.staging is None else S.staging.numel() * 2) + S.pad.numel() * 2
    return {"copy": copy, "held_bytes": held, "scratch_bytes": scratch, **marked}


def release(w) -> int:
    """Drop the bf16 copies (A/B runs: ``dequant`` from then on); returns the bytes freed."""

    freed = staging = 0
    for _, lin, _ in faces(w):
        held = getattr(lin, "thor_w", None)
        if held is not None:
            freed += held.numel() * 2
            staging = max(staging, held.numel())
            del lin.thor_w
    if staging and (S.staging is None or S.staging.numel() < staging):
        S.staging = torch.empty((staging,), dtype=torch.bfloat16, device=S.pad.device)
    return freed


def admission(geometry, model_dir: str | Path, mtp: bool):
    """The startup estimate plus what ``prepare`` will hold: bf16 copies of the checkpoint's MXFP8 DeltaNet, attention
    and shared-expert weights (``copy``), else a staging buffer of the largest stacked projection."""

    from tensorfold.cuda.geometry import with_fixed
    from tensorfold.families.qwen4_exp.host_table import read_header

    elements = 0
    names = {}
    for path in sorted(Path(model_dir).glob("model*.safetensors")):
        names.update(read_header(path))
    for name, info in names.items():
        if not isinstance(info, dict) or info.get("dtype") != "F8_E4M3" or not name.endswith(".weight"):
            continue
        if name[:-len("weight")] + "weight_scale" not in names or (not mtp and ("mtp." in name)):
            continue
        if any(part in name for part in (".linear_attn.", ".self_attn.", ".shared_expert.")):
            count = 1
            for d in info["shape"]:
                count *= int(d)
            elements += count

    def extra(t: dict) -> int:
        if copies():
            return 2 * elements
        d = int(t["hidden_size"])
        gdn = 2 * int(t["linear_num_key_heads"]) * int(t["linear_key_head_dim"]) + \
            2 * int(t["linear_num_value_heads"]) * (int(t["linear_value_head_dim"]) + 1)
        return 2 * d * gdn + 2 * PAD * 4 * d

    return lambda text: with_fixed(geometry(text), extra(text))
