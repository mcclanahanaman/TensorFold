"""NVFP4 activation quantization in plain torch, by TRT-LLM / FlashInfer's recipe, for the W4A4 candidates' probes.

A 16-input block's scale is e4m3(gs * amax / 6) (round to nearest), its values x * gs / scale rounded to the E2M1 grid
(nearest, ties to even, saturating at 6); the value back is code * scale / gs. ``gs`` is one fp32 global scale
(FlashInfer's convention: 448 * 6 / amax). ``swizzle`` lays scales out as cuBLASLt / CUTLASS read them (128 x 4 tiles).
"""

from __future__ import annotations

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MAX, E4M3_MAX = 6.0, 448.0


def global_scale(amax: float) -> float:
    return E4M3_MAX * FP4_MAX / max(float(amax), 1e-30)


def quantize(x: torch.Tensor, gs: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [M, K] (any float) -> packed codes uint8 [M, K/2] (low nibble the even input), scales e4m3 [M, K/16], and the
    dequantized values fp32 [M, K]."""

    m, k = x.shape
    xb = x.float().view(m, k // 16, 16)
    amax = xb.abs().amax(-1)
    sf = (amax * (gs / FP4_MAX)).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    sff = sf.float()
    inv = torch.where(sff > 0, gs / sff, torch.zeros_like(sff))
    q = xb * inv[..., None]
    a = q.abs()
    code = ((a > 0.25).to(torch.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5) + (a >= 3.5) + (a > 5.0))
    vals = torch.tensor(E2M1_VALUES, device=x.device)[code.long()]
    neg = q < 0
    deq = torch.where(neg, -vals, vals) * (sff / gs)[..., None]
    nib = (code | (neg.to(torch.uint8) << 3)).view(m, k)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).contiguous()
    return packed, sf, deq.view(m, k)


def swizzle(sf: torch.Tensor) -> torch.Tensor:
    """e4m3 scales [..., R, C] -> the 128 x 4 tiled order (rows padded to 128, columns to 4), flattened per leading index."""

    lead = sf.shape[:-2]
    r, c = sf.shape[-2:]
    rp, cp = -(-r // 128) * 128, -(-c // 4) * 4
    s = sf.view(torch.uint8).reshape(-1, r, c)
    if (rp, cp) != (r, c):
        pad = torch.zeros((s.shape[0], rp, cp), dtype=torch.uint8, device=sf.device)
        pad[:, :r, :c] = s
        s = pad
    b = s.shape[0]
    s = s.view(b, rp // 128, 4, 32, cp // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous()
    return s.view(*lead, rp * cp).view(torch.float8_e4m3fn) if lead else s.view(rp * cp).view(torch.float8_e4m3fn)


def dequantize(packed: torch.Tensor, sf: torch.Tensor, gs: float) -> torch.Tensor:
    """packed [M, K/2] uint8 and linear e4m3 scales [M, K/16] -> fp32 [M, K] (code * scale / gs)."""

    m, k2 = packed.shape
    lo, hi = packed & 0xF, packed >> 4
    nib = torch.stack([lo, hi], dim=-1).view(m, 2 * k2)
    vals = torch.tensor(E2M1_VALUES + tuple(-v for v in E2M1_VALUES), device=packed.device)[nib.long()]
    return vals * (sf.float() / gs).repeat_interleave(16, dim=1)
