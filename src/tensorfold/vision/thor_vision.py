"""Qwen's vision-tower attention on Jetson Thor (sm_110) (THOR-VISION-ATTN).

transformers' ``Qwen3_5VisionAttention`` (sdpa) rotates q and k through about seven full-size fp32 elementwise passes,
then attends one segment (an image, or one frame of a frame group) at a time: a host sync a layer for the segment
lengths and one SDPA call a segment (15 a layer in each tower call of a 60 s 1080p video). On Thor most of a video's
attention-layer time was those fp32 passes, not the flash kernel [s4-dev-r3]. Here, per layer:
- the rotary embedding is one CUDA pass (``thor_rope.cu``): transformers' operations in its order, the fp32 products and
  their sum rounded as it rounds them, then bf16 by round to nearest even; the same q and k bits;
- the attention is ONE variable-length call of the flash kernel SDPA already runs (``aten._flash_attention_forward``
  with the layer's cumulative segment lengths: the same kernel, tiles and scale; head_dim 72 is not a kernel tile, so
  both take its uneven-shape variant), and every segment's rows are computed as when it is called alone, so the output
  has the same bits; ``_encode`` passes the longest segment from the host, so no layer waits on the GPU.
Each switch is checked at load against transformers' forward on a small random layer (uneven segments) and refused,
with a line saying so, unless the bits match.
``TENSORFOLD_THOR_VISION_ROPE=0`` keeps transformers' rotary passes, ``TENSORFOLD_THOR_VISION_VARLEN=0`` its per-segment
SDPA calls, and ``TENSORFOLD_THOR_VISION=0`` its forward untouched; any other GPU, and an offloaded tower, never install
this path. tests/test_vision_thor_attn.py checks the bits on Thor.
"""

from __future__ import annotations

import functools
import os

ENV = "TENSORFOLD_THOR_VISION"
ENV_ROPE = "TENSORFOLD_THOR_VISION_ROPE"
ENV_VARLEN = "TENSORFOLD_THOR_VISION_VARLEN"
CAPABILITY = (11, 0)
CHECK_SEGMENTS = (64, 200, 33)      # the load check: uneven lengths, tails off the flash kernel's 128 x 64 tiles


def _on(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "off", "false", "no")


def available(device=None) -> bool:
    """sm_110 and not switched off."""
    try:
        import torch

        return _on(ENV) and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY
    except (ImportError, RuntimeError, ValueError, AssertionError):
        return False


@functools.lru_cache(maxsize=1)
def _rope_ext():
    from pathlib import Path

    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_vision_thor_rope", sources=[str(here / "thor_rope.cpp"), str(here / "thor_rope.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def flash_kernels():
    """The SDPA backends ``_encode`` allows (flash first): the per-segment calls this path must match."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])


def make_forward(attn, rope_ext, varlen: bool):
    """A forward for one ``Qwen3_5VisionAttention`` (same arguments, same result bits)."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    heads, dim, scale = attn.num_heads, attn.head_dim, attn.scaling

    def forward(hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
        rows = hidden_states.shape[0]
        qkv = attn.qkv(hidden_states)
        cos, sin = position_embeddings
        if rope_ext is not None:
            q, k = rope_ext.rope(qkv, cos.float().contiguous(), sin.float().contiguous(), heads, dim)
            v = qkv.view(rows, 3, heads, dim)[:, 2]          # transformers' value view: rows 3 * heads * dim apart
        else:
            q, k, v = qkv.reshape(rows, 3, heads, -1).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        if varlen:
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
            out = torch.ops.aten._flash_attention_forward(q, k, v, cu_seqlens, cu_seqlens, int(max_seqlen),
                                                          int(max_seqlen), 0.0, False, False, scale=scale)[0]
        else:                                                # transformers' sdpa calls, one a segment
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            parts = [torch.split(t.transpose(0, 1).unsqueeze(0), lengths, dim=2) for t in (q, k, v)]
            out = torch.cat([torch.nn.functional.scaled_dot_product_attention(
                a, b, c, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=False).transpose(1, 2).contiguous()
                for a, b, c in zip(*parts)], dim=1)
        return attn.proj(out.reshape(rows, -1).contiguous())

    return forward


def check(attn, forward, device) -> str | None:
    """One random layer input through transformers' forward and ``forward``: None when every bit matches."""
    import torch

    g = torch.Generator().manual_seed(110)
    rows, dim = sum(CHECK_SEGMENTS), attn.head_dim
    hidden = torch.randn(rows, attn.dim, generator=g).to(device=device, dtype=torch.bfloat16)
    angles = torch.rand(rows, dim // 2, generator=g) * 64.0
    cos, sin = (f(torch.cat([angles, angles], dim=-1)).to(device) for f in (torch.cos, torch.sin))
    cu = torch.tensor([0, *torch.tensor(CHECK_SEGMENTS).cumsum(0).tolist()], dtype=torch.int32, device=device)
    with torch.inference_mode(), flash_kernels():
        want = type(attn).forward(attn, hidden, cu_seqlens=cu, position_embeddings=(cos, sin))
        got = forward(hidden, cu, position_embeddings=(cos, sin), max_seqlen=max(CHECK_SEGMENTS))
    if got.shape != want.shape or got.dtype != want.dtype:
        return f"shape {tuple(got.shape)} {got.dtype} vs {tuple(want.shape)} {want.dtype}"
    if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
        diff = (got.float() - want.float()).abs()
        return f"{int((diff > 0).sum())} of {diff.numel()} values differ (max {float(diff.max()):.3g})"
    return None


def install(tower, device, *, rope: bool | None = None, varlen: bool | None = None) -> dict | None:
    """THOR-VISION-ATTN on every attention layer of a transformers Qwen vision tower, built and checked here (at load).
    Returns the switches in force, or None when the tower keeps transformers' forward (another GPU, switched off, or a
    check that failed). ``rope``/``varlen`` override the environment (A/B runs); a re-install starts from scratch."""
    layers = [blk.attn for blk in tower.blocks]
    for attn in layers:
        attn.__dict__.pop("forward", None)
    if not available(device):
        return None
    on = {"rope": _on(ENV_ROPE) if rope is None else bool(rope),
          "varlen": _on(ENV_VARLEN) if varlen is None else bool(varlen)}
    if not any(on.values()):
        return None
    ext = _rope_ext() if on["rope"] else None
    why = check(layers[0], make_forward(layers[0], ext, on["varlen"]), device)
    if why is not None:
        print(f"[tensorfold] Thor (sm_110): vision attention kept on transformers' path: the load check found {why}",
              flush=True)
        return None
    for attn in layers:
        attn.forward = make_forward(attn, ext, on["varlen"])
    return on
