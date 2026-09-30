"""FlashInfer's CuTe-DSL NVFP4 x NVFP4 (W4A4) grouped MoE on Thor, as a Phase A probe for the native FP4 speed.

Upstream: flashinfer-python 0.7.0.post1 ``fused_moe/cute_dsl/fused_moe.py::_moe_core_impl`` (gather grouped GEMM with
fused SwiGLU and FP4 output quantization, then the finalize grouped GEMM). Its routing helpers (``moe_sort``,
``moe_unpermute``, the output memset) live in a C++ module that FlashInfer builds from headers it downloads; offline,
this probe swaps them for torch equivalents with the same outputs. Activations are quantized here (tools/thor/
nvfp4_quant.py, TRT-LLM's recipe) with one global scale from the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_common as mc  # noqa: E402
import nvfp4_quant as nq  # noqa: E402
from fi_w4a16_moe import interleave_rows  # noqa: E402


def torch_moe_sort(token_selected_experts, token_final_scales, num_experts, top_k, local_expert_offset=0,
                   num_local_experts=None, tile_tokens_dim=128, enable_pdl=False, **_):
    ids = token_selected_experts
    flat = ids.flatten().long()
    p = flat.numel()
    tile = tile_tokens_dim
    counts = torch.bincount(flat, minlength=num_experts)
    tiles = (counts + tile - 1) // tile
    pad_start = torch.cumsum(tiles * tile, 0) - tiles * tile
    order = torch.argsort(flat, stable=True)
    e_sorted = flat[order]
    rank = torch.arange(p, device=ids.device) - (torch.cumsum(counts, 0) - counts)[e_sorted]
    pos = pad_start[e_sorted] + rank
    from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens

    slots = get_max_num_permuted_tokens(ids.shape[0], top_k, num_experts, tile)
    ntiles = slots // tile
    p2e = torch.full((slots,), -1, dtype=torch.int32, device=ids.device)
    p2e[pos] = order.to(torch.int32)
    e2p = torch.empty(p, dtype=torch.int32, device=ids.device)
    e2p[order] = pos.to(torch.int32)
    cum_tiles = torch.cumsum(tiles, 0)
    t = torch.arange(ntiles, device=ids.device)
    te = torch.searchsorted(cum_tiles, t, right=True).clamp(max=num_experts - 1)
    limit = (pad_start + counts)[te]
    used = cum_tiles[-1].to(torch.int32).reshape(1)
    total = (cum_tiles[-1] * tile).to(torch.int32).reshape(1)
    return te.to(torch.int32), limit.to(torch.int32), e2p.view(ids.shape), p2e, total, used


def torch_unpermute(permuted_input, output, expanded_idx_to_permuted_idx, topk_scales, num_tokens, top_k,
                    input_is_expanded=False, enable_pdl=False, **_):
    rows = permuted_input[:num_tokens * top_k] if input_is_expanded else \
        permuted_input[expanded_idx_to_permuted_idx.flatten().long()]
    y = rows.view(num_tokens, top_k, -1).float() * topk_scales[:num_tokens].float()[..., None]
    output[:num_tokens].copy_(y.sum(1).to(output.dtype))
    return output


def torch_memset(output):
    output.zero_()


class FlashInferW4A4:
    def __init__(self, layer: mc.Layer, tactic=None) -> None:
        import flashinfer.fused_moe.cute_dsl.fused_moe as fm

        fm.moe_sort = torch_moe_sort
        fm.moe_unpermute = torch_unpermute
        fm.moe_output_memset_inplace = torch_memset
        self.fm = fm
        self.shared_gu_scale = bool(torch.equal(layer.gate.s2, layer.up.s2))
        self.w1 = interleave_rows(torch.cat([layer.up.w, layer.gate.w], 1))
        self.w1_sf = nq.swizzle(interleave_rows(torch.cat([layer.up.s, layer.gate.s], 1).view(torch.uint8))
                                .view(torch.float8_e4m3fn)).contiguous()
        self.w1_s2 = layer.gate.s2.float()
        self.w2 = layer.down.w.contiguous()
        self.w2_sf = nq.swizzle(layer.down.s.contiguous()).contiguous()
        self.w2_s2 = layer.down.s2.float()
        self.tactic = tactic or dict(tile_size=128, gemm1_mma_tiler_mn=(128, 128), gemm1_cluster_shape_mn=(1, 1),
                                     gemm2_mma_tiler_mn=(128, 128), gemm2_cluster_shape_mn=(1, 1))
        self.a2_gs = None

    @torch.no_grad()
    def __call__(self, x, ids, w, h_amax: float | None = None):
        a1 = nq.global_scale(float(x.abs().max()))
        a2 = nq.global_scale(h_amax if h_amax is not None else (self.a2_gs or 64.0))
        xq, xsf, _ = nq.quantize(x, a1)
        x_sf = nq.swizzle(xsf).contiguous()
        return self.fm._moe_core_impl(
            x=xq, x_sf=x_sf, token_selected_experts=ids, token_final_scales=w, w1_weight=self.w1,
            w1_weight_sf=self.w1_sf, w1_alpha=(self.w1_s2 / a1).contiguous(),
            fc2_input_scale=torch.tensor([a2], dtype=torch.float32, device=x.device), w2_weight=self.w2,
            w2_weight_sf=self.w2_sf, w2_alpha=(self.w2_s2 / a2).contiguous(), num_experts=mc.E, top_k=ids.shape[1],
            num_local_experts=mc.E, use_fused_finalize=False, use_async_memset=False, enable_pdl=False,
            **self.tactic)
