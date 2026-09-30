"""FlashInfer's CuTe-DSL SM100 W4A16 grouped MoE (NVFP4 weights x bf16 activations) on Thor, as a Phase A probe.

Upstream: flashinfer-python 0.7.0.post1, ``flashinfer/fused_moe/cute_dsl/blackwell/moe_w4a16_kernel.py``
(``Sm100W4A16GroupedGemmKernel``, NVIDIA, BSD-3): each 16-value E2M1 block and its E4M3 scale are decoded to bf16 in
shared memory, then bf16 tcgen05 MMA with fp32 accumulation; weight_scale_2 is the per-expert alpha. The public entry
(``cute_dsl_fused_moe``) is gated to compute capability 10.0/10.3/10.7 by a Python decorator; this probe calls the
launcher underneath it (``launch_w4a16_moe``), or, when FlashInfer's routing module cannot build offline, its grouped
GEMM (``_run_grouped_gemm``) with the routing metadata made here in torch (same semantics as ``moe_sort``).

Weights as FlashInfer's ``prepare_cute_dsl_weights`` lays them out for NVFP4 x BF16: gemm1 rows [up; gate] interleaved
in groups of 64 (``_interleave_linear_and_gate``), scales in the 128 x 4 tiled order, one alpha an expert per GEMM
(so gate and up must share weight_scale_2; the probe checks).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_common as mc  # noqa: E402
import nvfp4_quant as nq  # noqa: E402

SM100_SMEM = 232448


def patch_smem() -> dict:
    """The kernel sizes its pipeline for sm_100's shared memory; clamp to this GPU's opt-in maximum when smaller."""

    import cutlass.utils as cu

    have = int(torch.cuda.get_device_properties(0).shared_memory_per_block_optin)
    orig = cu.get_smem_capacity_in_bytes
    try:
        assumed = int(orig("sm_100"))
    except Exception:  # noqa: BLE001
        assumed = SM100_SMEM
    info = {"device_smem_optin": have, "dsl_sm100_value": assumed, "patched": False}
    if have < assumed:
        def clamp(*args, _orig=orig, _have=have, **kwargs):
            return min(int(_orig(*args, **kwargs)), _have)

        cu.get_smem_capacity_in_bytes = clamp
        info["patched"] = True
    return info


def combine_rows(y: torch.Tensor, e2p: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[t] = sum_k w[t, k] * y[e2p[t, k]] in fp32, slot order, one rounding to bf16 (a Triton kernel)."""

    import triton

    m, top = e2p.shape
    out = torch.empty(m, y.shape[1], dtype=torch.bfloat16, device=y.device)
    _combine_rows[(m, triton.cdiv(y.shape[1], 512))](y, e2p, w, out, D=y.shape[1], TOP=top, BLOCK=512,
                                                        num_warps=4)
    return out


def _make_combine():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(Y, IDX, W, OUT, D: tl.constexpr, TOP: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = d < D
        acc = tl.zeros((BLOCK,), tl.float32)
        for k in tl.static_range(TOP):
            row = tl.load(IDX + r * TOP + k).to(tl.int64)
            acc += tl.load(W + r * TOP + k) * tl.load(Y + row * D + d, mask=ok, other=0.0).to(tl.float32)
        tl.store(OUT + r * D + d, acc.to(tl.bfloat16), mask=ok)

    return kernel


_combine_rows = _make_combine()


def interleave_rows(t: torch.Tensor, group: int = 64) -> torch.Tensor:
    """[E, 2I, ...] with rows [up; gate] -> [E, 2I, ...] rows [up g0; gate g0; up g1; gate g1; ...], ``group`` rows each."""

    e, r = t.shape[:2]
    rest = t.shape[2:]
    return t.view(e, 2, r // (2 * group), group, *rest).transpose(1, 2).contiguous().view(e, r, *rest)


class FlashInferW4A16:
    def __init__(self, layer: mc.Layer, mode: str = "auto") -> None:
        from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16 as fw
        from flashinfer.tllm_enums import ActivationType

        self.fw, self.act = fw, ActivationType.Swiglu
        self.shared_gu_scale = bool(torch.equal(layer.gate.s2, layer.up.s2))
        up_gate_w = torch.cat([layer.up.w, layer.gate.w], 1)                  # FlashInfer: linear (up) then gate
        up_gate_s = torch.cat([layer.up.s, layer.gate.s], 1).view(torch.uint8)
        self.w1 = interleave_rows(up_gate_w)                                    # [E, 2I, D/2] uint8
        self.w1_sf = nq.swizzle(interleave_rows(up_gate_s).view(torch.float8_e4m3fn)).contiguous()
        self.w1_alpha = layer.gate.s2.float().contiguous()
        self.w2 = layer.down.w.contiguous()
        self.w2_sf = nq.swizzle(layer.down.s.contiguous()).contiguous()
        self.w2_alpha = layer.down.s2.float().contiguous()
        self.tactic = fw.DEFAULT_W4A16_MOE_TACTIC
        self.mode = mode
        self.used = None

    # -- routing metadata in torch, as moe_sort defines it: each expert's rows padded to the route tile ----------
    def sort(self, ids: torch.Tensor, tile: int):
        flat = ids.flatten().long()
        p = flat.numel()
        counts = torch.bincount(flat, minlength=mc.E)
        tiles = (counts + tile - 1) // tile
        pad_start = torch.cumsum(tiles * tile, 0) - tiles * tile
        order = torch.argsort(flat, stable=True)
        e_sorted = flat[order]
        rank = torch.arange(p, device=ids.device) - (torch.cumsum(counts, 0) - counts)[e_sorted]
        pos = pad_start[e_sorted] + rank
        from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens

        slots = get_max_num_permuted_tokens(ids.shape[0], ids.shape[1], mc.E, tile)
        ntiles = slots // tile
        p2e = torch.full((slots,), -1, dtype=torch.int32, device=ids.device)
        p2e[pos] = order.to(torch.int32)
        e2p = torch.empty(p, dtype=torch.int32, device=ids.device)
        e2p[order] = pos.to(torch.int32)
        cum_tiles = torch.cumsum(tiles, 0)
        t = torch.arange(ntiles, device=ids.device)
        te = torch.searchsorted(cum_tiles, t, right=True).clamp(max=mc.E - 1)
        limit = (pad_start + counts)[te]
        used = cum_tiles[-1].to(torch.int32).reshape(1)
        return te.to(torch.int32), limit.to(torch.int32), e2p.view(ids.shape), p2e, used, pos, order, slots

    def prepare(self, x, ids):
        """Routing metadata (torch, moe_sort's semantics), the permuted rows and the scratch for this tactic."""

        g1, _ = self.tactic
        te, limit, e2p, p2e, used, pos, order, slots = self.sort(ids, g1[0][1])
        hidden = torch.empty(slots, mc.D, dtype=torch.bfloat16, device=x.device)
        hidden[pos] = x[order // ids.shape[1]]
        inter = torch.empty(slots, mc.NI, dtype=torch.bfloat16, device=x.device)
        y = torch.empty(slots, mc.D, dtype=torch.bfloat16, device=x.device)
        return dict(te=te, limit=limit, used=used, hidden=hidden, inter=inter, y=y, e2p=e2p)

    def gemms(self, prep) -> torch.Tensor:
        """The two grouped GEMMs alone (gemm1 with its fused SwiGLU, then gemm2) on prepared metadata."""

        g1, g2 = self.tactic
        common = dict(tile_idx_to_expert_idx=prep["te"], tile_idx_to_mn_limit=prep["limit"],
                      num_non_exiting_tiles=prep["used"], num_local_experts=mc.E,
                      swiglu_alpha=self._d("DEFAULT_SWIGLU_ALPHA"), swiglu_beta=self._d("DEFAULT_SWIGLU_BETA"),
                      swiglu_limit=self._d("DEFAULT_SWIGLU_LIMIT"), situ_beta=None, situ_linear_beta=None,
                      use_fused_finalize=False, permuted_idx_to_expanded_idx=None, token_final_scales=None,
                      enable_pdl=False)
        self.fw._run_grouped_gemm(weight=self.w1, weight_sf=self.w1_sf, activations=prep["hidden"],
                                  alpha=self.w1_alpha, output=prep["inter"], activation_type=self.act, tactic=g1,
                                  **common)
        self.fw._run_grouped_gemm(weight=self.w2, weight_sf=self.w2_sf, activations=prep["inter"],
                                  alpha=self.w2_alpha, output=prep["y"], activation_type=None, tactic=g2, **common)
        return prep["y"]

    def _torch_path(self, x, ids, w):
        prep = self.prepare(x, ids)
        y = self.gemms(prep)
        return combine_rows(y, prep["e2p"], w)

    def _d(self, name):
        from flashinfer import tllm_enums

        return getattr(tllm_enums, name)

    def _launch(self, x, ids, w):
        out = torch.empty(x.shape[0], mc.D, dtype=torch.bfloat16, device=x.device)
        return self.fw.launch_w4a16_moe(x, ids, w, self.w1, self.w1_sf, self.w1_alpha, self.w2, self.w2_sf,
                                        self.w2_alpha, num_experts=mc.E, num_local_experts=mc.E,
                                        local_expert_offset=0, moe_output=out, use_fused_finalize=False,
                                        enable_pdl=False, activation_type=self.act, tactic=self.tactic)

    @torch.no_grad()
    def __call__(self, x, ids, w):
        if self.mode in ("auto", "launch") and self.used != "torch":
            try:
                r = self._launch(x, ids, w)
                self.used = "launch"
                return r
            except Exception as exc:  # noqa: BLE001
                if self.mode == "launch":
                    raise
                self.launch_error = repr(exc)[:2000]
                self.used = "torch"
        return self._torch_path(x, ids, w)


def sweep_tactics(m: int = 2048) -> list:
    """A reduced tactic list: route tiles 8/16/32 for decode-sized calls (m <= 64 tokens), else 32/64/128; 1-CTA 128,
    2-CTA 256 and 1-CTA 128 in 2x1 clusters, the same for both GEMMs."""

    from flashinfer.fused_moe.cute_dsl.tuner import W4A16_MOE_TACTICS

    tiles = (8, 16, 32) if m <= 64 else (32, 64, 128)
    keep = []
    for g1, g2 in W4A16_MOE_TACTICS:
        if g1[0][1] in tiles and g1[0][0] == g2[0][0] and g1[1] == g2[1]:
            keep.append((g1, g2))
    return keep


def best_time(path: FlashInferW4A16, x, ids, w, reps: int) -> dict:
    """Each tactic of ``sweep_tactics`` timed on the two grouped GEMMs alone (prepared metadata, 3 reps after 1
    warm-up); the best one again with ``reps`` reps as GEMMs only and as the full block (torch routing metadata with
    a host sync, the permute gather, both GEMMs, the Triton combine)."""

    tried = []
    best = None
    for tac in sweep_tactics(x.shape[0]):
        path.tactic = tac
        try:
            prep = path.prepare(x, ids)
            t = mc.time_cuda(lambda: path.gemms(prep), reps=3, warmup=1)
        except Exception as exc:  # noqa: BLE001
            tried.append({"tactic": tac, "error": repr(exc)[:300]})
            continue
        tried.append({"tactic": tac, "gemms_ms": t["median_ms"]})
        if best is None or t["median_ms"] < best[1]:
            best = (tac, t["median_ms"])
    if best is None:
        raise RuntimeError(f"no W4A16 tactic ran: {tried[:3]}")
    path.tactic = best[0]
    prep = path.prepare(x, ids)
    g = mc.time_cuda(lambda: path.gemms(prep), reps=reps)
    t = mc.time_cuda(lambda: path(x, ids, w), reps=reps)
    t["gemms_only"] = g
    t["tactic"] = best[0]
    t["tried"] = tried
    t["mode"] = path.used
    return t


def first_working(path: FlashInferW4A16, x, ids, w):
    """Run with the default tactic, else the first of ``sweep_tactics`` that runs; returns (output, tactic)."""

    errors = []
    for tac in [path.fw.DEFAULT_W4A16_MOE_TACTIC] + sweep_tactics(x.shape[0]):
        path.tactic = tac
        try:
            out = path(x, ids, w)
            torch.cuda.synchronize()
            return out, tac, errors
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tac}: {exc!r}"[:400])
    raise RuntimeError("no W4A16 tactic ran: " + " | ".join(errors[:4]))
