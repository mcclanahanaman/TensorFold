"""Routed NVFP4 experts on Jetson Thor (sm_110): FlashInfer's W4A16 grouped GEMM, laid out once at load.

Each e2m1 code times its e4m3 scale is exact in bf16, so the kernel decodes a weight tile to bf16 on its way to the bf16
tcgen05 MMAs, sums over the whole of K in fp32 and applies weight_scale_2 as the expert's alpha; SwiGLU runs on the
fp32 sums and rounds once. A pair's bits never depend on the call it runs in (the rows around it, the tile or the
tactic), so drafted rows equal serial ones and concurrent rows solo ones. The glue is device-only and fixed-shape a row
count (CUDA graphs capture it): the call's plan (``tensorfold.cuda.experts``, routed with this path's tile) gives each
item of one expert's pairs a tile of the permuted rows, and the outputs go back to their (row, slot) places.

``TENSORFOLD_THOR_FI_MOE=0`` keeps the portable kernel; any other GPU never reaches this module.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch

ENV = "TENSORFOLD_THOR_FI_MOE"
CAPABILITY = (11, 0)
GROUP = 64                  # FlashInfer's gated rows: 64 up rows, then the same 64 gate rows, in turn
K_TILE = 256
MIN_TILE = 16               # the plan's items buffer holds items of 16 pairs at the narrowest
# (most rows a call, route tile, MMA M, cluster M x N): the fastest measured on AGX Thor for Flash Next's routed
# shapes (tools/thor/decode_graph_bench.py, CUDA graphs); a tile of 64 keeps the prompt scratch to 64-row padding
TACTICS = ((4, 16, 128, (1, 1)), (64, 16, 128, (2, 1)), (512, 32, 128, (2, 1)), (2048, 64, 128, (1, 1)),
           (1 << 30, 64, 128, (2, 1)))
_COPY = 512                 # columns a glue step moves


def _fi():
    from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16

    return moe_w4a16


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110, FlashInfer's W4A16 MoE importable and not switched off: decided once, before any weight is laid out."""

    if _off() or not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != CAPABILITY:
        return False
    try:
        fw = _fi()
        from flashinfer.tllm_enums import ActivationType  # noqa: F401

        fw._run_grouped_gemm, fw.DEFAULT_W4A16_MOE_TACTIC   # noqa: B018  (the entry points this path calls)
    except Exception as exc:                                 # noqa: BLE001  (any failure keeps the portable kernel)
        print(f"[tensorfold] Thor NVFP4 experts off: FlashInfer's W4A16 MoE is not usable here ({exc!r}); the "
              "portable kernel runs", flush=True)
        return False
    return True


@dataclass
class ThorExperts:
    """One layer's routed experts as FlashInfer's W4A16 kernel reads them: the checkpoint's bytes, reordered."""

    w1: torch.Tensor          # uint8 [E, 2NI, D/2]: 64 up rows, the same 64 gate rows, in turn (the gated order)
    w1_sf: torch.Tensor       # uint8 [E, 2NI * D/16]: their e4m3 scales in the 128 x 4 tiles the MMA reads
    w1_alpha: torch.Tensor    # fp32 [E]: weight_scale_2 (gate's, which equals up's)
    w2: torch.Tensor          # uint8 [E, D, NI/2]
    w2_sf: torch.Tensor       # uint8 [E, D * NI/16], tiled
    w2_alpha: torch.Tensor    # fp32 [E]
    width: int                # NI
    dims: int                 # D
    limit: float = 0.0

    kernel = "thor"

    @property
    def count(self) -> int:
        return int(self.w1.shape[0])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w1, self.w1_sf, self.w1_alpha, self.w2, self.w2_sf,
                                                           self.w2_alpha))


def _gated(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """[c, NI, X] each -> [c, 2NI, X]: GROUP up rows, the same GROUP gate rows, in turn."""

    c, ni = up.shape[:2]
    return torch.stack([up.reshape(c, ni // GROUP, GROUP, -1), gate.reshape(c, ni // GROUP, GROUP, -1)],
                       dim=2).reshape(c, 2 * ni, -1)


def tile_scales(s: torch.Tensor) -> torch.Tensor:
    """e4m3 bytes [c, R, C] (R % 128 == 0, C % 4 == 0) -> [c, R * C] in the 128 x 4 tiles block-scaled MMAs read."""

    c, r, k = s.shape
    return s.reshape(c, r // 128, 4, 32, k // 4, 4).permute(0, 1, 4, 3, 2, 5).reshape(c, r * k)


def make(gate: tuple, up: tuple, down: tuple, *, limit: float = 0.0, chunk: int = 64) -> ThorExperts:
    """Each of gate, up, down: (words [E, N, K/2] uint8, e4m3 scales [E, N, K/16], per-expert scales [E] fp32)."""

    if limit:
        raise ValueError("a clipped SwiGLU is not qualified on the Thor experts")
    g2, u2 = torch.as_tensor(gate[2]).float(), torch.as_tensor(up[2]).float()
    if not torch.equal(g2, u2):
        bad = int((g2 != u2).sum())
        raise ValueError(f"{bad} experts' gate and up weight_scale_2 differ; FlashInfer's gated GEMM takes one an "
                         "expert")
    e, ni, d2 = gate[0].shape
    d = 2 * d2
    if ni % 128 or d % 128 or ni % GROUP or down[0].shape != (e, d, ni // 2):
        raise ValueError(f"experts [{e}, {ni}, {d}] with down {tuple(down[0].shape)} do not fit the Thor layout")
    dev = gate[0].device
    u8 = torch.uint8
    w1 = torch.empty((e, 2 * ni, d2), dtype=u8, device=dev)
    w1_sf = torch.empty((e, 2 * ni * d // 16), dtype=u8, device=dev)
    w2_sf = torch.empty((e, d * ni // 16), dtype=u8, device=dev)
    for e0 in range(0, e, chunk):                  # a chunk at a time: the transient stays a few hundred MB
        e1 = min(e, e0 + chunk)
        w1[e0:e1] = _gated(up[0][e0:e1], gate[0][e0:e1])
        w1_sf[e0:e1] = tile_scales(_gated(up[1][e0:e1].view(u8), gate[1][e0:e1].view(u8)))
        w2_sf[e0:e1] = tile_scales(down[1][e0:e1].view(u8))
    return ThorExperts(w1, w1_sf, g2.to(dev).contiguous(), down[0].contiguous(),
                       w2_sf, torch.as_tensor(down[2]).float().to(dev).contiguous(), int(ni), int(d), float(limit))


def dense(ex: ThorExperts, e: int, which: str) -> torch.Tensor:
    """Expert ``e``'s gate, up or down weight [N, K] fp32 from the Thor layout: code x e4m3 x scale (a reference)."""

    w, sf, alpha = (ex.w2, ex.w2_sf, ex.w2_alpha) if which == "down" else (ex.w1, ex.w1_sf, ex.w1_alpha)
    words = w[e]
    r, k2 = words.shape
    s = sf[e].view(r // 128, (2 * k2 // 16) // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(r, 2 * k2 // 16)
    codes = torch.stack([words & 0xF, words >> 4], dim=-1).reshape(r, 2 * k2).long()
    mags = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=w.device)
    vals = torch.cat([mags, -mags])[codes]
    full = vals * s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1) * alpha[e].float()
    if which == "down":
        return full
    half = 0 if which == "up" else 1
    return full.view(r // (2 * GROUP), 2, GROUP, -1)[:, half].reshape(r // 2, -1)


def tactic(rows: int) -> tuple[int, tuple]:
    """(route tile, (gemm1 tactic, gemm2 tactic)) for a call of ``rows`` rows."""

    for top, tile, mma, cluster in TACTICS:
        if rows <= top:
            t = ((mma, tile, K_TILE), cluster, True)
            return tile, (t, t)
    raise AssertionError


def max_tiles(pairs: int, experts: int, tile: int) -> int:
    """Route tiles any routing of ``pairs`` pairs over ``experts`` can need, each expert's pairs in whole tiles."""

    return pairs if pairs <= experts else (pairs + (tile - 1) * experts) // tile


def slots(rows: int, top_k: int, experts: int) -> int:
    """Permuted rows (tiles x tile) a call of ``rows`` rows can need."""

    tile, _ = tactic(rows)
    return max_tiles(rows * top_k, experts, tile) * tile


def scratch_rows(rows: int, top_k: int, experts: int) -> int:
    """The most permuted rows any call of 1..``rows`` rows can need (each tactic's range checked at its end)."""

    most, lo = 0, 1
    for top, *_ in TACTICS:
        if lo > rows:
            break
        most = max(most, slots(min(top, rows), top_k, experts))
        lo = top + 1
    return most


def scratch_bytes(rows: int, top_k: int, experts: int, width: int, dims: int) -> int:
    s = scratch_rows(rows, top_k, experts)
    return s * (dims + width) * 2 + 2 * (s // MIN_TILE) * 4 + 4


class Scratch:
    """A buffer set's permuted rows (the first GEMM's input, then the second's output) and the route tiles."""

    def __init__(self, rows: int, top_k: int, experts: int, width: int, dims: int, device) -> None:
        s = scratch_rows(rows, top_k, experts)
        self.rows = rows
        self.hidden = torch.empty((s, dims), dtype=torch.bfloat16, device=device)
        self.inter = torch.empty((s, width), dtype=torch.bfloat16, device=device)
        self.tile_expert = torch.zeros((s // MIN_TILE,), dtype=torch.int32, device=device)
        self.tile_limit = torch.zeros((s // MIN_TILE,), dtype=torch.int32, device=device)
        self.tiles = torch.zeros((1,), dtype=torch.int32, device=device)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.hidden, self.inter, self.tile_expert, self.tile_limit,
                                                           self.tiles))


@functools.lru_cache(maxsize=1)
def _glue():
    """The glue's two Triton kernels (built on first use: this module imports without Triton)."""

    import triton
    import triton.language as tl

    @triton.jit
    def gather(X, x_stride, MEMBERS, ITEMS, COUNTS, HIDDEN, TE, TLIM, NT, shared_items,
               SLOTS: tl.constexpr, T: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
        """Program i: item i (<= T pairs of one expert) -> route tile i: its expert, row limit and pairs' rows."""

        i = tl.program_id(0)
        n = tl.load(COUNTS) - shared_items          # the shared expert (the last id) has the last items; it runs apart
        if i == 0:
            tl.store(NT, n)
        if i < n:
            first = tl.load(ITEMS + 3 * i + 1)
            cnt = tl.load(ITEMS + 3 * i + 2)
            tl.store(TE + i, tl.load(ITEMS + 3 * i))
            tl.store(TLIM + i, i * T + cnt)
            r = tl.arange(0, T)
            ok = r < cnt
            p = tl.load(MEMBERS + first + r, mask=ok, other=0)
            src = (p // SLOTS).to(tl.int64) * x_stride
            dst = (i * T + r).to(tl.int64) * D
            for d0 in tl.static_range(0, D, BD):
                d = d0 + tl.arange(0, BD)
                v = tl.load(X + src[:, None] + d[None, :], mask=ok[:, None], other=0.0)
                tl.store(HIDDEN + dst[:, None] + d[None, :], v, mask=ok[:, None])

    @triton.jit
    def scatter(YP, MEMBERS, ITEMS, NT, Y, T: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
        """Program i: route tile i's output rows -> their pairs' rows of Y [pairs, D]."""

        i = tl.program_id(0)
        if i < tl.load(NT):
            first = tl.load(ITEMS + 3 * i + 1)
            cnt = tl.load(ITEMS + 3 * i + 2)
            r = tl.arange(0, T)
            ok = r < cnt
            p = tl.load(MEMBERS + first + r, mask=ok, other=0).to(tl.int64) * D
            src = (i * T + r).to(tl.int64) * D
            for d0 in tl.static_range(0, D, BD):
                d = d0 + tl.arange(0, BD)
                v = tl.load(YP + src[:, None] + d[None, :], mask=ok[:, None], other=0.0)
                tl.store(Y + p[:, None] + d[None, :], v, mask=ok[:, None])

    return gather, scatter


def _gemm(fw, w, sf, alpha, act_in, out, tiles, scr, experts: int, gated: bool, t) -> None:
    from flashinfer.tllm_enums import DEFAULT_SWIGLU_ALPHA, DEFAULT_SWIGLU_BETA, DEFAULT_SWIGLU_LIMIT, ActivationType

    fw._run_grouped_gemm(weight=w, weight_sf=sf, activations=act_in, tile_idx_to_expert_idx=scr.tile_expert[:tiles],
                         tile_idx_to_mn_limit=scr.tile_limit[:tiles], num_non_exiting_tiles=scr.tiles, alpha=alpha,
                         output=out, num_local_experts=experts,
                         activation_type=ActivationType.Swiglu if gated else None,
                         swiglu_alpha=DEFAULT_SWIGLU_ALPHA, swiglu_beta=DEFAULT_SWIGLU_BETA,
                         swiglu_limit=DEFAULT_SWIGLU_LIMIT, situ_beta=None, situ_linear_beta=None,
                         use_fused_finalize=False, permuted_idx_to_expanded_idx=None, token_final_scales=None,
                         enable_pdl=False, tactic=t)


def run(x: torch.Tensor, ex: ThorExperts, plan, y: torch.Tensor, rows: int, top_k: int, scr: Scratch) -> None:
    """x [R, D] bf16 -> y [R, slots, D] bf16: each routed pair's output in its slot; the shared slot is left alone.

    ``plan``: the call's pairs grouped by expert (``tensorfold.cuda.experts.route`` with ``tactic(rows)``'s tile), the
    shared expert (id E) in the last slot. No host sync: the grid and the shapes follow ``rows`` alone.
    """

    from tensorfold.cuda import experts as grouped

    if y.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        raise ValueError("the Thor experts write bf16 slots from bf16 rows (prompt buffers, or decode buffers built "
                         "with the prompt arithmetic)")
    if rows > scr.rows:
        raise ValueError(f"a call of {rows} rows exceeds the Thor scratch's {scr.rows}")
    fw = _fi()
    tile, (g1, g2) = tactic(rows)
    if plan.tile != tile:
        raise ValueError(f"the plan was routed with items of {plan.tile} pairs; this call's tile is {tile}")
    nslots = y.shape[1]
    s = max_tiles(rows * top_k, ex.count, tile) * tile
    tiles = s // tile
    grid = (grouped.max_items(rows * nslots, ex.count + 1, tile),)
    gather, scatter = _glue()
    gather[grid](x, x.stride(0), plan.members, plan.items, plan.counts, scr.hidden, scr.tile_expert, scr.tile_limit,
                  scr.tiles, -(-rows // tile), SLOTS=nslots, T=tile, D=ex.dims, BD=_COPY, num_warps=4)
    _gemm(fw, ex.w1, ex.w1_sf, ex.w1_alpha, scr.hidden[:s], scr.inter[:s], tiles, scr, ex.count, True, g1)
    _gemm(fw, ex.w2, ex.w2_sf, ex.w2_alpha, scr.inter[:s], scr.hidden[:s], tiles, scr, ex.count, False, g2)
    scatter[grid](scr.hidden, plan.members, plan.items, scr.tiles, y.view(-1, ex.dims), T=tile, D=ex.dims,
                   BD=_COPY, num_warps=4)


_WARM: set = set()


def warm(ex: ThorExperts, scr: Scratch, top_k: int) -> None:
    """Build every kernel a call up to ``scr.rows`` rows can reach (both GEMMs of each tactic, the glue) now, not on a
    request: one call at each tactic's smallest and largest row count, on synthetic routing."""

    from tensorfold.cuda import experts as grouped

    dev = scr.hidden.device
    lo = 1
    for top, *_ in TACTICS:
        if lo > scr.rows:
            break
        for rows in sorted({lo, min(top, scr.rows)}):
            if rows in _WARM:
                continue
            g = torch.Generator(device="cpu").manual_seed(rows)
            picks = torch.stack([torch.randperm(ex.count, generator=g)[:top_k] for _ in range(rows)])
            picks = torch.cat([picks, torch.full((rows, 1), ex.count)], 1).to(torch.int32).to(dev).contiguous()
            plan = grouped.Plan(rows, top_k + 1, ex.count + 1, dev, prefill=True)
            grouped.route(picks, plan, tactic(rows)[0])
            x = torch.zeros((rows, ex.dims), dtype=torch.bfloat16, device=dev)
            y = torch.zeros((rows, top_k + 1, ex.dims), dtype=torch.bfloat16, device=dev)
            run(x, ex, plan, y, rows, top_k, scr)
            _WARM.add(rows)
        lo = top + 1
    torch.cuda.synchronize(dev)


def admission(geometry, decode_rows: int, prompt_rows: int | None = None):
    """The engine's geometry plus the Thor experts' fixed scratch: a decode window's and a prompt chunk's rows."""

    from tensorfold.cuda.geometry import PREFILL_ROWS, with_fixed

    rows = (decode_rows, prompt_rows or PREFILL_ROWS)

    def extra(t: dict) -> int:
        experts, top_k = int(t["num_experts"]), int(t["num_experts_per_tok"])
        width, dims = int(t["moe_intermediate_size"]), int(t["hidden_size"])
        return sum(scratch_bytes(r, top_k, experts, width, dims) for r in rows)

    return lambda text: with_fixed(geometry(text), extra(text))
