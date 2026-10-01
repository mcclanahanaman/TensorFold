"""A higher-precision referee for Flash Next's routed experts (G-B divergence analysis): fp32 SwiGLU, fp32 outputs.

Reads the Thor layout already in memory (``tensorfold.cuda.nvfp4.thor_experts.ThorExperts``), so the whole model fits:
each weight tile is decoded to bf16 exactly (an e2m1 code times its e4m3 scale), gate and up sum x (bf16) over all of
K in fp32, SwiGLU stays fp32 (no bf16 rounding of g, u, silu or h), and the down GEMM takes h as an exact-to-2^-16
bf16 pair (hi + lo) summed in fp32; the expert outputs reach TensorFold's combine in fp32 slots. Everything else
(router, shared expert, attention, DeltaNet, hyper-connections) is TensorFold's own prompt arithmetic.

    import moe_reference; moe_reference.install()      # before a prompt pass; experts on the Thor layout only
"""

from __future__ import annotations

import functools

import torch

TILE = 16            # pairs a plan item holds: the kernels' token block
BW = 64              # weight rows a program


@functools.lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def e2m1(code):
        m = code & 7
        pat = tl.where(m == 0, 0, ((126 + (m >> 1)) << 7) | (tl.where(m >= 2, m & 1, 0) << 6)) | ((code & 8) << 12)
        return pat.to(tl.uint16).to(tl.bfloat16, bitcast=True)

    @triton.jit
    def e4m3(byte):
        e = (byte >> 3) & 0xF
        m = byte & 0x7
        normal = (((e + 120) << 7) | (m << 4)).to(tl.int32) << 16
        v = tl.where(e == 0, m.to(tl.float32) * (2.0 ** -9), normal.to(tl.float32, bitcast=True))
        return tl.where((byte & 0x80) != 0, -v, v)

    @triton.jit
    def weights(W, SF, e, rows, k0, K: tl.constexpr, R: tl.constexpr, BW: tl.constexpr, BK: tl.constexpr):
        """Rows ``rows`` [BW] of expert e, inputs k0 .. k0 + BK -> bf16 [BW, BK] (code x scale, exact)."""

        e64 = e.to(tl.int64)
        c = tl.load(W + e64 * R * (K // 2) + rows[:, None].to(tl.int64) * (K // 2)
                    + (k0 // 2 + tl.arange(0, BK // 2))[None, :]).to(tl.int32)
        codes = tl.reshape(tl.join(c & 0xF, c >> 4), (BW, BK))
        col = k0 // 16 + tl.arange(0, BK // 16)                      # scale columns (of K/16)
        cb = K // 16 // 4
        idx = (((rows // 128)[:, None] * cb + (col // 4)[None, :]) * 512 + (rows % 32)[:, None] * 16
               + ((rows % 128) // 32)[:, None] * 4 + (col % 4)[None, :])
        s = tl.load(SF + e64 * R * (K // 16) + idx.to(tl.int64)).to(tl.int32)
        sv = tl.reshape(tl.broadcast_to(e4m3(s).to(tl.bfloat16)[:, :, None], (BW, BK // 16, 16)), (BW, BK))
        return e2m1(codes) * sv

    @triton.jit
    def gate_up(X, x_stride, W1, SF1, A1, MEMBERS, ITEMS, COUNTS, H, shared,
                SLOTS: tl.constexpr, K: tl.constexpr, NI: tl.constexpr, BW: tl.constexpr, BT: tl.constexpr,
                BK: tl.constexpr):
        i = tl.program_id(0)
        if i < tl.load(COUNTS):
            e = tl.load(ITEMS + 3 * i)
            if e != shared:
                first = tl.load(ITEMS + 3 * i + 1)
                cnt = tl.load(ITEMS + 3 * i + 2)
                r = tl.arange(0, BT)
                ok = r < cnt
                pair = tl.load(MEMBERS + first + r, mask=ok, other=0)
                tok = (pair // SLOTS).to(tl.int64)
                j = tl.program_id(1) * BW + tl.arange(0, BW)          # output rows of NI
                up_rows = (j // 64) * 128 + (j % 64)                  # the gated order: 64 up rows, 64 gate rows
                gate_rows = up_rows + 64
                accg = tl.zeros((BW, BT), tl.float32)
                accu = tl.zeros((BW, BT), tl.float32)
                kk = tl.arange(0, BK)
                for k0 in range(0, K, BK):
                    x = tl.load(X + tok[None, :] * x_stride + (k0 + kk)[:, None], mask=ok[None, :], other=0.0)
                    accg = tl.dot(weights(W1, SF1, e, gate_rows, k0, K, 2 * NI, BW, BK), x, accg)
                    accu = tl.dot(weights(W1, SF1, e, up_rows, k0, K, 2 * NI, BW, BK), x, accu)
                a = tl.load(A1 + e)
                g = accg * a
                u = accu * a
                h = g / (1.0 + tl.exp(-g)) * u
                tl.store(H + pair[None, :].to(tl.int64) * NI + j[:, None], h, mask=ok[None, :])

    @triton.jit
    def down(H, W2, SF2, A2, MEMBERS, ITEMS, COUNTS, Y, shared,
             K: tl.constexpr, N: tl.constexpr, BW: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
        i = tl.program_id(0)
        if i < tl.load(COUNTS):
            e = tl.load(ITEMS + 3 * i)
            if e != shared:
                first = tl.load(ITEMS + 3 * i + 1)
                cnt = tl.load(ITEMS + 3 * i + 2)
                r = tl.arange(0, BT)
                ok = r < cnt
                pair = tl.load(MEMBERS + first + r, mask=ok, other=0).to(tl.int64)
                j = tl.program_id(1) * BW + tl.arange(0, BW)          # output rows of D
                acc = tl.zeros((BW, BT), tl.float32)
                kk = tl.arange(0, BK)
                for k0 in range(0, K, BK):
                    h = tl.load(H + pair[None, :] * K + (k0 + kk)[:, None], mask=ok[None, :], other=0.0)
                    hi = h.to(tl.bfloat16)
                    lo = (h - hi.to(tl.float32)).to(tl.bfloat16)
                    w = weights(W2, SF2, e, j, k0, K, N, BW, BK)
                    acc = tl.dot(w, hi, acc)
                    acc = tl.dot(w, lo, acc)
                y = acc * tl.load(A2 + e)
                tl.store(Y + pair[None, :] * N + j[:, None], y, mask=ok[None, :])

    return gate_up, down


def routed(x: torch.Tensor, ex, plan, y32: torch.Tensor, h32: torch.Tensor, rows: int, top_k: int) -> None:
    """x [R, D] bf16 -> y32 [R, slots, D] fp32 for the routed slots; ``plan`` routed with TILE-pair items."""

    from tensorfold.cuda import experts as grouped

    gate_up, down = _kernels()
    slots = y32.shape[1]
    items = grouped.max_items(rows * slots, ex.count + 1, TILE)
    gate_up[(items, ex.width // BW)](x, x.stride(0), ex.w1, ex.w1_sf, ex.w1_alpha, plan.members, plan.items,
                                     plan.counts, h32, ex.count, SLOTS=slots, K=ex.dims, NI=ex.width, BW=BW, BT=TILE,
                                     BK=64, num_warps=4)
    down[(items, ex.dims // BW)](h32, ex.w2, ex.w2_sf, ex.w2_alpha, plan.members, plan.items, plan.counts,
                                 y32.view(-1, ex.dims), ex.count, K=ex.width, N=ex.dims, BW=BW, BT=TILE, BK=64,
                                 num_warps=4)


_INSTALLED: dict = {}


def install() -> None:
    """Route every Thor-layout MoE layer through the referee (the MTP head's portable experts are left alone)."""

    from tensorfold.cuda import moe as moe_mod
    from tensorfold.cuda.nvfp4 import thor_experts as thor
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe

    if _INSTALLED:
        return
    original = nvfp4_moe.moe

    def moe(x, xs, router_rows, ex, buf, cfg):
        routed_ex = ex.routed_experts
        if not isinstance(routed_ex, thor.ThorExperts):
            return original(x, xs, router_rows, ex, buf, cfg)
        rows, top_k = x.shape[0], int(cfg.num_experts_per_tok)
        moe_mod.router(x, router_rows, buf.logits[:rows])
        moe_mod.select(buf.logits[:rows], buf, top_k, ex.routed, TILE)
        if getattr(buf, "ref_y32", None) is None:          # fp32 slots from here on: the combine reads them as fp32
            buf.ref_y32 = torch.zeros(buf.y.shape, dtype=torch.float32, device=buf.y.device)
            buf.ref_h32 = torch.zeros((buf.y.shape[0] * buf.y.shape[1], routed_ex.width), dtype=torch.float32,
                                      device=buf.y.device)
            buf.y = buf.ref_y32
        routed(x, routed_ex, buf.plan, buf.ref_y32, buf.ref_h32, rows, top_k)
        buf.act[:rows, top_k] = ex.shared_act(x, True)        # the shared expert as the prompt passes run it
        buf.ref_y32[:rows, top_k] = nvfp4_moe._shared(ex.shared.down, buf.act[:rows, top_k], True, False).float()
        return buf

    nvfp4_moe.moe = moe
    _INSTALLED["moe"] = original


def uninstall() -> None:
    from tensorfold.families.qwen4_exp.cuda import nvfp4_moe

    if _INSTALLED:
        nvfp4_moe.moe = _INSTALLED.pop("moe")
