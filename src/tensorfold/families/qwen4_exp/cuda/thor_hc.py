"""Flash Next's prompt hyper-connections on Jetson Thor (sm_110), fused with TensorFold's bits (THOR-HC-FUSE).

On an NVFP4 checkpoint a hyper-connection's read-out runs five kernels a prompt pass (``forward._readout_b16``): the
streams normalized into ``normed`` (``glue.hc_normed``), the down projection on the bf16 split-K kernel
(``bf16.matmul``: 32 K slices of 320, each slice's fp32 partials [R, 324] stored, then summed in slice order by
``_reduce``), SiLU and the inject gates (``glue.hc_act``), the up projection (cuBLAS, ``thor_dense``) and the mix
(``glue.hc_mix``). At 4,096 rows a pass the split-K partials alone move 4x the bytes of the down projection's input,
and ``normed`` is written once and read twice. Here, for prompt passes:

- ``_down_act``: the down projection reads the streams ``h`` and normalizes them as it loads them (the very
  expression of ``glue._hc_normed``: bf16(h x rinv x scale), rinv from the write-back's 256-column partial sums in
  order), sums the same 32 K slices in one program (each slice on its own fp32 accumulator, the same tensor-core dot
  of 64-column blocks as ``bf16._b16mm``, then added to the running total in slice order from slice 0, as ``_reduce``
  does) and applies ``glue._hc_act``'s activation and inject gates to the total: no partials, no ``normed``;
- ``_mix_h``: ``glue._hc_mix`` with ``normed`` recomputed from ``h`` (the same expression), after the up projection;
- ``_upmix_h`` (only where a startup check finds it bit-equal to the up projection it replaces, see ``prepare``): the
  up projection's 64-column blocks for the four streams of a tile on the tensor cores, rounded to bf16, then the mix
  in the same program: ``up`` is never stored.

Exactness. Every value these kernels store is the stored value of the kernels they replace, bit for bit: the same
products summed in the same order, the same roundings, the same elementwise expressions (``glue._bsig``,
``glue._bsilu``). The 32-column group sums the replaced kernels also write (``xs_normed``, ``xs_act``,
``xs_mixed``) feed only 4-bit (MLX affine) matmuls; an NVFP4 checkpoint has none after a hyper-connection, so they
are not written. Rows stay independent, so prompts keep their bits at any split. ``prepare`` checks each kernel
against the kernels it replaces on this device before turning the path on; tools/thor/hc_check.py checks them on real
prompts against TensorFold's kernels and fp64.

``TENSORFOLD_THOR_HC_FUSE=0`` keeps TensorFold's read-out; decode windows always keep it; any other GPU never reaches
this module.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

from .glue import _bsig, _bsilu

ENV = "TENSORFOLD_THOR_HC_FUSE"
CAPABILITY = (11, 0)
BM, BK = 128, 64                # the down projection's row tile and K block (bf16._b16mm's prompt tile)
DOWN_N, DOWN_WARPS, DOWN_STAGES = 64, 4, 3      # its column tile, warps and pipeline depth
MIX_BLOCK = 256                 # _mix_h: dims a program (glue.hc_mix's)
UP_BM, UP_DB, UP_WARPS, UP_STAGES = 128, 64, 4, 3   # _upmix_h: rows and dims a program, warps, pipeline depth


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110 and not switched off: decided once."""

    return not _off() and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY


class _State:
    ready = False           # ``prepare``'s check passed on this device
    active = False          # prompt hyper-connections take this path (``prepare`` turns it on)
    upmix = False           # the up projection and the mix in one kernel (``prepare``'s check passed for it too)
    note = ""               # what ``prepare`` found


S = _State()


def set_active(on: bool) -> None:
    """This path (True) or TensorFold's read-out (False), once prepared: for A/B runs and tests."""

    S.active = bool(on) and S.ready


def use(hc) -> bool:
    """Whether a prompt pass's hyper-connection ``hc`` runs here: bf16 down and up projections (NVFP4 checkpoints)."""

    return S.active and getattr(hc.down, "kernel", "") == "b16" and getattr(hc.up, "kernel", "") == "b16"


@triton.jit(do_not_specialize=["M"])
def _down_act(H, PSS, SCALE, W, ACT, INJ, M, eps,
              N: tl.constexpr, D: tl.constexpr, S: tl.constexpr, NC: tl.constexpr, SK: tl.constexpr,
              LOW: tl.constexpr, HAS_INJ: tl.constexpr, BM: tl.constexpr, BLOCK_N: tl.constexpr, BK: tl.constexpr):
    """Program (row tile, column tile), column tiles of a row tile adjacent: the normalized streams [BM, S*D] @ W.T
    for W's first N rows, K in bf16.split_k's SK slices (each on a fresh fp32 accumulator, BK blocks in order, then
    added to the total in slice order), then hc_act's SiLU (columns < LOW) and inject gates (LOW..N-1, N <= LOW + S)
    of the total."""

    K: tl.constexpr = S * D
    NT: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    KS: tl.constexpr = K // SK                  # a slice's K
    NB: tl.constexpr = KS // BK                 # BK blocks a slice
    PER: tl.constexpr = D // KS                 # slices a stream
    pid = tl.program_id(0)
    pid_m = pid // NT
    pid_n = pid % NT
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    total = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for st in range(S):
        t = tl.zeros((BM,), dtype=tl.float32)
        for c in range(NC):                     # glue._hc_normed's rinv: the stream's partial sums in order
            t += tl.load(PSS + (rm * NC + c) * S + st, mask=m_ok, other=0.0)
        rinv = 1.0 / tl.sqrt(t / D + eps)
        for j in range(PER):
            acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
            for i in range(NB):
                k0 = (st * PER + j) * KS + i * BK
                hv = tl.load(H + rm[:, None] * K + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
                sc = tl.load(SCALE + k0 + rk).to(tl.float32)
                x = (hv * rinv[:, None] * sc[None, :]).to(tl.bfloat16)
                w = tl.load(W + rn[:, None] * K + (k0 + rk)[None, :], mask=n_ok[:, None], other=0.0)
                acc = tl.dot(x, tl.trans(w), acc)
            total = tl.where(st * PER + j == 0, acc, total + acc)     # _reduce: slice 0, then + each slice in order
    v = (total / S).to(tl.bfloat16).to(tl.float32)
    a = _bsilu(v)
    tl.store(ACT + rm[:, None] * LOW + rn[None, :], a.to(tl.bfloat16), mask=m_ok[:, None] & (rn < LOW)[None, :])
    if HAS_INJ:
        g = (2.0 * _bsig(v)).to(tl.bfloat16)
        tl.store(INJ + rm[:, None] * S + (rn - LOW)[None, :], g,
                 mask=m_ok[:, None] & ((rn >= LOW) & n_ok)[None, :])


@triton.jit
def _mix_h(UP, H, PSS, SCALE, MIXED, eps, D: tl.constexpr, S: tl.constexpr, NC: tl.constexpr, BLOCK: tl.constexpr):
    """Program (r, c): glue._hc_mix for dims [c BLOCK, (c + 1) BLOCK) of row r, each stream's normed value
    recomputed from h as glue._hc_normed stores it."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    d = c * BLOCK + tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(S):
        t = 0.0
        for k in range(NC):
            t += tl.load(PSS + (r * NC + k) * S + s)
        rinv = 1.0 / tl.sqrt(t / D + eps)
        hv = tl.load(H + r * (S * D) + s * D + d).to(tl.float32)
        w = tl.load(SCALE + s * D + d).to(tl.float32)
        n = (hv * rinv * w).to(tl.bfloat16).to(tl.float32)
        u = tl.load(UP + r * (S * D) + s * D + d).to(tl.float32)
        total += (_bsig(u) * n).to(tl.bfloat16).to(tl.float32)
    tl.store(MIXED + r * D + d, (total / S).to(tl.bfloat16))


@triton.jit(do_not_specialize=["M"])
def _upmix_h(ACT, W, H, PSS, SCALE, MIXED, UP, M, eps,
             D: tl.constexpr, S: tl.constexpr, LOW: tl.constexpr, NC: tl.constexpr,
             BM: tl.constexpr, DB: tl.constexpr, BK: tl.constexpr, STORE_UP: tl.constexpr):
    """Program (row tile, dim tile): for each stream in order, act [BM, LOW] @ W[s D + dims].T on the tensor cores (BK
    blocks in order on one fp32 accumulator), rounded to bf16 as the up projection stores it, then glue._hc_mix's
    sum with the normed streams recomputed from h. STORE_UP also stores the rounded projection (checks only)."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    dd = tl.program_id(1) * DB + tl.arange(0, DB)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    total = tl.zeros((BM, DB), dtype=tl.float32)
    for s in tl.static_range(S):
        n = s * D + dd
        acc = tl.zeros((BM, DB), dtype=tl.float32)
        for k0 in range(0, LOW, BK):
            a = tl.load(ACT + rm[:, None] * LOW + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + n[:, None] * LOW + (k0 + rk)[None, :])
            acc = tl.dot(a, tl.trans(w), acc)
        up = acc.to(tl.bfloat16)
        if STORE_UP:
            tl.store(UP + rm[:, None] * (S * D) + n[None, :], up, mask=m_ok[:, None])
        t = tl.zeros((BM,), dtype=tl.float32)
        for c in range(NC):
            t += tl.load(PSS + (rm * NC + c) * S + s, mask=m_ok, other=0.0)
        rinv = 1.0 / tl.sqrt(t / D + eps)
        hv = tl.load(H + rm[:, None] * (S * D) + n[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
        sc = tl.load(SCALE + n).to(tl.float32)
        nv = (hv * rinv[:, None] * sc[None, :]).to(tl.bfloat16).to(tl.float32)
        total += (_bsig(up.to(tl.float32)) * nv).to(tl.bfloat16).to(tl.float32)
    tl.store(MIXED + rm[:, None] * D + dd[None, :], (total / S).to(tl.bfloat16), mask=m_ok[:, None])


def down_act(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, down, act: torch.Tensor,
             inject: torch.Tensor | None, streams: int, low: int, eps: float, *, block_n: int = DOWN_N,
             num_warps: int = DOWN_WARPS, num_stages: int = DOWN_STAGES) -> None:
    """``down``: the hyper-connection's bf16 down (+ inject) rows (``bf16.B16``, [low (+ streams), S*D]); writes act
    [R, low] and the inject gates [R, streams] (when ``inject`` is given) as hc_normed, bf16.matmul(f32) and hc_act
    would."""

    from .bf16 import split_k

    rows, wide = h.shape
    d = wide // streams
    sk = split_k(down.n, down.k)                # the slices bf16.matmul sums: set by the whole matrix's shape
    used = low + streams if inject is not None else low     # the columns hc_act reads
    if down.k != wide or wide % sk or (wide // sk) % BK or d % (wide // sk) or pss.shape[1] * 256 != d:
        raise ValueError(f"thor_hc.down_act: {down.n}x{down.k} rows for streams {tuple(h.shape)}")
    if down.n < used:
        raise ValueError(f"thor_hc.down_act: {down.n} rows, inject gates need {used}")
    nt = -(-used // block_n)
    _down_act[(triton.cdiv(rows, BM) * nt,)](
        h, pss, scale, down.weight, act, inject if inject is not None else act, rows, eps,
        N=used, D=d, S=streams, NC=pss.shape[1], SK=sk, LOW=low, HAS_INJ=inject is not None, BM=BM,
        BLOCK_N=block_n, BK=BK, num_warps=num_warps, num_stages=num_stages)


def mix(up: torch.Tensor, h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, mixed: torch.Tensor,
        streams: int, eps: float) -> None:
    rows, wide = h.shape
    d = wide // streams
    _mix_h[(rows, d // MIX_BLOCK)](up, h, pss, scale, mixed, eps, D=d, S=streams, NC=pss.shape[1], BLOCK=MIX_BLOCK,
                                   num_warps=2)


def upmix(act: torch.Tensor, up_w: torch.Tensor, h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor,
          mixed: torch.Tensor, streams: int, eps: float, *, up: torch.Tensor | None = None, bm: int = UP_BM,
          db: int = UP_DB, num_warps: int = UP_WARPS, num_stages: int = UP_STAGES) -> None:
    """``up_w``: the up projection's bf16 rows [S*D, low]; ``up``: also store the rounded projection (checks)."""

    rows, wide = h.shape
    d, low = wide // streams, act.shape[1]
    if up_w.shape != (wide, low) or low % BK or d % db:
        raise ValueError(f"thor_hc.upmix: up rows {tuple(up_w.shape)} for streams {tuple(h.shape)}")
    _upmix_h[(triton.cdiv(rows, bm), d // db)](
        act, up_w, h, pss, scale, mixed, up if up is not None else mixed, rows, eps,
        D=d, S=streams, LOW=low, NC=pss.shape[1], BM=bm, DB=db, BK=BK, STORE_UP=up is not None,
        num_warps=num_warps, num_stages=num_stages)


def _up_ref(act: torch.Tensor, lin, out: torch.Tensor) -> torch.Tensor:
    """The up projection as TensorFold's prompt read-out runs it (forward._mm): cuBLAS (thor_dense) or bf16.matmul."""

    from tensorfold.cuda.nvfp4 import thor_dense

    from . import bf16

    if thor_dense.use(lin):
        return thor_dense.linear(act, lin, out=out)
    return bf16.matmul(act, lin.b, out=out)


def readout(hc, b, h: torch.Tensor, R: int, eps: float, streams: int, low: int, inject) -> None:
    """``forward._readout_b16`` for a prompt pass: b.act, ``inject`` and b.mixed[:R] with the same bits; b.normed,
    the group sums and (with the fused up projection) b.up are not written."""

    down_act(h[:R], b.pss[:R], hc.scale, hc.down.b, b.act[:R], inject, streams, low, eps)
    if S.upmix:
        upmix(b.act[:R], hc.up.weight, h[:R], b.pss[:R], hc.scale, b.mixed[:R], streams, eps)
        return
    _up_ref(b.act[:R], hc.up, b.up[:R])
    mix(b.up[:R], h[:R], b.pss[:R], hc.scale, b.mixed[:R], streams, eps)


@torch.no_grad()
def check(hc, rows: int, eps: float, streams: int, low: int, device, seed: int = 4111) -> dict:
    """This path against TensorFold's prompt read-out (glue.hc_normed, bf16.matmul f32, glue.hc_act, the up
    projection, glue.hc_mix) on random streams through the hyper-connection ``hc``'s weights: the bytes of every value
    both store (act, inject gates, mixed; the up projection's when fused)."""

    from . import bf16, glue

    wide = hc.scale.shape[0]
    d = wide // streams
    gen = torch.Generator(device=device).manual_seed(seed)
    h = (torch.randn((rows, wide), generator=gen, device=device) * 1.5).to(torch.bfloat16)
    pss = torch.empty((rows, d // 256, streams), dtype=torch.float32, device=device)
    glue.hc_writeback(h, h, pss, streams, 0)
    has_inj = hc.down.n > low
    normed = torch.empty_like(h)
    xs = torch.empty((rows, wide // 32), dtype=torch.float32, device=device)
    glue.hc_normed(h, pss, hc.scale, normed, xs, streams, eps)
    dn = bf16.matmul(normed, hc.down.b, f32=True)
    act0 = torch.empty((rows, low), dtype=torch.bfloat16, device=device)
    inj0 = torch.empty((rows, streams), dtype=torch.bfloat16, device=device) if has_inj else None
    glue.hc_act(dn, act0, torch.empty((rows, low // 32), dtype=torch.float32, device=device), inj0, streams, low)
    up0 = _up_ref(act0, hc.up, torch.empty((rows, wide), dtype=torch.bfloat16, device=device))
    mixed0 = torch.empty((rows, d), dtype=torch.bfloat16, device=device)
    glue.hc_mix(up0, normed, mixed0, torch.empty((rows, d // 32), dtype=torch.float32, device=device), streams)

    def same(a, b_):
        return bool(torch.equal(a.view(torch.uint8), b_.view(torch.uint8)))

    act1, inj1 = torch.empty_like(act0), (torch.empty_like(inj0) if has_inj else None)
    down_act(h, pss, hc.scale, hc.down.b, act1, inj1, streams, low, eps)
    mixed1 = torch.empty_like(mixed0)
    mix(up0, h, pss, hc.scale, mixed1, streams, eps)
    up2, mixed2 = torch.empty_like(up0), torch.empty_like(mixed0)
    upmix(act0, hc.up.weight, h, pss, hc.scale, mixed2, streams, eps, up=up2)
    out = {"rows": rows, "act": same(act0, act1), "inject": same(inj0, inj1) if has_inj else None,
           "mix": same(mixed0, mixed1), "up_fused": same(up0, up2), "mix_fused": same(mixed0, mixed2)}
    if not out["up_fused"]:
        diff = (up0.float() - up2.float()).abs()
        out["up_fused_mismatch"] = int((up0.view(torch.int16) != up2.view(torch.int16)).sum())
        out["up_fused_max_abs"] = float(diff.max())
    return out


def prepare(w) -> bool:
    """Compile and check the kernels on this device against TensorFold's read-out (rows 128, 1000 and 4096 through
    the first layer's hyper-connections), before the caches: the path turns on only if the down projection and the
    mix are bit-equal; the fused up projection only if it is too."""

    S.ready = S.active = S.upmix = False
    if not available():
        return False
    layer = w.layers[0]
    hcs = [hc for hc in (layer.attn_hc, layer.mlp_hc) if hc is not None]
    if not hcs or not all(getattr(hc.down, "kernel", "") == "b16" and getattr(hc.up, "kernel", "") == "b16"
                          for hc in hcs):
        S.note = "not a bf16 hyper-connection checkpoint"
        return False
    c = w.cfg
    device = w.inv_freq.device
    try:
        results = [check(hc, rows, c.eps, c.streams, c.low, device) for hc in hcs for rows in (128, 1000, 4096)]
    except Exception as exc:                # noqa: BLE001  (any failure keeps TensorFold's read-out)
        S.note = f"its kernels failed ({type(exc).__name__}: {exc})"
        return False
    finally:
        torch.cuda.empty_cache()            # the check's buffers do not stay reserved when the caches are sized
    exact = all(r["act"] and r["mix"] and r["inject"] in (True, None) for r in results)
    fused = exact and all(r["up_fused"] and r["mix_fused"] for r in results)
    if not exact:
        S.note = f"bytes differ from TensorFold's read-out: {results}"
        return False
    S.ready = S.active = True
    S.upmix = fused
    S.note = "down + activation fused" + (", up + mix fused" if fused else
                                          f"; up on the dense path (fused up differs: {results[0]})")
    return True
