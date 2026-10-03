"""Flash Next's prompt hyper-connection read-out on Jetson Thor (sm_110), warp-specialized (THOR-HC-FUSE-2).

``thor_hc`` runs a prompt pass's read-out as two Triton kernels (the down projection with its activation, then the up
projection with the mix). Here the same two steps run as Gluon kernels whose warps specialize, so TMA loads, the
tensor cores and the CUDA cores overlap:

- ``_down``: a CTA takes 128 rows and 176 of the down projection's columns (two CTAs a row tile). One warp loads h's
  and the weight's [*, 64] blocks with TMA; eight warps normalize each h block as ``glue._hc_normed`` stores it
  (bf16(h x rinv x scale), rinv from the write-back's partial sums in order) into the MMA's A block; one warp issues
  the tensor-core MMAs (N = 128, 32 and 16), each of ``bf16.split_k``'s K slices on a fresh accumulator in tensor
  memory (two buffers: a finished slice is added while the next one runs); four warps add each slice to the running
  sum in slice order (slice 0 as is, then + each slice, ``bf16._reduce``'s order; the sums stay in tensor memory) and
  apply ``glue._hc_act``'s activation and inject gates to the total.
- ``_upmix``: persistent CTAs over tiles of 128 rows and 64 dims: one warp loads act's and the up projection's
  blocks and the tile's four h blocks with TMA; one warp issues the four streams' MMAs (each stream on its own
  accumulator over the 64-column K blocks in order; two tiles in flight); eight warps round each projection to bf16
  and mix it with the normed streams recomputed from h (``glue._hc_mix`` on ``glue._hc_normed``'s values); the mix
  leaves through TMA.

Exactness. Every stored value is the stored value of TensorFold's read-out, bit for bit: the same products summed in
the same order (the tensor cores' per-element sums do not depend on the MMA's shape on sm_110; hcf-t1, hcf-t2), the
same roundings and the same elementwise expressions. ``thor_hc.prepare`` checks these kernels against TensorFold's
read-out on this device before turning them on; tests/cuda/test_thor_hc2.py checks them at any row count.

``TENSORFOLD_THOR_HC_FUSE2=0`` keeps thor_hc's Triton kernels; prompt passes under ``MIN_ROWS`` rows keep them too
(those are as fast there); other GPUs never reach this module.
"""

from __future__ import annotations

import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (TensorMemoryLayout, allocate_tensor_memory,
                                                                 get_tmem_reg_layout, tcgen05_commit, tcgen05_mma)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

ENV = "TENSORFOLD_THOR_HC_FUSE2"
MIN_ROWS = 256                  # fewer prompt rows: thor_hc's Triton kernels (hcf2-p2: as fast at 128 rows)
BM, BK = 128, 64                # rows a tile; the K block of every MMA (bf16._b16mm's)
TN = 176                        # _down: columns a CTA (MMAs of 128, 32 and 16)
_BM, _BK, _TN = gl.constexpr(BM), gl.constexpr(BK), gl.constexpr(TN)       # the kernels' copies
DOWN_STAGES, DOWN_NORM_WARPS, DOWN_NORM_REGS = 4, 8, 80     # hcf2-p2: 0.54 ms at 4,096 rows
UP_DB, UP_STAGES = 64, 2                                    # _upmix: dims a tile, operand stages


def off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


@gluon.jit
def _gsig(x):
    return (1.0 / (1.0 + gl.exp(-x))).to(gl.bfloat16).to(gl.float32)          # glue._bsig


@gluon.jit
def _gsilu(x):
    return (x / (1.0 + gl.exp(-x))).to(gl.bfloat16).to(gl.float32)            # glue._bsilu


@gluon.jit
def _rinv(PSS, rows, M, st, eps, D: gl.constexpr, S: gl.constexpr, NC: gl.constexpr):
    """glue._hc_normed's rinv for stream ``st`` of ``rows``: the NC partial sums added in order from 0."""

    m_ok = rows < M
    t = gl.zeros(rows.shape, gl.float32, rows.type.layout)
    for c in gl.static_range(NC):
        t += gl.load(PSS + (rows * NC + c) * S + st, mask=m_ok, other=0.0)
    return 1.0 / gl.sqrt(t / D + eps)


# ------------------------------------------------------------------ down projection with the activation
@gluon.jit
def _down_load(h_desc, w0_desc, w1_desc, w2_desc, h_bufs, w0_bufs, w1_bufs, w2_bufs, ready, empty, off_m, n0,
               NKB: gl.constexpr, NS: gl.constexpr):
    nbytes: gl.constexpr = (h_desc.block_type.nbytes + w0_desc.block_type.nbytes + w1_desc.block_type.nbytes
                            + w2_desc.block_type.nbytes)
    for kb in range(NKB):
        s = kb % NS
        mbarrier.wait(empty.index(s), ((kb // NS) & 1) ^ 1)
        bar = ready.index(s)
        mbarrier.expect(bar, nbytes)
        k0 = kb * _BK
        tma.async_copy_global_to_shared(h_desc, [off_m, k0], bar, h_bufs.index(s))
        tma.async_copy_global_to_shared(w0_desc, [n0, k0], bar, w0_bufs.index(s))
        tma.async_copy_global_to_shared(w1_desc, [n0 + 128, k0], bar, w1_bufs.index(s))
        tma.async_copy_global_to_shared(w2_desc, [n0 + 160, k0], bar, w2_bufs.index(s))


@gluon.jit
def _down_norm(h_bufs, a_bufs, ready, a_ready, empty, PSS, SCALE, M, eps, off_m, D: gl.constexpr, S: gl.constexpr,
               NC: gl.constexpr, NKB: gl.constexpr, NS: gl.constexpr, NW: gl.constexpr):
    """Each [128, 64] block of h, normalized with glue._hc_normed's expression, into the MMA's A block."""

    L: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    PERB: gl.constexpr = D // _BK                # K blocks a stream
    rows = off_m + gl.arange(0, _BM, gl.SliceLayout(1, L))
    cols = gl.arange(0, _BK, gl.SliceLayout(0, L))
    r0 = _rinv(PSS, rows, M, 0, eps, D, S, NC)
    r1 = _rinv(PSS, rows, M, 1, eps, D, S, NC)
    r2 = _rinv(PSS, rows, M, 2, eps, D, S, NC)
    r3 = _rinv(PSS, rows, M, 3, eps, D, S, NC)
    sc_next = gl.load(SCALE + cols).to(gl.float32)
    for kb in range(NKB):
        s = kb % NS
        sc = sc_next
        sc_next = gl.load(SCALE + ((kb + 1) % NKB) * _BK + cols).to(gl.float32)     # the next block's, ahead
        st = kb // PERB
        rinv = gl.where(st == 0, r0, gl.where(st == 1, r1, gl.where(st == 2, r2, r3)))
        mbarrier.wait(ready.index(s), (kb // NS) & 1)
        hv = h_bufs.index(s).load(L).to(gl.float32)
        a_bufs.index(s).store((hv * rinv[:, None] * sc[None, :]).to(gl.bfloat16))
        fence_async_shared()
        mbarrier.arrive(a_ready.index(s), count=1)
        mbarrier.arrive(empty.index(s), count=1)


@gluon.jit
def _down_mma(a_bufs, w0_bufs, w1_bufs, w2_bufs, acc0, acc1, acc2, ready, a_ready, empty, acc_ready, acc_empty,
              NKB: gl.constexpr, NB: gl.constexpr, NS: gl.constexpr):
    """Each K slice (NB blocks) on a fresh accumulator (use_acc off at its first block), the buffers alternating."""

    for kb in range(NKB):
        s = kb % NS
        ph = (kb // NS) & 1
        sl = kb // NB
        j = kb % NB
        b = sl % 2
        if j == 0:
            mbarrier.wait(acc_empty.index(b), ((sl // 2) & 1) ^ 1)
        mbarrier.wait(ready.index(s), ph)
        mbarrier.wait(a_ready.index(s), ph)
        a = a_bufs.index(s)
        use = j > 0
        tcgen05_mma(a, w0_bufs.index(s).permute((1, 0)), acc0.index(b), use_acc=use)
        tcgen05_mma(a, w1_bufs.index(s).permute((1, 0)), acc1.index(b), use_acc=use)
        tcgen05_mma(a, w2_bufs.index(s).permute((1, 0)), acc2.index(b), use_acc=use)
        tcgen05_commit(empty.index(s))
        if j == NB - 1:
            tcgen05_commit(acc_ready.index(b))


@gluon.jit
def _act_store(t, rows, cols, ACT, INJ, M, S: gl.constexpr, LOW: gl.constexpr, N: gl.constexpr,
               HAS_INJ: gl.constexpr):
    """glue._hc_act on the summed columns ``cols``: SiLU below LOW, the inject gates in LOW..N-1."""

    v = (t / S).to(gl.bfloat16).to(gl.float32)
    m_ok = rows < M
    gl.store(ACT + rows[:, None] * LOW + cols[None, :], _gsilu(v).to(gl.bfloat16),
             mask=m_ok[:, None] & (cols < LOW)[None, :])
    if HAS_INJ:
        gl.store(INJ + rows[:, None] * S + (cols - LOW)[None, :], (2.0 * _gsig(v)).to(gl.bfloat16),
                 mask=m_ok[:, None] & ((cols >= LOW) & (cols < N))[None, :])


@gluon.jit
def _down_sum(acc0, acc1, acc2, tot0, tot1, acc_ready, acc_empty, ACT, INJ, M, off_m, n0, S: gl.constexpr,
              LOW: gl.constexpr, N: gl.constexpr, SK: gl.constexpr, HAS_INJ: gl.constexpr, num_warps: gl.constexpr):
    """The slices summed in slice order (32 columns at a time; the sums of the first 160 columns in tensor memory,
    the last 16 in registers), then the activation and gates."""

    CW: gl.constexpr = 32
    qc: gl.constexpr = get_tmem_reg_layout(gl.float32, (_BM, CW), TensorMemoryLayout((_BM, CW), col_stride=1),
                                           num_warps)
    q2: gl.constexpr = get_tmem_reg_layout(gl.float32, (_BM, 16), TensorMemoryLayout((_BM, 16), col_stride=1),
                                           num_warps)
    mbarrier.wait(acc_ready.index(0), 0)
    for c in gl.static_range(0, 128, CW):
        tot0.slice(c, CW).store(acc0.index(0).slice(c, CW).load(qc))
    tot1.store(acc1.index(0).load(qc))
    t2 = acc2.index(0).load(q2)
    mbarrier.arrive(acc_empty.index(0), count=1)
    for sl in range(1, SK):
        b = sl % 2
        mbarrier.wait(acc_ready.index(b), (sl // 2) & 1)
        for c in gl.static_range(0, 128, CW):
            tot0.slice(c, CW).store(tot0.slice(c, CW).load(qc) + acc0.index(b).slice(c, CW).load(qc))
        tot1.store(tot1.load(qc) + acc1.index(b).load(qc))
        t2 = t2 + acc2.index(b).load(q2)
        mbarrier.arrive(acc_empty.index(b), count=1)
    rows = off_m + gl.arange(0, _BM, gl.SliceLayout(1, qc))
    for c in gl.static_range(0, 128, CW):
        _act_store(tot0.slice(c, CW).load(qc), rows, n0 + c + gl.arange(0, CW, gl.SliceLayout(0, qc)),
                   ACT, INJ, M, S, LOW, N, HAS_INJ)
    _act_store(tot1.load(qc), rows, n0 + 128 + gl.arange(0, CW, gl.SliceLayout(0, qc)), ACT, INJ, M, S, LOW, N,
               HAS_INJ)
    _act_store(t2, off_m + gl.arange(0, _BM, gl.SliceLayout(1, q2)), n0 + 160 + gl.arange(0, 16, gl.SliceLayout(0, q2)),
               ACT, INJ, M, S, LOW, N, HAS_INJ)


@gluon.jit
def _down(h_desc, w0_desc, w1_desc, w2_desc, PSS, SCALE, ACT, INJ, M, eps,
          D: gl.constexpr, S: gl.constexpr, NC: gl.constexpr, SK: gl.constexpr, LOW: gl.constexpr, N: gl.constexpr,
          HAS_INJ: gl.constexpr, NS: gl.constexpr, NW: gl.constexpr, NREG: gl.constexpr, num_warps: gl.constexpr):
    """CTA (row tile, column tile), the two column tiles of a row tile adjacent (h's blocks shared through L2)."""

    gl.static_assert(S == 4)
    NKB: gl.constexpr = S * D // _BK
    NB: gl.constexpr = NKB // SK                # K blocks a slice
    NT: gl.constexpr = (N + _TN - 1) // _TN
    pid = gl.program_id(0)
    off_m = (pid // NT) * _BM
    n0 = (pid % NT) * _TN
    h_bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, _BM, _BK], h_desc.layout)
    a_bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, _BM, _BK], h_desc.layout)
    w0_bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, 128, _BK], w0_desc.layout)
    w1_bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, 32, _BK], w1_desc.layout)
    w2_bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, 16, _BK], w2_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    a_ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NS):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(a_ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=2)          # the normalizer's arrival and the MMAs' commit
    acc_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(acc_ready.index(i), count=1)
        mbarrier.init(acc_empty.index(i), count=1)
    acc0 = allocate_tensor_memory(gl.float32, [2, _BM, 128], TensorMemoryLayout((_BM, 128), col_stride=1))
    acc1 = allocate_tensor_memory(gl.float32, [2, _BM, 32], TensorMemoryLayout((_BM, 32), col_stride=1))
    acc2 = allocate_tensor_memory(gl.float32, [2, _BM, 16], TensorMemoryLayout((_BM, 16), col_stride=1))
    tot0 = allocate_tensor_memory(gl.float32, [_BM, 128], TensorMemoryLayout((_BM, 128), col_stride=1))
    tot1 = allocate_tensor_memory(gl.float32, [_BM, 32], TensorMemoryLayout((_BM, 32), col_stride=1))
    gl.warp_specialize([
        (_down_sum, (acc0, acc1, acc2, tot0, tot1, acc_ready, acc_empty, ACT, INJ, M, off_m, n0, S, LOW, N, SK,
                     HAS_INJ, num_warps)),
        (_down_load, (h_desc, w0_desc, w1_desc, w2_desc, h_bufs, w0_bufs, w1_bufs, w2_bufs, ready, empty, off_m, n0,
                      NKB, NS)),
        (_down_mma, (a_bufs, w0_bufs, w1_bufs, w2_bufs, acc0, acc1, acc2, ready, a_ready, empty, acc_ready,
                     acc_empty, NKB, NB, NS)),
        (_down_norm, (h_bufs, a_bufs, ready, a_ready, empty, PSS, SCALE, M, eps, off_m, D, S, NC, NKB, NS, NW)),
    ], [1, 1, NW], [24, 24, NREG])


# ------------------------------------------------------------------ up projection with the mix
@gluon.jit
def _up_load(a_desc, w_desc, h_desc, a_bufs, w_bufs, h_s, ready, empty, h_ready, h_empty, M, D: gl.constexpr,
             NKB: gl.constexpr, NS: gl.constexpr, DB: gl.constexpr):
    NDT: gl.constexpr = D // DB
    nbytes: gl.constexpr = a_desc.block_type.nbytes + 4 * w_desc.block_type.nbytes
    ntiles = gl.cdiv(M, _BM) * NDT
    it = 0
    tc = 0
    for t in range(gl.program_id(0), ntiles, gl.num_programs(0)):
        off_m = (t // NDT) * _BM
        d0 = (t % NDT) * DB
        mbarrier.wait(h_empty, (tc & 1) ^ 1)
        mbarrier.expect(h_ready, 4 * h_desc.block_type.nbytes)
        for st in gl.static_range(4):
            tma.async_copy_global_to_shared(h_desc, [off_m, st * D + d0], h_ready, h_s.index(st))
        for kb in range(NKB):
            s = it % NS
            mbarrier.wait(empty.index(s), ((it // NS) & 1) ^ 1)
            bar = ready.index(s)
            mbarrier.expect(bar, nbytes)
            tma.async_copy_global_to_shared(a_desc, [off_m, kb * _BK], bar, a_bufs.index(s))
            for st in gl.static_range(4):
                tma.async_copy_global_to_shared(w_desc, [st * D + d0, kb * _BK], bar, w_bufs.index(s * 4 + st))
            it += 1
        tc += 1


@gluon.jit
def _up_mma(a_bufs, w_bufs, acc, ready, empty, acc_ready, acc_empty, M, D: gl.constexpr, NKB: gl.constexpr,
            NS: gl.constexpr, DB: gl.constexpr):
    NDT: gl.constexpr = D // DB
    ntiles = gl.cdiv(M, _BM) * NDT
    it = 0
    tc = 0
    for t in range(gl.program_id(0), ntiles, gl.num_programs(0)):
        b = tc % 2
        mbarrier.wait(acc_empty.index(b), ((tc // 2) & 1) ^ 1)
        for kb in range(NKB):
            s = it % NS
            mbarrier.wait(ready.index(s), (it // NS) & 1)
            a = a_bufs.index(s)
            use = kb > 0
            for st in gl.static_range(4):
                tcgen05_mma(a, w_bufs.index(s * 4 + st).permute((1, 0)), acc.index(b * 4 + st), use_acc=use)
            tcgen05_commit(empty.index(s))
            it += 1
        tcgen05_commit(acc_ready.index(b))
        tc += 1


@gluon.jit
def _mix_term(acc_s, hv, PSS, SCALE, rows, dd, st: gl.constexpr, M, eps, D: gl.constexpr, S: gl.constexpr,
              NC: gl.constexpr, q: gl.constexpr):
    """Stream st's term of glue._hc_mix: bf16(bf16(sigmoid(up)) * normed), normed as glue._hc_normed stores it."""

    rinv = _rinv(PSS, rows, M, st, eps, D, S, NC)
    sc = gl.load(SCALE + st * D + dd).to(gl.float32)
    nv = (hv.to(gl.float32) * rinv[:, None] * sc[None, :]).to(gl.bfloat16).to(gl.float32)
    up = acc_s.load(q).to(gl.bfloat16).to(gl.float32)
    return (_gsig(up) * nv).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _up_mix(acc, h_s, o_s, acc_ready, acc_empty, h_ready, h_empty, o_desc, PSS, SCALE, M, eps, D: gl.constexpr,
            S: gl.constexpr, NC: gl.constexpr, DB: gl.constexpr, num_warps: gl.constexpr):
    """A tile: h's four blocks into registers (its buffer freed for the next tile's), then glue._hc_mix's sum in stream
    order from 0 and its rounding (hcf2-p3: faster than normalizing before the projections are ready)."""

    NDT: gl.constexpr = D // DB
    q: gl.constexpr = get_tmem_reg_layout(gl.float32, (_BM, DB), TensorMemoryLayout((_BM, DB), col_stride=1),
                                          num_warps)
    ntiles = gl.cdiv(M, _BM) * NDT
    tc = 0
    for t in range(gl.program_id(0), ntiles, gl.num_programs(0)):
        off_m = (t // NDT) * _BM
        d0 = (t % NDT) * DB
        rows = off_m + gl.arange(0, _BM, gl.SliceLayout(1, q))
        dd = d0 + gl.arange(0, DB, gl.SliceLayout(0, q))
        mbarrier.wait(h_ready, tc & 1)
        h0 = h_s.index(0).load(q)
        h1 = h_s.index(1).load(q)
        h2 = h_s.index(2).load(q)
        h3 = h_s.index(3).load(q)
        fence_async_shared()
        mbarrier.arrive(h_empty, count=1)
        b = tc % 2
        mbarrier.wait(acc_ready.index(b), (tc // 2) & 1)
        total = gl.zeros([_BM, DB], gl.float32, q)
        total += _mix_term(acc.index(b * 4 + 0), h0, PSS, SCALE, rows, dd, 0, M, eps, D, S, NC, q)
        total += _mix_term(acc.index(b * 4 + 1), h1, PSS, SCALE, rows, dd, 1, M, eps, D, S, NC, q)
        total += _mix_term(acc.index(b * 4 + 2), h2, PSS, SCALE, rows, dd, 2, M, eps, D, S, NC, q)
        total += _mix_term(acc.index(b * 4 + 3), h3, PSS, SCALE, rows, dd, 3, M, eps, D, S, NC, q)
        mbarrier.arrive(acc_empty.index(b), count=1)
        tma.store_wait(pendings=0)
        o_s.store((total / S).to(gl.bfloat16))
        fence_async_shared()
        tma.async_copy_shared_to_global(o_desc, [off_m, d0], o_s)
        tc += 1
    tma.store_wait(pendings=0)


@gluon.jit
def _upmix(a_desc, w_desc, h_desc, o_desc, PSS, SCALE, M, eps, D: gl.constexpr, S: gl.constexpr, LOW: gl.constexpr,
           NC: gl.constexpr, DB: gl.constexpr, NS: gl.constexpr, num_warps: gl.constexpr):
    gl.static_assert(S == 4)
    NKB: gl.constexpr = LOW // _BK
    a_bufs = gl.allocate_shared_memory(gl.bfloat16, [NS, _BM, _BK], a_desc.layout)
    w_bufs = gl.allocate_shared_memory(gl.bfloat16, [NS * 4, DB, _BK], w_desc.layout)
    h_s = gl.allocate_shared_memory(gl.bfloat16, [4, _BM, DB], h_desc.layout)
    o_s = gl.allocate_shared_memory(gl.bfloat16, [_BM, DB], o_desc.layout)
    ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NS):
        mbarrier.init(ready.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    acc_ready = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_empty = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(2):
        mbarrier.init(acc_ready.index(i), count=1)
        mbarrier.init(acc_empty.index(i), count=1)
    h_ready = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    h_empty = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(h_ready, count=1)
    mbarrier.init(h_empty, count=1)
    acc = allocate_tensor_memory(gl.float32, [8, _BM, DB], TensorMemoryLayout((_BM, DB), col_stride=1))
    gl.warp_specialize([
        (_up_mix, (acc, h_s, o_s, acc_ready, acc_empty, h_ready, h_empty, o_desc, PSS, SCALE, M, eps, D, S, NC, DB,
                   num_warps)),
        (_up_load, (a_desc, w_desc, h_desc, a_bufs, w_bufs, h_s, ready, empty, h_ready, h_empty, M, D, NKB, NS, DB)),
        (_up_mma, (a_bufs, w_bufs, acc, ready, empty, acc_ready, acc_empty, M, D, NKB, NS, DB)),
    ], [1, 1], [24, 24])


# ------------------------------------------------------------------ host side
_DESC: dict = {}
_SMS: dict = {}


def _desc(t: torch.Tensor, block: list[int]) -> TensorDescriptor:
    """A TMA descriptor of ``t`` with a [block] box, kept per (address, shape, strides, box): a descriptor holds only
    those, so a cached one stays valid for any tensor that matches them."""

    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), tuple(block), t.dtype)
    d = _DESC.get(key)
    if d is None:
        if len(_DESC) > 512:
            _DESC.clear()
        d = _DESC[key] = TensorDescriptor.from_tensor(t, block, gl.NVMMASharedLayout.get_default_for(block,
                                                                                                    gl.bfloat16))
    return d


def fits(rows: int, wide: int, streams: int, low: int, nc: int, used: int, sk: int) -> bool:
    """Whether these kernels take a read-out of this shape (four streams, whole 64-column blocks, slices of whole
    blocks); else thor_hc's Triton kernels run it."""

    d = wide // streams
    return (rows >= MIN_ROWS and streams == 4 and d % BK == 0 and low % BK == 0 and nc * 256 == d and used <= 2 * TN
            and wide % sk == 0 and (wide // sk) % BK == 0)


def down_act(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, down, act: torch.Tensor,
             inject: torch.Tensor | None, streams: int, low: int, eps: float, sk: int):
    """thor_hc.down_act's outputs (act [R, low], the inject gates [R, streams]) with the same bits; ``down`` is the
    bf16 down projection (``bf16.B16``), ``sk`` its slices (``bf16.split_k``)."""

    rows, wide = h.shape
    used = low + streams if inject is not None else low
    w = down.weight[:used]
    return _down[(triton.cdiv(rows, BM) * -(-used // TN),)](
        _desc(h, [BM, BK]), _desc(w, [128, BK]), _desc(w, [32, BK]), _desc(w, [16, BK]), pss, scale, act,
        inject if inject is not None else act, rows, eps, D=wide // streams, S=streams, NC=pss.shape[1], SK=sk,
        LOW=low, N=used, HAS_INJ=inject is not None, NS=DOWN_STAGES, NW=DOWN_NORM_WARPS, NREG=DOWN_NORM_REGS,
        num_warps=4, maxnreg=128)


def upmix(act: torch.Tensor, up_w: torch.Tensor, h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor,
          mixed: torch.Tensor, streams: int, eps: float):
    """thor_hc.upmix's mix [R, D] with the same bits; ``up_w`` the up projection's bf16 rows [S*D, low]."""

    rows, wide = h.shape
    d = wide // streams
    dev = h.device.index or 0
    if dev not in _SMS:
        _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    tiles = triton.cdiv(rows, BM) * (d // UP_DB)
    return _upmix[(min(_SMS[dev], tiles),)](
        _desc(act, [BM, BK]), _desc(up_w, [UP_DB, BK]), _desc(h, [BM, UP_DB]), _desc(mixed, [BM, UP_DB]), pss, scale,
        rows, eps, D=d, S=streams, LOW=act.shape[1], NC=pss.shape[1], DB=UP_DB, NS=UP_STAGES, num_warps=8,
        maxnreg=168)
