"""Flash Next's sparse-attention prompt kernels on Jetson Thor (sm_110) (THOR-K4-QSA).

A prompt pass's attention layers run TensorFold's QSA in 256-row blocks: ``qsa_rows`` scores every complete 4-key
block for each sparse row (the indexer: 4 heads of relu(q . k) summed, one program a row and 64 blocks, the dot
products on CUDA cores) and selects each row's top 512; ``attention`` then reads each row's 2,048 selected keys, one
program a row and KV head. On Thor the 8-bit cache makes the attention kernel spend about a third of its time
dequantizing keys and values that every row reads again, and the scores kernel runs at a tenth of the tensor cores'
rate. Here, for prompt passes:

- the layer's 8-bit keys and values are dequantized once into a bf16 shadow (the very values the attention kernel
  dequantizes in registers, ``kvquant.dequant_group_8``): a segment of several row blocks stages its whole context
  (``stage_all``), a shorter one, such as a follow-up's rows at a long context, only the 4-key blocks its rows read
  (``stage``); TensorFold's own attention kernels then read the shadow (their bf16-cache path, two pipeline stages)
  and rotate the output back as for the 8-bit cache;
- the indexer's block scores are tensor-core dot products (``_scores``: 32 rows x 4 heads against 128 pooled keys a
  program, fp32 sums), and TensorFold's own selection (``attention._launch_select``) picks the blocks.

Exactness. Only sparse rows (a row past ``budget + ratio - 1`` keys, from prompt position ``first_sparse``) take this
path; a dense row keeps TensorFold's kernels, also inside a row block that holds both (the block is split at the
first sparse row), so a prompt under 2,052 tokens, and the first 2,051 rows of any prompt, have TensorFold's bits
exactly, with or without this path. (The same kernel compiled for a bf16 operand and two pipeline stages sums the same
products in another order: on dense rows that gave rounding-level differences that flipped near-tie replies of short
prompts. Keeping them costs 2.0% of the 8k prompt rate, 0.5% at 32k and 0.1% at 128k on Thor.) Every row is still
computed on its own: whether it is sparse, its scores, its selection and its attention depend only on the row's
position, its query and the cache (the shadow holds the same values however it was staged), so prompts resume, fill
together and fill with or without drafts with the same bits, at any split; decode windows keep TensorFold's kernels.
On sparse rows the values are TensorFold's up to rounding: the same products summed in another order.
tools/thor/qsa_check.py measures both paths against fp64 on real prompts.

THOR-K4-QSA-ATTN: a sparse row's attention is one program a row and KV head over its whole key list (``_row_attn``:
the online softmax of TensorFold's ``attention._tile`` over the list's 64-key tiles in order, no chunk partials and no
merge), and its selection is TensorFold's search with 32-bit keys and fewer warps a row (``_select_rows``: the same cut
and the same lists, bit for bit). Both depend only on the row, so the contract above holds.
``TENSORFOLD_THOR_QSA_ATTN=0`` keeps THOR-K4-QSA's kernels (chunks, merge and TensorFold's selection).

THOR-K4-QSA-SEL: the selection is a CUDA radix select (``thor_select.cu``: one CTA a row, the row's keys in registers,
the TOP-th largest key in three digit steps, then the blocks placed in block order by warp ballots), TensorFold's cut
and lists bit for bit, with or without THOR-K4-QSA-ATTN. ``TENSORFOLD_THOR_QSA_SEL=0`` keeps the Triton selection
(``_select_rows`` with THOR-K4-QSA-ATTN, else TensorFold's).

``TENSORFOLD_THOR_QSA=0`` keeps TensorFold's prompt kernels; any other GPU never reaches this module.
"""

from __future__ import annotations

import functools
import os
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

from . import attention as attn_mod
from .attention import _tile
from .kvquant import dequant_group_8, h32

ENV = "TENSORFOLD_THOR_QSA"
ENV_ATTN = "TENSORFOLD_THOR_QSA_ATTN"
ENV_SEL = "TENSORFOLD_THOR_QSA_SEL"
CAPABILITY = (11, 0)
STAGES = 2                  # the attention chunk kernel's pipeline depth on the shadow (bf16 loads)
SCORE_ROWS, SCORE_BLOCKS = 32, 128
ROW_TILE, ROW_WARPS, ROW_STAGES = 64, 4, 2      # _row_attn: keys a tile, warps, pipeline depth (k4qa-r1 sweep)
SELECT_WARPS = {16384: 4, 32768: 8}    # _select_rows' warps by row width (blocks, a power of 2; 4 below; k4qa-r1)
SEL_VARIANT = 0             # thor_select.cu's launch variant (0: the served kernels; others in tuning builds)


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110 and not switched off: decided once."""

    return not _off() and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY


class _State:
    active = False          # prompt passes take this path (``prepare`` turns it on)
    shadow = False          # the cache is 8-bit: prompt buffers hold a bf16 shadow of one layer's keys and values
    attn = os.environ.get(ENV_ATTN, "1").strip().lower() not in ("0", "off", "false", "no")   # THOR-K4-QSA-ATTN
    sel = os.environ.get(ENV_SEL, "1").strip().lower() not in ("0", "off", "false", "no")     # THOR-K4-QSA-SEL


S = _State()


def use() -> bool:
    """Whether prompt passes run here."""

    return S.active


DENSE_ON_TF = True          # dense rows keep TensorFold's kernels (False: every prompt row here, as first gated)


def first_sparse(scratch) -> int | None:
    """The first prompt position whose row is sparse (it reads more than ``budget + ratio - 1`` keys, so more than
    ``budget / ratio`` complete blocks), or None with QSA off (a cache within the budget): 2,051 for Flash Next."""

    return (scratch.budget // scratch.ratio + 1) * scratch.ratio - 1 if scratch.qsa else None


def rows_from(scratch, k: int) -> SimpleNamespace:
    """The prompt block's scratch as seen by a launch whose row 0 is the block's row ``k`` (its selections, key counts
    and partial sums from there on)."""

    return SimpleNamespace(nch=scratch.nch, qsa=scratch.qsa, budget=scratch.budget, ratio=scratch.ratio,
                           idw=scratch.idw, po=scratch.po[k:], pm=scratch.pm[k:], pl=scratch.pl[k:],
                           ids=scratch.ids[k:], nk=scratch.nk[k:], sparse=scratch.sparse[k:])


def set_active(on: bool) -> None:
    """This path (True) or TensorFold's prompt kernels (False), once prepared: for A/B runs and tests."""

    S.active = bool(on) and available()


def set_attn(on: bool) -> None:
    """THOR-K4-QSA-ATTN's kernels (True) or THOR-K4-QSA's (False): for A/B runs and tests."""

    S.attn = bool(on)


def set_sel(on: bool) -> None:
    """THOR-K4-QSA-SEL's CUDA selection (True) or the Triton one (False): for A/B runs and tests."""

    S.sel = bool(on)


@functools.lru_cache(maxsize=1)
def _sel_ext():
    from pathlib import Path

    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_qwen4_exp_thor_select", sources=[str(here / "thor_select.cpp"),
                                                                    str(here / "thor_select.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def shadow_bytes(capacity: int, kv_heads: int, head_dim: int) -> int:
    """A bf16 shadow of one layer's keys and values at ``capacity`` cache rows."""

    return 2 * capacity * kv_heads * head_dim * 2


def admission(geometry, kv_bits: int):
    """The engine's geometry plus the prompt buffers' shadow (an 8-bit cache only): one layer's keys and values in
    bf16 at the cache capacity, the prompt passes' workspace."""

    from tensorfold.cuda.capacity import Geometry

    if kv_bits != 8:
        return geometry

    def with_shadow(text: dict):
        g = geometry(text)
        hd = int(text.get("head_dim") or int(text["hidden_size"]) // int(text["num_attention_heads"]))
        hk = int(text["num_key_value_heads"])
        return Geometry(lambda slots: g.bytes_at(slots) + shadow_bytes(slots, hk, hd), g.reserve, g.minimum_slots)

    return with_shadow


def prepare(kv_bits: int) -> bool:
    """Turn the path on for this engine's cache width (8-bit: through the shadow; bf16: the cache itself); a 4-bit
    cache keeps TensorFold's kernels. Before the buffers are built: prompt buffers then hold the shadow."""

    S.shadow = kv_bits == 8
    S.active = available() and kv_bits in (8, 16)
    if S.active and S.sel:                  # built now, before the caches
        try:
            _sel_ext()
        except Exception as exc:            # noqa: BLE001  (any failure keeps the Triton selection)
            print(f"[tensorfold] Thor QSA CUDA selection off: its build failed ({exc!r}); the Triton selection runs",
                  flush=True)
            S.sel = False
    return S.active


class Shadow:
    """One layer's keys and values as bf16 [capacity, kv heads, head dim], staged for a prompt segment block by block:
    after each block of rows is selected, ``stage`` dequantizes the 4-key blocks those rows read that the segment has
    not staged yet (``want`` and ``held``: bitmaps of blocks), so a short pass at a long context stages only what its
    rows read."""

    def __init__(self, capacity: int, kv_heads: int, head_dim: int, device, ratio: int = 4) -> None:
        self.k = torch.empty((capacity, kv_heads, head_dim), dtype=torch.bfloat16, device=device)
        self.v = torch.empty_like(self.k)
        self.ratio = ratio
        words = -(-capacity // ratio) // 32 + 2
        self.want = torch.zeros((words,), dtype=torch.int32, device=device)    # blocks the current rows read
        self.held = torch.zeros_like(self.want)                                 # blocks staged this segment


def attach(w, capacity: int) -> Shadow | None:
    """Prompt buffers' shadow: None off this path or for a bf16 cache."""

    if not S.active or not S.shadow:
        return None
    c = w.cfg
    return Shadow(capacity, c.kv_heads, c.head_dim, w.device, c.index_ratio)


# --------------------------------------------------------------------------------------------- the 8-bit shadow

@triton.jit
def _want(IDS, NKR, SPR, POS0, WANT, RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr,
          DW: tl.constexpr):
    """The 4-key blocks row r reads (``attention._chunk``'s keys): a sparse row's TOP selected blocks and its
    unfinished tail, a dense row's every block to its position."""

    r = tl.program_id(0)
    end = tl.load(POS0) + r + 1
    complete = end // RATIO
    sparse = False
    if QSA:
        sparse = tl.load(SPR + r) != 0
    if sparse:
        t = tl.arange(0, TOP)
        blk = tl.load(IDS + r * IDW + t * RATIO) // RATIO
        tl.atomic_or(WANT + (blk >> 5), (1 << (blk & 31)).to(tl.int32), sem="relaxed")
        if end > complete * RATIO:
            tl.atomic_or(WANT + (complete >> 5), (1 << (complete & 31)).to(tl.int32), sem="relaxed")
    else:
        n = (end - 1) // RATIO + 1                      # blocks [0, n): every key to the row's own
        w = tl.arange(0, DW)
        part = (n & 31).to(tl.int32)
        val = tl.where(w < (n >> 5), -1, tl.where(w == (n >> 5), (1 << part) - 1, 0)).to(tl.int32)
        tl.atomic_or(WANT + w, val, mask=w <= (n >> 5), sem="relaxed")


@triton.jit(do_not_specialize=["NROWS"])
def _stage(WANT, HELD, KC, KS, VC, VS, SK, SV, NROWS, HK: tl.constexpr, D: tl.constexpr, RATIO: tl.constexpr):
    """Word w's wanted, not yet held blocks of 8-bit keys and values (KV head hk) -> bf16 in the shadow, still rotated:
    the values ``attention._chunk`` dequantizes in registers (a block's keys past the cache's NROWS rows are left)."""

    w = tl.program_id(0)
    hk = tl.program_id(1)
    word = tl.load(WANT + w) & ~tl.load(HELD + w)
    if word != 0:
        b = tl.arange(0, 32)
        on = ((word >> b) & 1) != 0
        keys = tl.reshape((w * 32 + b)[:, None] * RATIO + tl.arange(0, RATIO)[None, :], (32 * RATIO,))
        ok = tl.reshape(tl.broadcast_to(on[:, None], (32, RATIO)), (32 * RATIO,)) & (keys < NROWS)
        d = tl.arange(0, D)
        gs = tl.arange(0, D // 32)
        row = keys[:, None].to(tl.int64) * HK + hk
        sc = tl.load(KS + row * (D // 32) + gs[None, :], mask=ok[:, None], other=0.0)
        code = tl.load(KC + row * D + d[None, :], mask=ok[:, None], other=0)
        tl.store(SK + row * D + d[None, :], dequant_group_8(code, sc, M=32 * RATIO, W=D), mask=ok[:, None])
        sc = tl.load(VS + row * (D // 32) + gs[None, :], mask=ok[:, None], other=0.0)
        code = tl.load(VC + row * D + d[None, :], mask=ok[:, None], other=0)
        tl.store(SV + row * D + d[None, :], dequant_group_8(code, sc, M=32 * RATIO, W=D), mask=ok[:, None])


@triton.jit(do_not_specialize=["n"])
def _dequant(KC, KS, OUT, n, HK: tl.constexpr, D: tl.constexpr, BK: tl.constexpr):
    """Cache rows [0, n) of 8-bit codes and fp16 scales -> bf16 [n, HK, D], still rotated (as ``_stage``)."""

    i = tl.program_id(0) * BK + tl.arange(0, BK)
    hk = tl.program_id(1)
    ok = i < n
    d = tl.arange(0, D)
    gs = tl.arange(0, D // 32)
    row = i[:, None].to(tl.int64) * HK + hk
    code = tl.load(KC + row * D + d[None, :], mask=ok[:, None], other=0)
    sc = tl.load(KS + row * (D // 32) + gs[None, :], mask=ok[:, None], other=0.0)
    tl.store(OUT + row * D + d[None, :], dequant_group_8(code, sc, M=BK, W=D), mask=ok[:, None])


def stage_all(cache, shadow: Shadow, context: int) -> None:
    """The whole of an 8-bit ``cache``'s rows [0, context) into the shadow: for a segment of several row blocks, whose
    rows between them read nearly every block, one pass is cheaper than block by block."""

    hk, d = cache.k.shape[1], cache.k.shape[2]
    if context > 0:
        grid = (triton.cdiv(context, 32), hk)
        _dequant[grid](cache.k, cache.ks, shadow.k, context, HK=hk, D=d, BK=32, num_warps=4)
        _dequant[grid](cache.v, cache.vs, shadow.v, context, HK=hk, D=d, BK=32, num_warps=4)


def begin(shadow: Shadow) -> None:
    """A new prompt segment (a layer's or a stream's cache): nothing staged."""

    shadow.held.zero_()


def stage(cache, shadow: Shadow, scratch, pos0: torch.Tensor, rows: int, context: int) -> None:
    """Into the shadow, the blocks of an 8-bit ``cache`` that rows P0 .. P0 + rows - 1 read (their selections in
    ``scratch``) and the segment has not staged yet; ``context`` (P0 + rows) bounds the blocks."""

    ratio = shadow.ratio
    top = scratch.budget // ratio
    _want[(rows,)](scratch.ids, scratch.nk, scratch.sparse, pos0, shadow.want, RATIO=ratio, TOP=top,
                   IDW=scratch.idw, QSA=scratch.qsa, DW=triton.next_power_of_2(top // 32 + 2), num_warps=4)
    words = min(shadow.want.shape[0], -(-context // ratio) // 32 + 1)
    hk, d = cache.k.shape[1], cache.k.shape[2]
    _stage[(words, hk)](shadow.want, shadow.held, cache.k, cache.ks, cache.v, cache.vs, shadow.k, shadow.v,
                        min(cache.k.shape[0], shadow.k.shape[0]), HK=hk, D=d, RATIO=ratio, num_warps=4)
    shadow.held.bitwise_or_(shadow.want)
    shadow.want.zero_()


# ------------------------------------------------------------------------------------------- the indexer's scores

@triton.jit(do_not_specialize=["R"])
def _scores(IQ, POOLED, POS0, SC, NB, R, HI: tl.constexpr, DI: tl.constexpr, RATIO: tl.constexpr,
            TOP: tl.constexpr, TR: tl.constexpr, BB: tl.constexpr):
    """``attention._scores`` on tensor cores: TR rows x HI indexer heads against BB pooled keys a program; each sparse
    row's score of each complete block: the fp32 sum over heads of relu(q . k), over sqrt(DI)."""

    t = tl.program_id(0)
    j = tl.program_id(1)
    p0 = tl.load(POS0)
    last = (p0 + tl.minimum(t * TR + TR, R)) // RATIO           # the tile's last row's complete blocks
    if (last > TOP) & (j * BB < last):
        m = tl.arange(0, TR * HI)
        r = t * TR + m // HI
        h = m % HI
        d = tl.arange(0, DI)
        q = tl.load(IQ + (r[:, None] * HI + h[:, None]) * DI + d[None, :], mask=(r < R)[:, None], other=0.0)
        b = j * BB + tl.arange(0, BB)
        k = tl.load(POOLED + b[:, None].to(tl.int64) * DI + d[None, :], mask=(b < last)[:, None], other=0.0)
        s = tl.maximum(tl.dot(q, tl.trans(k)), 0.0)
        tot = tl.sum(tl.reshape(s, (TR, HI, BB)), axis=1) / tl.sqrt(DI * 1.0)
        rr = t * TR + tl.arange(0, TR)
        complete = (p0 + rr + 1) // RATIO
        ok = (rr < R)[:, None] & (complete > TOP)[:, None] & (b[None, :] < complete[:, None])
        tl.store(SC + rr[:, None].to(tl.int64) * NB + b[None, :], tot, mask=ok)


@triton.jit
def _select_rows(SC, POS0, IDS, NKR, SPR, NB, RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr,
                 BLOCK: tl.constexpr):
    """``attention._select`` with 32-bit keys (its search finds the same cut: the largest t with at least TOP keys
    >= t), so a row needs half the registers and an SM holds several rows: the same lists, bit for bit."""

    r = tl.program_id(0)
    end = tl.load(POS0) + r + 1
    complete = end // RATIO
    if complete <= TOP:
        tl.store(NKR + r, end)
        tl.store(SPR + r, 0)
    else:
        b = tl.arange(0, BLOCK)
        ok = b < complete
        v = tl.load(SC + r.to(tl.int64) * NB + b, mask=ok, other=0.0)
        bits = v.to(tl.uint32, bitcast=True)
        key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
        key = tl.where(ok, key, 0)
        lo = tl.zeros((), dtype=tl.uint32)
        hi = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
        for _ in range(32):                             # 2^32 candidates halve to one
            span = hi - lo
            mid = lo + span // 2 + (span & 1)           # ceil((lo + hi) / 2) without overflow
            count = tl.sum(tl.where(key >= mid, 1, 0), axis=0)
            take = count >= TOP
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        cut = lo
        above = ok & (key > cut)
        equal = ok & (key == cut)
        need = TOP - tl.sum(above.to(tl.int32), axis=0)
        rank = tl.cumsum(equal.to(tl.int32), axis=0)
        chosen = above | (equal & (rank <= need))
        place = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
        for k in tl.static_range(RATIO):
            tl.store(IDS + r * IDW + place * RATIO + k, b * RATIO + k, mask=chosen)
        t = tl.arange(0, RATIO)
        tail = RATIO * complete + t
        tl.store(IDS + r * IDW + TOP * RATIO + t, tail, mask=tail < end)
        tl.store(NKR + r, TOP * RATIO + end - RATIO * complete)
        tl.store(SPR + r, 1)


def select(scratch, pos0: torch.Tensor, rows: int, blocks: int, warps: int | None = None,
           variant: int | None = None) -> None:
    """``attention._launch_select``'s lists: THOR-K4-QSA-SEL's CUDA radix select (``warps`` None), else
    ``_select_rows`` up to 32,768 blocks a row and TensorFold's tiled selection past them."""

    if S.sel and warps is None:
        _sel_ext().select(scratch.scores, pos0, scratch.ids, scratch.nk, scratch.sparse, int(rows), int(blocks),
                          scratch.ratio, scratch.budget // scratch.ratio, SEL_VARIANT if variant is None else variant)
        return
    width = max(16, triton.next_power_of_2(blocks))
    if width > attn_mod.SELECT_REGS:
        attn_mod._launch_select(scratch, pos0, rows, blocks)
        return
    _select_rows[(rows,)](scratch.scores, pos0, scratch.ids, scratch.nk, scratch.sparse, scratch.nb,
                          RATIO=scratch.ratio, TOP=scratch.budget // scratch.ratio, IDW=scratch.idw, BLOCK=width,
                          num_warps=warps or SELECT_WARPS.get(width, 4))


def qsa_rows(iq: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, scratch, rows: int, *,
             context: int | None = None) -> None:
    """``attention.qsa_rows`` with tensor-core scores: the blocks of rows P0 .. P0 + rows - 1, selected as
    TensorFold selects them (``select``; THOR-K4-QSA: TensorFold's own selection)."""

    ratio = scratch.ratio
    blocks = scratch.nb if context is None else min(scratch.nb, max(1, triton.cdiv(context, ratio)))
    _scores[(triton.cdiv(rows, SCORE_ROWS), triton.cdiv(blocks, SCORE_BLOCKS))](
        iq, pooled, pos0, scratch.scores, scratch.nb, rows, HI=iq.shape[1], DI=pooled.shape[1], RATIO=ratio,
        TOP=scratch.budget // ratio, TR=SCORE_ROWS, BB=SCORE_BLOCKS, num_warps=8)
    if S.sel or S.attn:
        select(scratch, pos0, rows, blocks)
    else:
        attn_mod._launch_select(scratch, pos0, rows, blocks)


# ----------------------------------------------------------------------------------------------------- attention

@triton.jit
def _row_attn(Q, KC, VC, POS0, IDS, NKR, SPR, OUT, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
              G: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr,
              BT: tl.constexpr):
    """Row r's attention for KV head hk over its whole key list (a sparse row's through IDS) in one pass: the online
    softmax of ``attention._tile`` over the list's BT-key tiles in order, then o / l, rotated back for an 8-bit
    cache's shadow. Only the row's own keys, in its own order, enter its sums."""

    r = tl.program_id(0)
    hk = tl.program_id(1)
    n = tl.load(POS0) + r + 1
    sparse = False
    if QSA:
        sparse = tl.load(SPR + r) != 0
        n = tl.where(sparse, tl.load(NKR + r), n)
    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    q = tl.load(Q + (r * H + hk * G + gg[:, None]) * D + d[None, :], mask=gg[:, None] < G, other=0.0)
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for t in range(0, tl.cdiv(n, BT)):
        ki = t * BT + tl.arange(0, BT)
        valid = ki < n
        if QSA:
            if sparse:
                ki = tl.load(IDS + r * IDW + ki, mask=valid, other=0)
        off = (ki[:, None].to(tl.int64) * HK + hk) * D + d[None, :]
        kk = tl.load(KC + off, mask=valid[:, None], other=0.0)
        vv = tl.load(VC + off, mask=valid[:, None], other=0.0)
        m, l, o = _tile(q, kk, vv, m, l, o, valid, SCALE)
    result = o / l[:, None]
    if BITS:
        result = tl.reshape(h32(tl.reshape(result, (16 * D // 32, 32)), M=16 * D // 32), (16, D))
    tl.store(OUT + (r * H + hk * G + gg[:, None]) * D + d[None, :], result.to(tl.bfloat16), mask=gg[:, None] < G)


def row_attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, pos0: torch.Tensor, scratch, rows: int,
                  scale: float, out: torch.Tensor, *, bits: int, tile: int | None = None, warps: int | None = None,
                  stages: int | None = None) -> torch.Tensor:
    """``_row_attn`` for rows P0 .. P0 + rows - 1 on bf16 keys and values (the shadow or a bf16 cache)."""

    _, h, d = q.shape
    hk = kc.shape[1]
    _row_attn[(rows, hk)](q, kc, vc, pos0, scratch.ids, scratch.nk, scratch.sparse, out, H=h, HK=hk, D=d, G=h // hk,
                          SCALE=scale, IDW=scratch.idw, QSA=scratch.qsa, BITS=bits, BT=tile or ROW_TILE,
                          num_warps=warps or ROW_WARPS, num_stages=stages or ROW_STAGES)
    return out


def attention(q: torch.Tensor, cache, shadow: Shadow | None, pos0: torch.Tensor, scratch, rows: int, scale: float,
              out: torch.Tensor, *, context: int, bits: int) -> torch.Tensor:
    """``attention.attention`` for a prompt block on the bf16 shadow (an 8-bit cache) or the bf16 cache:
    ``row_attention`` (THOR-K4-QSA: TensorFold's chunk kernel, two pipeline stages, then its merge, which rotates
    the output back for the 8-bit cache)."""

    _, h, d = q.shape
    hk = cache.k.shape[1]
    kc, vc = (shadow.k, shadow.v) if bits else (cache.k, cache.v)
    if S.attn:
        return row_attention(q, kc, vc, pos0, scratch, rows, scale, out, bits=bits)
    nch = scratch.nch
    keys = context
    if scratch.qsa:
        keys = min(keys, (scratch.budget // scratch.ratio + 1) * scratch.ratio - 1)
    chunks = min(nch, triton.cdiv(keys, attn_mod.CHUNK))
    attn_mod._chunks[(rows, hk, chunks)](q, kc, vc, kc, vc, pos0, scratch.po, scratch.pm, scratch.pl, scratch.ids,
                                         scratch.nk, scratch.sparse, H=h, HK=hk, D=d, G=h // hk, CH=attn_mod.CHUNK,
                                         NCH=nch, SCALE=scale, IDW=scratch.idw, QSA=scratch.qsa, BITS=0,
                                         num_warps=4, num_stages=STAGES)
    attn_mod._merge[(rows, hk)](scratch.po, scratch.pm, scratch.pl, pos0, out, scratch.nk, scratch.sparse, H=h,
                                HK=hk, D=d, G=h // hk, CH=attn_mod.CHUNK, NCH=nch, QSA=scratch.qsa, BITS=bits,
                                num_warps=4)
    return out
