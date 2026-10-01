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

Exactness. Every row is still computed on its own: its scores, its selection and its attention read only the row's
query and the cache (the shadow holds the same values however it was staged), so prompts resume, fill together and
fill with or without drafts with the same bits, at any split; decode windows keep TensorFold's kernels. The values
are TensorFold's up to rounding: the same products summed in another order. tools/thor/qsa_check.py measures both
paths against fp64 on real prompts.

``TENSORFOLD_THOR_QSA=0`` keeps TensorFold's prompt kernels; any other GPU never reaches this module.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

from . import attention as attn_mod
from .kvquant import dequant_group_8

ENV = "TENSORFOLD_THOR_QSA"
CAPABILITY = (11, 0)
STAGES = 2                  # the attention chunk kernel's pipeline depth on the shadow (bf16 loads)
SCORE_ROWS, SCORE_BLOCKS = 32, 128


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110 and not switched off: decided once."""

    return not _off() and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY


class _State:
    active = False          # prompt passes take this path (``prepare`` turns it on)
    shadow = False          # the cache is 8-bit: prompt buffers hold a bf16 shadow of one layer's keys and values


S = _State()


def use() -> bool:
    """Whether prompt passes run here."""

    return S.active


def set_active(on: bool) -> None:
    """This path (True) or TensorFold's prompt kernels (False), once prepared: for A/B runs and tests."""

    S.active = bool(on) and available()


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


def qsa_rows(iq: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, scratch, rows: int, *,
             context: int | None = None) -> None:
    """``attention.qsa_rows`` with tensor-core scores: the blocks of rows P0 .. P0 + rows - 1, selected by
    TensorFold's own selection."""

    ratio = scratch.ratio
    blocks = scratch.nb if context is None else min(scratch.nb, max(1, triton.cdiv(context, ratio)))
    _scores[(triton.cdiv(rows, SCORE_ROWS), triton.cdiv(blocks, SCORE_BLOCKS))](
        iq, pooled, pos0, scratch.scores, scratch.nb, rows, HI=iq.shape[1], DI=pooled.shape[1], RATIO=ratio,
        TOP=scratch.budget // ratio, TR=SCORE_ROWS, BB=SCORE_BLOCKS, num_warps=8)
    attn_mod._launch_select(scratch, pos0, rows, blocks)


# ----------------------------------------------------------------------------------------------------- attention

def attention(q: torch.Tensor, cache, shadow: Shadow | None, pos0: torch.Tensor, scratch, rows: int, scale: float,
              out: torch.Tensor, *, context: int, bits: int) -> torch.Tensor:
    """``attention.attention`` for a prompt block: TensorFold's chunk kernel on the bf16 shadow (an 8-bit cache) or
    the bf16 cache, two pipeline stages, then its merge, which rotates the output back for the 8-bit cache."""

    _, h, d = q.shape
    hk = cache.k.shape[1]
    kc, vc = (shadow.k, shadow.v) if bits else (cache.k, cache.v)
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
