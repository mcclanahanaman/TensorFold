"""Flash Next's routed-expert glue on Jetson Thor (sm_110), with TensorFold's bits (THOR-EXPERT-GLUE).

On the Thor experts (``tensorfold.cuda.nvfp4.thor_experts``) a prompt pass's MoE layer runs the router, the top-k and
the plan, then ``tiles`` and ``gather`` (each row of x to its pairs' permuted rows), FlashInfer's two W4A16 grouped
GEMMs (the second's outputs in the permuted rows) and ``scatter`` (each permuted row to its (row, slot) place of
y [R, slots, D]); the shared expert follows (cuBLAS gate/up, ``MoE4.shared_act``'s SwiGLU as ten torch elementwise
kernels, a copy into its act slot, cuBLAS down, a copy into y's last slot). The next write-back
(``glue._hc_writeback``, mode 2) then sums y's slots, weighted, in fp32 in slot order. At 4,096 rows a pass the scatter
reads and writes the 40,960 routed rows (420 MB, 1.83 ms a call) and the shared expert's elementwise kernels and copies
take 0.89 ms [hcf-s2]. Here, for prompt passes on one GPU:

- no scatter: the write-back (``_writeback_rows``) reads each routed slot from its permuted row, through the pairs'
  rows the plan's tiles already store for the gather (``Scratch.pos``), and the shared slot from the shared expert's
  own output;
- the shared expert's SwiGLU is one kernel (``_swiglu``) writing its act slot, and its down projection writes a
  contiguous [R, D] block (the first rows of ``buf.y``, which the routed slots no longer use) instead of y's strided
  slot: the same cuBLAS call, with an output.

Exactness. The write-back sums the same values in the same order with ``glue._hc_writeback``'s mode-2 expressions and
layout (256 dims a program, 2 warps: the streams' square sums keep their order); ``_swiglu`` is ``shared_act``'s
expression operation by operation (fp32 gate and up, the exp and the IEEE division torch's kernels compute, the same
bf16 roundings). Every stored value is TensorFold's, bit for bit. ``prepare`` checks both kernels on this device
against the kernels they replace before turning the path on (the SwiGLU on every bf16 gate value);
tools/thor/glue_check.py checks the layer on real prompts.

``TENSORFOLD_THOR_MOE_GLUE=0`` keeps the scatter and the shared expert's kernels; decode windows (and the MTP head's
fp32 slots) always keep them; any other GPU never reaches this module.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENV = "TENSORFOLD_THOR_MOE_GLUE"
CAPABILITY = (11, 0)
WB_BLOCK, WB_WARPS = 256, 2     # the write-back's dims a program and warps (glue.hc_writeback's: pss's sum order)
ACT_BLOCK = 128                 # _swiglu: act columns a program


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110 and not switched off: decided once."""

    return not _off() and torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY


class _State:
    ready = False           # ``prepare``'s check passed on this device
    active = False          # prompt passes take this path (``prepare`` turns it on)
    note = ""               # what ``prepare`` found


S = _State()


def set_active(on: bool) -> None:
    """This path (True) or the scatter and the shared expert's kernels (False), once prepared: for A/B runs and tests."""

    S.active = bool(on) and S.ready


@triton.jit
def _writeback_rows(H, HOUT, PSS, INJ, YP, POS, YS, WTS,
                    D: tl.constexpr, S: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
    """``glue._hc_writeback`` mode 2 with routed slot k of row r read from its permuted row YP[POS[r SLOTS + k]] and
    the shared slot from YS[r]: the same fp32 sum in slot order, rounded once, the same stream updates and sums."""

    r = tl.program_id(0)
    c = tl.program_id(1)
    NC: tl.constexpr = D // BLOCK
    d = c * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK + 1):
        wk = tl.load(WTS + r * SLOTS + k)
        if k < TOPK:
            p = tl.load(POS + r * SLOTS + k).to(tl.int64)
            yk = tl.load(YP + p * D + d).to(tl.float32)
        else:
            yk = tl.load(YS + r.to(tl.int64) * D + d).to(tl.float32)
        acc = acc + yk * wk
    branch = acc.to(tl.bfloat16).to(tl.float32)
    for s in tl.static_range(S):
        hv = tl.load(H + r * (S * D) + s * D + d).to(tl.float32)
        inj = tl.load(INJ + r * S + s).to(tl.float32)
        hv = (hv + (branch * inj).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
        tl.store(HOUT + r * (S * D) + s * D + d, hv.to(tl.bfloat16))
        tl.store(PSS + (r * NC + c) * S + s, tl.sum(hv * hv, axis=0))


@triton.jit
def _swiglu(G, g_stride, OUT, out_stride, W: tl.constexpr, BLOCK: tl.constexpr):
    """Program (r, j): act columns [j BLOCK, (j + 1) BLOCK) of row r from G's bf16 gate (its first W columns) and up
    (the next W): bf16(bf16(gate / (1 + exp(-gate))) x up) in fp32, ``MoE4.shared_act``'s operations in its order."""

    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < W
    gate = tl.load(G + r * g_stride + c, mask=ok, other=0.0).to(tl.float32)
    up = tl.load(G + r * g_stride + W + c, mask=ok, other=0.0).to(tl.float32)
    silu = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate))
    act = (silu.to(tl.bfloat16).to(tl.float32) * up).to(tl.bfloat16)
    tl.store(OUT + r * out_stride + c, act, mask=ok)


def swiglu(g: torch.Tensor, width: int, out: torch.Tensor) -> torch.Tensor:
    """g [R, 2 width] bf16 (gate columns, then up) -> out [R, width] bf16 (any row stride): ``shared_act``'s bits."""

    rows = g.shape[0]
    if g.dtype != torch.bfloat16 or g.shape[1] != 2 * width or g.stride(1) != 1:
        raise ValueError(f"swiglu: g {tuple(g.shape)} {g.dtype} is not [R, {2 * width}] bf16 with unit column stride")
    if out.dtype != torch.bfloat16 or tuple(out.shape) != (rows, width) or out.stride(1) != 1:
        raise ValueError(f"swiglu: out {tuple(out.shape)} {out.dtype} is not [{rows}, {width}] bf16")
    if rows:
        _swiglu[(rows, triton.cdiv(width, ACT_BLOCK))](g, g.stride(0), out, out.stride(0), W=width, BLOCK=ACT_BLOCK,
                                                       num_warps=1)
    return out


def swiglu_ref(g: torch.Tensor, width: int) -> torch.Tensor:
    """``MoE4.shared_act``'s expression after its gate/up projection, as the r28 path runs it (torch kernels)."""

    gate, up = g[:, :width].to(torch.float32), g[:, width:].to(torch.float32)
    return ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32) * up).to(torch.bfloat16)


def writeback(h: torch.Tensor, hout: torch.Tensor, pss: torch.Tensor, streams: int, rows_view: "Rows",
              inject: torch.Tensor, wts: torch.Tensor) -> None:
    """``glue.hc_writeback(h, hout, pss, streams, 2, inject=inject, y=<the slots>, wts=wts)`` on slots kept as
    ``rows_view``: the routed ones in the experts' permuted rows, the shared one in its own [R, D]."""

    rows, wide = h.shape
    d = wide // streams
    slots = wts.shape[1]
    if slots != rows_view.slots or rows_view.shared.shape[0] != rows or d % WB_BLOCK:
        raise ValueError(f"thor_glue.writeback: {rows} rows of {slots} slots do not match the kept rows "
                         f"({rows_view.shared.shape[0]} rows of {rows_view.slots})")
    if not rows:
        return
    pos = rows_view.pos[rows_view.first * slots:]
    _writeback_rows[(rows, d // WB_BLOCK)](h, hout, pss, inject, rows_view.hidden, pos, rows_view.shared, wts,
                                           D=d, S=streams, TOPK=slots - 1, SLOTS=slots, BLOCK=WB_BLOCK,
                                           num_warps=WB_WARPS)


@dataclass(frozen=True)
class Rows:
    """A prompt pass's MoE slots, kept for the write-back: the routed outputs in the experts' permuted rows
    (``hidden``; pair row x slots + slot sits in row ``pos[pair]``) and the shared expert's [R, D] (``shared``).
    ``first``: the pass row of row 0 (a row slice keeps the pairs' numbering). Valid until the buffers' next MoE call."""

    hidden: torch.Tensor
    pos: torch.Tensor
    shared: torch.Tensor
    slots: int
    first: int = 0

    def __getitem__(self, rows: slice) -> "Rows":
        start, stop, step = rows.indices(self.shared.shape[0])
        if step != 1:
            raise ValueError("kept MoE slots slice by contiguous rows only")
        return Rows(self.hidden, self.pos, self.shared[start:stop], self.slots, self.first + start)

    def writeback(self, h, hout, pss, streams: int, mode: int, inject, wts) -> None:
        if mode != 2 or inject is None or wts is None:
            raise ValueError(f"kept MoE slots are written back in mode 2 with inject gates and weights (mode {mode})")
        writeback(h, hout, pss, streams, self, inject, wts)


def check(rows: int, d: int, streams: int, slots: int, width: int, device, seed: int = 6007) -> dict:
    """Both kernels against the kernels they replace, on random data shaped like the model's: the write-back's bytes
    (streams and square sums) against ``glue.hc_writeback`` mode 2 on the scattered slots; the SwiGLU's against
    ``swiglu_ref`` with every bf16 bit pattern as a gate (NaN where both are NaN)."""

    from . import glue

    gen = torch.Generator(device=device).manual_seed(seed + rows)
    top = slots - 1
    h = (torch.randn((rows, streams * d), generator=gen, device=device) * 1.5).to(torch.bfloat16)
    inj = (torch.rand((rows, streams), generator=gen, device=device) * 2.0).to(torch.bfloat16)
    wts = torch.softmax(torch.randn((rows, slots), generator=gen, device=device), dim=-1)
    pairs = rows * top
    extra = 3 * slots                                   # unused permuted rows, as a plan's tiles leave
    hidden = torch.randn((pairs + extra, d), generator=gen, device=device).to(torch.bfloat16)
    perm = torch.randperm(pairs + extra, generator=gen, device=device)[:pairs].to(torch.int32)
    pos = torch.zeros((rows * slots,), dtype=torch.int32, device=device)
    ids = (torch.arange(rows, device=device)[:, None] * slots + torch.arange(top, device=device)[None, :]).reshape(-1)
    pos[ids] = perm
    shared = (torch.randn((rows, d), generator=gen, device=device) * 0.3).to(torch.bfloat16)
    y = torch.empty((rows, slots, d), dtype=torch.bfloat16, device=device)
    y[:, :top] = hidden[pos.view(rows, slots)[:, :top].long()]
    y[:, top] = shared
    h0, h1 = h.clone(), h.clone()
    pss0 = torch.empty((rows, d // WB_BLOCK, streams), dtype=torch.float32, device=device)
    pss1 = torch.full_like(pss0, float("nan"))
    glue.hc_writeback(h0, h0, pss0, streams, 2, inject=inj, y=y, wts=wts)
    Rows(hidden, pos, shared, slots).writeback(h1, h1, pss1, streams, 2, inj, wts)
    # a row slice of kept slots writes back the rows it names (a mixed pass's split)
    half = rows // 2
    h2 = h.clone()
    pss2 = torch.full_like(pss0, float("nan"))
    kept = Rows(hidden, pos, shared, slots)
    kept[:half].writeback(h2[:half], h2[:half], pss2[:half], streams, 2, inj[:half], wts[:half])
    kept[half:].writeback(h2[half:], h2[half:], pss2[half:], streams, 2, inj[half:], wts[half:])

    def same(a, b):
        return bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8)))

    bits = torch.arange(-32768, 32768, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
    grows = -(-bits.numel() // width)
    gate = torch.zeros(grows * width, dtype=torch.bfloat16, device=device)
    gate[:bits.numel()] = bits
    up = (torch.randn((grows, width), generator=gen, device=device) * 2.0).to(torch.bfloat16)
    g = torch.cat([gate.view(grows, width), up], dim=1).contiguous()
    a0 = swiglu_ref(g, width)
    a1 = swiglu(g, width, torch.empty((grows, width), dtype=torch.bfloat16, device=device))
    nan0, nan1 = torch.isnan(a0), torch.isnan(a1)
    act_ok = bool(torch.equal(nan0, nan1)) and same(a0[~nan0], a1[~nan1])
    # the strided act slot the engine writes
    act_slot = torch.zeros((grows, slots, width), dtype=torch.bfloat16, device=device)
    swiglu(g, width, act_slot[:, top])
    out = {"rows": rows, "streams": same(h0, h1), "pss": same(pss0, pss1),
           "split": same(h0, h2) and same(pss0, pss2), "act": act_ok,
           "act_slot": same(act_slot[:, top][~nan0], a0[~nan0]) and bool(torch.isnan(act_slot[:, top][nan0]).all())}
    if not act_ok:
        diff = (a0.view(torch.int16) != a1.view(torch.int16)) & ~(nan0 & nan1)
        out["act_mismatch"] = int(diff.sum())
        out["act_mismatch_gates"] = [float(v) for v in g[:, :width][diff][:8].float().tolist()]
    return out


def prepare(w) -> bool:
    """Compile and check the kernels on this device (``check`` at 128 and 1,000 rows of the model's shapes): the path
    turns on only if every byte matches. The engine calls it after the caches' room is set, before its warm-up."""

    S.ready = S.active = False
    if not available():
        return False
    from . import nvfp4_moe

    layers = nvfp4_moe.thor_layers(w)
    if not layers:
        S.note = "no routed experts on the Thor path"
        return False
    if getattr(w, "comm", None) is not None:
        S.note = "more than one GPU: the slots are gathered across ranks"
        return False
    moe = layers[0].moe.experts
    c = w.cfg
    device = w.inv_freq.device
    try:
        results = [check(rows, moe.dims, c.streams, int(c.top_k) + 1, moe.width, device) for rows in (128, 1000)]
    except Exception as exc:                # noqa: BLE001  (any failure keeps the scatter and the torch kernels)
        S.note = f"its kernels failed ({type(exc).__name__}: {exc})"
        return False
    finally:
        torch.cuda.empty_cache()            # the check's buffers do not stay reserved
    keys = ("streams", "pss", "split", "act", "act_slot")
    if not all(r[k] for r in results for k in keys):
        S.note = f"bytes differ from the scatter path: {results}"
        return False
    S.ready = S.active = True
    S.note = "no scatter (routed slots read from the experts' rows), the shared SwiGLU in one kernel"
    return True
