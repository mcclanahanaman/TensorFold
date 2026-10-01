"""Flash Next's DeltaNet prompt chain on Jetson Thor (sm_110): FlashInfer's chunked gated delta rule (THOR-K4).

A prompt pass's DeltaNet recurrence runs as FlashInfer's chunked kernel for Blackwell
(``flashinfer.gdn_kernels.chunk_gated_delta_rule_sm100``: CuTe DSL, tcgen05 MMAs; FlashInfer's public entry dispatches
on compute capability 9, 10 and 12, so it is called directly) instead of TensorFold's chain of one fp32 step a row.
Each 64-row chunk is solved as tensor-core GEMMs on fp16 q, k and v (the front rounds its fp32 q and k once to fp16;
its bf16 v is exact in fp16) with fp32 sums, the state stays fp32 from chunk to chunk and between calls, and y comes
back in fp16 for the back (the gated RMSNorm), which reads it as it reads TensorFold's bf16 y. The gate is passed
linear (the kernel takes its log) and the scale is 1 (the front scales q). Decode keeps TensorFold's kernels.

Exactness: a row's bits depend on where its 64-row chunk starts. A chain split at a multiple of 64 rows from the
stream's first row keeps every bit (the state crosses a call as it crosses a chunk, in fp32), so while this path is on
a kept point below LIMIT is the last multiple of 64 before the prompt's end (none under 65 tokens), and a pass ends a
prompt piece below LIMIT on such a row unless the piece ends the prompt (``kept_point``, ``piece_rows``); past LIMIT
TensorFold's chain keeps its bits at any split, so kept points and pieces there are TensorFold's own. Prompts
therefore resume, run concurrently and run with or without drafts with the same bits.

Drift: the chunked kernel's state carries fp16-operand rounding from chunk to chunk, and in long-memory heads it grows
with the prompt. On real weights over a 128k prompt (tools/thor/gdn_accuracy.py, every DeltaNet layer against fp64),
the worst layer's decode read-out from the state reached 1.27x TensorFold's error at 32k rows, 1.37x at 48k, 1.60x at
56k and 2.4x at 128k, and its prompt read-outs 1.25x at 64k. So the chunked kernel takes a prompt's rows below LIMIT
and TensorFold's fp32 chain the rest (``chunked_rows``); LIMIT is on the 64-row grid, so both splits keep every bit.

``TENSORFOLD_THOR_GDN=0`` keeps TensorFold's chain; any other GPU never reaches this module.
"""

from __future__ import annotations

import functools
import os

import torch

ENV = "TENSORFOLD_THOR_GDN"
CAPABILITY = (11, 0)
ALIGN = 64                  # the kernel's chunk: a chain split on this grid keeps every bit
LIMIT = 32768               # a prompt's rows from here on run TensorFold's chain (the fp16 state's drift, see above)
ROWS_PER_BLOCK = 8          # the DeltaNet front and back on this path: 8 rows a block (the same values, fewer blocks)


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


def _fi():
    from flashinfer.gdn_kernels import chunk_gated_delta_rule_sm100

    if chunk_gated_delta_rule_sm100 is None:
        raise RuntimeError("FlashInfer's SM100 GDN prefill kernel is unavailable in this install")
    return chunk_gated_delta_rule_sm100


@functools.lru_cache(maxsize=None)
def available(device: int = 0) -> bool:
    """sm_110, FlashInfer's chunked GDN kernel importable and not switched off: decided once."""

    if _off() or not torch.cuda.is_available() or torch.cuda.get_device_capability(device) != CAPABILITY:
        return False
    try:
        _fi()
    except Exception as exc:                                 # noqa: BLE001  (any failure keeps TensorFold's chain)
        print(f"[tensorfold] Thor DeltaNet prompt chain off: FlashInfer's chunked kernel is not usable here "
              f"({exc!r}); TensorFold's chain runs", flush=True)
        return False
    return True


class _State:
    active = False          # prompt chains take this path (``prepare`` turns it on)
    ready = False           # the kernel compiled for this process's shapes
    dtype = torch.float16   # q, k, v and y (bf16 gave 2.2x TensorFold's read-out error in tools/thor/gdn_kernel_bench)


S = _State()


def use() -> bool:
    """Whether prompt chains run here."""

    return S.active


def set_active(on: bool) -> None:
    """This path (True) or TensorFold's chain (False), once prepared: for A/B runs and tests."""

    S.active = bool(on) and S.ready


def prepare(nk: int, nv: int, device=None) -> bool:
    """Compile (or load FlashInfer's cached build of) the kernel for nk key and nv value heads, then turn the path on:
    at load, so no request compiles it. A failure keeps TensorFold's chain (False)."""

    try:
        dev = torch.device("cuda") if device is None else torch.device(device)
        rows = 2 * ALIGN + 3
        q = torch.zeros((rows, nk, 128), dtype=S.dtype, device=dev)
        v = torch.zeros((rows, nv, 128), dtype=S.dtype, device=dev)
        g = torch.ones((rows, nv), dtype=torch.float32, device=dev)
        state = torch.zeros((nv, 128, 128), dtype=torch.float32, device=dev)
        chain(q, q, v, g, torch.zeros_like(g), state, torch.empty_like(state))
        torch.cuda.synchronize(dev)
    except Exception as exc:                                 # noqa: BLE001  (any failure keeps TensorFold's chain)
        print(f"[tensorfold] Thor DeltaNet prompt chain off: FlashInfer's chunked kernel failed to build or run "
              f"({exc!r}); TensorFold's chain runs", flush=True)
        S.ready = S.active = False
        return False
    S.ready = True
    S.active = True
    return True


def chain(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
          state: torch.Tensor, final: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """TensorFold's ``gdn.chain`` on this path: q, k [W, Hk, 128] and v [W, Hv, 128] in one 16-bit type, g (linear) and
    beta fp32 [W, Hv], from ``state`` (fp32 [Hv, 128, 128], read only) to ``final``; returns y [W, Hv, 128] in q's
    type (``out`` if given)."""

    rows, hv = v.shape[0], v.shape[1]
    y = torch.empty((rows, hv, 128), dtype=q.dtype, device=q.device) if out is None else out
    if rows == 0:
        final.copy_(state)
        return y
    _fi()(q, k, v, g, beta, y, _bounds(rows, q.device), state.unsqueeze(0), final.unsqueeze(0), 1.0)
    return y


@functools.lru_cache(maxsize=256)
def _bounds(rows: int, device) -> torch.Tensor:
    """The kernel's cu_seqlens for one sequence of ``rows`` rows, kept on the device."""

    return torch.tensor([0, rows], dtype=torch.int32, device=device)


def chunked_rows(pos: int, n: int) -> int:
    """How many of a prompt piece's ``n`` rows, from prompt row ``pos``, run on the chunked kernel (those below
    LIMIT)."""

    return max(0, min(n, LIMIT - pos))


def kept_point(n: int) -> int | None:
    """Where a prompt of ``n`` tokens keeps its state on this path: one token early (``entry_end``'s point) when that
    row is at or past LIMIT, where TensorFold's chain keeps its bits at any split; below it, the last multiple of ALIGN
    before that point, or None under ALIGN + 1 tokens."""

    if n - 1 >= LIMIT:
        return n - 1
    k = (n - 1) // ALIGN * ALIGN
    return k if k > 0 else None


def stops(points, lo: int, end: int | None) -> list[int]:
    """A prompt's message-start keep points on this path (TensorFold 0.6.1 keeps states at message starts): each below
    LIMIT moved back to the last multiple of ALIGN at or before it, so the prompt's pieces and kept states stay on the
    grid; past LIMIT as given. Those in [lo, end), once each; none when the prompt keeps nothing (``end`` None)."""

    if end is None:
        return []
    return sorted({q for p in points if lo <= (q := p if p >= LIMIT else p // ALIGN * ALIGN) < end})


def piece_rows(start: int, n: int, total: int) -> int:
    """A pass's rows of a prompt piece from ``start``: ``n`` if it ends the prompt or reaches LIMIT (the chunked rows
    then end on the grid), else cut back to the grid (0: the piece waits for the next pass)."""

    end = start + n
    if end >= total or end >= LIMIT:
        return n
    return max(0, end // ALIGN * ALIGN - start)
