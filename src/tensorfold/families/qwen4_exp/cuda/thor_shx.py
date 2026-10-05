"""The shared expert's elementwise tail on Jetson Thor (sm_110), one kernel, TensorFold's bits.

A decode window's MoE (bf16 slots, so the shared expert's MXFP8 projections take cuBLAS) runs, after the routed
experts' scatter and before ``glue._hc_writeback``, thirteen torch elementwise kernels [par1-s6b]:

1-2. ``g[:, :width].to(float32)`` and ``g[:, width:].to(float32)`` (``nvfp4_moe.MoE4.shared_act``)
3-6. ``-gate``, ``exp``, ``1 + exp``, ``gate / (1 + exp)`` in fp32, that order
7-10. ``.to(bfloat16)``, ``.to(float32)``, ``* up``, ``.to(bfloat16)``
11. the store into ``buf.act[:rows, top_k]`` (``nvfp4_moe.moe``)
12. ``thor_dense.linear``'s copy of that slot into the PAD-row buffer, the down GEMM's input
13. the store of the down GEMM into ``buf.y[:rows, top_k]``

The two cuBLAS GEMMs stay. Kernels 1-12 become one launch: the SwiGLU ``thor_glue._swiglu`` already checks bit for bit
(fp32 gate and up, libdevice's exp, IEEE division, the same two bf16 roundings), writing the act slot and the same
PAD-row prefix the copy wrote, and leaving every other PAD row untouched. Kernel 13 stays the assignment into y.
The down GEMM is ``thor_dense.linear``'s ``torch.mm`` on that buffer. No new buffer: the PAD rows and the act slot
already exist, and the GEMM outputs are the ones the graph already captured.

``TENSORFOLD_THOR_SHX_FUSE=0`` keeps today's kernels. Unset is on for sm_110. MTP buffers (fp32 slots, the lane
matmul), a window of PAD rows or more, and ``--prefill-fp8`` (those prompts stay on ``prefill8``) keep today's
kernels too. ``stats()["calls"]`` counts tails that ran: eager launches, and each replay of a graph that captured
them (the capture itself runs nothing).
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorfold.cuda import prompt_precision
from tensorfold.cuda.nvfp4 import thor_dense

ENV = "TENSORFOLD_THOR_SHX_FUSE"
CAPABILITY = (11, 0)
BLOCK = 128


def _off() -> bool:
    return os.environ.get(ENV, "1").strip().lower() in ("0", "off", "false", "no")


@functools.lru_cache(maxsize=None)
def _sm110(device: int = 0) -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) == CAPABILITY


def active(device: int = 0) -> bool:
    """Unset on sm_110; ``0`` keeps today's elementwise kernels. Read on each call (tests flip it)."""

    return _sm110(device) and not _off()


class _State:
    calls = 0       # fused tails that ran: eager launches, plus replays (``add_replays``)
    replays = 0     # of those, the ones a CUDA-graph replay ran
    pending = 0     # ``store`` calls recorded in the capture underway (a capture runs nothing)


S = _State()


def stats() -> dict:
    """``calls``: fused tails that ran (eager launches and graph replays). ``replays``: the replayed ones."""

    return {"on": active(), "calls": S.calls, "replays": S.replays}


def begin_capture() -> None:
    """A CUDA-graph capture is starting. ``store`` during it is recorded, not an execution."""

    S.pending = 0


def take_capture() -> int:
    """How many fused tails the capture that just ended recorded (0 if it recorded none)."""

    n = S.pending
    S.pending = 0
    return n


def add_replays(n: int) -> None:
    """``n`` fused tails ran because a graph that captured them replayed."""

    S.replays += n
    S.calls += n


def want(ex, x: torch.Tensor, buf, rows: int, top_k: int, prompt: bool) -> bool:
    """The thirteen-kernel tail: bf16 slots, both projections on the cuBLAS path, fewer than PAD rows.

    ``--prefill-fp8`` is refused here, before any other guard: those runs keep torch's thirteen kernels and
    ``Mx8Linear.prefill8``. The fused down GEMM is cuBLAS bf16, which does not match that FP8 prompt GEMM.
    """

    if prompt_precision.fp8():                  # --prefill-fp8: today's prefill8 path. This is the only new check.
        return False
    if not prompt or rows <= 0 or rows >= thor_dense.PAD or not active(x.device.index or 0):
        return False
    gu, down = ex.shared.gu, ex.shared.down
    width = ex.width
    if not (thor_dense.use(gu) and thor_dense.use(down)):
        return False
    if getattr(gu, "n", None) != 2 * width or getattr(down, "k", None) != width:
        return False
    pad = thor_dense.S.pad
    if pad is None or pad.numel() < thor_dense.PAD * width or down.n != ex.dims:
        return False
    slot = buf.act[:rows, top_k]
    y = buf.y[:rows, top_k]
    return (slot.dtype == torch.bfloat16 and slot.shape == (rows, width) and slot.stride(-1) == 1
            and y.dtype == torch.bfloat16 and y.shape == (rows, ex.dims))


@triton.jit
def _tail(G, g_stride, PAD, pad_stride, ACT, act_stride, W: tl.constexpr, BLOCK: tl.constexpr):
    """Row r, columns [j BLOCK, (j + 1) BLOCK): ``shared_act``'s SwiGLU, stored to the PAD row and the act slot."""

    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = c < W
    gate = tl.load(G + r * g_stride + c, mask=ok, other=0.0).to(tl.float32)
    up = tl.load(G + r * g_stride + W + c, mask=ok, other=0.0).to(tl.float32)
    silu = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate))
    act = (silu.to(tl.bfloat16).to(tl.float32) * up).to(tl.bfloat16)
    tl.store(PAD + r * pad_stride + c, act, mask=ok)
    tl.store(ACT + r * act_stride + c, act, mask=ok)


def store(g: torch.Tensor, width: int, act_slot: torch.Tensor, rows: int) -> None:
    """``g`` [rows, 2 width] bf16 -> the SwiGLU in ``act_slot`` [rows, width] and in the down GEMM's PAD rows."""

    if g.dtype != torch.bfloat16 or g.shape != (rows, 2 * width) or g.stride(1) != 1:
        raise ValueError(f"shx store: g {tuple(g.shape)} {g.dtype} is not [{rows}, {2 * width}] bf16")
    pad = thor_dense.S.pad[:thor_dense.PAD * width].view(thor_dense.PAD, width)
    _tail[(rows, triton.cdiv(width, BLOCK))](g, g.stride(0), pad, pad.stride(0), act_slot, act_slot.stride(0),
                                              W=width, BLOCK=BLOCK, num_warps=1)
    if torch.cuda.is_current_stream_capturing():
        S.pending += 1                          # recorded; the kernel runs when the graph replays
    else:
        S.calls += 1


def project(lin, rows: int) -> torch.Tensor:
    """``thor_dense.linear``'s padded cuBLAS down GEMM. The first ``rows`` of the PAD buffer are the SwiGLU."""

    k = lin.k
    pad = thor_dense.S.pad[:thor_dense.PAD * k].view(thor_dense.PAD, k)
    return torch.mm(pad, thor_dense._weight(lin).t())[:rows]
