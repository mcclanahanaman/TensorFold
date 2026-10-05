"""Flash Next's NVFP4 MoE: routed experts on the grouped NVFP4 kernel, the bf16 shared expert on the FP4 tables.

On Jetson Thor (sm_110) the routed experts run FlashInfer's W4A16 grouped GEMM instead (``thor_experts``): laid out
once at load, for prompt chunks and decode windows alike (both keep bf16 slots); TENSORFOLD_THOR_FI_MOE=0 keeps the
portable kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4 import thor_experts as thor

from . import nvfp4, thor_glue, thor_shx


@dataclass
class Expert4:
    """One expert: gate/up rows in one grid (gate rows first), and the down grid."""

    gu: nvfp4.FP4         # [2*NI, K]
    down: nvfp4.FP4       # [K, NI]


@dataclass
class MoE4:
    """One layer's experts: the routed ones as grouped NVFP4 blocks, the shared one as bf16 tables."""

    routed_experts: nvx.Experts4 | thor.ThorExperts
    shared: Expert4
    kernel: str = "nvfp4"

    capturable = True     # the plan stays on the device

    @property
    def routed(self) -> int:
        return self.routed_experts.count

    @property
    def count(self) -> int:
        return self.routed + 1

    @property
    def width(self) -> int:
        return self.routed_experts.width                     # NI: a gate row's inputs, down's outputs

    @property
    def dims(self) -> int:
        return self.routed_experts.dims                      # D: x's inputs, down's outputs

    def nbytes(self) -> int:
        return self.routed_experts.nbytes() + self.shared.gu.nbytes() + self.shared.down.nbytes()

    def shared_act(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        """x [R, D] bf16 -> the shared expert's SwiGLU [R, NI] bf16, bf16(bf16(silu(g)) * u) on bf16 g and u."""

        g = _shared(self.shared.gu, x, prompt, False)
        gate, up = g[:, :self.width].to(torch.float32), g[:, self.width:].to(torch.float32)
        return ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32) * up).to(torch.bfloat16)


def routed_dense(routed, e: int, which: str) -> torch.Tensor:
    """Expert ``e``'s gate, up or down weight [N, K] fp32 from either layout (a reference)."""

    return (thor.dense if isinstance(routed, thor.ThorExperts) else nvx.dense)(routed, e, which)


def _shared(lin, x: torch.Tensor, prompt: bool, f32: bool, out: torch.Tensor | None = None) -> torch.Tensor:
    """The shared expert's matmul: bf16 weights on the FP4 kernel's exact tables, MXFP8 ones on the lane matmul."""

    if isinstance(lin, nvfp4.FP4):
        return nvfp4.matmul(x, lin, f32=f32, out=out)
    return lin.prefill(x, out) if prompt else lin(x, out)


def expert4_from_bf16(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> Expert4:
    """The shared expert from bf16 weights (torch linear layout [out, in]): identity-scaled FP4 tables."""

    gu = nvfp4.fp4_from_bf16(torch.cat([gate, up], dim=0).contiguous())
    return Expert4(gu, nvfp4.fp4_from_bf16(down.contiguous()))


_NOTED: set = set()


def moe4_from_checkpoint(gate: tuple, up: tuple, down: tuple,
                         shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """A layer from stacked checkpoint arrays (words, e4m3 scales, per-expert scales) and the bf16 shared expert."""

    shared = shared if isinstance(shared, Expert4) else expert4_from_bf16(*shared)
    if thor.available(torch.cuda.current_device()):
        try:
            routed = thor.make(gate, up, down)
            if "on" not in _NOTED:
                _NOTED.add("on")
                print(f"[tensorfold] Thor (sm_110): routed NVFP4 experts on FlashInfer's W4A16 grouped GEMM "
                      f"({thor.ENV}=0 keeps the portable kernel)", flush=True)
            return MoE4(routed, shared)
        except ValueError as exc:                  # a layer the Thor layout cannot hold exactly keeps the portable one
            if str(exc) not in _NOTED:
                _NOTED.add(str(exc))
                print(f"[tensorfold] Thor NVFP4 experts: a layer keeps the portable kernel: {exc}", flush=True)
    return MoE4(nvx.make(gate, up, down), shared)


def moe4_from_bf16(gate_up: torch.Tensor, down: torch.Tensor,
                   shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """The MTP layer's bf16 experts (gate_up [E, 2NI, K], down [E, K, NI]) in NVFP4: they only draft."""

    ni = gate_up.shape[1] // 2
    gate, up = nvx.quantize(gate_up[:, :ni]), nvx.quantize(gate_up[:, ni:])
    return MoE4(nvx.make(gate, up, nvx.quantize(down)),
                shared if isinstance(shared, Expert4) else expert4_from_bf16(*shared))


def moe4_from_experts(gate: list, up: list, down: list, shared: tuple) -> MoE4:
    """Per-expert FP4 triples and the bf16 shared expert -> a layer (the test path)."""

    def stack(items):
        return (torch.stack([t[0] for t in items]), torch.stack([t[1].view(torch.uint8) for t in items]),
                torch.stack([torch.as_tensor(t[2], dtype=torch.float32).reshape(()) for t in items]).to(
                    items[0][0].device))

    return moe4_from_checkpoint(stack(gate), stack(up), stack(down), shared)


def moe(x: torch.Tensor, xs: torch.Tensor, router_rows: torch.Tensor, ex: MoE4, buf, cfg, *,
        unscattered: bool = False) -> None:
    """Route rows x [R, D] and run their experts into buf (the shared ``moe`` contract); ``xs`` unused: no biases.

    ``unscattered`` (one GPU): a Thor prompt pass (thor_glue) may leave the routed outputs in the experts' permuted rows
    and the shared expert's in a block of its own; ``buf.kept`` then holds them for the write-back (a ``thor_glue.Rows``,
    valid until the next call), else it is None and buf.y holds every slot."""

    from tensorfold.cuda import moe as moe_mod

    rows, top_k = x.shape[0], int(cfg.num_experts_per_tok)
    routed = ex.routed_experts
    on_thor = isinstance(routed, thor.ThorExperts)
    moe_mod.router(x, router_rows, buf.logits[:rows])
    moe_mod.select(buf.logits[:rows], buf, top_k, ex.routed, thor.tactic(rows)[0] if on_thor else nvx.PREFILL_TILE)
    prompt = buf.y.dtype != torch.float32                  # a prompt's buffers keep bf16 slots
    keep = on_thor and unscattered and prompt and buf.plan.prefill and thor_glue.S.active
    buf.kept = None
    if on_thor:
        scratch = getattr(buf, "thor", None)
        if scratch is None:
            raise RuntimeError("these buffers were built without the Thor experts' scratch (attach_thor at load)")
        thor.run(x, routed, buf.plan, buf.y, rows, top_k, scratch, scatter=not keep)
    else:
        nvx.gate_up(x, routed, buf.plan, buf.act.view(-1, ex.width), rows, skip=ex.routed)
    if keep:                     # the same SwiGLU in one kernel; the down projection into y's (unused) first rows
        thor_glue.swiglu(_shared(ex.shared.gu, x, prompt, False), ex.width, buf.act[:rows, top_k])
        shared = buf.y.view(-1, ex.dims)[:rows]
        _shared(ex.shared.down, buf.act[:rows, top_k], prompt, False, out=shared)
        buf.kept = thor_glue.Rows(scratch.hidden, scratch.pos, shared, buf.slots)
        return buf
    if thor_shx.want(ex, x, buf, rows, top_k, prompt):
        # kernels 1-12 of the elementwise tail, then the routed down (portable experts), then kernel 13
        thor_shx.store(_shared(ex.shared.gu, x, prompt, False), ex.width, buf.act[:rows, top_k], rows)
        if not on_thor:
            nvx.down(buf.act.view(-1, ex.width), routed, buf.plan, buf.y.view(-1, ex.dims), rows, skip=ex.routed)
        buf.y[:rows, top_k] = thor_shx.project(ex.shared.down, rows)
        return buf
    buf.act[:rows, top_k] = ex.shared_act(x, prompt)
    if not on_thor:
        nvx.down(buf.act.view(-1, ex.width), routed, buf.plan, buf.y.view(-1, ex.dims), rows, skip=ex.routed)
    buf.y[:rows, top_k] = _shared(ex.shared.down, buf.act[:rows, top_k], prompt, not prompt)
    return buf


def thor_layers(w) -> list:
    """The layers (the MTP head's included) whose routed experts run on the Thor path."""

    mtp = getattr(getattr(w, "mtp", None), "layer", None)
    return [layer for layer in [*w.layers, *([mtp] if mtp is not None else [])]
            if isinstance(getattr(getattr(getattr(layer, "moe", None), "experts", None), "routed_experts", None),
                          thor.ThorExperts)]


def attach_thor(w, buf, rows: int) -> None:
    """A buffer set gets the Thor experts' scratch and warms their kernels, at load, not on a request."""

    layers = thor_layers(w)
    if not layers:
        return
    ex = layers[0].moe.experts.routed_experts
    top_k = buf.slots - 1
    buf.thor = thor.Scratch(rows, top_k, ex.count, ex.width, ex.dims, buf.y.device)
    thor.warm(ex, buf.thor, top_k, prompt=buf.y.dtype == torch.bfloat16)
