"""Shared by the Thor MoE probes: Flash Next's routed NVFP4 experts read from the checkpoint, the fp32 reference, routing,
metrics and CUDA-event timing. Nothing here depends on a candidate kernel.

The reference dequantizes every expert with ``tensorfold.cuda.nvfp4.format.dequant`` (numpy, independent of the CUDA
packing) and runs the block in fp32: g = x W_gate^T, u = x W_up^T, h = silu(g) * u, y = h W_down^T, out = sum_k w_k y_k.
"""

from __future__ import annotations

import json
import statistics
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

E, TOP, D, NI = 512, 10, 2560, 640
PROJS = ("gate_proj", "up_proj", "down_proj")


def flops(m: int, top: int = TOP) -> float:
    """Routed-expert FLOPs of an M-token block: per (token, slot) gate+up [2*NI x D] and down [D x NI], 2 a MAC."""

    return 2.0 * m * top * (2 * NI * D + D * NI)


@dataclass
class Proj:
    w: torch.Tensor           # uint8 [E, N, K/2], low nibble the even input
    s: torch.Tensor           # float8_e4m3fn [E, N, K/16]
    s2: torch.Tensor          # fp32 [E] weight_scale_2 (dequant = code * s * s2)
    a: torch.Tensor | None    # fp32 [E] input_scale (the checkpoint's static activation scale), when stored


@dataclass
class Layer:
    index: int
    gate: Proj
    up: Proj
    down: Proj
    router: torch.Tensor      # bf16 [E, D]
    names: dict


class Checkpoint:
    def __init__(self, model: str | Path) -> None:
        from safetensors import safe_open

        self.root = Path(model)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.config = json.loads((self.root / "config.json").read_text())
        self._open = safe_open
        self._files: dict[str, object] = {}
        probe = [k for k in self.index if k.endswith("layers.0.mlp.experts.0.gate_proj.weight")]
        if not probe:
            raise SystemExit("no per-expert NVFP4 tensors (layers.0.mlp.experts.0.gate_proj.weight) in the index")
        self.prefix = probe[0][: -len("layers.0.mlp.experts.0.gate_proj.weight")]

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str) -> torch.Tensor:
        f = self.index[name]
        if f not in self._files:
            self._files[f] = self._open(str(self.root / f), framework="pt", device="cpu")
        return self._files[f].get_tensor(name)

    def layer(self, i: int, device: str = "cuda") -> Layer:
        base = f"{self.prefix}layers.{i}.mlp."

        def proj(p: str) -> Proj:
            ws = [self.get(f"{base}experts.{e}.{p}.weight") for e in range(E)]
            ss = [self.get(f"{base}experts.{e}.{p}.weight_scale") for e in range(E)]
            s2 = [self.get(f"{base}experts.{e}.{p}.weight_scale_2").float().reshape(()) for e in range(E)]
            an = f"{base}experts.0.{p}.input_scale"
            a = (torch.stack([self.get(f"{base}experts.{e}.{p}.input_scale").float().reshape(()) for e in range(E)])
                 if self.has(an) else None)
            w = torch.stack(ws)
            s = torch.stack(ss)
            if w.dtype != torch.uint8 or s.dtype != torch.float8_e4m3fn:
                raise SystemExit(f"{base}experts.*.{p}: weight {w.dtype}, scale {s.dtype}: not NVFP4 as expected")
            return Proj(w.to(device), s.to(device), torch.stack(s2).to(device), None if a is None else a.to(device))

        names = {p: f"{base}experts.<e>.{p}.(weight|weight_scale|weight_scale_2|input_scale)" for p in PROJS}
        names["router"] = f"{base}gate.weight"
        return Layer(i, proj("gate_proj"), proj("up_proj"), proj("down_proj"),
                     self.get(f"{base}gate.weight").to(torch.bfloat16).to(device), names)


def reference_weights(layer: Layer, device: str = "cuda", threads: int = 8) -> dict[str, torch.Tensor]:
    """fp32 [E, N, K] of each projection by TensorFold's numpy dequantizer (format.dequant), expert by expert."""

    from tensorfold.cuda.nvfp4 import format as fmt

    out = {}
    for name in ("gate", "up", "down"):
        p: Proj = getattr(layer, name)
        w = p.w.cpu().numpy()
        s = p.s.cpu().view(torch.uint8).numpy()
        s2 = p.s2.cpu().numpy()

        def one(e: int, w=w, s=s, s2=s2) -> np.ndarray:
            return fmt.dequant("nvfp4", w[e], s[e], float(s2[e]))

        with ThreadPoolExecutor(threads) as pool:
            mats = list(pool.map(one, range(E)))
        out[name] = torch.from_numpy(np.stack(mats)).to(device)
        del mats
    return out


def activations(m: int, seed: int, device: str = "cuda") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(m, D, generator=g) * 0.5).to(torch.bfloat16).to(device)


def route(x: torch.Tensor, router: torch.Tensor, top: int = TOP) -> tuple[torch.Tensor, torch.Tensor]:
    """The layer's own router on x: fp32 logits, top-k largest, softmax over the k (bf16-rounded, as TensorFold)."""

    logits = x.float() @ router.float().t()
    vals, ids = torch.topk(logits, top, dim=-1)
    w = torch.softmax(vals, dim=-1).to(torch.bfloat16).float()
    return ids.to(torch.int32).contiguous(), w.contiguous()


def load_stats(ids: torch.Tensor) -> dict:
    counts = torch.bincount(ids.flatten().long(), minlength=E)
    used = counts[counts > 0].float()
    return {"pairs": int(ids.numel()), "experts_used": int((counts > 0).sum()), "rows_mean_used": float(used.mean()),
            "rows_max": int(counts.max()), "rows_min_used": int(used.min()),
            "rows_p90": float(torch.quantile(used, 0.9))}


def groups(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Pairs (flattened [M * top]) sorted by expert (stable), per-expert offsets, and the experts that have pairs."""

    flat = ids.flatten().long()
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E)
    offs = torch.zeros(E + 1, dtype=torch.long, device=ids.device)
    offs[1:] = torch.cumsum(counts, 0)
    used = torch.nonzero(counts).flatten().tolist()
    return order, offs, used


@torch.no_grad()
def reference(x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, ref: dict, dtype=torch.float32,
              want_h: bool = False):
    """fp32 (or fp64) routed block; also |h| max over pairs (for a best-case static FP4 scale of the down input)."""

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    m = x.shape[0]
    xf = x.to(dtype)
    out = torch.zeros(m, D, dtype=dtype, device=x.device)
    order, offs, used = groups(ids)
    offs_h = offs.tolist()
    wf = w.flatten().to(dtype)
    hmax = 0.0
    for e in used:
        sel = order[offs_h[e]:offs_h[e + 1]]
        t = sel // ids.shape[1]
        xe = xf[t]
        g = xe @ ref["gate"][e].to(dtype).t()
        u = xe @ ref["up"][e].to(dtype).t()
        h = torch.nn.functional.silu(g) * u
        if want_h:
            hmax = max(hmax, float(h.abs().max()))
        y = h @ ref["down"][e].to(dtype).t()
        out.index_add_(0, t, y * wf[sel][:, None])
    torch.backends.cuda.matmul.allow_tf32 = prev
    return (out, hmax) if want_h else out


def metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    g, r = got.double(), ref.double()
    diff = g - r
    return {"rel_fro": float(diff.norm() / r.norm()), "max_abs": float(diff.abs().max()),
            "ref_max_abs": float(r.abs().max()),
            "cosine": float((g * r).sum() / (g.norm() * r.norm()))}


def time_cuda(fn, reps: int = 10, warmup: int = 3) -> dict:
    """CUDA-event time of fn() (a full call, host launch cost included), warm-up excluded; ms."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return {"median_ms": statistics.median(times), "min_ms": min(times), "mean_ms": statistics.fmean(times),
            "max_ms": max(times), "reps": reps}


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))
