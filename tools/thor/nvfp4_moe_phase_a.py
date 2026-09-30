"""THOR-K1-A Phase A: Flash Next's routed NVFP4 experts on Thor (sm_110), TensorFold's grouped kernel against native
candidates, for accuracy (gate G-A) against an fp32 reference built from the checkpoint's own bytes, and for speed.

    python tools/thor/nvfp4_moe_phase_a.py --model /hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/<rev> \
        --out /out/phase_a.json [--layers 0,23,47] [--acc-ms 128,512,2048,4096] [--speed-ms 512,2048,4096,8192] \
        [--paths tf,emu1,emu2,emu3,smm1,smm2,smm3,fi,fi-ckpt] [--speed-layer 23] [--reps 10]

Every path gives the routed experts' combined output [M, D] (bf16): sum over the top-10 slots of weight * expert(x),
x = randn * 0.5 in bf16, routing by the layer's own router on x (fp32 logits, softmax over the top 10, bf16 weights).

  tf       TensorFold 0.3.6.3's nvfp4_expert_kernel (W4A16: bf16 rows x e2m1 codes on bf16 mma.sync, fp32 sums,
           e4m3 scale per 16 inputs in fp32), its plan, then its Triton combine
  emuN     torch emulation of W4A4: activations (x, then h) as N NVFP4 terms (residuals), fp32 matmuls; the error an
           FP4-activation path cannot beat, whatever its kernel
  smmN     per expert, torch._scaled_mm (cuBLASLt block-scaled NVFP4 x NVFP4) on the N activation terms stacked along
           the rows (weights read once), terms summed in fp32 with their global scales
  fi       FlashInfer cutlass_fused_moe, NVFP4 weights x NVFP4 activations (W4A4), activation global scales from the
           data (x amax, reference h amax: the best case)
  fi-ckpt  the same with the checkpoint's static input_scale (max over experts, as vLLM does)
  fiw4a16  FlashInfer 0.7.0.post1's CuTe-DSL SM100 W4A16 grouped MoE (NVFP4 weights decoded to bf16 in SMEM, bf16
           tcgen05 MMA, fp32 sums): tools/thor/fi_w4a16_moe.py
  fiw4a4   FlashInfer 0.7.0.post1's CuTe-DSL NVFP4 x NVFP4 grouped MoE (W4A4; its C++ routing helpers replaced by
           torch equivalents offline): tools/thor/fi_w4a4_moe.py
  tw4a16   a Triton W4A16 probe (tools/thor/triton_w4a16_moe.py): e2m1 x e4m3 widened to bf16 per tile (exact),
           bf16 tl.dot with fp32 sums, swap-AB tiles of one expert's pairs

G-A: a path PASSes at an M when its relative Frobenius error vs the fp32 reference is <= 1.5x TensorFold's.
Speed: CUDA events around a full call (host launch cost included), 3 warm-ups excluded, --reps reps; TFLOPS counts
the routed FLOPs 2 * M * 10 * (2 * 640 * 2560 + 2560 * 640).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_common as mc  # noqa: E402
import nvfp4_quant as nq  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.set_float32_matmul_precision("highest")


def log(*a) -> None:
    print(f"[{mc.stamp()}]", *a, flush=True)


# -- TensorFold -------------------------------------------------------------------------------------------------------
class TensorFoldPath:
    def __init__(self, layer: mc.Layer) -> None:
        from tensorfold.cuda import experts as grouped
        from tensorfold.cuda import moe as moe_mod
        from tensorfold.cuda.nvfp4 import experts as nvx

        self.grouped, self.nvx, self.moe_mod = grouped, nvx, moe_mod
        p = lambda q: (q.w, q.s, q.s2)  # noqa: E731
        self.ex = nvx.make(p(layer.gate), p(layer.up), p(layer.down))
        self.bufs: dict[int, tuple] = {}

    def check_packing(self, ref: dict) -> dict:
        got = {}
        for e in (0, 257, 511):
            for which in ("gate", "up", "down"):
                got[f"{which}{e}"] = bool(torch.equal(self.nvx.dense(self.ex, e, which), ref[which][e]))
        return got

    def _buf(self, m: int):
        if m not in self.bufs:
            plan = self.grouped.Plan(m, mc.TOP + 1, mc.E + 1, "cuda", prefill=True)
            act = torch.zeros(m * (mc.TOP + 1), mc.NI, dtype=torch.bfloat16, device="cuda")
            y = torch.zeros(m, mc.TOP + 1, mc.D, dtype=torch.bfloat16, device="cuda")   # slot 10 stays 0 (shared)
            self.bufs[m] = (plan, act, y)
        return self.bufs[m]

    def prepare(self, x, ids, w):
        m = x.shape[0]
        picks = torch.cat([ids, torch.full((m, 1), mc.E, dtype=torch.int32, device="cuda")], 1).contiguous()
        wts = torch.cat([w, torch.zeros(m, 1, device="cuda")], 1).contiguous()
        return picks, wts

    def kernels(self, x, picks):
        m = x.shape[0]
        plan, act, y = self._buf(m)
        self.grouped.route(picks, plan)
        self.nvx.gate_up(x, self.ex, plan, act, m, skip=mc.E)
        self.nvx.down(act, self.ex, plan, y.view(-1, mc.D), m, skip=mc.E)
        return y

    def __call__(self, x, picks, wts):
        return self.moe_mod.combine(self.kernels(x, picks), wts)


# -- W4A4 emulation and per-expert torch._scaled_mm -------------------------------------------------------------------
def terms_of(x: torch.Tensor, n: int, gs_first: float | None = None):
    """x [R, K] -> n NVFP4 terms (packed, e4m3 scales, gs), each quantizing the residual the previous ones left."""

    out, r = [], x.float()
    for i in range(n):
        gs = gs_first if (i == 0 and gs_first is not None) else nq.global_scale(float(r.abs().max()))
        packed, sf, deq = nq.quantize(r, gs)
        out.append((packed, sf, gs))
        r = r - deq
    return out


def fake(x: torch.Tensor, n: int) -> torch.Tensor:
    r, acc = x.float(), torch.zeros_like(x, dtype=torch.float32)
    for _ in range(n):
        _, _, deq = nq.quantize(r, nq.global_scale(float(r.abs().max())))
        acc += deq
        r = r - deq
    return acc


@torch.no_grad()
def emulate(x, ids, w, ref, n: int):
    """W4A4 in fp32 arithmetic: x and every pair's h through n NVFP4 terms (per-tensor global scales), exact weights."""

    m = x.shape[0]
    xq = fake(x, n)
    order, offs, used = mc.groups(ids)
    o = offs.tolist()
    h_all = torch.empty(m * mc.TOP, mc.NI, device=x.device)
    for e in used:
        sel = order[o[e]:o[e + 1]]
        xe = xq[sel // mc.TOP]
        h_all[sel] = torch.nn.functional.silu(xe @ ref["gate"][e].t()) * (xe @ ref["up"][e].t())
    hq = fake(h_all.to(torch.bfloat16), n)          # the intermediate is bf16 before the down input's FP4
    out = torch.zeros(m, mc.D, device=x.device)
    wf = w.flatten()
    for e in used:
        sel = order[o[e]:o[e + 1]]
        out.index_add_(0, sel // mc.TOP, (hq[sel] @ ref["down"][e].t()) * wf[sel][:, None])
    return out.to(torch.bfloat16)


class ScaledMMPath:
    """Per expert: _scaled_mm(NVFP4 activations (n terms stacked), NVFP4 weights) for gate+up and for down."""

    def __init__(self, layer: mc.Layer, terms: int) -> None:
        self.n = terms
        self.w1 = torch.cat([layer.gate.w, layer.up.w], 1).contiguous()                  # [E, 2NI, D/2]
        self.s1 = nq.swizzle(torch.cat([layer.gate.s, layer.up.s], 1).contiguous())      # [E, 2NI * D/16]
        self.a1 = torch.cat([layer.gate.s2[:, None].expand(mc.E, mc.NI),
                             layer.up.s2[:, None].expand(mc.E, mc.NI)], 1).contiguous()  # [E, 2NI]
        self.w2 = layer.down.w.contiguous()                                              # [E, D, NI/2]
        self.s2 = nq.swizzle(layer.down.s.contiguous())
        self.a2 = layer.down.s2
        self.out_dtype = torch.float32
        self.fp4 = torch.float4_e2m1fn_x2

    def _mm(self, a_packed, a_sf, w, w_sf):
        a = a_packed.view(self.fp4)
        b = w.view(self.fp4).t()
        try:
            return torch._scaled_mm(a, b, nq.swizzle(a_sf), w_sf, out_dtype=self.out_dtype)
        except (RuntimeError, ValueError):
            if self.out_dtype != torch.float32:
                raise
            self.out_dtype = torch.bfloat16
            log("  _scaled_mm: fp32 out refused, using bf16 out")
            return torch._scaled_mm(a, b, nq.swizzle(a_sf), w_sf, out_dtype=self.out_dtype)

    @torch.no_grad()
    def __call__(self, x, ids, w):
        m = x.shape[0]
        order, offs, used = mc.groups(ids)
        o = offs.tolist()
        xt = terms_of(x, self.n)
        h = torch.empty(m * mc.TOP, mc.NI, dtype=torch.bfloat16, device="cuda")
        for e in used:
            sel = order[o[e]:o[e + 1]]
            t = sel // mc.TOP
            me = t.numel()
            a = torch.cat([p[t] for p, _, _ in xt]) if self.n > 1 else xt[0][0][t]
            s = torch.cat([f[t] for _, f, _ in xt]) if self.n > 1 else xt[0][1][t]
            r = self._mm(a, s, self.w1[e], self.s1[e]).float().view(self.n, me, 2 * mc.NI)
            gu = r[0] * (1.0 / xt[0][2])
            for i in range(1, self.n):
                gu = gu + r[i] * (1.0 / xt[i][2])
            gu = gu * self.a1[e]
            h[sel] = (torch.nn.functional.silu(gu[:, :mc.NI]) * gu[:, mc.NI:]).to(torch.bfloat16)
        ht = terms_of(h, self.n)
        y = torch.empty(m * mc.TOP, mc.D, dtype=torch.float32, device="cuda")
        for e in used:
            sel = order[o[e]:o[e + 1]]
            me = sel.numel()
            a = torch.cat([p[sel] for p, _, _ in ht]) if self.n > 1 else ht[0][0][sel]
            s = torch.cat([f[sel] for _, f, _ in ht]) if self.n > 1 else ht[0][1][sel]
            r = self._mm(a, s, self.w2[e], self.s2[e]).float().view(self.n, me, mc.D)
            acc = r[0] * (1.0 / ht[0][2])
            for i in range(1, self.n):
                acc = acc + r[i] * (1.0 / ht[i][2])
            y[sel] = acc * self.a2[e]
        return (y.view(m, mc.TOP, mc.D) * w[..., None]).sum(1).to(torch.bfloat16)


# -- FlashInfer -------------------------------------------------------------------------------------------------------
class FlashInferPath:
    def __init__(self, layer: mc.Layer) -> None:
        from flashinfer.fused_moe import cutlass_fused_moe

        self.fn = cutlass_fused_moe
        same = torch.equal(layer.gate.s2, layer.up.s2)
        self.shared_gu_scale = bool(same)
        # FlashInfer (TRT-LLM) wants fc1 = [up; gate] (act on the second half) and one fc1 global scale an expert
        self.fc1 = torch.cat([layer.up.w, layer.gate.w], 1).contiguous().view(torch.long)
        self.fc1_sf = nq.swizzle(torch.cat([layer.up.s, layer.gate.s], 1).contiguous()).view(
            mc.E, 2 * mc.NI, mc.D // 16).view(torch.int32)
        self.fc2 = layer.down.w.contiguous().view(torch.long)
        self.fc2_sf = nq.swizzle(layer.down.s.contiguous()).view(mc.E, mc.D, mc.NI // 16).view(torch.int32)
        self.gs1_w = layer.gate.s2 if same else torch.maximum(layer.gate.s2, layer.up.s2)
        self.gs2_w = layer.down.s2
        self.ckpt_a = (layer.gate.a, layer.down.a)
        self.scales = None

    def set_scales(self, a1_gs: float, a2_gs: float) -> None:
        a1 = torch.tensor(a1_gs, dtype=torch.float32, device="cuda")       # 0-d: one scale for every expert
        a2 = torch.tensor(a2_gs, dtype=torch.float32, device="cuda")
        self.scales = [a1, self.fc1_sf, (self.gs1_w / a1_gs).float().contiguous(), a2, self.fc2_sf,
                       (self.gs2_w / a2_gs).float().contiguous()]

    def __call__(self, x, ids, w):
        r = self.fn(x, ids, w, self.fc1, self.fc2, torch.bfloat16, quant_scales=self.scales)
        return r[0] if isinstance(r, (list, tuple)) else r


# -- driver -----------------------------------------------------------------------------------------------------------
def gate_row(results: dict, ms, paths) -> None:
    for p in paths:
        if p == "tf":
            continue
        row = []
        for m in ms:
            a, t = results.get((p, m)), results.get(("tf", m))
            if a is None or t is None:
                row.append(f"{'n/a':>22}")
                continue
            ratio = a["rel_fro"] / t["rel_fro"]
            row.append(f"{a['rel_fro']:.3e} x{ratio:6.2f} {'PASS' if ratio <= 1.5 else 'FAIL'}")
        print(f"  {p:8s} " + " | ".join(row), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/out/phase_a.json")
    ap.add_argument("--layers", default="0,23,47")
    ap.add_argument("--acc-ms", default="128,512,2048,4096")
    ap.add_argument("--speed-ms", default="512,2048,4096,8192")
    ap.add_argument("--speed-layer", type=int, default=23)
    ap.add_argument("--paths", default="tf,emu1,emu2,emu3,smm1,smm2,smm3,fi,fi-ckpt")
    ap.add_argument("--speed-paths", default="tf,smm1,smm2,fi")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--no-speed", action="store_true")
    args = ap.parse_args()

    paths = args.paths.split(",")
    layers = [int(v) for v in args.layers.split(",")] if args.layers else []
    acc_ms = [int(v) for v in args.acc_ms.split(",")]
    speed_ms = [int(v) for v in args.speed_ms.split(",")]
    ck = mc.Checkpoint(args.model)
    dev = torch.cuda.get_device_properties(0)
    report = {"device": dev.name, "cc": [dev.major, dev.minor], "sms": dev.multi_processor_count,
              "torch": torch.__version__, "cuda": torch.version.cuda, "prefix": ck.prefix, "accuracy": {},
              "speed": {}, "errors": {}, "args": vars(args)}
    try:
        import flashinfer

        report["flashinfer"] = getattr(flashinfer, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        report["flashinfer"] = f"import failed: {exc!r}"
    log("device", dev.name, (dev.major, dev.minor), "torch", torch.__version__, "prefix", repr(ck.prefix))

    def save() -> None:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))

    def attempt(key: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - a candidate that fails is a finding, not a stop
            report["errors"][key] = f"{exc!r}\n{traceback.format_exc()[-3000:]}"
            log(f"  {key}: FAILED {exc!r}")
            save()
            return None

    run_layers = [] if args.no_accuracy else layers
    if not args.no_speed and args.speed_layer not in run_layers:
        run_layers.append(args.speed_layer)
    for li in run_layers:
        t0 = time.time()
        layer = ck.layer(li)
        info = {"gate_up_scale2_equal": bool(torch.equal(layer.gate.s2, layer.up.s2)),
                "gate_s2_range": [float(layer.gate.s2.min()), float(layer.gate.s2.max())],
                "up_s2_range": [float(layer.up.s2.min()), float(layer.up.s2.max())],
                "down_s2_range": [float(layer.down.s2.min()), float(layer.down.s2.max())],
                "input_scale": {k: (None if a is None else [float(a.min()), float(a.max())])
                                for k, a in (("gate", layer.gate.a), ("up", layer.up.a), ("down", layer.down.a))},
                "names": layer.names}
        log(f"layer {li} loaded in {time.time() - t0:.1f}s", json.dumps({k: v for k, v in info.items()
                                                                          if k != 'names'}))
        do_acc = (not args.no_accuracy) and li in layers
        ref = None
        if do_acc:
            t0 = time.time()
            ref = mc.reference_weights(layer)
            log(f"  fp32 reference weights (format.dequant) in {time.time() - t0:.1f}s")
        cands: dict[str, object] = {}
        if "tf" in paths or "tf" in args.speed_paths.split(","):
            cands["tf"] = attempt(f"L{li}.tf.build", lambda: TensorFoldPath(layer))
            if ref is not None and cands["tf"] is not None:
                info["tf_packing_exact"] = cands["tf"].check_packing(ref)
                log("  TensorFold packing equals format.dequant:", info["tf_packing_exact"])
        if "fiw4a16" in paths + args.speed_paths.split(","):
            def make_fiw4a16():
                import fi_w4a16_moe as fw

                info["fiw4a16_smem"] = fw.patch_smem()
                return fw.FlashInferW4A16(layer)

            cands["fiw4a16"] = attempt(f"L{li}.fiw4a16.build", make_fiw4a16)
            if cands["fiw4a16"] is not None:
                info["fiw4a16_shared_gu_scale2_exact"] = cands["fiw4a16"].shared_gu_scale
        if "fiw4a4" in paths + args.speed_paths.split(","):
            def make_fiw4a4():
                import fi_w4a16_moe as fw
                import fi_w4a4_moe as f4

                info.setdefault("fiw4a16_smem", fw.patch_smem())
                return f4.FlashInferW4A4(layer)

            cands["fiw4a4"] = attempt(f"L{li}.fiw4a4.build", make_fiw4a4)
        if "tw4a16" in paths + args.speed_paths.split(","):
            def make_triton():
                from triton_w4a16_moe import TritonW4A16

                return TritonW4A16(layer)

            cands["tw4a16"] = attempt(f"L{li}.tw4a16.build", make_triton)
        for n in (1, 2, 3):
            if f"smm{n}" in paths or f"smm{n}" in args.speed_paths.split(","):
                cands[f"smm{n}"] = attempt(f"L{li}.smm{n}.build", lambda n=n: ScaledMMPath(layer, n))
        if any(p in ("fi", "fi-ckpt") for p in paths + args.speed_paths.split(",")):
            cands["fi"] = attempt(f"L{li}.fi.build", lambda: FlashInferPath(layer))
            if cands["fi"] is not None:
                info["fi_fc1_shared_scale2_exact"] = cands["fi"].shared_gu_scale
        report["accuracy"][f"layer{li}"] = {"info": info}
        save()

        if do_acc:
            res: dict = {}
            for m in acc_ms:
                x = mc.activations(m, seed=1000 + m)
                ids, w = mc.route(x, layer.router)
                t0 = time.time()
                refo, hmax = mc.reference(x, ids, w, ref, want_h=True)
                entry = {"load": mc.load_stats(ids), "h_amax": hmax, "x_amax": float(x.abs().max())}
                if m == acc_ms[0]:
                    r64 = mc.reference(x, ids, w, ref, dtype=torch.float64)
                    entry["ref_fp32_vs_fp64"] = mc.metrics(refo, r64)
                    del r64
                log(f"  M={m}: reference in {time.time() - t0:.1f}s, load {entry['load']}")
                for p in paths:
                    got = None
                    if p == "tf" and cands.get("tf") is not None:
                        picks, wts = cands["tf"].prepare(x, ids, w)
                        got = attempt(f"L{li}.M{m}.tf", lambda: cands["tf"](x, picks, wts))
                    elif p.startswith("emu"):
                        got = attempt(f"L{li}.M{m}.{p}", lambda: emulate(x, ids, w, ref, int(p[3:])))
                    elif p == "fiw4a16" and cands.get(p) is not None:
                        c = cands[p]
                        c.tactic = c.fw.DEFAULT_W4A16_MOE_TACTIC
                        got = attempt(f"L{li}.M{m}.{p}", lambda: c(x, ids, w))
                        entry["fiw4a16_mode"] = c.used
                        entry["fiw4a16_tactic"] = str(c.tactic)
                        if getattr(c, "launch_error", None):
                            entry["fiw4a16_launch_error"] = c.launch_error
                    elif p == "fiw4a4" and cands.get(p) is not None:
                        got = attempt(f"L{li}.M{m}.{p}", lambda: cands[p](x, ids, w, h_amax=hmax))
                    elif p == "tw4a16" and cands.get(p) is not None:
                        got = attempt(f"L{li}.M{m}.{p}", lambda: cands[p](x, ids, w))
                    elif p.startswith("smm") and cands.get(p) is not None:
                        got = attempt(f"L{li}.M{m}.{p}", lambda: cands[p](x, ids, w))
                    elif p in ("fi", "fi-ckpt") and cands.get("fi") is not None:
                        fi = cands["fi"]
                        if p == "fi":
                            fi.set_scales(nq.global_scale(float(x.abs().max())), nq.global_scale(hmax))
                        else:
                            ga, da = fi.ckpt_a
                            if ga is None or da is None:
                                continue
                            fi.set_scales(1.0 / float(ga.max()), 1.0 / float(da.max()))
                        got = attempt(f"L{li}.M{m}.{p}", lambda: fi(x, ids, w))
                    if got is None:
                        continue
                    torch.cuda.synchronize()
                    met = mc.metrics(got.float(), refo)
                    entry[p] = met
                    res[(p, m)] = met
                    log(f"    {p:8s} rel {met['rel_fro']:.4e}  max|d| {met['max_abs']:.4e}  cos {met['cosine']:.8f}")
                report["accuracy"][f"layer{li}"][f"M{m}"] = entry
                save()
                del refo
            print(f"\n  G-A layer {li}: rel_fro vs fp32 ref, ratio to TensorFold (PASS <= 1.5x), M = {acc_ms}")
            gate_row(res, acc_ms, [p for p in paths])
            print(flush=True)

        if not args.no_speed and li == args.speed_layer:
            sp = {}
            for m in speed_ms:
                x = mc.activations(m, seed=2000 + m)
                ids, w = mc.route(x, layer.router)
                row = {"load": mc.load_stats(ids), "gflop": mc.flops(m) / 1e9}
                for p in args.speed_paths.split(","):
                    c = cands.get("fi" if p in ("fi", "fi-ckpt") else p)
                    if c is None:
                        continue
                    if p == "tf":
                        picks, wts = c.prepare(x, ids, w)
                        t = attempt(f"S.M{m}.tf", lambda: mc.time_cuda(lambda: c(x, picks, wts), args.reps))
                        k = attempt(f"S.M{m}.tf_kernels", lambda: mc.time_cuda(lambda: c.kernels(x, picks),
                                                                                 args.reps))
                        if k is not None:
                            k["tflops"] = mc.flops(m) / (k["median_ms"] * 1e-3) / 1e12
                            row["tf_kernels"] = k
                    elif p == "fiw4a16":
                        import fi_w4a16_moe as fw

                        t = attempt(f"S.M{m}.{p}", lambda: fw.best_time(c, x, ids, w, args.reps))
                    elif p.startswith("fi"):
                        c.set_scales(nq.global_scale(float(x.abs().max())), nq.global_scale(64.0))

                        def tuned():
                            try:
                                from flashinfer.autotuner import autotune

                                with autotune(True):
                                    c(x, ids, w)
                            except ImportError:
                                c(x, ids, w)
                            return mc.time_cuda(lambda: c(x, ids, w), args.reps)

                        t = attempt(f"S.M{m}.{p}", tuned)
                    else:
                        t = attempt(f"S.M{m}.{p}", lambda: mc.time_cuda(lambda: c(x, ids, w), args.reps))
                    if t is not None:
                        t["tflops"] = mc.flops(m) / (t["median_ms"] * 1e-3) / 1e12
                        row[p] = t
                        log(f"  speed M={m} {p:6s} {t['median_ms']:9.3f} ms  {t['tflops']:7.2f} TFLOPS")
                if "tf_kernels" in row:
                    log(f"  speed M={m} tf kernels only {row['tf_kernels']['median_ms']:9.3f} ms "
                        f"{row['tf_kernels']['tflops']:7.2f} TFLOPS")
                sp[f"M{m}"] = row
                report["speed"] = sp
                save()
        del cands, ref, layer
        gc.collect()
        torch.cuda.empty_cache()
    save()
    log("done; errors:", list(report["errors"]))


if __name__ == "__main__":
    main()
