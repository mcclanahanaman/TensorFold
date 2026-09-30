"""vLLM's fused Marlin MoE (NVFP4 weights x bf16 activations, W4A16 on mma.sync) on Flash Next's real experts: the
upstream weight-only path vLLM falls back to on Thor. Run in the runner's ``vllm`` image:

    PYTHONPATH=/src/src:/src/tools/thor python tools/thor/marlin_moe_probe.py --model /hf/... --out /out/marlin.json

It prints the vLLM API it finds, prepares one layer's experts the way vLLM's ModelOpt NVFP4 MoE does for Marlin
(``prepare_moe_fp4_layer_for_marlin``), and measures accuracy against the same fp32 reference as the Phase A harness
(format.dequant weights, fp32 SwiGLU and sums) and speed with CUDA events.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import moe_common as mc  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def log(*a):
    print(f"[{mc.stamp()}]", *a, flush=True)


def sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except Exception as exc:  # noqa: BLE001
        return f"<{exc!r}>"


class FakeLayer(torch.nn.Module):
    pass


def build(layer: mc.Layer, dtype=torch.bfloat16):
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mu

    lay = FakeLayer()
    lay.num_experts = mc.E
    lay.local_num_experts = mc.E
    lay.global_num_experts = mc.E
    lay.hidden_size = mc.D
    lay.intermediate_size_per_partition = mc.NI
    lay.params_dtype = dtype
    lay.moe_config = None
    p = torch.nn.Parameter
    # vLLM's w13 = [w1 (gate); w3 (up)] rows, one weight_scale_2 an expert (it takes gate's: [:, 0])
    lay.w13_weight = p(torch.cat([layer.gate.w, layer.up.w], 1).contiguous(), requires_grad=False)
    lay.w13_weight_scale = p(torch.cat([layer.gate.s, layer.up.s], 1).contiguous(), requires_grad=False)
    lay.w13_weight_scale_2 = p(layer.gate.s2.clone(), requires_grad=False)
    lay.w2_weight = p(layer.down.w.contiguous(), requires_grad=False)
    lay.w2_weight_scale = p(layer.down.s.contiguous(), requires_grad=False)
    lay.w2_weight_scale_2 = p(layer.down.s2.clone(), requires_grad=False)
    kw = {}
    params = inspect.signature(mu.prepare_moe_fp4_layer_for_marlin).parameters
    if "input_dtype" in params:
        kw["input_dtype"] = dtype
    mu.prepare_moe_fp4_layer_for_marlin(lay, **kw)
    return lay


def run_marlin(lay, x, ids, w):
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    params = inspect.signature(fused_marlin_moe).parameters
    kw = dict(hidden_states=x, w1=lay.w13_weight, w2=lay.w2_weight, w1_scale=lay.w13_weight_scale,
              w2_scale=lay.w2_weight_scale, topk_weights=w, topk_ids=ids,
              global_scale1=lay.w13_weight_scale_2, global_scale2=lay.w2_weight_scale_2,
              quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=mc.E, workspace=lay.workspace)
    if "bias1" in params:
        kw["bias1"] = None
        kw["bias2"] = None
    if "gating_output" in params:
        kw["gating_output"] = None
    kw = {k: v for k, v in kw.items() if k in params}
    return fused_marlin_moe(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/out/marlin.json")
    ap.add_argument("--layers", default="0,23,47")
    ap.add_argument("--acc-ms", default="128,512,2048,4096")
    ap.add_argument("--speed-ms", default="512,2048,4096,8192")
    ap.add_argument("--speed-layer", type=int, default=23)
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()
    rep = {"accuracy": {}, "speed": {}, "errors": {}}
    out = Path(args.out)

    def save():
        out.write_text(json.dumps(rep, indent=1, default=str))

    import vllm

    rep["vllm"] = vllm.__version__
    rep["torch"] = torch.__version__
    try:
        from vllm.model_executor.layers.fused_moe import fused_marlin_moe as fmm
        from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4 as mu

        rep["api"] = {"fused_marlin_moe": sig(fmm.fused_marlin_moe),
                      "prepare_moe_fp4_layer_for_marlin": sig(mu.prepare_moe_fp4_layer_for_marlin),
                      "marlin_utils_fp4": [n for n in dir(mu) if not n.startswith("_")]}
        try:
            rep["api"]["prepare_source"] = inspect.getsource(mu.prepare_moe_fp4_layer_for_marlin)[:6000]
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        rep["errors"]["api"] = f"{exc!r}\n{traceback.format_exc()[-3000:]}"
    save()
    log("vllm", rep["vllm"], json.dumps(rep.get("api", {}), default=str)[:3000])

    ck = mc.Checkpoint(args.model)
    layers = [int(v) for v in args.layers.split(",")]
    if args.speed_layer not in layers:
        layers.append(args.speed_layer)
    for li in layers:
        layer = ck.layer(li)
        try:
            t0 = time.time()
            lay = build(layer)
            log(f"layer {li}: Marlin prep in {time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            rep["errors"][f"L{li}.build"] = f"{exc!r}\n{traceback.format_exc()[-3000:]}"
            log("build failed", repr(exc))
            save()
            break
        entry = {}
        if li in [int(v) for v in args.layers.split(",")]:
            ref = mc.reference_weights(layer)
            for m in [int(v) for v in args.acc_ms.split(",")]:
                x = mc.activations(m, seed=1000 + m)
                ids, w = mc.route(x, layer.router)
                refo = mc.reference(x, ids, w, ref)
                try:
                    got = run_marlin(lay, x, ids, w)
                    torch.cuda.synchronize()
                    met = mc.metrics(got.float(), refo)
                    entry[f"M{m}"] = met
                    log(f"  L{li} M={m} marlin rel {met['rel_fro']:.4e} max|d| {met['max_abs']:.4e} "
                        f"cos {met['cosine']:.8f}")
                except Exception as exc:  # noqa: BLE001
                    rep["errors"][f"L{li}.M{m}"] = f"{exc!r}\n{traceback.format_exc()[-3000:]}"
                    log("  run failed", repr(exc))
                    save()
                    break
            del ref
        rep["accuracy"][f"layer{li}"] = entry
        save()
        if li == args.speed_layer:
            for m in [int(v) for v in args.speed_ms.split(",")]:
                x = mc.activations(m, seed=2000 + m)
                ids, w = mc.route(x, layer.router)
                try:
                    t = mc.time_cuda(lambda: run_marlin(lay, x, ids, w), args.reps)
                    t["tflops"] = mc.flops(m) / (t["median_ms"] * 1e-3) / 1e12
                    rep["speed"][f"M{m}"] = t
                    log(f"  speed M={m} marlin {t['median_ms']:.3f} ms {t['tflops']:.2f} TFLOPS")
                except Exception as exc:  # noqa: BLE001
                    rep["errors"][f"S.M{m}"] = f"{exc!r}\n{traceback.format_exc()[-3000:]}"
                    log("  speed failed", repr(exc))
                save()
        del lay, layer
        torch.cuda.empty_cache()
    save()
    log("done; errors:", list(rep["errors"]))


if __name__ == "__main__":
    main()
