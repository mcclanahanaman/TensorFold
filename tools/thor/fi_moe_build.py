"""Build (JIT) and smoke-test FlashInfer's CUTLASS fused MoE (NVFP4 x NVFP4) on this GPU, recording what happens.

    FLASHINFER_WORKSPACE_BASE=/cache/fi FLASHINFER_CUDA_ARCH_LIST=11.0a MAX_JOBS=12 \
        python tools/thor/fi_moe_build.py --out /out

Writes <out>/fi_build.json (versions, arch flags, whether a prebuilt module was found, build seconds, the error tail
when it fails) and, on success, a small NVFP4 MoE call checked against a torch emulation of the same W4A4 math.
The container has no network: a module that wants to download cubins fails here, and the error says which file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nvfp4_quant as nq  # noqa: E402


def smoke(fused_moe) -> dict:
    """E=8 experts, top-2, K=256, N=128: the fused call vs the same NVFP4 math in fp32."""

    g = torch.Generator().manual_seed(0)
    e, k, n, m, top = 8, 256, 128, 64, 2
    w1 = (torch.randn(e, 2 * n, k, generator=g) * 0.05).cuda()     # [up; gate] rows
    w2 = (torch.randn(e, k, n, generator=g) * 0.05).cuda()
    q1 = [nq.quantize(w1[i], nq.global_scale(float(w1[i].abs().max()))) for i in range(e)]
    q2 = [nq.quantize(w2[i], nq.global_scale(float(w2[i].abs().max()))) for i in range(e)]
    gs1 = torch.tensor([nq.global_scale(float(w1[i].abs().max())) for i in range(e)], device="cuda")
    gs2 = torch.tensor([nq.global_scale(float(w2[i].abs().max())) for i in range(e)], device="cuda")
    fc1 = torch.stack([p for p, _, _ in q1]).view(torch.long)
    fc2 = torch.stack([p for p, _, _ in q2]).view(torch.long)
    sf1 = nq.swizzle(torch.stack([s for _, s, _ in q1])).view(e, 2 * n, k // 16).view(torch.int32)
    sf2 = nq.swizzle(torch.stack([s for _, s, _ in q2])).view(e, k, n // 16).view(torch.int32)
    x = (torch.randn(m, k, generator=g) * 0.5).to(torch.bfloat16).cuda()
    ids = torch.stack([torch.randperm(e, generator=g)[:top] for _ in range(m)]).to(torch.int32).cuda()
    wts = torch.softmax(torch.randn(m, top, generator=g), -1).cuda()
    a1 = nq.global_scale(float(x.abs().max()))
    w1d = torch.stack([d for _, _, d in q1])
    w2d = torch.stack([d for _, _, d in q2])
    # emulation: x -> NVFP4 (a1), h -> NVFP4 (a2 from the data), exact dequantized weights, fp32 sums
    _, _, xq = nq.quantize(x.float(), a1)
    h = torch.empty(m, top, n, device="cuda")
    for r in range(m):
        for s in range(top):
            o = xq[r] @ w1d[ids[r, s]].t()
            h[r, s] = torch.nn.functional.silu(o[n:]) * o[:n]
    a2 = nq.global_scale(float(h.abs().max()))
    _, _, hq = nq.quantize(h.view(-1, n).to(torch.bfloat16).float(), a2)
    hq = hq.view(m, top, n)
    ref = torch.zeros(m, k, device="cuda")
    for r in range(m):
        for s in range(top):
            ref[r] += wts[r, s] * (hq[r, s] @ w2d[ids[r, s]].t())
    scales = [torch.tensor(a1, device="cuda"), sf1, (1.0 / (a1 * gs1)).float(), torch.tensor(a2, device="cuda"),
              sf2, (1.0 / (a2 * gs2)).float()]
    out = fused_moe(x, ids, wts, fc1, fc2, torch.bfloat16, quant_scales=scales)
    out = out[0] if isinstance(out, (list, tuple)) else out
    torch.cuda.synchronize()
    rel = float((out.float() - ref).norm() / ref.norm())
    swapped = None
    if rel > 0.05:   # the gated half the other way round?
        h2 = torch.empty(m, top, n, device="cuda")
        for r in range(m):
            for s in range(top):
                o = xq[r] @ w1d[ids[r, s]].t()
                h2[r, s] = torch.nn.functional.silu(o[:n]) * o[n:]
        swapped = "see rel_if_gate_first"
        a2b = nq.global_scale(float(h2.abs().max()))
        _, _, hq2 = nq.quantize(h2.view(-1, n).to(torch.bfloat16).float(), a2b)
        ref2 = torch.zeros(m, k, device="cuda")
        for r in range(m):
            for s in range(top):
                ref2[r] += wts[r, s] * (hq2.view(m, top, n)[r, s] @ w2d[ids[r, s]].t())
        swapped = float((out.float() - ref2).norm() / ref2.norm())
    return {"rel_vs_emulation": rel, "rel_if_gate_first": swapped, "out_finite": bool(torch.isfinite(out).all())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    rep: dict = {"env": {k: os.environ.get(k) for k in ("FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CUDA_ARCH_LIST",
                                                        "MAX_JOBS", "CUDA_HOME")}}
    path = Path(args.out) / "fi_build.json"

    def save():
        path.write_text(json.dumps(rep, indent=1, default=str))

    import flashinfer
    from flashinfer.compilation_context import CompilationContext
    from flashinfer.fused_moe import core
    from flashinfer.jit import env as jit_env

    rep["flashinfer"] = flashinfer.__version__
    rep["torch"] = torch.__version__
    rep["cc"] = list(torch.cuda.get_device_capability())
    ctx = CompilationContext()
    rep["target_archs"] = sorted(map(list, ctx.TARGET_CUDA_ARCHS))
    rep["nvcc_flags_sm100_module"] = ctx.get_nvcc_flags_list(supported_major_versions=[10, 11],
                                                              map_sm107_to_100f=True)
    rep["jit_dir"] = str(jit_env.FLASHINFER_JIT_DIR)
    from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm100_module

    spec = gen_cutlass_fused_moe_sm100_module()
    rep["spec_name"] = spec.name
    rep["sources"] = len(spec.sources) if hasattr(spec, "sources") else None
    try:
        rep["prebuilt_or_cached"] = spec.try_load() is not None
    except Exception as exc:  # noqa: BLE001
        rep["prebuilt_or_cached"] = f"try_load raised {exc!r}"
    save()
    print("flashinfer", flashinfer.__version__, "archs", rep["target_archs"], "flags", rep["nvcc_flags_sm100_module"],
          "cached", rep["prebuilt_or_cached"], "sources", rep["sources"], flush=True)
    t0 = time.time()
    try:
        major, minor = torch.cuda.get_device_capability()
        core.get_cutlass_fused_moe_module(f"{major * 10 + minor}")
        rep["build"] = "ok"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        rep["build"] = "failed"
        rep["error_head"] = msg[:4000]
        tail = msg[-20000:]
        (Path(args.out) / "fi_build_error.txt").write_text(msg[-400000:])
        errs = [ln for ln in msg.splitlines() if "error" in ln.lower()][:80]
        rep["error_lines"] = errs
        rep["traceback"] = traceback.format_exc()[-4000:]
        print("BUILD FAILED:", "\n".join(errs[:40]), "\n...", tail[-3000:], flush=True)
    rep["build_seconds"] = round(time.time() - t0, 1)
    save()
    print("build", rep["build"], "in", rep["build_seconds"], "s", flush=True)
    if rep["build"] == "ok" and not args.no_smoke:
        try:
            rep["smoke"] = smoke(core.cutlass_fused_moe)
        except Exception as exc:  # noqa: BLE001
            rep["smoke"] = f"failed: {exc!r}\n{traceback.format_exc()[-4000:]}"
        print("smoke", rep["smoke"], flush=True)
        save()


if __name__ == "__main__":
    main()
