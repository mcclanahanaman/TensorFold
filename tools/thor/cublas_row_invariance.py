"""Is a cuBLAS / cuBLASLt bf16 GEMM (fp32 sums) row-invariant at Flash Next's dense prompt shapes?

    python tools/thor/cublas_row_invariance.py OUT.json

A row's output must not depend on the call's row count M or where its rows start: TensorFold's prompt kernels
guarantee it, and "resend == fresh" needs it. Per BLAS backend and shape: x[:8192] @ W.T once, then x[:m] @ W.T for
many m and a shifted block, compared bit for bit; each one's time at M=2048. Then the cost of dequantizing each MXFP8
projection to bf16 once a prompt pass (a fused Triton kernel), the alternative to holding bf16 copies.
"""
import json
import statistics
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
SHAPES = {"gdn_in": (16480, 2560), "gdn_out": (2560, 6144), "attn_qkv": (13952, 2560), "attn_o": (2560, 6144),
          "hc_down": (324, 10240), "hc_up": (10240, 320), "shared_gu": (1280, 2560), "shared_down": (2560, 640),
          "ple_key": (10240, 2560), "mtp_fc": (2560, 2560)}
MS = (1, 2, 3, 7, 16, 17, 31, 64, 100, 127, 128, 129, 255, 256, 500, 512, 1000, 1024, 1500, 2047, 2048, 4096)


def timed(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    r = []
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        r.append(s.elapsed_time(e) / iters)
    return statistics.median(r)


backends = {}
for lib in ("cublas", "cublaslt"):
    try:
        torch.backends.cuda.preferred_blas_library(lib)
        backends[lib] = lambda a, b: a @ b.t()
    except Exception as exc:  # noqa: BLE001
        print(f"[rowinv] {lib}: {exc!r}")
try:
    import flashinfer
    mm = getattr(flashinfer, "mm_bf16", None) or getattr(getattr(flashinfer, "gemm", None), "mm_bf16", None)
    print(f"[rowinv] flashinfer mm_bf16: {mm}")
except Exception as exc:  # noqa: BLE001
    mm = None
    print(f"[rowinv] flashinfer import: {exc!r}")
t_end = __import__("time").time() + 3.0
z = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
while __import__("time").time() < t_end:
    z @ z
out = {}
g = torch.Generator(device="cuda").manual_seed(0)
for name, (n, k) in SHAPES.items():
    x = torch.randn(8192, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * k ** -0.5).to(torch.bfloat16)
    for lib in ("cublas", "cublaslt"):
        if lib not in backends:
            continue
        torch.backends.cuda.preferred_blas_library(lib)
        ref = x @ w.t()
        bad = []
        for m in MS:
            if not torch.equal(x[:m] @ w.t(), ref[:m]):
                bad.append(m)
        if not torch.equal(x[5:5 + 2048] @ w.t(), ref[5:5 + 2048]):
            bad.append("shift5")
        ms = timed(lambda: x[:2048] @ w.t())
        out[f"{name}/{lib}"] = {"n": n, "k": k, "not_invariant_at": bad, "ms_2048": ms,
                                "tflops": 2 * 2048 * n * k / ms / 1e9}
        print(f"[rowinv] {name} [{n} x {k}] {lib}: {ms:.3f} ms at M=2048 ({2 * 2048 * n * k / ms / 1e9:.0f} TF); "
              + ("row-invariant over M 1..4096 and a shifted block" if not bad else f"bits differ at M = {bad}"),
              flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)

# MXFP8 weights dequantized to bf16 once a prompt pass (a fused Triton kernel): its cost
import triton
import triton.language as tl


@triton.jit
def _dq(W, S, OUT, K: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    w = tl.load(W + r * K + k).to(tl.float32)
    s = tl.load(S + r * (K // 32) + k // 32).to(tl.float32)
    tl.store(OUT + r * K + k, (w * tl.exp2(s - 127.0)).to(tl.bfloat16))


dq = {}
for name in ("gdn_in", "gdn_out", "attn_qkv", "attn_o", "shared_gu", "shared_down"):
    n, k = SHAPES[name]
    w8 = (torch.randn(n, k, device="cuda") * 8).to(torch.float8_e4m3fn)
    sc = torch.randint(110, 130, (n, k // 32), device="cuda", dtype=torch.uint8)
    o = torch.empty((n, k), device="cuda", dtype=torch.bfloat16)
    fn = lambda: _dq[(n, k // 128)](w8, sc, o, K=k, BK=128, num_warps=1)  # noqa: E731
    ms = timed(fn)
    ref = (w8.float() * torch.exp2(sc.float() - 127).repeat_interleave(32, 1)).to(torch.bfloat16)
    ok = bool(torch.equal(o, ref))
    gbs = n * k * (1 + 2 + 1 / 32) / ms / 1e6
    dq[name] = {"ms": ms, "gb_s": gbs, "exact": ok}
    print(f"[dequant] {name} [{n} x {k}]: {ms:.3f} ms, {gbs:.0f} GB/s, exact {ok}", flush=True)
out["dequant"] = dq
json.dump(out, open(sys.argv[1], "w"), indent=1)
