"""QSA tensor-core proxies on Jetson Thor, per layer and 2048-row prompt pass:

    python tools/thor/qsa_proxy_bench.py OUT.json

  scores  the indexer's block scores as one cuBLAS bf16 GEMM: 2048 rows x 4 index heads x 128 dims against
          context/4 pooled keys (TensorFold's _scores kernel does this one row a program)
  attn    dense GQA attention at the QSA core's FLOPs: 24 query heads over 2 KV heads, head_dim 256, 2048 rows x 2048
          keys (a long-context row reads its 2048-key budget), through each SDPA backend this torch has on sm_110
"""
import json
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def spin(seconds=2.0):
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    end = time.time() + seconds
    while time.time() < end:
        x @ x
    torch.cuda.synchronize()


def timed(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    spin(0.5)
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


out = {}
spin(3.0)
for ctx in (8192, 32768, 131072):
    nb = ctx // 4
    q = torch.randn(2048 * 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(nb, 128, device="cuda", dtype=torch.bfloat16)
    ms = timed(lambda: q @ k.t())
    fl = 2.0 * 2048 * 4 * nb * 128
    out[f"scores_ctx{ctx}"] = {"ms_bf16_out": ms, "tflops": fl / ms / 1e9, "out_mb": 2048 * 4 * nb * 2 / 1e6}
    print(f"[qsa] scores at context {ctx} ({nb} blocks): cuBLAS bf16 {ms:.3f} ms ({fl / ms / 1e9:.0f} TF)", flush=True)
q = torch.randn(1, 24, 2048, 256, device="cuda", dtype=torch.bfloat16)
k = torch.randn(1, 2, 2048, 256, device="cuda", dtype=torch.bfloat16)
v = torch.randn(1, 2, 2048, 256, device="cuda", dtype=torch.bfloat16)
fl = 4.0 * 24 * 2048 * 2048 * 256
for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
    be = getattr(SDPBackend, name, None)
    if be is None:
        continue
    try:
        with sdpa_kernel(be):
            fn = lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)  # noqa: E731
            ms = timed(fn, iters=5)
        out[f"attn_{name}"] = {"ms": ms, "tflops": fl / ms / 1e9}
        print(f"[qsa] dense GQA attention 2048 x 2048, hd 256, {name}: {ms:.3f} ms ({fl / ms / 1e9:.0f} TF)",
              flush=True)
    except Exception as exc:  # noqa: BLE001
        out[f"attn_{name}"] = {"error": repr(exc)[:300]}
        print(f"[qsa] {name}: {repr(exc)[:200]}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
