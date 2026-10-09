"""Flash Next's dense MXFP8 projections at decode verification rows on Jetson Thor: qmmf's row tiles (DPUSH-5).

    python tools/thor/qmmf_rows_bench.py OUT.json [--rows 4,8,16,24,32,44,48,64] [--tiles 64,48,32,16] [--iters 20]

Each shape (n, k) runs as nvfp4.linear._matmul runs it (MXFP8 words and e8m0 block scales, K slices from the shape,
the slices meeting through ``part`` and the reduce on Thor) with every row tile ``bm`` (tiles side by side for more
rows); tiles never change a row's bits, which ``--tiles`` checks against the first tile on
the same inputs. Weights are random e4m3 bytes (no NaN codes) and scales near 1; ``layers`` copies of each weight (more
bytes than the L2 holds) run in turn in one CUDA graph; time = CUDA events / launch (median of 3), GB/s = weight and
scale bytes / time.
"""

from __future__ import annotations

import argparse
import json

import torch

SHAPES = {"in_proj": (16480, 2560), "out_proj": (2560, 6144)}


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rows", default="16,24,32,36,40,44,48,56,64")
    ap.add_argument("--tiles", default="64,48,32,16")
    ap.add_argument("--shapes", default="in_proj,out_proj")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    from tensorfold.cuda.kernels import qmm
    from tensorfold.cuda.nvfp4 import linear

    ext = linear._ext()
    gen = torch.Generator(device="cuda").manual_seed(7)
    a_ = torch.randn((2048, 2048), device="cuda")
    for _ in range(200):
        a_ = (a_ @ a_).clamp_(-1, 1)
    result = {"cases": []}
    for name in a.shapes.split(","):
        n, k = SHAPES[name]
        npad = -(-n // 64) * 64
        sk = qmm.split_k(n, k)
        meet = sk > 8 or (sk > 1 and linear.thor_slices(0))
        ws, bss = [], []
        for _ in range(a.layers):
            w = torch.randint(0, 126, (npad * k,), generator=gen, device="cuda", dtype=torch.int32)
            w = (w | (torch.randint(0, 2, w.shape, generator=gen, device="cuda", dtype=torch.int32) << 7)).to(torch.uint8)
            ws.append(w)
            bss.append(torch.randint(124, 131, (npad * (k // 64) * 2,), generator=gen, device="cuda",
                                     dtype=torch.int32).to(torch.uint8))
        wbytes = npad * k + npad * (k // 64) * 2
        for m in ints(a.rows):
            x = (torch.randn((m, k), generator=gen, device="cuda") * 0.5).bfloat16()
            row = {"shape": name, "n": n, "k": k, "sk": sk, "rows": m, "us": {}, "GBps": {}, "same_bits": {}}
            ref = None
            for bm in ints(a.tiles):
                ys = [torch.empty((m, n), dtype=torch.bfloat16, device="cuda") for _ in range(a.layers)]
                parts = [torch.empty((sk, m, n), dtype=torch.float32, device="cuda") if meet else None
                         for _ in range(a.layers)]

                def call(i: int) -> None:
                    ext.qmmf(x, ws[i], bss[i], 1.0, ys[i], parts[i], linear.MXFP8, n, sk, npad, bm, False)

                for i in range(a.layers):
                    call(i)
                torch.cuda.synchronize()
                if ref is None:
                    ref = ys[0].clone()
                row["same_bits"][bm] = bool(torch.equal(ys[0].view(torch.int16), ref.view(torch.int16)))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for i in range(a.layers):
                        call(i)
                graph.replay()
                torch.cuda.synchronize()
                times = []
                for _ in range(3):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(a.iters):
                        graph.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    times.append(e0.elapsed_time(e1) * 1e3 / (a.iters * a.layers))
                us = sorted(times)[1]
                row["us"][bm], row["GBps"][bm] = round(us, 2), round(wbytes / us / 1e3, 1)
            result["cases"].append(row)
            print(f"[qmmf] {name} m{m} sk{sk}: " + " ".join(
                f"bm{b} {row['us'][b]:.1f}us {row['GBps'][b]:.0f}GB/s{'' if row['same_bits'][b] else ' !BITS'}"
                for b in row["us"]), flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[qmmf] done: {len(result['cases'])} cases, bit mismatches "
          f"{sum(not v for c in result['cases'] for v in c['same_bits'].values())}")


if __name__ == "__main__":
    main()
